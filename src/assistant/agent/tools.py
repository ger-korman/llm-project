"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]


def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    """Собирает единообразное описание функции в OpenAI-совместимом формате tool calling."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


TOOL_SCHEMAS: list[dict] = [
    # Два инструмента GraphRAG дают агенту конкретный и обзорный режим поиска
    _schema(
        "search_knowledge",
        "Найти факты о конкретной модели, ошибке или детали в графе знаний.",
        {"question": {"type": "string"}},
        ["question"],
    ),
    _schema(
        "summarize_topic",
        "Ответить на обзорный или статистический вопрос по сводкам графа.",
        {"question": {"type": "string"}},
        ["question"],
    ),
    _schema(
        "check_stock",
        "Проверить остаток детали на складе ERP.",
        {"part_id": {"type": "string"}, "warehouse": {"type": "string", "enum": WAREHOUSES}},
        ["part_id", "warehouse"],
    ),
    _schema(
        "get_price",
        "Получить текущую цену детали из ERP.",
        {"part_id": {"type": "string"}},
        ["part_id"],
    ),
    _schema(
        "create_service_order",
        "Создать заказ детали после подтверждения пользователем.",
        {
            "part_id": {"type": "string"},
            "warehouse": {"type": "string", "enum": WAREHOUSES},
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string"},
        },
        ["part_id", "warehouse", "qty", "model"],
    ),
]


_PART = re.compile(r"P-\d{4}[A-Z]?")


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP

    def search_knowledge(question: str) -> dict:
        """Оборачивает local search в сериализуемый результат инструмента."""
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        """Оборачивает global search для обзорных вопросов агента."""
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def valid_part(part_id: object) -> str | None:
        """Проверяет строгий формат номера детали до любого обращения к ERP."""
        return part_id if isinstance(part_id, str) and _PART.fullmatch(part_id) else None

    def erp_error(exc: Exception) -> dict:
        """Сохраняет различие между отсутствием записи и осмысленным отказом ERP."""
        prefix = "данных нет" if isinstance(exc, ErpNotFound) else "запрос отклонён"
        return {"error": f"{prefix}: {exc}"}

    def check_stock(part_id: str, warehouse: str) -> dict:
        """Возвращает остаток, включая корректный нулевой остаток, либо словарь error."""
        if not valid_part(part_id):
            return {"error": "некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "неизвестный склад"}
        try:
            return erp.get_stock(part_id, warehouse)
        except (ErpNotFound, ErpRejected) as exc:
            return erp_error(exc)

    def get_price(part_id: str) -> dict:
        """Возвращает текущую цену и признак discontinued либо словарь error."""
        if not valid_part(part_id):
            return {"error": "некорректный номер детали"}
        try:
            return erp.get_price(part_id)
        except (ErpNotFound, ErpRejected) as exc:
            return erp_error(exc)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        """Проверяет изменяющую операцию и только затем передаёт её клиенту ERP."""
        if not valid_part(part_id):
            return {"error": "некорректный номер детали"}
        if warehouse not in WAREHOUSES:
            return {"error": "неизвестный склад"}
        if not isinstance(qty, int) or isinstance(qty, bool) or qty <= 0:
            return {"error": "количество должно быть положительным целым числом"}
        try:
            return erp.create_order(part_id, warehouse, qty, model)
        except (ErpNotFound, ErpRejected) as exc:
            return erp_error(exc)

    # ErpUnavailable здесь намеренно не перехватывается: три попытки реализует run_agent по FR-8
    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
