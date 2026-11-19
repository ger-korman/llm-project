"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes
from assistant.infra.llm import LLMClient, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

_WITH_UNIT = re.compile(r"(?<![\w-])(\d[\d\s  ]*\d|\d+)\s*(?:руб|₽|шт)", re.IGNORECASE)


def _serialized(results: list[dict]) -> str:
    """Стабильно сериализует наблюдения для кеш-ключей и проверок происхождения фактов."""
    return json.dumps(results, ensure_ascii=False, sort_keys=True)


def _grounded(answer: str, question: str, results: list[dict]) -> bool:
    """Проверяет по FR-10, что коды и числа с единицами подтверждены диалогом."""
    # Код допустим, только если пользователь назвал его сам или его вернул выполненный инструмент
    evidence = question + " " + _serialized(results)
    allowed_codes = set(find_codes(evidence))
    if any(code not in allowed_codes for code in find_codes(answer)):
        return False
    evidence_numbers = set(re.findall(r"\d+", _serialized(results)))
    # Проверяются именно числа с единицами «руб.» и «шт.», то есть потенциальные цена и остаток
    for match in _WITH_UNIT.finditer(answer):
        number = re.sub(r"\D", "", match.group(1))
        if number not in evidence_numbers:
            return False
    return True


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа

    # FR-7: системная инструкция задаёт политику, а фактические ограничения дублируются кодом ниже
    system = (
        "Ты ассистент сервисной службы. Используй инструменты для фактов из документов и ERP. "
        "Не выдумывай номера деталей, цены и остатки. Различай отсутствие данных и нулевой остаток. "
        "Заказ предлагай только с явно названным пользователем количеством."
    )
    messages: list[dict] = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    steps: list[ToolStep] = []
    results: list[dict] = []
    sources: set[str] = set()
    # Кеш действует в пределах одного диалога и исключает повторное выполнение одинакового вызова
    cache: dict[str, dict] = {}
    tool_calls_done = 0
    response = None

    while True:
        # Один проход цикла: решение модели -> вызовы инструментов -> наблюдения в истории сообщений
        response = llm.chat(messages, tools=TOOL_SCHEMAS)
        messages.append(response.as_message())
        if not response.tool_calls:
            break
        for call in response.tool_calls:
            if tool_calls_done >= max_steps:
                # Лимит считает попытки вызова, включая отклонённые, чтобы модель не зациклилась
                messages.append(
                    tool_message(call, {"error": "лимит шагов исчерпан; дай ответ по уже полученным данным"})
                )
                continue
            tool_calls_done += 1
            args = call.arguments
            error = None
            result = None
            if call.name not in tools:
                # Ошибка возвращается модели как tool message, после чего она может выбрать другой инструмент
                error = f"неизвестный инструмент: {call.name}"
            elif not isinstance(args, dict) or "_raw" in args:
                error = "аргументы инструмента не удалось разобрать"
            else:
                part_id = args.get("part_id")
                known_text = question + " " + _serialized(results)
                # FR-7: запрещаем отправлять в ERP номер детали, которого ещё не было в доказательствах
                if isinstance(part_id, str) and part_id not in find_codes(known_text):
                    error = "номер детали не подтверждён; сначала вызови search_knowledge"
                elif call.name == "create_service_order" and str(args.get("qty")) not in re.findall(r"\d+", question):
                    # Количество нельзя угадывать даже при confirm=True: его должен назвать пользователь
                    error = "пользователь не назвал количество; задай уточняющий вопрос"
                elif call.name == "create_service_order" and not confirm:
                    # FR-9: возвращаем описание ожидающего действия, не вызывая изменяющий инструмент
                    return AgentResult(
                        "Требуется подтверждение создания заказа.",
                        steps,
                        sorted(sources),
                        needs_confirmation=True,
                        pending_action={"tool": call.name, "args": args},
                    )
                else:
                    key = json.dumps([call.name, args], ensure_ascii=False, sort_keys=True)
                    if key in cache:
                        result = cache[key]
                    else:
                        # FR-8: временно недоступная ERP получает всего три попытки; прочие ошибки
                        # уже превращены инструментами в обычный словарь result
                        for attempt in range(3):
                            try:
                                result = tools[call.name](**args)
                                cache[key] = result
                                break
                            except ErpUnavailable as exc:
                                if attempt == 2:
                                    error = f"данные ERP получить не удалось: {exc}"
            step = ToolStep(call.name, args, result, error)
            # Каждый разрешённый или отклонённый вызов остаётся в трассе для отчёта и аудита
            steps.append(step)
            payload = (
                result if error is None and result is not None else {"error": error or "инструмент не вернул результат"}
            )
            messages.append(tool_message(call, payload))
            if result is not None:
                # Источники GraphRAG накапливаются отдельно от наблюдений ERP
                results.append(result)
                sources.update(result.get("sources", []))
        if tool_calls_done >= max_steps:
            # После исчерпания лимита инструменты больше не предлагаются: модель обязана подвести итог
            messages.append(
                {"role": "user", "content": "Лимит вызовов инструментов исчерпан. Ответь по уже полученным данным."}
            )
            response = llm.chat(messages)
            messages.append(response.as_message())
            break

    answer = response.content.strip() if response else ""
    refused = any(result.get("refused") for result in results) and "не найден" in answer.lower()
    if not _grounded(answer, question, results):
        # FR-10: одна попытка самокоррекции, затем безопасный отказ без неподтверждённых значений.
        messages.append(
            {
                "role": "user",
                "content": (
                    "Исправь ответ: используй только коды, цены и остатки из вопроса и результатов инструментов."
                ),
            }
        )
        corrected = llm.chat(messages, tools=TOOL_SCHEMAS).content.strip()
        if not _grounded(corrected, question, results):
            return AgentResult("В базе знаний не найдено.", steps, sorted(sources), refused=True)
        answer = corrected
    if not answer:
        answer = "Работа прервана: ответ сформировать не удалось."
    return AgentResult(answer, steps, sorted(sources), refused=refused)
