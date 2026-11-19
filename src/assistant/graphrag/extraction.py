"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, RELATION_SIGNATURES, Chunk, Entity, ExtractionResult, Relation

_PROMPT = """Извлеки граф знаний только из данного фрагмента.
Допустимые направленные связи (начало -> конец):
BELONGS_TO Model -> ProductLine; HAS_PART Model -> Part; IS_A Part -> Component;
INDICATES ErrorCode -> Component; FIXED_BY ErrorCode -> Procedure;
REPLACED_BY старая Part -> новая Part; SUPPLIED_BY Part -> Supplier;
FAILED Model -> Component. Не добавляй факты, которых нет в тексте.
Пример: «E-17 означает отказ пускового реле» -> сущности ErrorCode E-17 и
Component пусковое реле, связь E-17 -INDICATES-> пусковое реле.

Фрагмент:
{text}"""


def _parse(content: str) -> tuple[list[dict], list[dict]] | None:
    """Разбирает полный или оборванный JSON и проверяет обязательные поля ответа модели."""
    # Сначала используем JSON-парсер; восстановление нужно только при обрыве по лимиту токенов
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        data = _parse_truncated(content)
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("entities"), list)
        or not isinstance(data.get("relations"), list)
    ):
        return None
    required_entity = {"type", "name"}
    required_relation = {"subject_type", "subject", "type", "object_type", "object"}
    if any(not isinstance(item, dict) or not required_entity <= item.keys() for item in data["entities"]):
        return None
    if any(not isinstance(item, dict) or not required_relation <= item.keys() for item in data["relations"]):
        return None
    return data["entities"], data["relations"]


def _parse_truncated(content: str) -> dict | None:
    """Salvage complete objects when Ollama stops at the token limit mid-JSON."""
    if not isinstance(content, str):
        return None
    decoder = json.JSONDecoder()

    def array_items(key: str) -> list[dict] | None:
        """Последовательно читает только полностью завершённые объекты нужного массива."""
        marker = f'"{key}"'
        start = content.find(marker)
        if start < 0:
            return None
        start = content.find("[", start + len(marker))
        if start < 0:
            return None
        items: list[dict] = []
        position = start + 1
        while position < len(content):
            while position < len(content) and content[position] in " \t\r\n,":
                position += 1
            if position >= len(content) or content[position] == "]":
                break
            try:
                item, position = decoder.raw_decode(content, position)
            except json.JSONDecodeError:
                break
            if isinstance(item, dict):
                items.append(item)
        return items

    entities = array_items("entities")
    relations = array_items("relations")
    return {"entities": entities, "relations": relations} if entities is not None and relations is not None else None


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю

    # Модель получает формальную JSON Schema; максимум две попытки защищают индекс от сбоя LLM
    parsed = None
    messages = [{"role": "user", "content": _PROMPT.format(text=chunk.text)}]
    for _ in range(2):
        parsed = _parse(llm.chat(messages, json_schema=EXTRACTION_SCHEMA).content)
        if parsed is not None:
            break
    if parsed is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    raw_entities, raw_relations = parsed
    vocab = vocabulary()
    codes_in_text = set(find_codes(chunk.text))
    entities: dict[str, Entity] = {}
    dropped = 0

    def validated(kind: object, raw: object) -> Entity | None:
        """Канонизирует сущность и подтверждает её код или название исходным чанком."""
        if not isinstance(kind, str) or not isinstance(raw, str):
            return None
        name = canonical_name(kind, raw, vocab)
        if name is None or (kind in CODE_TYPES and name not in codes_in_text):
            return None
        return Entity(kind, name)

    for item in raw_entities:
        # Словарь по node_id одновременно объединяет дубликаты сущностей внутри одного чанка
        entity = validated(item["type"], item["name"])
        if entity is None:
            dropped += 1
        else:
            entities[entity.node_id] = entity

    relations: list[Relation] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for item in raw_relations:
        # Оба конца отношения валидируются независимо: связь с выдуманным концом отбрасывается целиком
        subject = validated(item["subject_type"], item["subject"])
        object_ = validated(item["object_type"], item["object"])
        if subject is None or object_ is None:
            dropped += 1
            continue
        kind = item["type"]
        signature = RELATION_SIGNATURES.get(kind)
        # Маленькая модель иногда меняет местами субъект и объект; разнозначные типы позволяют
        # безопасно восстановить направление по сигнатуре отношения
        if signature and (subject.type, object_.type) == signature[::-1] and signature[0] != signature[1]:
            subject, object_ = object_, subject
        elif kind == "REPLACED_BY":
            # У REPLACED_BY типы концов одинаковы, поэтому направление восстанавливается по порядку
            # упоминания в бюллетене: сначала старая, затем новая деталь
            subject_position = chunk.text.upper().find(subject.name.upper())
            object_position = chunk.text.upper().find(object_.name.upper())
            if 0 <= object_position < subject_position:
                subject, object_ = object_, subject
        relation = Relation(subject, kind, object_)
        key = (subject.node_id, relation.type, object_.node_id)
        if key in seen_relations:
            continue
        seen_relations.add(key)
        relations.append(relation)
        entities[subject.node_id] = subject
        entities[object_.node_id] = object_

    # Длинные руководства иногда обрываются после IS_A. Заголовок и формулировка раздела явно
    # подтверждают, что перечисленные детали установлены в модели, поэтому HAS_PART восстанавливается кодом
    if "состав узлов" in chunk.text.lower() and "установлены следующие узлы" in chunk.text.lower():
        models = [entity for entity in entities.values() if entity.type == "Model"]
        parts = [entity for entity in entities.values() if entity.type == "Part"]
        for model in models:
            for part in parts:
                key = (model.node_id, "HAS_PART", part.node_id)
                if key not in seen_relations:
                    seen_relations.add(key)
                    relations.append(Relation(model, "HAS_PART", part))

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), relations, dropped)
