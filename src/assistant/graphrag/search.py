"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase


def _refusal() -> Answer:
    """Создаёт единообразный контролируемый отказ по FR-10."""
    return Answer("В базе знаний не найдено.", refused=True)


def _documents(graph: nx.DiGraph, edges: set[tuple[str, str]]) -> list[str]:
    """Преобразует источники выбранных рёбер из chunk_id в уникальные имена документов."""
    chunks = {chunk for source, target in edges for chunk in graph.edges[source, target].get("sources", [])}
    return sorted({chunk.split("#", 1)[0] for chunk in chunks})


def _edge_for_pair(graph: nx.DiGraph, left: str, right: str) -> tuple[str, str] | None:
    """Находит реальное направление ребра после ненаправленного поиска пути."""
    if graph.has_edge(left, right):
        return left, right
    if graph.has_edge(right, left):
        return right, left
    return None


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    # TODO: опорные сущности, отбор фактов, запрос к модели, источники
  
    # FR-4: anchors находятся детерминированно, поэтому модель не может придумать начальную сущность
    graph = kb.graph
    codes = find_codes(question)
    anchors = [f"{code_type(code)}:{code}" for code in codes]
    if any(anchor not in graph for anchor in anchors):
        # FR-10: неизвестный код приводит к отказу до вызова LLM.
        return _refusal()
    anchors.extend(f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary))
    anchors = list(dict.fromkeys(anchor for anchor in anchors if anchor in graph))
    if not anchors:
        return _refusal()

    undirected = graph.to_undirected()
    edges: set[tuple[str, str]] = set()
    path_nodes: set[str] = set(anchors)
    if len(anchors) >= 2:
        # Для нескольких опор собираются все короткие пути, а не только один кратчайший
        for index, left in enumerate(anchors):
            for right in anchors[index + 1 :]:
                try:
                    paths = nx.all_simple_paths(undirected, left, right, cutoff=3)
                    for path in paths:
                        path_nodes.update(path)
                        for a, b in zip(path, path[1:], strict=False):
                            edge = _edge_for_pair(graph, a, b)
                            if edge:
                                edges.add(edge)
                except nx.NetworkXNoPath:
                    continue
        # Цепочка замены лежит за пределами пути между моделью и ошибкой, поэтому добавляется отдельно
        queue = [node for node in path_nodes if graph.nodes[node].get("type") == "Part"]
        visited = set(queue)
        while queue:
            node = queue.pop(0)
            for successor in graph.successors(node):
                data = graph.edges[node, successor]
                if data.get("type") == "REPLACED_BY":
                    edges.add((node, successor))
                    if successor not in visited:
                        visited.add(successor)
                        queue.append(successor)
    else:
        # Для одного кода или названия контекстом служит локальная окрестность радиуса два
        nodes = set(nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2))
        for source, target in graph.edges:
            if source in nodes and target in nodes:
                edges.add((source, target))

    if not edges:
        return _refusal()
    ordered = sorted(edges)
    facts = [
        f"{graph.nodes[source]['name']} -{graph.edges[source, target]['type']}-> {graph.nodes[target]['name']}"
        for source, target in ordered
    ]
    numbered = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, 1))[:8000]
    # Модель видит только отобранные факты; ограничение размера защищает контекст от разрастания
    prompt = (
        "Ответь на вопрос только по пронумерованным фактам. Не выдумывай коды или числа. "
        "Дай полный ответ словами в одном или нескольких предложениях; не отвечай номером пункта. "
        "Если есть цепочка REPLACED_BY, для заказа назови последнюю деталь этой цепочки. "
        "Например: «Нужно заказать деталь P-1042B».\n"
        f"Вопрос: {question}\nФакты:\n{numbered}"
    )
    text = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    return Answer(text, _documents(graph, edges), facts)


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    # TODO: частичные ответы по сводкам, объединение
    
    # FR-5 map: каждая сводка оценивается независимо и обязана вернуть структурированный JSON
    useful: list[tuple[str, int]] = []
    useful_nodes: set[str] = set()
    for community in kb.communities:
        prompt = (
            "Ответь на вопрос только по сводке. Верни JSON: answer — частичный ответ, "
            "score — полезность сводки от 0 до 10.\n"
            f"Вопрос: {question}\nСводка: {community.summary}"
        )
        response = llm.chat([{"role": "user", "content": prompt}], json_schema=MAP_SCHEMA).content
        try:
            item = json.loads(response)
            score = item["score"]
            answer = item["answer"]
            if isinstance(score, int) and score > 0 and isinstance(answer, str):
                useful.append((answer, score))
                useful_nodes.update(community.nodes)
        except (json.JSONDecodeError, KeyError, TypeError):
            # Повреждённый частичный ответ не срывает обработку остальных сообществ
            continue
    if not useful:
        return _refusal()
    partials = "\n".join(f"- [{score}/10] {answer}" for answer, score in useful)
    # FR-5 reduce: в итоговый prompt попадают только частичные ответы с положительной полезностью
    prompt = (
        "Объедини полезные частичные ответы в один точный ответ на вопрос. "
        "Не добавляй фактов, которых нет в частичных ответах.\n"
        f"Вопрос: {question}\nЧастичные ответы:\n{partials}"
    )
    text = llm.chat([{"role": "user", "content": prompt}]).content.strip()
    sources = sorted(
        {chunk.split("#", 1)[0] for node in useful_nodes for chunk in kb.graph.nodes[node].get("sources", [])}
    )
    # FR-10: источниками служат документы узлов только тех сообществ, которые участвовали в reduce
    return Answer(text, sources)
