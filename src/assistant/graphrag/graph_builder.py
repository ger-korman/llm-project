"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    
    # TODO: узлы, рёбра с весами, проверка схемы, разбор встречных замен

    # Сначала накапливаем уникальные источники, а затем одним проходом создаём взвешенные рёбра
    graph = nx.DiGraph(conflicts=0)
    edge_sources: dict[tuple[str, str, str], set[str]] = {}

    for result in results:
        for entity in result.entities:
            # Одинаковый node_id из разных чанков становится одним узлом с объединёнными источниками
            if entity.node_id not in graph:
                graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
            sources = set(graph.nodes[entity.node_id]["sources"])
            sources.add(result.chunk_id)
            graph.nodes[entity.node_id]["sources"] = sorted(sources)
        for relation in result.relations:
            # Схемная проверка не позволяет, например, связать модель отношением REPLACED_BY
            if not relation.matches_schema():
                continue
            for entity in (relation.subject, relation.object):
                if entity.node_id not in graph:
                    graph.add_node(entity.node_id, type=entity.type, name=entity.name, sources=[])
                sources = set(graph.nodes[entity.node_id]["sources"])
                sources.add(result.chunk_id)
                graph.nodes[entity.node_id]["sources"] = sorted(sources)
            key = (relation.subject.node_id, relation.object.node_id, relation.type)
            edge_sources.setdefault(key, set()).add(result.chunk_id)

    for (source, target, kind), sources in edge_sources.items():
        # Повтор отношения в одном чанке не увеличивает вес, потому что sources хранится как set
        graph.add_edge(source, target, type=kind, weight=len(sources), sources=sorted(sources))

    checked: set[frozenset[str]] = set()
    # Встречные замены противоречат смыслу: сохраняем более подтверждённое направление или удаляем оба
    for source, target, data in list(graph.edges(data=True)):
        pair = frozenset((source, target))
        if data["type"] != "REPLACED_BY" or pair in checked or not graph.has_edge(target, source):
            continue
        reverse = graph.edges[target, source]
        if reverse.get("type") != "REPLACED_BY":
            continue
        checked.add(pair)
        graph.graph["conflicts"] += 1
        if data["weight"] > reverse["weight"]:
            graph.remove_edge(target, source)
        elif data["weight"] < reverse["weight"]:
            graph.remove_edge(source, target)
        else:
            graph.remove_edge(source, target)
            graph.remove_edge(target, source)
    return graph
