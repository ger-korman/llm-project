"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    
    # Алгоритм Лувена работает с ненаправленным представлением, но учитывает число подтверждений связи
    if not graph:
        return []
    groups = nx.community.louvain_communities(graph.to_undirected(), weight="weight", seed=7)
    # Сортировка и фиксированный seed обеспечивают воспроизводимые номера сообществ между запусками
    ordered = sorted((sorted(group) for group in groups), key=lambda nodes: nodes[0])
    return [Community(number, nodes) for number, nodes in enumerate(ordered)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    
    # Числа берёт код из weight; языковая модель лишь сжимает готовые факты в связный текст
    members = set(community.nodes)
    facts = []
    for source, target, data in graph.edges(data=True):
        # В текущую сводку входят только внутренние рёбра выбранного сообщества
        if source in members and target in members:
            left = graph.nodes[source]["name"]
            right = graph.nodes[target]["name"]
            facts.append(f"{left} -{data['type']}-> {right}; число подтверждений: {data.get('weight', 1)}")
    entities = [f"{graph.nodes[node]['type']}: {graph.nodes[node]['name']}" for node in community.nodes]
    prompt = (
        "Напиши на русском краткую фактическую сводку сообщества графа, не более 150 слов. "
        "Не пересчитывай и не изменяй числа. Укажи основные сущности, связи и самые частые отказы.\n"
        "Сущности:\n" + "\n".join(entities) + "\nСвязи и рассчитанные кодом числа:\n" + "\n".join(facts)
    )
    text = llm.chat([{"role": "user", "content": prompt}], max_tokens=400).content.strip()
    # Ограничение повторяется, даже если модель проигнорировала требование в промпт
    return " ".join(text.split()[:150])
