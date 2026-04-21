---
type: concept
title: Data Flywheel
slug: data-flywheel
date: 2026-04-20
updated: 2026-04-20
aliases: [data flywheel, self-improving data loop, 数据飞轮]
tags: [training, scalability, agents, retrieval]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Data Flywheel** (数据飞轮) — a self-reinforcing feedback loop in which system usage generates training data that improves the system, which in turn generates better usage data, enabling iterative and scalable improvement without additional human annotation.

## Key Points

- LRAT frames agent trajectories as the agent-era counterpart of human click logs in web search, enabling a sustainable data flywheel driven by agent–retriever interactions.
- Even trajectories from agents that produce incorrect final answers contain useful intermediate relevance signals, allowing the flywheel to operate without filtering for correctness.
- Simulated flywheel experiments show steady gains in both task success rate and evidence recall across iterative retriever updates, confirming that the positive feedback loop is real and not a one-time effect.
- The flywheel scales with volume: retrievers trained on progressively larger trajectory datasets show no early saturation up to 30K trajectories tested.
- The mechanism mirrors the historical development of web search, where user click logs provided self-sustaining signal for iterative ranker improvement.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2026-retrieve-2604-04949]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2026-retrieve-2604-04949]].
