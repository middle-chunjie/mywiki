---
type: concept
title: Knowledge Graph Question Answering
slug: knowledge-graph-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [KGQA, Knowledge Graph QA, 知识图谱问答]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Graph Question Answering** (知识图谱问答) — the task of answering natural-language questions by reasoning over entities and relations in a structured knowledge graph.

## Key Points

- The paper formalizes KGQA over `G subseteq E x R x E`, where answers are entity sets `A_q` predicted from a question-specific subgraph rather than the full KG.
- It emphasizes the practical split between subgraph extraction and answer reasoning, especially for large KGs where full-graph inference is infeasible.
- The work studies IR-style KGQA instead of semantic-parsing pipelines, trading away gold query supervision for learned retrieval and reasoning.
- For complex questions, the paper argues that answer quality depends not just on relevant facts but on whether those facts form the correct structural evidence.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2024-enhancing-2402-02175]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2024-enhancing-2402-02175]].
