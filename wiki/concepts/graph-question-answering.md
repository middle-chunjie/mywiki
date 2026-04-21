---
type: concept
title: Graph Question Answering
slug: graph-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [GraphQA, graph QA, 图问答]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Graph Question Answering** (图问答) — the task of answering natural-language questions about graph-structured data by identifying and reasoning over relevant nodes, edges, and their attributes.

## Key Points

- The paper frames graph question answering over textual graphs as a flexible conversational task rather than a narrow classification problem.
- It introduces the [[graphqa]] benchmark by standardizing [[explagraphs]], [[scenegraphs]], and [[webquestionssp]] into one graph-plus-question format.
- The benchmark spans commonsense reasoning, scene understanding, and knowledge-graph reasoning, so graph QA is evaluated across multiple graph modalities.
- G-Retriever improves over prompt tuning on all three datasets by combining retrieval, connected subgraph extraction, and LLM generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-gretriever-2402-07630]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-gretriever-2402-07630]].
