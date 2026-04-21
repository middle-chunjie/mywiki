---
type: concept
title: Agentic RAG
slug: agentic-rag
date: 2026-04-20
updated: 2026-04-20
aliases: [agentic retrieval-augmented generation, agentic retrieval augmented generation, 智能体式检索增强生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Agentic RAG** (智能体式检索增强生成) — a retrieval-augmented generation setup in which the model decides when to search, what to query, and how often retrieval should be invoked inside an ongoing reasoning process.

## Key Points

- Search-o1 contrasts agentic RAG with standard one-shot RAG, arguing that complex reasoning needs different evidence at different steps.
- The framework lets the model emit search queries inline during reasoning and trigger retrieval multiple times within one solution attempt.
- Retrieved documents are conditioned on the local reasoning state rather than only the original user question.
- In the paper's experiments, agentic RAG already improves QwQ-32B over standard RAG on most reasoning tasks, and Search-o1 further improves it by refining documents before injection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-searcho-2501-05366]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-searcho-2501-05366]].
