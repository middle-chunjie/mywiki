---
type: concept
title: Proposition Retrieval
slug: proposition-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [proposition-level retrieval, 命题检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Proposition retrieval** (命题检索) — a retrieval setup in which the corpus is indexed and ranked at the proposition level rather than at sentence or passage level.

## Key Points

- The paper uses proposition retrieval as an inference-time alternative for dense retrievers without changing retriever parameters.
- Passage scores are recovered by taking the maximum similarity over propositions belonging to the same passage.
- Proposition retrieval improves average Recall@20 for both unsupervised and supervised retrievers, with larger gains for unsupervised models.
- It is especially beneficial for long-tail entity questions, where fine-grained facts are easier to surface than longer passages.
- The same proposition units also improve retrieval-augmented QA under fixed prompt token budgets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2024-dense-2312-06648]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2024-dense-2312-06648]].
