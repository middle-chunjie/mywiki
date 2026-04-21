---
type: concept
title: LM-Supervised Retrieval
slug: lm-supervised-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [language-model-supervised retrieval, 语言模型监督检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**LM-Supervised Retrieval** (语言模型监督检索) — a retriever-training paradigm that uses a language model's scoring signal over retrieved contexts as supervision for which documents should be preferred.

## Key Points

- RePlug LSR adapts the retriever to a frozen LM instead of adapting the LM to retrieval.
- The method converts LM quality on the ground-truth continuation into a document distribution `Q(d | x, y)`.
- Retriever scores define a second distribution `P_R(d | x)`, and training minimizes `KL(P_R || Q)` so retrieval rankings better match the LM's preferences.
- The approach improves over frozen RePlug across language modeling, MMLU, and open-domain QA, indicating that LM-aligned retrieval is beneficial even without LM fine-tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2023-replug-2301-12652]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2023-replug-2301-12652]].
