---
type: concept
title: Naive Identifier
slug: naive-identifier
date: 2026-04-20
updated: 2026-04-20
aliases: [Naive ID, 朴素标识符]
tags: [retrieval, decoding]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Naive Identifier** (朴素标识符) — a document identifier scheme that uses the corpus's original textual id string directly and decodes it token by token through the model vocabulary.

## Key Points

- The paper uses the original document id as a string, such as a numeric identifier decomposed into SentencePiece tokens.
- Naive IDs avoid the large corpus-proportional parameter cost of Atomic IDs, making them attractive for million-scale corpora.
- At full MS MARCO scale, naive scaling with larger T5 backbones is the strongest setting studied, peaking at `26.7` MRR@10 with `T5-XL`.
- Performance is not monotonic with parameter count: `T5-XXL` Naive IDs underperform `T5-XL`, so simple model growth does not fully solve the task.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[pradeep-2023-how]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[pradeep-2023-how]].
