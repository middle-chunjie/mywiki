---
type: concept
title: Semantic Textual Similarity
slug: semantic-textual-similarity
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 语义文本相似度
  - STS
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Textual Similarity** (语义文本相似度) — the task of assigning a graded similarity score to a pair of texts based on meaning rather than surface overlap.

## Key Points

- The paper trains a dedicated STS adapter because semantic similarity is symmetric and should not share the same encoding behavior as asymmetric retrieval.
- When graded labels are available, the adapter uses CoSENT ranking loss; otherwise it falls back to InfoNCE plus distillation with `lambda_NCE : lambda_D = 1 : 2`.
- STS training data includes multilingual annotated datasets such as STS12 and SICK, plus machine-translated and paraphrase-style pairs for broader coverage.
- The resulting models score `78.9` and `78.2` on MMTEB STS, and `88.1` and `88.3` on English MTEB STS for the small and nano variants respectively.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[akram-2026-jinaembeddingsvtext-2602-15547]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[akram-2026-jinaembeddingsvtext-2602-15547]].
