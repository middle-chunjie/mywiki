---
type: concept
title: Generative Retrieval
slug: generative-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [GR, 生成式检索]
tags: [retrieval, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Generative Retrieval** (生成式检索) — an information retrieval paradigm in which a generative model directly produces document identifiers or equivalent targets for relevant documents instead of scoring an external index.

## Key Points

- The paper frames GR as unifying indexing and retrieval within a single model, rather than separating inverted-index construction from query-time ranking.
- Prior GR systems mostly rely on training-based indexing, where a seq2seq model is fine-tuned on large query-docid pairs.
- Few-Shot GR shows that GR can also be realized with prompting-only indexing by generating free-text docids without parameter updates.
- The paper argues that GR becomes more practical when indexing cost and corpus-update friction are reduced.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[askari-2024-generative-2408-02152]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[askari-2024-generative-2408-02152]].
