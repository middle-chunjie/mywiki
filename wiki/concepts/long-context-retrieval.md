---
type: concept
title: Long-Context Retrieval
slug: long-context-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [long document retrieval, 长上下文检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Context Retrieval** (长上下文检索) — retrieval over queries and documents whose relevant evidence is distributed across long sequences, so success depends on encoding broad context rather than only short local spans.

## Key Points

- The paper argues that long-context retrieval fails on standard short-context benchmarks because many relevant cues appear near the beginning of documents, making truncation deceptively strong.
- LoCoV1 operationalizes the setting with `12` tasks drawn from law, medicine, science, finance, government, and programming, with many documents well beyond `10k` tokens.
- On LoCoV1, retrieval quality correlates much more strongly with maximum sequence length than it does on BEIR, indicating that context budget materially affects performance.
- `M2-BERT-32k` improves over E5-Mistral by `23.3` average points, showing that explicit long-context encoding matters when chunking or truncation is insufficient.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[saad-falcon-2024-benchmarking-2402-07440]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[saad-falcon-2024-benchmarking-2402-07440]].
