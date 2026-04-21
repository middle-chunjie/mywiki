---
type: concept
title: Nonparametric Language Model
slug: nonparametric-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [non-parametric language model, 非参数语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Nonparametric Language Model** (非参数语言模型) — a language model whose effective capacity depends on an external datastore available at inference time rather than being bounded only by fixed neural parameters.

## Key Points

- The paper positions `∞`-gram as a nonparametric LM that reads directly from a massive text corpus at inference time.
- Unlike vector-retrieval methods that store one embedding per token or chunk, infini-gram uses suffix-array indexing over raw token IDs.
- The approach scales reference data to `1.8T` tokens while keeping storage at roughly `7` bytes per token, which is presented as much cheaper than large vector datastores.
- The paper shows that this nonparametric component remains useful even when paired with neural LMs as large as `70B`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-infinigram-2401-17377]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-infinigram-2401-17377]].
