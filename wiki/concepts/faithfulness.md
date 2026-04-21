---
type: concept
title: Faithfulness
slug: faithfulness
date: 2026-04-20
updated: 2026-04-20
aliases: [groundedness]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Faithfulness** (忠实性) — the degree to which claims made in a generated answer are supported by the evidence provided in the conditioning context.

## Key Points

- RAGAS defines faithfulness with respect to the retrieved context rather than world knowledge in the abstract.
- The metric first decomposes an answer into focused statements before verifying each statement against the context.
- The reported score is `F = |V| / |S|`, the fraction of extracted statements judged supported.
- On WikiEval, faithfulness is the strongest of the three RAGAS dimensions, reaching `0.95` agreement with human preferences.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[es-2023-ragas-2309-15217]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[es-2023-ragas-2309-15217]].
