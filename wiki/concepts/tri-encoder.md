---
type: concept
title: Tri-Encoder
slug: tri-encoder
date: 2026-04-20
updated: 2026-04-20
aliases: [Tri Encoder, 三编码器]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tri-Encoder** (三编码器) — an architecture that encodes two target texts and a conditioning text separately, then composes the condition with each target representation before computing similarity.

## Key Points

- The paper proposes tri-encoders to better match C-STS's three-input structure: sentence 1, sentence 2, and condition.
- Each input is encoded independently as a vector, then a transformation `h: R^(2d) -> R^d` combines the condition with each sentence representation.
- The authors test both MLP and Hadamard-style composition operators and train these models with MSE, Quad, or `Quad + MSE`.
- Despite the architectural motivation, tri-encoders underperform bi-encoders in the reported experiments, with the best test Spearman only `35.3 ± 1.0`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deshpande-2023-csts-2305-15093]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deshpande-2023-csts-2305-15093]].
