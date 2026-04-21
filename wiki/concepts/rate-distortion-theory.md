---
type: concept
title: Rate-Distortion Theory
slug: rate-distortion-theory
date: 2026-04-20
updated: 2026-04-20
aliases: [RD theory, 率失真理论]
tags: [information-theory, compression]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Rate-Distortion Theory** (率失真理论) — an information-theoretic framework that studies the minimum representation rate required to encode data under a specified level of distortion.

## Key Points

- The paper treats prior GDR indexing methods as optimizing document codes with respect to `X` alone, which matches a rate-distortion view.
- In this framing, the compactness term is `I(X;T)`, where `T` is the document identifier.
- The authors argue that RD theory is insufficient for retrieval because it does not directly account for the query variable `Q`.
- Their BMI formulation extends RD theory into an information-bottleneck setting that explicitly incorporates query predictability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[du-2024-bottleneckminimal-2405-10974]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[du-2024-bottleneckminimal-2405-10974]].
