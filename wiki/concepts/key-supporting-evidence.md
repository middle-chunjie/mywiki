---
type: concept
title: Key Supporting Evidence
slug: key-supporting-evidence
date: 2026-04-20
updated: 2026-04-20
aliases: [KSE, key evidence, 关键支撑证据]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Key Supporting Evidence** (关键支撑证据) — the minimal set of external evidence that best supports a downstream generator in producing a correct answer.

## Key Points

- BIDER defines KSE as evidence aligned with what the generator actually needs, not merely what a retriever ranks highly.
- The paper constructs oracle KSE through a three-step synthesis process: nugget extraction, nugget refinement, and nugget cleaning.
- KSE is used as the supervised target for training a seq2seq refiner from raw retrieval results.
- The approach treats compactness as a feature rather than a loss, cutting average input length from `759` to `90` tokens while improving downstream accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jin-2024-bider-2402-12174]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jin-2024-bider-2402-12174]].
