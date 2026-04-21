---
type: concept
title: RASP
slug: rasp
date: 2026-04-20
updated: 2026-04-20
aliases: [RASP]
tags: [transformer, programming-language, interpretability]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**RASP** — a programming language for expressing sequence operations with Transformer-aligned primitives such as `select`, `aggregate`, and elementwise transformations.

## Key Points

- [[friedman-2023-transformer-2306-01128]] uses RASP as the main conceptual bridge between Transformer components and human-readable code.
- In the paper's framing, `select` corresponds to attention pattern construction and `aggregate` corresponds to reading values through that pattern.
- The paper builds on RASP to argue that some Transformers can be constrained so every learned module has a discrete program interpretation.
- The algorithmic experiments reuse the suite of RASP tasks from Weiss et al. to test whether learned programs recover useful sequence algorithms.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-transformer-2306-01128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-transformer-2306-01128]].
