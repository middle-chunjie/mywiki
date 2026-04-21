---
type: concept
title: Neural Program Synthesis
slug: neural-program-synthesis
date: 2026-04-20
updated: 2026-04-20
aliases: [neural program synthesis, 神经程序合成]
tags: [program-synthesis, neural-methods]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Neural Program Synthesis** (神经程序合成) — program synthesis in which learned models predict, rank, or parameterize candidate programs while symbolic mechanisms enforce search or execution constraints.

## Key Points

- [[fijalkow-2022-scaling]] treats neural guidance as a probability-labeling step that converts a CFG into a PCFG conditioned on input-output examples.
- The paper argues that neural inference is often cheap relative to combinatorial search, so scaling requires better search algorithms rather than only better predictors.
- Its learned benchmark uses a one-layer GRU and a 3-layer MLP to predict derivation-rule probabilities for DreamCoder-style synthesis tasks.
- The work shows that a strong symbolic back-end such as Heap Search can materially improve neural program synthesis without changing the model architecture.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fijalkow-2022-scaling]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fijalkow-2022-scaling]].
