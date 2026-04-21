---
type: concept
title: Parallel Adapter
slug: parallel-adapter
date: 2026-04-20
updated: 2026-04-20
aliases: [parallel adapters, 并行适配器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parallel Adapter** (并行适配器) — an adapter architecture that adds a learnable bottleneck branch in parallel with an existing backbone sublayer and combines its output with the original model path.

## Key Points

- The paper writes Parallel Adapter as `H_o <- H_o + f(H_i W_down) W_up`, highlighting that the adapter branch operates on the layer input in parallel with the native transformation.
- In the placement study on `LLaMA-7B`, inserting Parallel Adapter into the `MLP` layer yields the strongest average math-reasoning performance at `61.7%`.
- The configuration sweep shows the best bottleneck size is `bn = 256`; larger `bn = 512` degrades average math performance to `58.0%`.
- On commonsense reasoning, `LLaMA-13B + Parallel Adapter` is the strongest reported model with `81.5%` average accuracy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hu-2023-llmadapters-2304-01933]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hu-2023-llmadapters-2304-01933]].
