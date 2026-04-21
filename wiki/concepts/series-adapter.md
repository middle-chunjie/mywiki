---
type: concept
title: Series Adapter
slug: series-adapter
date: 2026-04-20
updated: 2026-04-20
aliases: [series adapters, 串行适配器]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Series Adapter** (串行适配器) — an adapter module inserted sequentially after a backbone sublayer, typically using down-projection, nonlinearity, and up-projection to add task-specific capacity with few trainable parameters.

## Key Points

- LLM-Adapters formulates the series adapter as `H_o <- H_o + f(H_o W_down) W_up`, where the bottleneck size controls the adapter capacity.
- The placement study finds that, for `LLaMA-7B` on math reasoning, placing the Series Adapter after the `MLP` layer works best.
- With `bn = 256`, Series Adapter reaches `59.5%` average accuracy on the six math reasoning datasets in the configuration analysis.
- On commonsense reasoning, `LLaMA-13B + Series Adapter` achieves `79.5%` average accuracy, outperforming ChatGPT in the paper's in-distribution setting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hu-2023-llmadapters-2304-01933]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hu-2023-llmadapters-2304-01933]].
