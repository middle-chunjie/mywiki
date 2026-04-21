---
type: concept
title: Memory Distillation
slug: memory-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [memory filtering]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Memory Distillation** — a policy that filters a retrieved memory set to the subset most relevant for answering a current query before final reasoning.

## Key Points

- In Memory-R1, the Answer Agent receives `60` retrieved candidate memories and distills them before generating the answer.
- The paper motivates memory distillation as a defense against noisy RAG contexts that can distract the model.
- Distillation is optimized with the same outcome-based RL signal as answer generation, using exact-match reward on the final answer.
- Removing memory distillation lowers performance from `37.51 / 45.02 / 62.74` to `34.37 / 40.95 / 60.14` on `F1 / BLEU-1 / Judge`.
- The concept operationalizes the intuition that useful memory systems need both broad retrieval and selective integration.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yan-2026-memoryr-2508-19828]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yan-2026-memoryr-2508-19828]].
