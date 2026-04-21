---
type: concept
title: Parallel Context Windows
slug: parallel-context-windows
date: 2026-04-20
updated: 2026-04-20
aliases: [PCW, parallel context windows, 并行上下文窗口]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Parallel Context Windows** (并行上下文窗口) — an inference-time method that partitions long context into multiple parallel windows, reuses positional embeddings across windows, and restricts attention so task tokens can access all windows without retraining the model.

## Key Points

- [[ratner-2023-parallel-2212-10947]] expands usable context from `C` tokens to `B·C` context tokens plus `T` task tokens by instantiating `B` window replicas.
- The method is training-free and only changes positional assignment and attention masking at inference time.
- In the paper's main experiments, `B = 3` already gives large gains on high-cardinality classification tasks and information extraction.
- PCW is most effective when the information in different windows can be processed independently and aggregated only at the task tokens.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ratner-2023-parallel-2212-10947]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ratner-2023-parallel-2212-10947]].
