---
type: concept
title: Warm-start Training
slug: warm-start-training
date: 2026-04-20
updated: 2026-04-20
aliases: [warm start, 热启动训练]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Warm-start Training** (热启动训练) — an optimization strategy that initializes a new model from a compatible pretrained checkpoint instead of learning all parameters from scratch.

## Key Points

- FLOATER is designed so that setting the positional dynamics to zero recovers the original Transformer, enabling checkpoint-compatible initialization.
- On WMT, the authors first train a standard Transformer, copy its best checkpoint into FLOATER, lower the peak learning rate, and retrain briefly.
- On GLUE, RACE, and SQuAD, the method warm-starts from pretrained RoBERTa and only learns the new flow-based positional parameters on top.
- This design reduces the otherwise large training burden of introducing Neural ODE-based positional layers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2020-encode-2003-09229]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2020-encode-2003-09229]].
