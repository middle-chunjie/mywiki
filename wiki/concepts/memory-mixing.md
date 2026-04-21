---
type: concept
title: Memory Mixing
slug: memory-mixing
date: 2026-04-20
updated: 2026-04-20
aliases: [memory interaction, 记忆混合]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Memory Mixing** (记忆混合) — recurrent interaction between memory cells or hidden channels that lets updates depend on previously stored state rather than only on the current token.

## Key Points

- sLSTM keeps and restructures memory mixing through recurrent matrices `R_z, R_i, R_f, R_o`, while mLSTM removes it to gain full sequence parallelism.
- The paper claims memory mixing is crucial for state-tracking tasks and formal-language generalization where purely parallel sequence mixers underperform.
- In the vectorized sLSTM, these recurrent matrices are block-diagonal, reducing interaction cost to roughly `d^2 / N_h` with head-wise structure.
- The authors treat memory mixing as a major reason recurrent models remain more expressive than state-space models and Transformers on certain symbolic tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
