---
type: concept
title: State Space Model
slug: state-space-model
date: 2026-04-20
updated: 2026-04-20
aliases: [SSM, state space model, 状态空间模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**State Space Model** (状态空间模型) — a sequence model that represents history through latent state transitions and is often used as a linear-time alternative to attention for long-context processing.

## Key Points

- The paper compares xLSTM against state-space models such as H3 and Mamba in matched language-modeling setups.
- mLSTM uses a pre-up-projection residual block that the paper explicitly describes as architecturally similar to state-space model blocks.
- xLSTM outperforms the compared SSM baselines in the reported SlimPajama validation perplexity tables at both `15B` and `300B` training-token scales.
- The authors argue that state-space models without memory mixing remain weaker on state-tracking tasks than recurrent architectures like sLSTM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
