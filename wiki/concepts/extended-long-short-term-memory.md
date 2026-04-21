---
type: concept
title: Extended Long Short-Term Memory
slug: extended-long-short-term-memory
date: 2026-04-20
updated: 2026-04-20
aliases: [xLSTM, xLSTM architecture, extended LSTM]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Extended Long Short-Term Memory** (扩展长短期记忆) — a residual recurrent architecture family that modernizes LSTMs with exponential gating and new memory cells so they can scale competitively for language modeling.

## Key Points

- xLSTM introduces two cell variants: sLSTM for normalized scalar memory with memory mixing, and mLSTM for matrix memory with a covariance-style update rule.
- The architecture embeds these cells inside residual blocks instead of using standalone recurrent layers, aligning LSTM-style recurrence with modern LLM backbones.
- Mixed architectures are denoted `xLSTM[a:b]`, where `a:b` gives the ratio of mLSTM to sLSTM blocks; `xLSTM[7:1]` is the paper's main hybrid configuration.
- The paper reports that xLSTM outperforms matched Transformer, state-space, and recurrent baselines on SlimPajama validation perplexity at both `15B` and `300B` training-token scales.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
