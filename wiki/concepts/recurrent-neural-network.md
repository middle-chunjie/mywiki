---
type: concept
title: Recurrent Neural Network
slug: recurrent-neural-network
date: 2026-04-20
updated: 2026-04-20
aliases: [RNN, recurrent neural network, 循环神经网络]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Recurrent Neural Network** (循环神经网络) — a sequence model that updates its hidden state step by step so later predictions depend on a compressed summary of prior tokens.

## Key Points

- The paper positions xLSTM as a modern recurrent alternative to Transformers and state-space models for language modeling.
- It argues that recurrent architectures with memory mixing remain stronger for state-tracking problems than architectures that only use parallel sequence mixing.
- xLSTM is compared against RWKV, HGRN, HGRN2, and LSTM-based baselines, outperforming the recurrent baselines in perplexity at matched scale.
- The work shows that recurrence is not abandoned, but redesigned to fit residual backbones, normalization, and large-scale pretraining.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
