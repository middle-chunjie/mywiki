---
type: concept
title: Exponential Gating
slug: exponential-gating
date: 2026-04-20
updated: 2026-04-20
aliases: [exp gating, exponential gate, 指数门控]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Exponential Gating** (指数门控) — a gating scheme that uses exponential activations for recurrent update weights so a model can rescale memory updates more flexibly than with standard sigmoid gates.

## Key Points

- In xLSTM, the input gate is `i_t = exp(\tilde{i}_t)` and the forget gate may be either sigmoid- or exponential-activated depending on the variant.
- The paper introduces exponential gating to help recurrent models revise previous storage decisions instead of being restricted to classic LSTM gating behavior.
- Because raw exponentials can overflow, xLSTM adds a stabilizer state `m_t` and computes normalized effective gates without changing the exact network output.
- Exponential gating is shared by both sLSTM and mLSTM and is one of the paper's two central architectural changes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[beck-2024-xlstm-2405-04517]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[beck-2024-xlstm-2405-04517]].
