---
type: concept
title: Contextual Calibration
slug: contextual-calibration
date: 2026-04-20
updated: 2026-04-20
aliases: [context calibration, 上下文校准]
tags: [calibration, few-shot-learning, in-context-learning, prompting]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contextual Calibration** (上下文校准) — a data-free post-processing method for few-shot in-context learning that estimates a language model's output bias by querying it with a content-free input, then applies a diagonal affine transformation to flatten the predicted distribution before decoding.

## Key Points

- The calibration parameters are estimated without any labeled data: replace the test input with a content-free string (e.g., "N/A", "[MASK]", or the empty string) to obtain a baseline probability vector `p_cf`; set `W = diag(p_cf)^{-1}` and `b = 0`.
- Calibrated prediction uses `q_hat = softmax(W * p_hat + b)` for any test input's raw probability `p_hat`; the argmax of `q_hat` is the final prediction.
- Addresses three biases simultaneously: [[majority-label-bias]] (over-prediction of frequent labels), [[recency-bias]] (over-weight of end-of-prompt examples), and [[common-token-bias]] (pre-training frequency preference).
- Reduces variance across prompt permutations and formats without modifying model weights; compatible with black-box API access as long as token probabilities are returned.
- Ensembling predictions from multiple content-free inputs ("N/A", "[MASK]", empty string) further stabilizes estimates; the best single recipe in the paper.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhao-2021-calibrate-2102-09690]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhao-2021-calibrate-2102-09690]].
