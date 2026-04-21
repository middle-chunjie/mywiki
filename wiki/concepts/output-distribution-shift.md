---
type: concept
title: Output Distribution Shift
slug: output-distribution-shift
date: 2026-04-20
updated: 2026-04-20
aliases: [prediction distribution shift, output bias shift, 输出分布偏移]
tags: [bias, calibration, language-model]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Output Distribution Shift** (输出分布偏移) — the phenomenon where a language model's predicted probability distribution over answer tokens is systematically offset from a uniform or task-appropriate distribution, causing the model to systematically favor certain answers regardless of the test input content.

## Key Points

- In few-shot in-context learning, the combined effect of [[majority-label-bias]], [[recency-bias]], and [[common-token-bias]] manifests as a shift in the output distribution: the model assigns consistently high probability to a subset of labels across all test inputs.
- The shift is *contextual*: it depends on the specific prompt (training examples, permutation, format), so the same model exhibits different shifts for different prompts; there is no single fixed correction that works for all prompts.
- Visualized in the paper as a scatter plot of per-example Positive class probability: an uncalibrated, biased prompt clusters test probabilities near 0.8, while the optimal threshold would be ~0.68 for near-perfect accuracy.
- Motivates the [[contextual-calibration]] approach: because the shift is approximately constant across test inputs (for a given prompt), it can be estimated from a single content-free input and subtracted.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhao-2021-calibrate-2102-09690]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhao-2021-calibrate-2102-09690]].
