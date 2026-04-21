---
type: concept
title: Recency Bias
slug: recency-bias
date: 2026-04-20
updated: 2026-04-20
aliases: [recency effect, positional recency bias, 近因偏差]
tags: [bias, few-shot-learning, in-context-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Recency Bias** (近因偏差) — the tendency of left-to-right autoregressive language models to disproportionately repeat or favor the answer tokens that appear near the *end* of the in-context prompt, due to the sequential attention update of hidden states.

## Key Points

- In GPT-3 few-shot prompts, training examples closer to the test position are weighted more heavily in prediction; for 4-shot LAMA, the model overpredicts the 4th (last) training answer by 16.1% vs. only 8.5% for the 1st.
- The effect can override majority label bias: a prompt with three Positive examples followed by one Negative (P P P N) can lead to ~90% Negative predictions, despite 3/4 examples being Positive.
- Explains why the ordering/permutation of few-shot examples is a critical hyperparameter — simply reversing two examples in an SST-2 prompt drops accuracy from 88.5% to 51.3%.
- Contrasts with standard supervised learning where training example order is typically an afterthought.
- Addressed at inference time by [[contextual-calibration]] (which absorbs positional shift into the content-free estimate) rather than by reordering examples.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhao-2021-calibrate-2102-09690]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhao-2021-calibrate-2102-09690]].
