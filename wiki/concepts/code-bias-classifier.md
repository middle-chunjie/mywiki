---
type: concept
title: Code Bias Classifier
slug: code-bias-classifier
date: 2026-04-20
updated: 2026-04-20
aliases: [bias scorer, 代码偏见分类器]
tags: [classification, fairness, code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Bias Classifier** (代码偏见分类器) — a classifier trained to distinguish biased from acceptable code completions for automated fairness evaluation.

## Key Points

- [[liu-nd-uncovering]] builds a dedicated classifier because existing fairness scorers are designed for natural language rather than generated code.
- The paper compares LSTM with random initialization, LSTM with word2vec embeddings, and BERT-Base, selecting BERT-Base as the strongest automatic scorer.
- Classifier outputs are thresholded in the CBS formula through `P_cls(code_i) >= 0.5`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-nd-uncovering]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-nd-uncovering]].
