---
type: concept
title: Lexical Complexity Prediction
slug: lexical-complexity-prediction
date: 2026-04-20
updated: 2026-04-20
aliases: [LCP, word complexity prediction, 词汇复杂度预测]
tags: [nlp, readability, complex-word-identification]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Lexical Complexity Prediction** (词汇复杂度预测) — the NLP regression sub-task of assigning a continuous complexity score in `[0, 1]` to a target word or phrase given its sentence context, where annotations are normalized from Likert-scale human judgments.

## Key Points

- Formulated as a regression problem rather than binary classification; evaluation uses Pearson Correlation Coefficient and Mean Absolute Error against human-annotated complexity scores.
- The SemEval-2021 Task 1 (LCP 2021) defines the benchmark: CompLex corpus with English single-word (7,662 train) and multi-word (1,517 train) subsets across biblical, biomedical, and parliamentary domains.
- Lexical complexity is highly domain-dependent: a term like "sitosterolemia" is trivial in a biomedical paper but maximally complex in a news article; models must be robust across domains.
- Strong baselines combine target word features (character n-grams, word frequency, syllable count) with contextual sentence embeddings; the best LCP 2021 system achieves `.7886` Pearson on single-word test.
- Closely related to complex word identification; the distinction is granularity: LCP predicts a score, CWI classifies binary complex/not-complex.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
