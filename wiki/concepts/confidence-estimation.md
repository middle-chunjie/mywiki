---
type: concept
title: Confidence Estimation
slug: confidence-estimation
date: 2026-04-20
updated: 2026-04-20
aliases: [uncertainty estimation, 置信度估计]
tags: [llm, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Confidence Estimation** (置信度估计) — estimating how likely a prediction, answer, or question outcome is to be correct, typically with an explicit probability.

## Key Points

- [[kadavath-2022-language-2207-05221]] distinguishes answer-level confidence `P(True)` from question-level confidence `P(IK)`.
- The paper evaluates confidence quality with ECE, RMS calibration error, AUROC, and Brier score rather than accuracy alone.
- Confidence estimates improve with better prompting and larger models, but they remain fragile out of distribution.
- The work shows that confidence can incorporate context, since `P(IK)` rises when relevant documents or hints are provided.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
