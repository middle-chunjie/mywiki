---
type: concept
title: Temperature Scaling
slug: temperature-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [logit rescaling, 温度缩放]
tags: [llm, calibration]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Temperature Scaling** (温度缩放) — a post-hoc calibration method that rescales logits or probabilities by a temperature parameter to make confidence better match observed accuracy.

## Key Points

- [[kadavath-2022-language-2207-05221]] shows that RLHF policy miscalibration can be reduced substantially with temperature `T = 2.5`.
- The result demonstrates that poor confidence estimates do not always require retraining; some errors are due to badly scaled output distributions.
- The paper separately uses `T = 1` sampling to define ground-truth answerability for `P(IK)` training, so generation temperature is part of the task definition.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
