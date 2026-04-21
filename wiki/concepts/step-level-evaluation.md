---
type: concept
title: Step-Level Evaluation
slug: step-level-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [step-wise evaluation, process error identification, 步骤级评估]
tags: [evaluation, process-reward-model, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Step-Level Evaluation** (步骤级评估) — an evaluation paradigm that measures a model's ability to identify the first erroneous intermediate step in a reasoning trace, as opposed to judging only the final answer.

## Key Points

- [[zhang-2025-lessons-2501-07301]] advocates combining response-level Best-of-N evaluation with step-level evaluation using [[processbench]] to avoid the [[bon-prm-misalignment]] problem.
- PROCESSBENCH formulates step-level evaluation as: given a solution, identify the first step that contains an error, or report that all steps are correct; PRMs locate errors by finding the step with the minimum predicted score.
- The paper shows that BoN and PROCESSBENCH scores can be inversely correlated: MC-trained PRMs achieve relatively high BoN but low PROCESSBENCH F1, while human-annotated PRMs show the reverse.
- Qwen2.5-Math-PRM-7B achieves `73.5` avg F1 on PROCESSBENCH, surpassing GPT-4o-0806 (`61.9`) despite being a much smaller model.
- ORM score decomposition (scoring each individual step with an ORM) is proposed as a simple baseline and surprisingly outperforms many dedicated PRMs on step-level metrics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2025-lessons-2501-07301]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2025-lessons-2501-07301]].
