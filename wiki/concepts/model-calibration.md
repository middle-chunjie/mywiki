---
type: concept
title: Model Calibration
slug: model-calibration
date: 2026-04-20
updated: 2026-04-20
aliases: [calibration, 模型校准]
tags: [llm, evaluation, uncertainty]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model Calibration** (模型校准) — the degree to which a model's predicted probabilities match empirical correctness frequencies.

## Key Points

- The paper studies calibration to understand why stronger teachers can become harder for students to imitate in distillation.
- Teacher and student calibration are analyzed against data labels, teacher top-1 outputs, and full teacher distributions.
- The calibration analyses show that distilled students can become overconfident or underconfident depending on the target they are evaluated against.
- Calibration is used as a diagnostic lens on the capacity gap rather than as the main optimization objective.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[busbridge-2025-distillation-2502-08606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[busbridge-2025-distillation-2502-08606]].
