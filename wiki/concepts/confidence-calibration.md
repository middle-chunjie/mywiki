---
type: concept
title: Confidence Calibration
slug: confidence-calibration
date: 2026-04-20
updated: 2026-04-20
aliases: [decision calibration, 置信度校准]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Confidence Calibration** (置信度校准) — the alignment between a model's confidence signal and the actual correctness of its decisions or predictions.

## Key Points

- SMART evaluates calibration at the action level: whether the model is confident for the correct choice between internal reasoning and a tool call.
- The authors add special tokens such as `[Reasoning]`, `[Search]`, and `[AskUser]` so token logits can be used as confidence probes.
- On sampled Time and Intention decisions, correct choices receive higher confidence than incorrect ones, indicating better calibrated switching behavior.
- This analysis supports the claim that SMART improves decision quality, not only raw task accuracy or tool suppression.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[qian-2025-smart-2502-11435]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[qian-2025-smart-2502-11435]].
