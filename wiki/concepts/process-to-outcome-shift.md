---
type: concept
title: Process-to-Outcome Shift
slug: process-to-outcome-shift
date: 2026-04-20
updated: 2026-04-20
aliases: [outcome drift, PRM degradation, 过程转结果退化]
tags: [process-reward-model, evaluation, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Process-to-Outcome Shift** (过程转结果退化) — a degradation mode in which a PRM optimized solely on Best-of-N rewards gradually concentrates its minimum step-score on the final answer step, effectively behaving as an outcome reward model rather than a step-level verifier.

## Key Points

- [[zhang-2025-lessons-2501-07301]] diagnoses this shift by analyzing the distribution of minimum step scores: if the minimum score most often falls on the final step, the PRM is de facto an ORM.
- Multiple open-source PRMs (EurusPRM, Math-Shepherd-PRM-7B, Skywork-PRM-7B) show >40% of responses with their minimum score on the final step, confirming the shift.
- The proposed Qwen2.5-Math-PRM-7B/72B models exhibit a significantly lower proportion of final-step minimums, indicating retained process-level discrimination.
- For BoN selection, the minimum step score acts as the effective ranking key regardless of whether product or minimum aggregation is used, so final-step concentration undermines process awareness.
- The shift is a consequence of BoN being a response-level objective: rewarding correct-answer responses regardless of intermediate step quality teaches the PRM to focus on the final step.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2025-lessons-2501-07301]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2025-lessons-2501-07301]].
