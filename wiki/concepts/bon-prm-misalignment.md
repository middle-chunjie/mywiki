---
type: concept
title: BoN-PRM Misalignment
slug: bon-prm-misalignment
date: 2026-04-20
updated: 2026-04-20
aliases: [BoN evaluation bias, Best-of-N PRM misalignment, BoN偏差]
tags: [evaluation, process-reward-model, reasoning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**BoN-PRM Misalignment** (BoN偏差) — the systematic divergence between Best-of-N evaluation performance and the actual process-verification capability of a PRM, caused by policy models frequently generating responses that have correct final answers but incorrect intermediate steps.

## Key Points

- [[zhang-2025-lessons-2501-07301]] demonstrates that real policy models (e.g., `Qwen2.5-Math-7B-Instruct`) regularly produce responses with correct answers but flawed reasoning steps, especially on harder benchmarks.
- A PRM that cannot distinguish correct-answer-flawed-process responses from genuinely correct responses will tolerate these responses and award them high scores, inflating its BoN performance.
- Manual annotation shows the proportion of correct-answer-but-wrong-process responses increases with problem difficulty (more pronounced on OlympiadBench and Omni-MATH than on GSM8K).
- PRMs trained purely on MC-estimation data exhibit this pattern more severely, as MC estimation also conflates answer correctness with step correctness.
- The paper advocates supplementing BoN with step-level benchmarks such as [[processbench]] to measure true process-verification capability.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2025-lessons-2501-07301]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2025-lessons-2501-07301]].
