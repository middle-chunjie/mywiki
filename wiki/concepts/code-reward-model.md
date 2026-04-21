---
type: concept
title: Code Reward Model
slug: code-reward-model
date: 2026-04-20
updated: 2026-04-20
aliases: [coding reward model, reward model for code, 代码奖励模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Reward Model** (代码奖励模型) — a learned verifier that scores a candidate program for correctness or quality directly from the problem and solution text, without executing hidden oracle tests at evaluation time.

## Key Points

- [[ficek-2025-scoring-2502-13820]] compares code reward models against generated-test verifiers on the same ranking and scoring benchmarks.
- Reward scores are normalized per problem using the highest and lowest model outputs before being compared with oracle pass-rate rankings.
- AceCodeRM-32B and Nemotron4-340B-Reward are the strongest reward-model baselines reported, but they generally trail the best reasoning-based test generators on Top-1 and rank correlation.
- The paper argues transformed scoring benchmarks make direct, model-agnostic comparison between reward models and test-case generation methods possible.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ficek-2025-scoring-2502-13820]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ficek-2025-scoring-2502-13820]].
