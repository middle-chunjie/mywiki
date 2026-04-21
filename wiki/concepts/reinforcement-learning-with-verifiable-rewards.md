---
type: concept
title: Reinforcement Learning with Verifiable Rewards
slug: reinforcement-learning-with-verifiable-rewards
date: 2026-04-20
updated: 2026-04-20
aliases: [RLVR, 可验证奖励强化学习]
tags: [rl, evaluation, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reinforcement Learning with Verifiable Rewards** (可验证奖励强化学习) — a reinforcement-learning setup where rewards come from externally checkable outcomes such as exact answers, program outputs, or other easily verifiable signals.

## Key Points

- The paper positions RLVR as the dominant recipe behind prior open deep-research models trained on short-form QA proxies.
- RLVR works well when correctness is easy to verify, but the authors argue that it does not extend cleanly to open-ended long-form research tasks.
- Long-form deep research needs evaluation of completeness, depth, citation quality, and evidence use, which are not captured by simple exact-match style rewards.
- Static RLVR-style rewards are especially vulnerable when tasks are under-specified and admit many plausible good answers.
- RLER is proposed as an alternative that keeps RL but replaces brittle proxy verification with evolving, search-grounded rubrics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2025-dr-2511-19399]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2025-dr-2511-19399]].
