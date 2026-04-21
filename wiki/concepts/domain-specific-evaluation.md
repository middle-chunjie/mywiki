---
type: concept
title: Domain-Specific Evaluation
slug: domain-specific-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [domain-aware evaluation, 领域特定评测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Domain-Specific Evaluation** (领域特定评测) — an evaluation protocol that measures system performance separately across task domains instead of relying only on aggregate scores.

## Key Points

- EvoCodeBench assigns each task a domain label under a `10`-domain taxonomy so code LLMs can be compared on subsets such as Database, Internet, or Scientific Engineering.
- The paper reports per-domain Pass@1 and introduces DSI to quantify how relatively strong a model is within a given domain.
- A threshold of `10%` is used to define comfort domains and strange domains, making per-model strengths and weaknesses easier to interpret.
- The domain-wise analysis shows that global ranking can be misleading: `gpt-4` is strongest overall but lags peers in the Internet domain.
- The benchmark is intended to support practitioner model selection for specialized development settings, not just headline leaderboard comparison.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-evocodebench-2410-22821]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-evocodebench-2410-22821]].
