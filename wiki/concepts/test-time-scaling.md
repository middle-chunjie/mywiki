---
type: concept
title: Test-Time Scaling
slug: test-time-scaling
date: 2026-04-20
updated: 2026-04-20
aliases: [inference scaling, test-time compute scaling]
tags: [llm, inference]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Test-Time Scaling** (测试时扩展) — improving model performance by spending more inference-time computation, such as generating longer reasoning traces or sampling multiple solutions.

## Key Points

- The paper frames controllable reasoning length as a practical way to allocate test-time compute under explicit token budgets.
- L1 exhibits a roughly log-linear accuracy improvement as requested reasoning length increases.
- Sequential scaling through longer CoT generation outperforms parallel majority-vote scaling at the same total token budget in the appendix experiments.
- The main benefit of LCPO is not only higher performance, but also a predictable accuracy-versus-cost frontier.
- OOD tasks such as GPQA and LSAT still benefit from increased token budgets under the learned control policy.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[aggarwal-2025-l-2503-04697]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[aggarwal-2025-l-2503-04697]].
