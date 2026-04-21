---
type: concept
title: Deterministic Verification
slug: deterministic-verification
date: 2026-04-20
updated: 2026-04-20
aliases: [确定性验证]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Deterministic Verification** (确定性验证) — evaluation by programmatic assertions that produce reproducible pass/fail outcomes without relying on subjective model judging.

## Key Points

- Every SkillsBench task includes a deterministic verifier with programmatic assertions and numeric tolerances where needed.
- Oracle solutions must reach `100%` verifier pass rate before a task can enter the benchmark.
- The benchmark uses deterministic verification to avoid the instability of LLM-as-a-judge scoring and to support paired comparisons between no-skills and with-skills settings.
- Failure analysis is grounded in verifier outputs, enabling the paper to separate timeout, execution, coherence, and verification failures.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-skillsbench-2602-12670]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-skillsbench-2602-12670]].
