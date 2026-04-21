---
type: concept
title: Pass Rate
slug: pass-rate
date: 2026-04-20
updated: 2026-04-20
aliases: [通过率]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Pass Rate** (通过率) — the proportion of evaluation instances that satisfy all verification criteria, typically averaged across repeated trials.

## Key Points

- SkillsBench uses pass rate as its primary metric, averaging binary rewards over `5` trials per task and then averaging across `84` tasks.
- The paper reports both absolute pass-rate deltas and normalized gain to distinguish genuine scaffolding from ceiling effects.
- Mean pass rate rises from `24.3%` without skills to `40.6%` with curated skills, while self-generated skills average `21.0%`.
- Domain-level pass rates reveal that skill efficacy is highly heterogeneous, ranging from modest gains in software engineering to very large gains in healthcare and manufacturing.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2026-skillsbench-2602-12670]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2026-skillsbench-2602-12670]].
