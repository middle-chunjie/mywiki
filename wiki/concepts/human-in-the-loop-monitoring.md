---
type: concept
title: Human-in-the-Loop Monitoring
slug: human-in-the-loop-monitoring
date: 2026-04-20
updated: 2026-04-20
aliases: [HiLT monitoring, human escalation, 人在回路监控]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Human-in-the-Loop Monitoring** (人在回路监控) — a monitoring workflow in which human reviewers inspect or challenge automated monitor outputs before a final decision is made.

## Key Points

- The paper places humans after an LLM monitor rather than using humans as the first-line monitor.
- Human reviewers can interrogate the monitor, inspect trajectories, and provide feedback that updates the final suspiciousness score.
- Naive human feedback hurts overall reliability, while selective escalation with strong priors improves low-FPR recall.
- The paper treats escalation policy design as a major component of effective human oversight.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
