---
type: concept
title: Planning Hallucination
slug: planning-hallucination
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: [agents, planning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Planning Hallucination** — a failure mode where an agent generates plausible-looking plans or actions that are poorly grounded in the actual environment dynamics, constraints, or task state.

## Key Points

- [[xiong-2025-mpo-2503-02682]] frames planning hallucination as a key weakness of implicit plan-on-the-fly agent frameworks such as ReAct-style execution.
- The paper argues that abstract external guidance can reduce myopic or misleading action sequences without requiring the full agent to be retrained.
- MPO addresses this failure mode by conditioning action generation on a vetted meta plan rather than leaving all planning burden to the acting policy at inference time.
- The empirical gains on unseen ALFWorld and ScienceWorld splits support the claim that better-guided planning reduces brittle exploration behavior.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2025-mpo-2503-02682]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2025-mpo-2503-02682]].
