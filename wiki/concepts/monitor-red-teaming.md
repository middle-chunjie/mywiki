---
type: concept
title: Monitor Red Teaming
slug: monitor-red-teaming
date: 2026-04-20
updated: 2026-04-20
aliases: [MRT, monitor robustness evaluation, 监控红队测试]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Monitor Red Teaming** (监控红队测试) — the adversarial evaluation of a monitoring system's ability to detect harmful or deceptive agent behavior under explicit attacker threat models.

## Key Points

- The paper defines MRT as red teaming the **monitor**, not just the agent, making monitor reliability the primary object of evaluation.
- MRT is parameterized by agent awareness, monitor awareness, evasion strategies, and monitoring scaffolding.
- The workflow uses complete trajectories with ground-truth labels for whether the agent attempted the harmful side task.
- The paper argues that `AUC-ROC` and `TPR @ FPR = 0.01` are more deployment-relevant than raw detection rate.
- MRT is used to compare baseline, hierarchical, sequential, and hybrid monitors on both tool-calling and computer-use agents.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
