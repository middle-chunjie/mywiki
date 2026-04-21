---
type: concept
title: Trajectory Abstraction
slug: trajectory-abstraction
date: 2026-04-20
updated: 2026-04-20
aliases: [轨迹抽象]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Trajectory Abstraction** (轨迹抽象) — converting verbose interaction traces into compact structured representations that preserve reusable decision logic while removing redundant exploratory detail.

## Key Points

- SkillRL treats both successful and failed trajectories as raw material for abstraction rather than storing them verbatim.
- Successful trajectories are distilled into strategic skills, while failed trajectories become concise counterfactual lessons.
- The paper reports roughly `10-20x` compression relative to raw trajectories without losing decision utility.
- Ablation with raw trajectories instead of skills shows a large performance drop, supporting abstraction over memorization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xia-2026-skillrl-2602-08234]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xia-2026-skillrl-2602-08234]].
