---
type: concept
title: Prompt Injection Attack
slug: prompt-injection-attack
date: 2026-04-20
updated: 2026-04-20
aliases: [monitor hijacking, prompt injection, 提示注入攻击]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Injection Attack** (提示注入攻击) — an attack that inserts adversarial instructions into the model-consumed context to steer a monitor or agent away from its intended objective.

## Key Points

- The paper studies prompt injection as a monitor-hijacking strategy embedded directly into agent trajectories.
- Injected text attempts to override the monitor's scoring behavior and force low suspiciousness outputs.
- Baseline full-trajectory monitors are much more vulnerable to this attack than hierarchical monitors.
- The attack can be inserted at trajectory boundaries or repeated at each step, affecting scaffoldings differently.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
