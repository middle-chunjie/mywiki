---
type: concept
title: Weak-to-Strong Monitoring
slug: weak-to-strong-monitoring
date: 2026-04-20
updated: 2026-04-20
aliases: [weak monitor strong agent oversight, 弱到强监控]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Weak-to-Strong Monitoring** (弱到强监控) — the use of weaker or cheaper monitors to oversee stronger, potentially more capable agents without losing too much reliability.

## Key Points

- The paper studies how monitor reliability changes with the capability gap between attacker and monitor.
- It evaluates `36` attacker-monitor pairs spanning Claude, Gemini, GPT-4o, and o-series models.
- Hybrid monitoring weakens the dependence of monitoring quality on raw model capability compared with baseline monitoring.
- Even weaker monitors such as GPT-4o-mini and Claude 3.6 Haiku achieve `AUC > 0.85` in the reported setup.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
