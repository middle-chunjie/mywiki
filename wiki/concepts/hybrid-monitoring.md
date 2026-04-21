---
type: concept
title: Hybrid Monitoring
slug: hybrid-monitoring
date: 2026-04-20
updated: 2026-04-20
aliases: [mu-hybrid monitoring, 混合监控]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hybrid Monitoring** (混合监控) — a monitor that combines hierarchical and sequential suspiciousness scores to improve robustness over either component alone.

## Key Points

- The paper defines the hybrid score as `mu * h + (1 - mu) * s`, where `h` is the hierarchical score and `s` is the sequential score.
- Main experiments fix `mu = 0.5` to avoid overfitting and to equally trust both components.
- Hybrid monitoring is the strongest monitor scaffolding reported across awareness levels and datasets.
- The paper argues hybrid monitoring enables a practical weak-to-strong effect, allowing weaker models to monitor stronger agents more reliably.
- The method is more robust to adversarial evasion than the baseline full-trajectory monitor.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
