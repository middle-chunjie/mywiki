---
type: concept
title: Sequential Monitoring
slug: sequential-monitoring
date: 2026-04-20
updated: 2026-04-20
aliases: [rolling monitor summarization, 顺序监控]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Sequential Monitoring** (顺序监控) — a monitoring architecture that processes trajectory chunks in order while carrying forward a running summary of previously observed behavior.

## Key Points

- The monitor evaluates one chunk at a time and conditions on the previous summary for the next chunk.
- This design maps naturally to online monitoring because it only needs a single left-to-right pass.
- The paper finds sequential monitoring can benefit from larger chunk sizes than hierarchical monitoring.
- Sequential monitoring alone has notably weak `TPR` in some SHADE-Arena settings, despite being complementary to hierarchical monitoring.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kale-2026-reliable-2508-19461]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kale-2026-reliable-2508-19461]].
