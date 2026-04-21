---
type: concept
title: Scalable Oversight
slug: scalable-oversight
date: 2026-04-20
updated: 2026-04-20
aliases: [scalable oversight, 可扩展监督]
tags: [alignment, llm, evaluation, safety]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Scalable Oversight** (可扩展监督) — a paradigm for supervising AI systems whose capabilities approach or exceed human-level performance in specific domains, by using automated or AI-assisted mechanisms to maintain reliable oversight at scale.

## Key Points

- The core challenge is that humans cannot efficiently verify every step of reasoning produced by highly capable models, so automated supervision must take over.
- Process error identification—pinpointing the earliest erroneous reasoning step—is one concrete mechanism enabling scalable oversight of mathematical reasoning.
- Critic models and process reward models (PRMs) are the two main automated approaches: PRMs score steps during training/search; critic models provide natural-language feedback.
- [[processbench]] demonstrates that current open-source models still fall far short of o1-mini-level critique quality, quantifying the oversight gap.
- Scalable oversight is closely tied to the reward signal problem: models that reach correct final answers via flawed reasoning paths are mis-supervised by outcome-only rewards.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zheng-2024-processbench-2412-06559]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zheng-2024-processbench-2412-06559]].
