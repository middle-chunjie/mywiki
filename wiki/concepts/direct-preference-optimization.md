---
type: concept
title: Direct Preference Optimization
slug: direct-preference-optimization
date: 2026-04-20
updated: 2026-04-20
aliases: [DPO, 直接偏好优化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Direct Preference Optimization** (直接偏好优化) — a preference-learning method that directly optimizes a model toward preferred completions over rejected ones without explicit reward-model fitting.

## Key Points

- Phi-4 uses two DPO rounds after SFT: a first round built from pivotal-token pairs and a second round built from full-response comparisons judged by GPT-4o.
- The first DPO stage is especially beneficial on reasoning-heavy tasks, while the judge-guided second stage is especially helpful on ArenaHard-style judged evaluations.
- Safety and anti-hallucination data are mixed into both DPO stages rather than treated as a separate alignment pipeline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdin-2024-phi-2412-08905]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdin-2024-phi-2412-08905]].
