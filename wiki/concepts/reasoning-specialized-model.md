---
type: concept
title: Reasoning-Specialized Model
slug: reasoning-specialized-model
date: 2026-04-20
updated: 2026-04-20
aliases: [reasoning model, test-time compute model, 推理专用模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Reasoning-Specialized Model** (推理专用模型) — a language model optimized to spend additional test-time compute on intermediate reasoning steps in order to solve harder problems more reliably.

## Key Points

- The paper contrasts general-purpose models with reasoning-specialized models such as o3-mini (high), using BBEH as a broad stress test beyond math and coding.
- Reasoning-specialized models substantially outperform general-purpose models overall on BBEH, reaching `44.8%` adjusted harmonic mean versus `9.8%` for the best general model.
- Their gains are largest on formal tasks involving counting, planning, arithmetic, and algorithmic reasoning.
- Their gains are much smaller on softer skills such as humour, sarcasm, commonsense, and causal understanding.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kazemi-2025-bigbench-2502-19187]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kazemi-2025-bigbench-2502-19187]].
