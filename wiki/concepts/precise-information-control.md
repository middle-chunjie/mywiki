---
type: concept
title: Precise Information Control
slug: precise-information-control
date: 2026-04-20
updated: 2026-04-20
aliases: [PIC, precise information control, 精确信息控制]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Precise Information Control** (精确信息控制) — a claim-level generation objective requiring a model to produce responses grounded only in an explicitly provided set of verifiable claims.

## Key Points

- The paper defines two PIC modes: full PIC requires including exactly all input claims, while partial PIC permits only a relevant subset with no unsupported additions.
- PIC treats controllability as an information-grounding problem rather than a general factuality check against external knowledge.
- The formulation operates on extracted claim sets `C` and `C'`, enabling precision, recall, and `F_1@K` evaluation at the claim level.
- PIC is designed to prioritize user-provided contextual knowledge over a model's parametric memory, especially when the two conflict.
- The paper turns PIC into both a benchmark target through PIC-Bench and a post-training target through PIC-LM.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2025-precise-2506-06589]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2025-precise-2506-06589]].
