---
type: concept
title: Information-Seeking Formalization
slug: information-seeking-formalization
date: 2026-04-20
updated: 2026-04-20
aliases: [IS formalization, formalized information seeking, 信息寻求形式化]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Information-Seeking Formalization** (信息寻求形式化) — a structured specification of an information-seeking question as recursive compositions of set-valued relations over entities, so question difficulty and reasoning structure can be controlled before retrieval.

## Key Points

- WebShaper models the answer set `T` with intersections and unions of Knowledge Projections rather than free-form natural-language templates.
- The formalization supports recursion by replacing constants with intermediate target sets such as `T_1, \dots, T_k`.
- The paper introduces a triplet-based KP representation to flatten recursive structures into a list the Expander can interpret.
- This formal layer is intended to reduce reasoning-structure mismatches and answer inconsistencies common in information-driven synthesis.
- Ablation results report that formalized synthesis outperforms a natural-language-only expansion variant across all tested backbones.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tao-2025-webshaper-2507-15061]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tao-2025-webshaper-2507-15061]].
