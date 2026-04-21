---
type: concept
title: Knowledge Concept Filtering
slug: knowledge-concept-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [knowledge concept filtering, concept-consistency filtering, 知识概念过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Concept Filtering** (知识概念过滤) — a retrieval post-filtering step that keeps only evidence whose semantic concept matches the fine-grained concept label of the current problem.

## Key Points

- [[dong-2024-progressive-2412-14835]] introduces this module because retrieved evidence with the wrong fine-grained concept can hurt multimodal reasoning even when it is superficially similar.
- The filter keeps only retrieved items satisfying both the original retrieval threshold `T_r` and the concept-consistency threshold `T_kc`.
- The paper motivates the design with math benchmarks where labels such as "Angles and Length" expose concept mismatch that raw similarity alone misses.
- Ablation results show that removing concept filtering lowers performance on MathVista, We-Math, and GAOKAO-MM, indicating the filter reduces retrieval noise.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dong-2024-progressive-2412-14835]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dong-2024-progressive-2412-14835]].
