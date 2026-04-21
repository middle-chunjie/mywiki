---
type: concept
title: Objectivity
slug: objectivity
date: 2026-04-20
updated: 2026-04-20
aliases: [neutrality, 客观性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Objectivity** (客观性) — the degree to which a generated response remains neutral and minimally influenced by subjective sentiment or user bias.

## Key Points

- The paper measures objectivity in longform generation by first asking GPT-4 for a sentiment score `S in [-5, 5]` and then computing `5 - |S|`.
- Baseline longform generations score `2.23` on objectivity, oracle prompts score `3.0`, and S2A reaches `3.82`.
- S2A improves objectivity even when overall answer quality stays roughly unchanged at `4.6-4.7` out of `5`.
- The reported gains suggest that filtering opinionated context can matter independently of final response fluency.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[weston-2023-system-2311-11829]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[weston-2023-system-2311-11829]].
