---
type: concept
title: Utility-Aware Retrieval
slug: utility-aware-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [效用感知检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Utility-Aware Retrieval** (效用感知检索) — a retrieval strategy that ranks candidate memories using both semantic relevance and an empirical estimate of downstream usefulness.

## Key Points

- D2Skill first filters candidate skills by cosine similarity and then re-ranks them with a score combining normalized similarity, learned utility, and a UCB-style exploration bonus.
- The utility term favors skills that have historically improved rollout outcomes rather than only matching the query surface form.
- The exploration bonus prevents low-frequency skills from being ignored before they are adequately evaluated.
- Ablation results show that removing utility-aware retrieval lowers ALFWORLD validation success from `72.7` to `64.8`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tu-2026-dynamic-2603-28716]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tu-2026-dynamic-2603-28716]].
