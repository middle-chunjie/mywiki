---
type: concept
title: User Profiling
slug: user-profiling
date: 2026-04-20
updated: 2026-04-20
aliases: [user profiling, 用户画像]
tags: [recommendation, personalization]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**User Profiling** (用户画像) — constructing a compact textual or structured description of a user's preferences from observed behaviors and auxiliary information for downstream personalization.

## Key Points

- RLMRec generates each user profile with an LLM rather than relying only on sparse user IDs or one-hot attributes.
- The user prompt is built from sampled interacted items, generated item profiles, and the user's own review text, so the profile reflects both collaborative and textual evidence.
- The paper treats user profiling as a denoising step that can expose stable preference semantics even when implicit feedback is noisy.
- Generated user profiles are embedded into semantic vectors and aligned with collaborative embeddings during recommender training.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2024-representation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2024-representation]].
