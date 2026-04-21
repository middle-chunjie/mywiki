---
type: concept
title: Prompt Rewriting
slug: prompt-rewriting
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt rewrite, 提示词改写]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Prompt Rewriting** (提示词改写) — an inference-time intervention that transforms a user's prompt into a more policy-compliant or generic alternative before generation.

## Key Points

- The paper simulates DALL-E-style prompt rewriting with a GPT-4 pipeline adapted from an extracted system prompt.
- Rewriting is intended to preserve user intent while reducing similarity to copyrighted characters by broadening or genericizing the request.
- On Playground v2.5, rewriting alone lowers `DETECT` from `30.33` to `14.33`, but this still leaves many failures.
- Failed rewritten prompts tend to retain more top LAION keywords and higher embedding similarity to the original character name.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-fantastic-2406-14526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-fantastic-2406-14526]].
