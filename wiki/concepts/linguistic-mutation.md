---
type: concept
title: Linguistic Mutation
slug: linguistic-mutation
date: 2026-04-20
updated: 2026-04-20
aliases: [prompt mutation, 语言变体扰动]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Linguistic Mutation** (语言变体扰动) — a controlled transformation of a prompt's wording, style, language, or encoding used to test model sensitivity to surface-form variation.

## Key Points

- The benchmark defines `20` mutations covering writing style, persuasion, encoding/encryption, and multilingual translation.
- Applying these mutations to the `440` base prompts yields `8,800` additional unsafe instructions.
- The paper shows that seemingly superficial mutations can move fulfillment rates by large margins, especially persuasion and technical-term rewrites.
- The mutations are meant to capture realistic user-side phrasing diversity rather than full worst-case jailbreak attacks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xie-2024-sorrybench-2406-14598]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xie-2024-sorrybench-2406-14598]].
