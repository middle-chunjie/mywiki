---
type: concept
title: Style Transfer
slug: style-transfer
date: 2026-04-20
updated: 2026-04-20
aliases: [text style transfer, 风格迁移]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Style Transfer** (风格迁移) — the task of rewriting text so that stylistic form changes while core semantic content remains substantially preserved.

## Key Points

- [[gou-2023-diversify]] treats diverse question generation as a style transfer problem from a base question template to alternative retrieved templates.
- The transfer model conditions on both context `x` and retrieved style `z`, so stylistic change is constrained by answer consistency.
- The paper avoids predefined style labels; instead, style is represented by automatically extracted question templates.
- Diversity reward encourages the generated question to mimic the retrieved template's surface form without fully copying it.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gou-2023-diversify]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gou-2023-diversify]].
