---
type: concept
title: Negative Prompt
slug: negative-prompt
date: 2026-04-20
updated: 2026-04-20
aliases: [negative prompting, 负向提示词]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Negative Prompt** (负向提示词) — a text condition supplied during generation to discourage specific attributes, concepts, or targets from appearing in the output.

## Key Points

- The paper studies negative prompts as a copyright-mitigation mechanism rather than as a pure aesthetic-control feature.
- It tests negative prompts containing the target name, "Copyrighted character", and top-ranked keywords from LM, embedding, and co-occurrence methods.
- Including the character name in the negative prompt consistently reduces `DETECT` more than generic negative prompts alone.
- The strongest setting combines rewritten prompts with target name plus `5` EmbeddingSim and `5` Co-Occurrence keywords, substantially lowering `DETECT` while preserving `CONS`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-fantastic-2406-14526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-fantastic-2406-14526]].
