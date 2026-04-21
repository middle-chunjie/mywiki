---
type: concept
title: Character Name Anchoring
slug: character-name-anchoring
date: 2026-04-20
updated: 2026-04-20
aliases: [name anchoring, 角色名锚定]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Character Name Anchoring** (角色名锚定) — the direct elicitation of a copyrighted character by explicitly including that character's name in the generation prompt.

## Key Points

- The paper uses character-name anchoring as the obvious baseline against which indirect anchoring and mitigations are compared.
- On the studied models, direct name prompts generate roughly `~60%` of the tested copyrighted characters.
- The authors find that this mode is brittle to misspellings: replacing three letters in a name can collapse generation success from `33` to `1` detected characters on Playground v2.5.
- DALL-E 3 blocks this mode more effectively than indirect anchoring because it explicitly filters prompts naming copyrighted characters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2024-fantastic-2406-14526]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2024-fantastic-2406-14526]].
