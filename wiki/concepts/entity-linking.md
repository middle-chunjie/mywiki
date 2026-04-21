---
type: concept
title: Entity Linking
slug: entity-linking
date: 2026-04-20
updated: 2026-04-20
aliases: [实体链接]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Entity Linking** (实体链接) — the task of mapping a mention or query to the correct entity in a fixed candidate inventory.

## Key Points

- The paper uses zero-shot entity linking on ZeShEL as one of its two main evaluation settings for cross-encoder `k`-NN search.
- In this setup, the items being indexed are entity candidates rather than generic text passages.
- The work reports that improvements in nearest-neighbor recall can translate into better entity-linking accuracy, though exact cross-encoder search is not always optimal downstream.
- YuGiOh and Star Trek are the concrete ZeShEL domains used to test the proposed indexing and adaptive retrieval methods.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-adaptive-2405-03651]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-adaptive-2405-03651]].
