---
type: concept
title: Interactive Retrieval
slug: interactive-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [interactive retrieval, 交互式检索]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Interactive Retrieval** (交互式检索) — a retrieval setting in which the system and user exchange feedback across multiple turns to refine the search result.

## Key Points

- [[levy-nd-chatting]] operationalizes interactive retrieval through alternating question generation and image ranking rounds.
- The system actively asks for missing evidence instead of waiting for the user to repeatedly rewrite full queries.
- Success is measured cumulatively by whether the target image enters top-`10` at any round, matching the user behavior of stopping once the target is found.
- Later rounds still help, but gains eventually saturate, highlighting the need for better question selection policies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[levy-nd-chatting]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[levy-nd-chatting]].
