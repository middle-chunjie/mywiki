---
type: concept
title: User Intent
slug: user-intent
date: 2026-04-20
updated: 2026-04-20
aliases: [用户意图, search intent]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**User Intent** (用户意图) — the underlying information need a user aims to satisfy with a query, which may be more specific than the surface query wording.

## Key Points

- This paper treats the gap between query text and the actual information need as a measurable failure mode for ranking systems.
- DL-MIA annotates multiple explicit intents for ambiguous TREC-DL queries rather than assigning only query-level relevance labels.
- Candidate intents are first generated with an LLM and then refined through human review to remove hallucinated, redundant, or answer-like formulations.
- Passage-level labels are attached to individual intents so that the same query can support intent-specific ranking evaluation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[anand-2024-understanding-2408-17103]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[anand-2024-understanding-2408-17103]].
