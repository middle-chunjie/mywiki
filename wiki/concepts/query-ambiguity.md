---
type: concept
title: Query Ambiguity
slug: query-ambiguity
date: 2026-04-20
updated: 2026-04-20
aliases: [查询歧义, ambiguous query]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Query Ambiguity** (查询歧义) — the property that a query can correspond to multiple plausible information needs or interpretations.

## Key Points

- The paper focuses on TREC-DL queries that express at least two distinct intents after generation and filtering.
- Ambiguity is operationalized through multiple passage-supported intents instead of only a taxonomy label.
- Candidate ambiguous queries are selected by generating intents, clustering them, and retaining only queries with `>= 2` intents after manual screening.
- Example cases such as "slow cooking food" and "what is 311 for" illustrate how a short query can hide multiple valid intent readings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[anand-2024-understanding-2408-17103]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[anand-2024-understanding-2408-17103]].
