---
type: concept
title: Bootstrap Query Corpus
slug: bootstrap-query-corpus
date: 2026-04-20
updated: 2026-04-20
aliases: [引导查询语料]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Bootstrap Query Corpus** (引导查询语料) — a trusted set of query texts used to define the target semantic distribution for filtering or model initialization.

## Key Points

- The paper constructs its bootstrap corpus from StackOverflow Java titles starting with `how to`.
- Titles are additionally filtered by the rule-based syntactic cleaner before being used for semantic modeling.
- The final corpus contains `168,779` titles out of `1,709,703` Java-related candidates.
- This corpus provides the reference distribution that the VAE uses to judge whether GitHub comments look like natural queries.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-importance-2202-06649]].
