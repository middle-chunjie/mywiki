---
type: concept
title: Semantic Filter
slug: semantic-filter
date: 2026-04-20
updated: 2026-04-20
aliases: [model-based semantic filter, semantics filter, 语义过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semantic Filter** (语义过滤) — a learned filter that keeps or discards inputs based on whether their meaning matches a target semantic distribution.

## Key Points

- The paper’s second-stage filter ranks comments by VAE reconstruction loss relative to a StackOverflow query corpus.
- It targets comments that are syntactically valid but still semantically unlike real developer search queries.
- Comments with lower anomaly scores are treated as more query-appropriate.
- EM-GMM is used to partition the ranked comments into qualified and unqualified groups automatically.
- The full pipeline performs better than using only the rule-based or only the model-based stage.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2022-importance-2202-06649]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2022-importance-2202-06649]].
