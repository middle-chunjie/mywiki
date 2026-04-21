---
type: concept
title: Consistency Filtering
slug: consistency-filtering
date: 2026-04-20
updated: 2026-04-20
aliases: [一致性过滤]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Consistency Filtering** (一致性过滤) — a data-cleaning procedure that removes weakly supervised pairs whose purported positive document is not sufficiently supported by a pretrained similarity model.

## Key Points

- The paper uses consistency filtering to clean large public pair datasets before contrastive pretraining.
- For each candidate pair, the authors embed queries and documents separately and keep the pair only if the labeled document falls within the query's top-`2` nearest neighbors.
- The filtering model is `gte-base`, chosen over `all-MiniLM-L6-v2` because it preserves more true retrieval positives with low lexical overlap.
- This step reduces the raw contrastive corpus from `470M` pairs to about `235M`.
- The paper reports abandoning threshold-based filtering because manual inspection and downstream retrieval scores were worse than with top-`k` filtering.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[nussbaum-2025-nomic-2402-01613]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[nussbaum-2025-nomic-2402-01613]].
