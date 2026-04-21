---
type: concept
title: Listwise Ranking
slug: listwise-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [list-wise ranking, 列表排序]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Listwise Ranking** (列表排序) — a ranking formulation that learns from the relative ordering of a full candidate list rather than from isolated positives or pairwise labels alone.

## Key Points

- UDR converts LM feedback on candidate demonstrations into ranks instead of binary keep/drop labels.
- The paper uses a LambdaRank-style objective where better-ranked demonstrations are pushed above worse-ranked ones with weights derived from rank gaps.
- This design captures finer supervision than EPR's binary labeling and is a main source of UDR's gains.
- Ablation shows removing the rank loss reduces average performance from `58.4` to `55.7` across the reported tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-unified]].
