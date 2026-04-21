---
type: concept
title: Same-Tower Negatives
slug: same-tower-negatives
date: 2026-04-20
updated: 2026-04-20
aliases: [query-query negatives]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Same-Tower Negatives** — negatives taken from representations produced by the same encoder tower, such as other queries in the batch for a dual-encoder retriever.

## Key Points

- [[lee-2024-gecko-2403-20327]] adds other queries `q_j` in the batch as negatives alongside positive and hard-negative passages.
- The loss masks out the current example with an indicator term so only `j != i` contributes as same-tower negatives.
- The paper argues that these negatives are especially useful for symmetric tasks such as [[semantic-textual-similarity]], where both sides are query-like texts.
- Gecko combines same-tower negatives with NLI data and reports improved STS performance in the ablation study.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-gecko-2403-20327]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-gecko-2403-20327]].
