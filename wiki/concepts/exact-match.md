---
type: concept
title: Exact Match
slug: exact-match
date: 2026-04-20
updated: 2026-04-20
aliases: [EM, 精确匹配]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Exact Match** (精确匹配) — an evaluation metric that counts a prediction as correct only when it matches the reference answer exactly after normalization.

## Key Points

- The paper uses EM together with F1 as the main evaluation metric family for multi-hop QA datasets.
- CoRAG improves EM by more than `10` points over a same-backbone fine-tuned baseline on 2WikiMultiHopQA, Bamboogle, and MuSiQue.
- On the KILT hidden test set, EM-style downstream scores include `63.1` on NQ and `60.6` on HoPo, supporting the claim that CoRAG generalizes beyond synthetic multi-hop settings.
- EM is also operationally relevant during rejection sampling because chain generation may stop when an intermediate sub-answer matches the gold final answer.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2025-chainofretrieval-2501-14342]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2025-chainofretrieval-2501-14342]].
