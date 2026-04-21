---
type: concept
title: Retrieval Robustness
slug: retrieval-robustness
date: 2026-04-20
updated: 2026-04-20
aliases: [retrieval robustness]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Robustness** — the property that a retrieval-augmented system benefits from relevant evidence but does not degrade when retrieved context is irrelevant or noisy.

## Key Points

- The paper defines robust behavior with two conditions: retrieved context should improve performance when relevant and should not hurt when irrelevant.
- Irrelevant retrieval can corrupt both final answers and intermediate decomposition steps, especially in multi-hop QA.
- An NLI-based entailment check is an effective but overly strict robustness mechanism because it blocks some genuinely useful retrieval.
- Fine-tuning on a mixture of top-1, low-ranked, and random passages makes the model substantially more stable under noisy retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-making-2310-01558]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-making-2310-01558]].
