---
type: concept
title: Retrieval Utility
slug: retrieval-utility
date: 2026-04-20
updated: 2026-04-20
aliases: [RU, 检索效用]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Retrieval Utility** (检索效用) — a metric that measures how much a retrieved document improves prediction relative to a null-document baseline.

## Key Points

- REALM defines `RU(z|x) = log p(y|z, x) - log p(y|∅, x)`.
- Positive RU indicates that a retrieved passage is more helpful than using the null document, while negative RU suggests the passage is unhelpful or unnecessary.
- The appendix reports that RU rises during pre-training and tracks downstream open-QA quality more closely than the overall log-likelihood.
- RU depends on both retrieval quality and whether the masked target truly requires external world knowledge.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guu-2020-realm-2002-08909]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guu-2020-realm-2002-08909]].
