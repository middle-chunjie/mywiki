---
type: concept
title: Proposition-Level Ranking
slug: proposition-level-ranking
date: 2026-04-20
updated: 2026-04-20
aliases: [atomic-fact ranking, 命题级排序]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Proposition-Level Ranking** (命题级排序) — ranking atomic propositions or facts rather than full sentences or passages, typically for fine-grained evidence retrieval and attribution.

## Key Points

- AGRaME uses the proposition tokens inside a sentence as the effective query while keeping the sentence encoding as the contextual representation.
- On PropSegmEnt, the paper shows that multi-vector retrieval already outperforms specialized proposition encoders such as SubEncoder.
- The reported gains suggest that retaining sentence context helps proposition token vectors disambiguate the intended claim.
- The paper argues proposition-level training supervision should improve this setting further, but leaves that extension as future work.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[reddy-2024-agrame-2405-15028]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[reddy-2024-agrame-2405-15028]].
