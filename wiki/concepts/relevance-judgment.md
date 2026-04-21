---
type: concept
title: Relevance Judgment
slug: relevance-judgment
date: 2026-04-20
updated: 2026-04-20
aliases: [相关性标注, relevance judgement]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Relevance Judgment** (相关性标注) — an annotation that specifies whether and how strongly a document or passage satisfies an information need.

## Key Points

- The paper converts crowd annotations into intent-aware QRel scores for each `(intent, passage)` pair.
- Scoring is ordinal: `0` if no annotator selected the intent, `1` if at least one annotator selected it, and `2` if all annotators selected it.
- Pairs with fewer than two annotators are dropped during cleanup to improve agreement quality.
- The resulting judgments enable evaluation of ranking quality against explicit user intents rather than generic query relevance alone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[anand-2024-understanding-2408-17103]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[anand-2024-understanding-2408-17103]].
