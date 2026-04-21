---
type: concept
title: Semantic Alignment
slug: semantic-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [semantic alignment, 语义对齐]
tags: [alignment, representation-learning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Semantic Alignment** (语义对齐) — the process of mapping representations from different views into a shared space so that semantically corresponding signals become comparable and mutually informative.

## Key Points

- RLMRec aligns collaborative user/item embeddings with LLM-derived semantic embeddings instead of using textual semantics as raw side features only.
- The paper formalizes alignment through mutual information maximization between collaborative representations `e` and semantic representations `s`.
- It studies two alignment mechanisms: contrastive matching with in-batch negatives and masked generative reconstruction in semantic space.
- The alignment objective is meant to filter noisy implicit-feedback signals by preserving information that is shared across collaborative and textual views.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ren-2024-representation]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ren-2024-representation]].
