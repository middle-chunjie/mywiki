---
type: concept
title: Retrieval-Oriented Pre-Training
slug: retrieval-oriented-pre-training
date: 2026-04-20
updated: 2026-04-20
aliases: [检索导向预训练, retrieval-aware pre-training]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Oriented Pre-Training** (检索导向预训练) — pre-training designed to improve query-document representation quality for retrieval, rather than optimizing only generic token prediction or generation objectives.

## Key Points

- RetroMAE is motivated by the claim that standard token-level pre-training under-develops sentence representations needed for dense retrieval.
- The method avoids reliance on sophisticated data augmentation and massive negative sampling used by self-contrastive retrieval pre-training.
- It uses unlabeled corpora to train an encoder whose sentence embedding must support reconstruction under difficult masking conditions.
- The reported gains appear in both BEIR transfer after MS MARCO supervision and supervised MS MARCO / Natural Questions retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2022-retromae-2205-12035]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2022-retromae-2205-12035]].
