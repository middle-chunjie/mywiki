---
type: concept
title: Language Mixing Strategy
slug: language-mixing-strategy
date: 2026-04-20
updated: 2026-04-20
aliases: [language mixing, batch language assignment, 语言混合策略]
tags: [training, multilingual, batching]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Language Mixing Strategy** (语言混合策略) — the rule for assigning languages to examples or passages inside a multilingual training mini-batch.

## Key Points

- [[yang-2024-distillation-2405-00977]] treats batch language assignment as a first-class design choice for multilingual late-interaction retrieval training.
- The paper compares `Mix Passages`, `Mix Entries`, and `Round Robin Entries`, which trade off direct multilingual ranking exposure, translation-bias control, and uniform language coverage.
- `Mix Passages` assigns languages per passage, whereas `Mix Entries` assigns one language to all passages under a query, and `Round Robin Entries` repeats queries across languages.
- The reported MAP and Recall scores are statistically similar across most collections, implying the model is robust to the exact strategy if several languages are present per batch.
- Round-robin mixing is the most memory-hungry variant because repeated queries reduce the number of distinct entries that fit on a GPU.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yang-2024-distillation-2405-00977]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yang-2024-distillation-2405-00977]].
