---
type: concept
title: Hard-Negative Mining
slug: hard-negative-mining
date: 2026-04-20
updated: 2026-04-20
aliases: [hard negative mining, 困难负例挖掘]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hard-Negative Mining** (困难负例挖掘) — a data curation strategy that selects difficult non-relevant examples that look plausible for a query, making contrastive training discriminate among semantically close candidates.

## Key Points

- [[lee-2024-nvembed-2405-17428]] adopts positive-aware hard-negative mining to curate retrieval training data for NV-Embed.
- The teacher model is [[e5-mistral-7b-instruct]], which scores candidate passages and filters likely false negatives using the positive passage score.
- The paper sets a maximum negative threshold of `` `pos_score * 0.95` `` under the TopKPercPos heuristic.
- Adding hard-negative mining raises retrieval performance from `59.22` to `61.52` in the staged data-curation ablation.
- The method is presented as one of the main reproducible ingredients behind the jump from the early recipe toward NV-Embed-v2.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[lee-2024-nvembed-2405-17428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[lee-2024-nvembed-2405-17428]].
