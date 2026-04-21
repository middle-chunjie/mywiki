---
type: concept
title: New Intent Discovery
slug: new-intent-discovery
date: 2026-04-20
updated: 2026-04-20
aliases: [NID, Intent Discovery, Open Intent Discovery, 新意图发现]
tags: [dialogue, intent, clustering, open-world, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**New Intent Discovery** (新意图发现, NID) — the task of automatically uncovering novel, previously unseen intent categories from unlabeled user utterances in a deployed dialogue system, enabling iterative expansion of the supported intent set.

## Key Points

- NID can be framed as unsupervised (no known-intent labels from current domain) or semi-supervised (some known intents and labeled examples provided); the proportion of known intents is called the Known Class Ratio (KCR).
- It differs from zero-shot learning in that no class information is assumed during training, and from standard OOD detection in that the goal is to discover the underlying cluster structure, not merely flag unknowns.
- Utterance representation quality is critical: vanilla PLM embeddings (e.g., raw BERT) perform poorly because they are not attuned to intent semantics, motivating task-specific pre-training strategies.
- Pseudo-labeling approaches (the pre-MTP-CLNN dominant paradigm) suffer from label noise that propagates through training; neighborhood-based contrastive learning avoids explicit label assignment.
- Performance degrades less gracefully with reduced KCR when the representation learning stage uses task-specific pre-training (MTP drops only ~8.5% NMI vs. ~21.6% for DAC on M-CID from KCR=75% to KCR=25%).

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-new-2205-12914]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-new-2205-12914]].
