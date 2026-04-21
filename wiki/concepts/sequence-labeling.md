---
type: concept
title: Sequence Labeling
slug: sequence-labeling
date: 2026-04-20
updated: 2026-04-20
aliases: [序列标注]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Sequence Labeling** (序列标注) — a modeling setup that assigns one structured label to each token in an input sequence while preserving dependencies across positions.

## Key Points

- [[fang-2021-guided]] casts concept extraction as sequence labeling over sentence tokens rather than phrase ranking or sequence generation.
- The label space is `S-CON`, `B-CON`, `I-CON`, `E-CON`, and `O`, allowing single-token and multi-token concepts to be represented explicitly.
- A Bi-LSTM encoder provides contextual token states, and a CRF decoder models output-label dependencies.
- Guided attention adds document-level and clue-word-level information before decoding, so the sequence labels are not predicted from local context alone.
- The paper compares this framing against earlier neural and non-neural extraction baselines and reports stronger F1 on all three datasets.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[fang-2021-guided]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[fang-2021-guided]].
