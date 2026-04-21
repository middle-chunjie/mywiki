---
type: concept
title: Clue Generation
slug: clue-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [query clue generation, 线索生成]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Clue Generation** (线索生成) — a retrieval-oriented generation step in which a model predicts short corpus-grounded phrases that help identify relevant documents before fuller evidence generation.

## Key Points

- RetroLLM generates clues under corpus-level FM-index constraints so each clue phrase must appear in the corpus.
- Generated clues are weighted by corpus frequency and document frequency, favoring rarer and more discriminative phrases.
- The model combines generated clues with auxiliary lexical clues from SPLADE-v3 to improve recall.
- These clues define the candidate document set that later constrains evidence generation and reduces false pruning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-retrollm-2412-11919]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-retrollm-2412-11919]].
