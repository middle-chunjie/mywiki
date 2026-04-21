---
type: concept
title: FM-Index
slug: fm-index
date: 2026-04-20
updated: 2026-04-20
aliases: [Ferragina-Manzini index, FM index, FM 索引]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**FM-Index** (FM 索引) — a compressed full-text index that supports substring existence, counting, locating, and extraction operations used to constrain generation to text spans that appear in a corpus.

## Key Points

- RetroLLM builds a corpus-level FM-index over the full Wikipedia corpus and a document-level FM-index for each candidate document.
- The corpus-level index constrains clue generation so predicted clue phrases must occur in the corpus.
- Document-level indexes support evidence generation by locating clue positions, extracting future windows, and restricting allowed next tokens.
- The paper treats FM-index operations such as `Loc` and `Ext` as the mechanism that turns retrieval into token-level constrained generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-retrollm-2412-11919]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-retrollm-2412-11919]].
