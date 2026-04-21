---
type: concept
title: Unsupervised Keyphrase Extraction
slug: unsupervised-keyphrase-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [Unsupervised Keyphrase Extraction, 无监督关键词抽取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Unsupervised Keyphrase Extraction** (无监督关键词抽取) — the task of identifying salient phrases from a document without supervised labels, usually by combining lexical, positional, or semantic signals.

## Key Points

- [[li-2024-chulo-2410-11119]] uses unsupervised keyphrase extraction to identify the most semantically important phrases in an entire long document.
- Its Semantic Keyphrase Prioritization algorithm adapts PromptRank from first-segment ranking to document-level ranking across all segments.
- Candidate phrases are obtained with POS-pattern matching and then scored with prompted likelihood plus a position-sensitive penalty.
- The top `15` phrases are used to increase the weight of key tokens inside each chunk representation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-chulo-2410-11119]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-chulo-2410-11119]].
