---
type: concept
title: Code Clone Detection
slug: code-clone-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [代码克隆检测]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Clone Detection** (代码克隆检测) — the task of finding code fragments that are identical, near-duplicate, or semantically equivalent across files, languages, or representations.

## Key Points

- XLIR is evaluated not only on binary-source matching but also on cross-language source-source clone detection.
- The paper reports substantially better precision, recall, and F1 than LICCA on C/C++, C/Java, and C++/Java clone detection.
- Threshold selection matters: higher cosine thresholds increase precision while lowering recall.
- The work frames clone detection as a retrieval problem over embeddings in a shared vector space.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gui-2022-crosslanguage-2201-07420]]
- [[ahmad-2021-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gui-2022-crosslanguage-2201-07420]].
