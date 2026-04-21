---
type: concept
title: Hierarchical Text Classification
slug: hierarchical-text-classification
date: 2026-04-20
updated: 2026-04-20
aliases: [HTC, hierarchical classification, 分层文本分类]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Hierarchical Text Classification** (分层文本分类) — a text classification setting where documents are assigned to one or more labels organized by a taxonomy or directed acyclic graph rather than an unstructured flat label set.

## Key Points

- [[liu-2023-enhancing]] treats HTC as multi-label prediction over a hierarchy `G_2 = (Y, A)` whose labels may span up to 4 levels.
- The paper argues HTC is harder than ordinary classification because models must respect label dependencies and distinguish increasingly fine-grained categories.
- K-HTC injects knowledge graphs into both document encoding and label learning instead of relying only on text and hierarchy structure.
- The reported gains are strongest on deeper BGC levels, suggesting external knowledge is especially useful when hierarchical decisions become more specific.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-enhancing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-enhancing]].
