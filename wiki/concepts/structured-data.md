---
type: concept
title: Structured Data
slug: structured-data
date: 2026-04-20
updated: 2026-04-20
aliases: [structured text, 结构化数据]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Structured Data** (结构化数据) — data whose meaning is partly encoded by explicit fields, syntax, or schema-like organization rather than by plain natural-language sequence alone.

## Key Points

- The paper treats code snippets and product descriptions as structured data because their semantics depend on identifiers, attributes, and boundaries in addition to raw tokens.
- Retrieval over structured data is harder than plain-text retrieval because pretrained language models often underfit the semantics conveyed by structure.
- SANTA uses aligned natural-language counterparts, such as documentation or bullet points, to improve structured-data representations.
- The paper's masked-entity objective assumes that semantically important units inside structured data can be detected and explicitly recovered during pretraining.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structureaware-2305-19912]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structureaware-2305-19912]].
