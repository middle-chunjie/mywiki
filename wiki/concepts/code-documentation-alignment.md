---
type: concept
title: Code-Documentation Alignment
slug: code-documentation-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [CDA, code documentation alignment, 代码-文档对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code-Documentation Alignment** (代码-文档对齐) — a contrastive pretraining objective that aligns code snippets and their natural-language documentation in a shared embedding space for retrieval.

## Key Points

- [[li-2024-building]] uses CDA to continuously pretrain CodeT5 so matched code and documentation pairs become close in embedding space while in-batch negatives are pushed away.
- The paper formulates CDA with a contrastive loss over `` `f(d_doc, d_code)` ``, treating documentation as the query side and code as the positive target.
- The objective is meant to bridge the modality gap between natural language and program language, which the authors identify as a core issue for dense code retrieval.
- Ablation results show CDA is the main source of retrieval gains: on Adv zero-shot retrieval, CodeT5 with CDA reaches `45.01` MRR while vanilla CodeT5 remains near zero.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-building]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-building]].
