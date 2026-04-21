---
type: concept
title: Machine Reading Comprehension
slug: machine-reading-comprehension
date: 2026-04-20
updated: 2026-04-20
aliases: [MRC, reading comprehension, 机器阅读理解]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Machine Reading Comprehension** (机器阅读理解) — a task family in which a model reads a passage and predicts an answer, often as a span extracted from the passage.

## Key Points

- The source data for S2M comes from the single-turn MRC dataset SQuAD 2.0 rather than from an already conversational corpus.
- The paper reviews span prediction with start and end logits followed by `softmax` over context tokens.
- Directly reusing single-turn MRC data for CQA yields only weak gains because of distribution shift.
- S2M treats MRC QA pairs as structured seeds that can be reorganized into multi-turn conversations.
- The work shows that stronger CQA augmentation requires more than standard MRC supervision; it also needs sequencing and rewriting.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-sm-2312-16511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-sm-2312-16511]].
