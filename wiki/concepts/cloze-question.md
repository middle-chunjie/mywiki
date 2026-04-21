---
type: concept
title: Cloze Question
slug: cloze-question
date: 2026-04-20
updated: 2026-04-20
aliases: [cloze QA, 完形填空式问题]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Cloze Question** (完形填空式问题) — a question formed by masking a plausible answer span inside a sentence so that the model must recover the missing content from context.

## Key Points

- The paper generates cloze QA pairs by taking a target-domain sentence and masking a sampled entity mention.
- Cloze QA is compared directly with standard question generation as a target-side augmentation method for both retrievers and readers.
- The method is computationally cheaper than standard question generation because it does not require an additional question-generation model.
- Empirically, cloze QA is competitive on multiple datasets, for example improving Quasar-S retriever `Acc@100` from `10.24` to `21.79`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dua-2023-adapt]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dua-2023-adapt]].
