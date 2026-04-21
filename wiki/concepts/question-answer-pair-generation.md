---
type: concept
title: Question-Answer Pair Generation
slug: question-answer-pair-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [QA pair generation, question-answer generation, 问答对生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Question-Answer Pair Generation** (问答对生成) — the construction of synthetic questions and answers from a document so model training explicitly reinforces factual recall and answer production for that document's content.

## Key Points

- For each document, the paper generates `3` QA pairs from the original text and combines them with rewritten document variants.
- The generated QA pairs are concatenated with document text during LoRA training so the adapter learns both knowledge storage and answer usage.
- Ablation results show that removing QA generation causes a larger performance drop than removing document rewriting alone.
- Warm-up with task-oriented QA data further improves adapter quality, suggesting QA-based supervision is a central driver of the method's gains.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[su-2025-parametric-2501-15915]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[su-2025-parametric-2501-15915]].
