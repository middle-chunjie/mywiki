---
type: concept
title: Unanswerable Questions
slug: unanswerable-questions
date: 2026-04-20
updated: 2026-04-20
aliases: [unanswerable QA, 不可回答问题]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Unanswerable Questions** (不可回答问题) — questions whose answer is absent from the provided or retrieved context, so a grounded QA model should abstain instead of generating unsupported content.

## Key Points

- ChatQA explicitly adds `1.5k` human-annotated unanswerable user-agent turns to stage-2 instruction tuning by deleting answer-supporting spans from context.
- The model is trained to output a fixed abstention response: `Sorry. I cannot find the answer based on the context`.
- The synthetic pipeline also creates unanswerable cases by chunking documents and checking answer overlap with 4-gram recall thresholds `> 0.5` and `< 0.1`.
- Evaluation on QuAC and DoQA averages performance on both answerable and unanswerable cases, treating abstention correctness as a first-class metric.
- The paper shows that more abstention data is not always better: `1.5k` samples outperform both smaller (`1k`) and larger (`2k`, `2.5k`) variants on the reported overall balance.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2024-chatqa-2401-10225]]
- [[katsis-2025-mtrag-2501-03468]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2024-chatqa-2401-10225]].
