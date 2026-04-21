---
type: concept
title: Question Generation
slug: question-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [问题生成]
tags: [generation, qa, synthetic-data]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Question Generation** (问题生成) — the task of producing plausible questions from source content such that the content contains or supports the answer.

## Key Points

- Here question generation is used to synthesize positive question-document pairs for retrieval training rather than to augment a QA benchmark directly.
- The generation prompt is document-conditioned and seeded by question initiators to bias outputs toward interrogative forms.
- The authors compare random sampling, contrastive search, and beam search as decoding strategies for producing multiple questions per selected document.
- Output quality is not trusted blindly; all generated questions are later filtered with retrieval-based criteria before downstream use.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[almeida-2024-exploring]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[almeida-2024-exploring]].
