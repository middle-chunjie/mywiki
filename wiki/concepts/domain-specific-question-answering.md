---
type: concept
title: Domain-Specific Question Answering
slug: domain-specific-question-answering
date: 2026-04-20
updated: 2026-04-20
aliases: [domain QA, domain-specific QA, 领域问答]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Domain-Specific Question Answering** (领域问答) — a QA task constrained to a particular vertical (e.g., cloud products, healthcare, legal) where answering requires specialized knowledge not well covered by general pre-training.

## Key Points

- KnowPAT targets cloud-product QA for Huawei's Network AI Engine, where a domain KG (CPKG) stores product-specific triples as background knowledge.
- Unlike open-domain QA, domain QA answers must be both factually correct (requiring selective use of retrieved KG triples) and stylistically appropriate for enterprise end-users.
- Vanilla fine-tuning on golden QA pairs is insufficient because the model learns neither when to rely on retrieved knowledge nor how to produce preferred answer styles.
- The proprietary nature of domain corpora severely limits public benchmarking; internal test sets of ~500 questions are typical.
- KnowPAT's preference alignment framework unifies style and knowledge correctness as dual human-preference alignment objectives.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-knowledgeable-2311-06503]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-knowledgeable-2311-06503]].
