---
type: concept
title: Human-Centric Evaluation
slug: human-centric-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [human-level evaluation, human-centric benchmark]
tags: [evaluation, benchmark, llm, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Human-Centric Evaluation** — a paradigm for assessing AI/LLM capabilities using tasks derived directly from real human examinations (e.g., college entrance tests, professional qualification tests), establishing human performance as the reference baseline.

## Key Points

- Contrasted with artificially curated benchmark datasets (SQuAD, GLUE): human-centric benchmarks source questions from official, high-stakes exams with millions of real participants, ensuring tasks reflect genuine human cognitive demands.
- Requires both average and top human performance as reference bounds, since the gap between them encodes task difficulty stratification.
- Covers multiple cognitive dimensions simultaneously: understanding (semantic comprehension), knowledge (domain recall), reasoning (multi-step logical inference), and calculation (numerical/symbolic manipulation).
- Bilingual and cross-cultural scope is a desirable property: tasks in multiple languages expose language-specific capability gaps (e.g., CoT effectiveness differences between Chinese and English math exams).
- Data contamination is a structural risk: questions from public exams may appear in LLM training data; timestamp-based analysis of uncontaminated subsets is needed to validate benchmark integrity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhong-2023-agieval-2304-06364]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhong-2023-agieval-2304-06364]].
