---
type: concept
title: Model-Based Evaluation
slug: model-based-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [llm-as-a-judge, reference-based-model-evaluation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model-Based Evaluation** (模型式评测) — an evaluation approach that uses a stronger language model with explicit judging prompts to score outputs against references or task-specific criteria.

## Key Points

- HELMET uses reference-based GPT-4o judging for NarrativeQA and summarization instead of depending on ROUGE alone.
- For QA, the judge combines a fluency score in `{0,1}` and a correctness score in `{0,1,2,3}` before normalization to `[0,100]`.
- For summarization, the paper decomposes references into atomic claims, measures precision and recall with GPT-4o, and multiplies their `F1` by a fluency term.
- The paper reports strong agreement with human judgments, including Cohen's `kappa = 0.91` for summary precision and `kappa = 0.76` for recall on InfiniteBench Sum.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yen-2024-helmet-2410-02694]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yen-2024-helmet-2410-02694]].
