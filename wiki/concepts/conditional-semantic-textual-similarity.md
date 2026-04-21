---
type: concept
title: Conditional Semantic Textual Similarity
slug: conditional-semantic-textual-similarity
date: 2026-04-20
updated: 2026-04-20
aliases: [C-STS, 条件语义文本相似度]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Conditional Semantic Textual Similarity** (条件语义文本相似度) — a sentence-pair similarity task in which the similarity score is defined with respect to an explicit natural-language condition specifying the aspect to compare.

## Key Points

- The paper introduces C-STS to replace ambiguous unconditional STS judgements with condition-specific similarity assessments.
- Each example is a quadruplet `{s_1, s_2, c, y}` containing two sentences, a condition sentence or phrase, and a Likert score from `1` to `5`.
- The benchmark is designed so the same sentence pair can receive very different labels under different conditions, exposing fine-grained semantic distinctions.
- C-STS-2023 contains `18,908` verified examples and is used to evaluate both fine-tuned sentence encoders and prompted LLMs.
- Even strong STS systems perform poorly on this formulation, indicating that aspect-conditioned similarity remains unsolved.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[deshpande-2023-csts-2305-15093]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[deshpande-2023-csts-2305-15093]].
