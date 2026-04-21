---
type: concept
title: Knowledge Grounding
slug: knowledge-grounding
date: 2026-04-20
updated: 2026-04-20
aliases: [evidence grounding, grounded revision]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Grounding** — the process of verifying or revising a model-generated statement by explicitly citing supporting evidence from external documents.

## Key Points

- GenGround grounds each hop-level question-answer pair `(q_i, a_i)` after the model first produces an immediate answer from parametric knowledge.
- The grounding instruction asks the LLM to cite the most relevant evidence from retrieved documents and then revise the answer into a corrected trajectory `\\tilde{a}_i`.
- If grounding cannot find relevant evidence, the model emits an `Empty` signal and keeps the generated answer as a fallback instead of forcing a revision.
- In the paper, grounding is the main mechanism for reducing non-factual hallucinations introduced either by the base LLM or by noisy retrieved passages.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shi-2024-generatethenground]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shi-2024-generatethenground]].
