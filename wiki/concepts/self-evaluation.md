---
type: concept
title: Self-Evaluation
slug: self-evaluation
date: 2026-04-20
updated: 2026-04-20
aliases: [answer self-verification, 自我评估]
tags: [llm, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Evaluation** (自我评估) — the process where a model estimates whether one of its own generated answers is correct instead of only producing the answer itself.

## Key Points

- [[kadavath-2022-language-2207-05221]] operationalizes self-evaluation as `P(True)`, a true/false probability assigned to a model-generated candidate answer.
- Showing `5` comparison samples from the same question materially improves self-evaluation, especially for short-answer tasks.
- Few-shot prompting mainly improves calibration of `P(True)` rather than only the AUROC for separating correct and incorrect samples.
- The paper argues that verification can improve faster than generation quality as model size increases.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
