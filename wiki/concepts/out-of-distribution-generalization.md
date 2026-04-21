---
type: concept
title: Out-of-Distribution Generalization
slug: out-of-distribution-generalization
date: 2026-04-20
updated: 2026-04-20
aliases: [ood generalization, 分布外泛化]
tags: [llm, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Out-of-Distribution Generalization** (分布外泛化) — the ability of a model or learned signal to remain useful when evaluated on tasks or data distributions different from the ones used for training.

## Key Points

- [[kadavath-2022-language-2207-05221]] trains `P(IK)` on TriviaQA and evaluates transfer to Lambada, arithmetic, GSM8k, HumanEval, and Python function synthesis.
- Larger models show stronger OOD separation ability, but calibration is still substantially worse than in-distribution performance.
- Training on all four non-trivia source tasks improves OOD behavior compared with TriviaQA-only training, showing a real but reducible generalization gap.
- The paper also demonstrates contextual OOD adaptation: `P(IK)` increases when relevant documents or useful hints are inserted into the prompt.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
