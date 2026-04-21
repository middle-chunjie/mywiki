---
type: concept
title: Self-Knowledge
slug: self-knowledge
date: 2026-04-20
updated: 2026-04-20
aliases: [metaknowledge, 自我知识]
tags: [llm, evaluation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Knowledge** (自我知识) — a model's ability to predict what it does and does not know, including whether it can answer a question correctly before producing an answer.

## Key Points

- [[kadavath-2022-language-2207-05221]] studies self-knowledge through `P(IK)`, the probability that the model knows the answer to a question.
- The paper trains `P(IK)` with a value head at the final token and uses `30` sampled answers per question to approximate soft labels.
- `P(IK)` carries some model-specific signal in cross-model experiments, so it is not reducible to a generic difficulty score alone.
- Generalization exists across trivia, math, code, and story completion, but calibration degrades under distribution shift.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[kadavath-2022-language-2207-05221]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[kadavath-2022-language-2207-05221]].
