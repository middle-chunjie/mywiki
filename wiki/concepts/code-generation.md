---
type: concept
title: Code Generation
slug: code-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [text-to-code generation]
tags: [software-engineering, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Generation** (代码生成) — the task of producing source code from a natural-language specification or other structured prompt.

## Key Points

- In [[ahmad-2021-unified]], PLBART is evaluated on Concode, where the input is English plus class context and the output is a Java method.
- PLBART reaches `36.69` BLEU and `38.52` CodeBLEU, outperforming strong decoder-only baselines on code quality metrics.
- The paper emphasizes that semantically correct code may diverge from the reference string, so EM alone is not sufficient.
- Ablations with `10K`, `20K`, and `50K` fine-tuning examples suggest pretraining injects useful syntax and data-flow knowledge before supervision.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ahmad-2021-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ahmad-2021-unified]].
