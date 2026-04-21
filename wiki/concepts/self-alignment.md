---
type: concept
title: Self-Alignment
slug: self-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [self alignment, 自对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Alignment** (自对齐) — a training paradigm in which a model improves its own instruction-following behavior using supervision signals generated or selected by the same model rather than by human annotators or a stronger external teacher.

## Key Points

- [[wei-2024-selfcodealign-2410-24198]] uses the same base code LLM throughout concept extraction, instruction generation, response generation, and test generation.
- The paper argues self-alignment is especially effective when the student can learn from data close to its own distribution instead of a shifted teacher distribution.
- In the main CodeQwen1.5-7B setting, self-alignment with execution filtering yields `67.1` HumanEval+ `pass@1`, outperforming matched-size GPT-3.5- and GPT-4o-based distillation baselines.
- Cross-model analysis shows self-generated data can beat slightly stronger teacher-generated data when the teacher-student performance gap is small, but much stronger teachers can still help weaker students.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wei-2024-selfcodealign-2410-24198]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wei-2024-selfcodealign-2410-24198]].
