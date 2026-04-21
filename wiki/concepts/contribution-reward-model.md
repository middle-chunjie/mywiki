---
type: concept
title: Contribution Reward Model
slug: contribution-reward-model
date: 2026-04-20
updated: 2026-04-20
aliases: [contribution RM, step-contribution reward model, 贡献奖励模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Contribution Reward Model** (贡献奖励模型) — a reward model that assigns dense scores to intermediate reasoning or code steps according to how much they appear to contribute to the quality of the final answer.

## Key Points

- DACO trains the contribution RM `R_c` on pairwise preferences between intermediate code steps rather than on full answers.
- The target signal is derived heuristically from `Sim(\mathbf{a}, \mathbf{o}_i)`, the similarity between the final answer and the execution output of step `i`.
- `R_c` scores the generated code `\mathbf{c}_i` without consuming `\mathbf{o}_i`, which simplifies deployment but can introduce misspecification.
- The model supplies dense feedback to intermediate steps, which the paper argues is necessary because answer-only rewards are too sparse for multi-turn code generation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-daco-2403-02528]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-daco-2403-02528]].
