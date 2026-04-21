---
type: concept
title: Two-stage Training
slug: two-stage-training
date: 2026-04-20
updated: 2026-04-20
aliases: [staged training, 两阶段训练]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Two-stage Training** (两阶段训练) — a training strategy that inserts an intermediate adaptation phase before the final instruction-tuning stage, rather than mixing all supervision into one step.

## Key Points

- The paper trains first on CodeI/O or CodeI/O++ and only afterward performs general instruction tuning.
- For Qwen 2.5 Coder 7B, fully separating the stages gives the best reported average score among the compared mixtures.
- All tested two-stage variants outperform the pure single-stage instruction-tuning baseline in the paper's analysis table.
- The authors position the first stage as a reasoning-oriented analogue of continual pretraining, but with more structured supervision than raw code LM loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-codeio-2502-07316]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-codeio-2502-07316]].
