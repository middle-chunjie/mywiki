---
type: concept
title: Few-Shot Learning
slug: few-shot-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [in-context learning from few examples, 小样本学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Few-Shot Learning** (小样本学习) — the ability of a model to perform a task from a small number of in-context examples without parameter updates.

## Key Points

- Pythia studies few-shot behavior over training rather than only at the final checkpoint.
- The paper evaluates arithmetic prompting and TriviaQA-style question answering under `k`-shot settings.
- Smaller models, especially below `1B`, often fail to perform these tasks well even with up to `16` few-shot examples.
- Correlation between pretraining term frequency and few-shot performance becomes much clearer in larger models and later checkpoints.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[biderman-2023-pythia-2304-01373]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[biderman-2023-pythia-2304-01373]].
