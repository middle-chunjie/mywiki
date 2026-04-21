---
type: concept
title: Fine-Tuning
slug: fine-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [微调, task-specific adaptation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Fine-Tuning** (微调) — adapting a pre-trained model on task-specific labeled data to improve performance on a downstream objective.

## Key Points

- BLANCA uses fine-tuning to test whether code-related text tasks can improve sentence encoders for code understanding.
- The paper fine-tunes BERTOverflow and CodeBERT on individual BLANCA tasks and on multi-task mixtures.
- Fine-tuning improves every evaluated task relative to the corresponding untuned base models.
- Population-based hyperparameter search did not beat the default model settings for the ranking and link tasks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[abdelaziz-2022-can]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[abdelaziz-2022-can]].
