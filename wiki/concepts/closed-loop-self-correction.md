---
type: concept
title: Closed-Loop Self-Correction
slug: closed-loop-self-correction
date: 2026-04-20
updated: 2026-04-20
aliases: [self-correcting loop, iterative self-debugging, 闭环自我纠错]
tags: [agents, debugging]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Closed-Loop Self-Correction** (闭环自我纠错) — an iterative generation-and-execution procedure in which a model evaluates runtime feedback, diagnoses failures, and revises its output until success criteria are met.

## Key Points

- TOOLMAKER restores a clean installed environment before each attempt, runs the candidate tool, and asks the model to assess both returned outputs and runtime logs.
- Failed attempts trigger a diagnosis step, re-implementation step, and a summary that is carried forward into later iterations.
- This loop is central to solving multi-step tasks whose dependencies or intermediate file requirements are not obvious from the initial prompt alone.
- The paper reports that some tasks require many such iterations, such as `9` iterations for `stamp_train_classification_model`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[w-lflein-2025-llm-2502-11705]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[w-lflein-2025-llm-2502-11705]].
