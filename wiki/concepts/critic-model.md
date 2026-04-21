---
type: concept
title: Critic Model
slug: critic-model
date: 2026-04-20
updated: 2026-04-20
aliases: [critic, value model, critique model]
tags: [reasoning, planning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Critic Model** (评价模型) — a model that scores candidate actions or intermediate states so a separate generator can choose the option with the highest expected downstream utility.

## Key Points

- CR-Planner uses a sub-goal critic to decide whether the next step should be reasoning, query generation, or retrieval.
- It also trains specialized execution critics for rationales, queries, and retrieved documents instead of relying on one generic scorer.
- The critics are trained on MCTS-derived preference pairs so they approximate long-horizon value, not just local fluency.
- In this paper, critic models are smaller than the generator and are LoRA-tuned from `Skywork-Reward-Llama-3.1-8B`.
- The paper shows domain-specific critics outperform using GPT-4 as an in-context critic on tasks such as StackBio.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2024-can-2410-01428]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2024-can-2410-01428]].
