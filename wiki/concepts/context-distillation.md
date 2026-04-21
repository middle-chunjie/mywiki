---
type: concept
title: Context Distillation
slug: context-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [context distillation, 上下文蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Context Distillation** (上下文蒸馏) — a training procedure that transfers behaviors induced by a long or specially engineered prompt into model parameters by supervising the model on prompt-conditioned teacher outputs.

## Key Points

- The paper uses context distillation in its verbose-cloning stage to make Dromedary produce longer and more direct answers without keeping the verbose prompt at inference time.
- A principle-engraved teacher, prompted with a verbose system prompt, generates `358,777` synthetic responses that become supervision for the final student model.
- The distilled model is trained for `1` epoch with LoRA-only updates on attention modules, batch size `768`, maximum sequence length `768`, and max learning rate `4e-4`.
- The paper reports that this stage improves open-ended response quality, but also introduces a "verbose tax" that slightly hurts some multiple-choice trustworthiness metrics.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[sun-2023-principledriven-2305-03047]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[sun-2023-principledriven-2305-03047]].
