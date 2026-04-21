---
type: concept
title: Rejection Sampling Fine-Tuning
slug: rejection-sampling-fine-tuning
date: 2026-04-20
updated: 2026-04-20
aliases: [RFT, rejection-sampling finetuning]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Rejection Sampling Fine-Tuning** (拒绝采样微调) — a supervision pipeline that filters or synthesizes candidate trajectories and retains high-quality examples for model fine-tuning.

## Key Points

- IterResearch uses RFT before RL because standard LLMs do not natively produce the paper's iterative research format.
- The paper starts from 30K high-quality QA pairs and uses Qwen3-235B-A22B to synthesize 110K iterative trajectories.
- The resulting data teaches the model to output reasoning, report updates, and actions in the intended structured format.
- RFT serves as the warm-up stage before EAPO-based reinforcement learning on a smaller question subset.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[chen-2025-iterresearch-2511-07327]]
- [[unknown-nd-bstar]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[chen-2025-iterresearch-2511-07327]].
