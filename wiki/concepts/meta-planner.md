---
type: concept
title: Meta Planner
slug: meta-planner
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: [agents, planning]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Meta Planner** — a model that generates meta plans from task instructions so that a separate agent can use them as explicit planning guidance.

## Key Points

- [[xiong-2025-mpo-2503-02682]] instantiates the meta planner with `[[llama-3.1-8b-instruct]]` and trains it first by supervised fine-tuning, then by DPO on contrastive plan pairs.
- The meta planner is intentionally lightweight and detachable from the rest of MPO, so it can be reused with new agent backbones after training.
- Its outputs are optimized against downstream execution feedback rather than against a learned reward model or human-written plan annotations alone.
- The paper argues that separating the meta planner from the acting agent makes planning improvement more portable across frameworks and parameter scales.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiong-2025-mpo-2503-02682]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiong-2025-mpo-2503-02682]].
