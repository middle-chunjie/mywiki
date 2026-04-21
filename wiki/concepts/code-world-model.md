---
type: concept
title: Code World Model
slug: code-world-model
date: 2026-04-20
updated: 2026-04-20
aliases: [CWM, code world model, code-based world model]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code World Model** (代码世界模型) — a world model represented as executable code that predicts environment transitions and rewards and can be called directly during planning.

## Key Points

- [[dainese-2024-generating-2405-15383]] instantiates a CWM as a Python `Environment` class with `__init__`, `set_state`, and `step` methods.
- A CWM is considered correct when it reproduces all transitions in the validation dataset, and partial correctness is measured by averaged next-state, reward, and done accuracy.
- The paper conditions CWM synthesis on both a natural-language environment description and offline trajectories, combining textual priors with executable feedback.
- Once synthesized, the CWM can replace repeated LLM calls inside planning, yielding four to seven orders of magnitude lower inference latency in the reported examples.
- The authors argue that code brings precision, interpretability, and reliability, but also note that rigid Python implementations struggle with stochastic, partially observed, or highly complex physics environments.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[dainese-2024-generating-2405-15383]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[dainese-2024-generating-2405-15383]].
