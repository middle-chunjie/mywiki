---
type: concept
title: DAPO
slug: dapo
date: 2026-04-20
updated: 2026-04-20
aliases: [Decoupled Clip and Dynamic Sampling Policy Optimization]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**DAPO** — an on-policy reinforcement learning objective that combines clipped policy optimization with dynamic prompt sampling to focus agent training on informative, non-trivial rollouts.

## Key Points

- WebDancer uses DAPO after supervised cold start to optimize multi-turn Thought-Action-Observation behavior in real web environments.
- The objective samples a group of candidate executions and applies clipped importance-ratio updates over model-generated tokens rather than tool responses.
- Dynamic sampling filters out prompts whose accuracy is always `0` or always `1`, improving robustness against noisy or trivial synthetic QA pairs.
- In this paper, DAPO is paired with answer-format rewards and answer-correctness rewards, with the final scalar reward weighted as `0.1 / 0.9`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2025-webdancer-2505-22648]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2025-webdancer-2505-22648]].
