---
type: concept
title: Evolving Rubrics
slug: evolving-rubrics
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic rubrics, 演化评分量表]
tags: [evaluation, rl, agents]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Evolving Rubrics** (演化评分量表) — rubric criteria that are repeatedly updated during training using current policy rollouts and newly retrieved evidence so the verifier stays informative and on-policy.

## Key Points

- The paper generates new rubrics from the prompt, sampled rollouts, and the existing rubric pool at every RL step.
- Evolving rubrics include both positive criteria that capture newly useful behaviors and negative criteria that penalize emerging reward-hacking patterns.
- Because rubric generation sees search traces, the criteria can incorporate information the model only discovered during interaction with tools.
- The active rubric buffer is pruned by reward variance, and only the top `K_max = 5` discriminative evolving rubrics are kept.
- In ablations, adding evolving rubrics improves over initial search-based rubrics alone on average after `650` training steps.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shao-2025-dr-2511-19399]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shao-2025-dr-2511-19399]].
