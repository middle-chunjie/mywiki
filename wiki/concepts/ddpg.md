---
type: concept
title: Deep Deterministic Policy Gradient
slug: ddpg
date: 2026-04-20
updated: 2026-04-20
aliases: [DDPG]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Deep Deterministic Policy Gradient** — an off-policy actor-critic reinforcement-learning algorithm for continuous action spaces that learns a deterministic policy with replay buffers and target networks.

## Key Points

- CoRAL uses DDPG because retrieval actions are continuous user/item query vectors rather than discrete choices over the full database.
- The actor predicts the next user and item embeddings, and nearest-neighbor lookup maps those continuous actions back to concrete users and items.
- The critic is trained with bootstrapped targets over replay-buffer transitions, while the actor is updated through policy gradients derived from the critic.
- Target networks and Ornstein-Uhlenbeck noise are used to stabilize training and support exploration in the embedding space.
- In the paper, DDPG serves as the core mechanism for learning minimal-sufficient collaborative evidence under prompt-capacity constraints.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-coral-2403-06447]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-coral-2403-06447]].
