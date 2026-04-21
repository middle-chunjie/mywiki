---
type: concept
title: Model-Agnostic Meta-Learning
slug: model-agnostic-meta-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [MAML, 模型无关元学习]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Model-Agnostic Meta-Learning** (模型无关元学习) — a meta-learning algorithm that learns an initialization from which a model can adapt to new tasks with only a few gradient updates.

## Key Points

- The paper uses MAML to adapt pre-trained code representations from Python and Java toward low-resource target languages.
- It constructs `k` local tasks by splitting source-language training data into batches and then splitting each batch into train and validation subsets.
- Inner-loop adaptation follows `` `theta_i = theta - alpha * grad_theta L_Ti(f_theta)` `` to obtain task-specific parameters.
- Outer-loop updates use validation gradients to optimize a global initialization that transfers better across tasks.
- The paper argues that MAML is especially useful when target-language supervision is scarce, and the experimental gains are largest in low-data settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[gu-2018-deep]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[gu-2018-deep]].
