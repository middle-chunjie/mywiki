---
type: concept
title: Gradient Reversal Layer
slug: gradient-reversal-layer
date: 2026-04-20
updated: 2026-04-20
aliases: [GRL, gradient reversal, 梯度反转层]
tags: [domain-adaptation, adversarial-training, nlp]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Gradient Reversal Layer** (梯度反转层) — a parameter-free neural network layer that acts as an identity transform during the forward pass but multiplies the gradient by `-1` (scaled by a factor `λ`) during backpropagation, enabling adversarial domain adaptation without an alternating minimax optimization loop.

## Key Points

- Introduced by Ganin and Lempitsky (2015) for unsupervised domain adaptation; widely adopted for domain-adversarial training in NLP.
- Placed between the feature extractor and a domain/language discriminator: the feature extractor is encouraged to produce representations that *maximize* discriminator loss, effectively removing domain-identifying information.
- The reversal strength `λ` is typically annealed from 0 to 1 during training (e.g., `λ = 2/(1 + exp(−γ·epoch)) − 1`) to prevent early-training instability.
- In the CWI setting, the combined loss is `L = L_r − β·λ·L_d`, where `L_r` is the task regression loss and `L_d` is the discriminator classification loss.
- Enables simultaneous training of the feature extractor and the discriminator in a single forward-backward pass, unlike alternating optimization.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zaharia-2022-domain-2205-07283]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zaharia-2022-domain-2205-07283]].
