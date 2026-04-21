---
type: concept
title: White-Box Attack
slug: white-box-attack
date: 2026-04-20
updated: 2026-04-20
aliases: [White Box Attack, White-box Adversarial Attack]
tags: [adversarial, security, nlp, deep-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**White-Box Attack** (白盒攻击) — An adversarial attack strategy in which the attacker has full access to the target model's architecture, parameters, loss function, activation functions, and training data, enabling gradient-based crafting of adversarial examples.

## Key Points

- Requires complete model knowledge: architecture, weights, loss function, and input/output data.
- Typically more effective than black-box attacks because the adversary can directly compute gradients of the loss with respect to the input.
- Major white-box NLP strategies: FGSM-based (gradient sign/magnitude), JSMA-based (Jacobian saliency maps), C&W-based (l_p-norm constrained optimization), direction-based (HotFlip directional derivatives), attention-based (perturbing high-attention tokens), adversarial reprogramming, and hybrid embedding-space methods.
- The key challenge for text is mapping gradient-derived perturbations back to valid vocabulary tokens, since text is discrete.
- White-box adversarial examples are typically used to stress-test model robustness and to generate training data for adversarial training defenses.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2019-adversarial-1901-06796]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2019-adversarial-1901-06796]].
