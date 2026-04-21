---
type: concept
title: Adversarial Attack
slug: adversarial-attack
date: 2026-04-20
updated: 2026-04-20
aliases: [adversarial example attack, 对抗攻击]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Adversarial Attack** (对抗攻击) — a method that perturbs an input in a constrained way to induce erroneous or degraded model behavior.

## Key Points

- [[jha-2023-codeattack-2206-00052]] studies adversarial attacks for pre-trained programming-language models in the natural channel of code rather than standard natural-language text.
- The paper formalizes attack success by requiring minimal perturbation, high similarity to the original code, and a sufficient downstream quality drop ``Q(F(X)) - Q(F(X_adv)) >= phi``.
- CodeAttack is evaluated on code translation, code repair, and code summarization instead of only classification.
- The work argues that code-specific attacks must preserve coding style and local token-class structure to remain imperceptible to humans.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jha-2023-codeattack-2206-00052]].
