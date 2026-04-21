---
type: concept
title: Black-Box Attack
slug: black-box-attack
date: 2026-04-20
updated: 2026-04-20
aliases: [black box attack, 黑盒攻击]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Black-Box Attack** (黑盒攻击) — an adversarial attack setting where the attacker cannot access model parameters or gradients and must rely on observable outputs.

## Key Points

- [[jha-2023-codeattack-2206-00052]] assumes realistic black-box access: the attacker can query the victim PL model and observe output probabilities, but not parameters, architecture, training data, gradients, or loss.
- Vulnerable-token ranking is therefore based on output-logit changes after masking, rather than gradient saliency.
- CodeAttack's low average query count relative to BERT-Attack is one of the paper's main empirical advantages.
- The paper positions black-box robustness as more operationally relevant for deployed code-generation systems than white-box attack settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jha-2023-codeattack-2206-00052]].
