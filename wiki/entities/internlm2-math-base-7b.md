---
type: entity
title: InternLM2-Math-base-7b
slug: internlm2-math-base-7b
date: 2026-04-20
entity_type: tool
aliases: [InternLM2-7b, InternLM2-Math-7b]
tags: []
---

## Description

InternLM2-Math-base-7b is the base language model instantiated by [[unknown-nd-leanstar]] for Lean theorem proving. It is the starting point for the SFT baseline, Lean-CoT, and the two rounds of Lean-STaR expert iteration.

## Key Contributions

- Serves as the main base model used to quantify gains from thought augmentation and expert iteration.
- Improves from `29.5%` SFT to `32.8%` Lean-CoT and `34.8%` Lean-STaR pass@32 on [[mini-f2f]].
- Demonstrates that a Lean-specialized open model can benefit materially from thought-conditioned tactic prediction.

## Related Concepts

- [[tactic-prediction]]
- [[chain-of-thought]]
- [[expert-iteration]]

## Sources

- [[unknown-nd-leanstar]]
