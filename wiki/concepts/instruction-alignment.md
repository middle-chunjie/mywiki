---
type: concept
title: Instruction Alignment
slug: instruction-alignment
date: 2026-04-20
updated: 2026-04-20
aliases: [instruction alignment, 指令对齐]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Instruction Alignment** (指令对齐) — the process of making a language model produce outputs that better match user instructions, preferences, and safety requirements.

## Key Points

- This paper frames instruction alignment as a bidirectional problem: one can either train the model toward human preference or rewrite the prompt toward the model's input preferences.
- BPO treats prompt optimization as an alignment intervention that does not require updating the target model's parameters.
- The method is designed for both closed API models and open-weight models, making alignment feasible even when white-box training access is unavailable.
- Empirically, prompt-side alignment is competitive with PPO and DPO, and can also add gains on top of those training-based methods.
- The paper emphasizes interpretability because the alignment change is visible directly in the rewritten instruction rather than hidden inside model weights.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2023-blackbox-2311-04155]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2023-blackbox-2311-04155]].
