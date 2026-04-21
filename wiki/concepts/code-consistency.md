---
type: concept
title: Code Consistency
slug: code-consistency
date: 2026-04-20
updated: 2026-04-20
aliases: [consistent adversarial code, 代码一致性]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Code Consistency** (代码一致性) — the degree to which a modified code snippet preserves the original code's style, local structure, and overall similarity.

## Key Points

- [[jha-2023-codeattack-2206-00052]] treats code consistency as a core quality requirement for adversarial examples, alongside minimal perturbation and fluency.
- The paper measures consistency quantitatively with ``CodeBLEU_q = CodeBLEU(X, X_adv)``.
- Operator-count and token-class constraints are used to keep substitutes structurally aligned with the original code.
- CodeAttack generally achieves higher or comparable `CodeBLEU_q` than stronger-drop NLP baselines, showing a better quality-effectiveness tradeoff.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[jha-2023-codeattack-2206-00052]].
