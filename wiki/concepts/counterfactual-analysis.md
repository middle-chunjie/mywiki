---
type: concept
title: Counterfactual Analysis
slug: counterfactual-analysis
date: 2026-04-20
updated: 2026-04-20
aliases: [conceptual counterfactual analysis, 反事实分析]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Counterfactual Analysis** (反事实分析) — a method for estimating how changing one controlled factor in an input alters model behavior while holding other relevant properties fixed.

## Key Points

- [[hooda-2024-do-2402-05980]] adapts counterfactual analysis from general ML to code generation by perturbing one programming concept at a time.
- The paper emphasizes three requirements for useful counterfactuals: correctness, validity, and specificity.
- In this work, counterfactual programs must remain executable solutions to the same task after mutation.
- The framework is designed for hard-label settings and does not require token probabilities or white-box access to the model.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hooda-2024-do-2402-05980]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hooda-2024-do-2402-05980]].
