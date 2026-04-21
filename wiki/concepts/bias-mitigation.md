---
type: concept
title: Bias Mitigation
slug: bias-mitigation
date: 2026-04-20
updated: 2026-04-20
aliases: [debiasing intervention, 偏见缓解]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Bias Mitigation** (偏见缓解) — interventions that reduce undesirable social or representational bias in model behavior while preserving useful task performance.

## Key Points

- Pythia studies bias mitigation through counterfactual pretraining-data interventions rather than only through post-hoc fine-tuning.
- The intervention swaps masculine pronouns to feminine counterparts in the last `7%` or `21%` of training for selected deduplicated models.
- The measured gender bias on WinoBias and CrowS-Pairs decreases after intervention, especially at larger scales.
- The paper reports only marginal degradation on LAMBADA, suggesting the intervention reduces bias without major language-modeling loss.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[biderman-2023-pythia-2304-01373]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[biderman-2023-pythia-2304-01373]].
