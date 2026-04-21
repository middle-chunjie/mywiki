---
type: entity
title: LeanDojo
slug: lean-dojo
date: 2026-04-20
entity_type: dataset
aliases: [Lean Dojo, LeanDojo Benchmark 4 v9]
tags: []
---

## Description

LeanDojo is the dataset and evaluation environment used in [[unknown-nd-leanstar]] for supervised fine-tuning data and an additional Lean test set. In this paper it operationalizes the standard proof-state and tactic format used for training theorem provers.

## Key Contributions

- Provides the `231,240` supervised `(state, tactic)` examples for the SFT baseline.
- Supplies successful proof trajectories from which the paper samples thought annotations and expert-iteration data.
- Contributes an additional test split on which Lean-STaR improves from `30.6%` to `39.4%`.

## Related Concepts

- [[proof-state]]
- [[tactic-prediction]]
- [[formal-theorem-proving]]

## Sources

- [[unknown-nd-leanstar]]
