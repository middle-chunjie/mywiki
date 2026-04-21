---
type: concept
title: Semi-Supervised Learning
slug: semi-supervised-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [SSL, 半监督学习]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Semi-Supervised Learning** (半监督学习) — a learning paradigm that improves a target model by combining a small labeled set with a larger unlabeled set during training.

## Key Points

- ASGN uses semi-supervised learning in the teacher model rather than directly optimizing property prediction on the scarce labeled subset alone.
- The teacher jointly learns from labeled and unlabeled molecules through property regression, node-level reconstruction, and graph-level clustering.
- Unlabeled molecules are treated as structurally informative even when their target properties are unknown.
- The paper argues that this setup reduces overfitting compared with purely supervised MPGNN training under label scarcity.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[hao-2020-asgn]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[hao-2020-asgn]].
