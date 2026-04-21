---
type: concept
title: Domain-Invariant Feature Learning
slug: domain-invariant-feature-learning
date: 2026-04-20
updated: 2026-04-20
aliases: [domain-invariant representations, invariant feature learning]
tags: [domain-adaptation, domain-generalization, representation-learning, transfer-learning]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Domain-Invariant Feature Learning** (领域不变特征学习) — the goal of learning feature representations whose distribution does not change across domains, enabling a classifier trained on source domains to generalize to unseen target domains.

## Key Points

- The central assumption: if feature distributions are aligned across domains, a classifier trained on labeled source data will perform well on an unlabeled target domain where labels shift but features do not.
- Multiple methods pursue this goal: Domain Adversarial Training (DAT) minimizes domain divergence via a discriminator; IRM enforces invariant predictors; CORAL aligns feature covariances; maximum mean discrepancy (MMD) minimizes kernel-based distribution distance.
- Perfect domain-invariant representations are unattainable when class-conditional distributions differ across domains (a phenomenon called negative transfer); in practice, methods tradeoff alignment tightness against task discrimination.
- Noisy environment labels undermine domain-invariant learning because the discriminator may overfit mislabelled samples; ELS mitigates this by reducing discriminator confidence on ambiguous boundary samples.
- Environment label noise worsens as the encoder improves: features from different domains become more similar, making some labels effectively incorrect even if originally accurate.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-free-2302-00194]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-free-2302-00194]].
