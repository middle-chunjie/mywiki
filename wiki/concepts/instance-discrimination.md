---
type: concept
title: Instance Discrimination
slug: instance-discrimination
date: 2026-04-20
updated: 2026-04-20
aliases: [实例判别]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Instance Discrimination** (实例判别) — a self-supervised objective that treats each training instance as its own class, pulling augmentations of the same instance together while pushing all other instances apart.

## Key Points

- The paper identifies instance discrimination as the underlying objective that creates the tolerance problem in unsupervised contrastive learning.
- Because every distinct sample is treated as a negative, semantically related neighbors can be pushed apart even when that harms downstream transfer.
- Small temperatures intensify this effect because the loss focuses more strongly on the nearest and most confusing negatives.
- The paper argues that improving contrastive learning will require objectives that explicitly model relations between different instances.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2021-understanding-2012-09740]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2021-understanding-2012-09740]].
