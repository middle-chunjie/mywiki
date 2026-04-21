---
type: concept
title: Tensor Factorization
slug: tensor-factorization
date: 2026-04-20
updated: 2026-04-20
aliases: [张量分解, TF]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Tensor Factorization** (张量分解) — a latent-variable method that decomposes multi-way relational data into lower-dimensional factors for each entity type.

## Key Points

- The paper uses a 3D plaintiff-defendant-patent tensor instead of a 2D interaction matrix.
- Factorization produces latent vectors `U_i`, `V_j`, and `P_k` for plaintiffs, defendants, and patents.
- The scoring function sums bilinear interactions across the three tensor modes, capturing cross-party and party-patent structure.
- In this paper, tensor factorization alone is weaker than the hybrid CTF model because it cannot use patent content directly.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2018-patent]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2018-patent]].
