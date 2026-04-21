---
type: concept
title: Soft-InfoNCE
slug: soft-infonce
date: 2026-04-20
updated: 2026-04-20
aliases: [Soft InfoNCE, Soft-InfoNCE loss]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Soft-InfoNCE** — a weighted variant of InfoNCE that scales the contribution of each negative pair according to estimated relevance instead of treating all negatives equally.

## Key Points

- The paper inserts negative-pair weights `w_ij` into the InfoNCE denominator while preserving total negative mass with normalization.
- The weighting scheme is designed to reduce over-penalization of false negatives and partially relevant code snippets during code-search fine-tuning.
- Similarity scores `sim_ij` are estimated with BM25, SimCSE, or a trained retrieval model, then normalized with softmax.
- The analysis connects Soft-InfoNCE to KL control over negative distributions and improved mutual-information estimation through importance sampling.
- Across CodeBERT, GraphCodeBERT, and UniXCoder, Soft-InfoNCE consistently beats vanilla InfoNCE on CodeSearchNet.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-rethinking-2310-08069]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-rethinking-2310-08069]].
