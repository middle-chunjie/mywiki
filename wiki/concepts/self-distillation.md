---
type: concept
title: Self-Distillation
slug: self-distillation
date: 2026-04-20
updated: 2026-04-20
aliases: [自蒸馏]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-distillation** (自蒸馏) — training a model variant to match the output distribution of a stronger or less constrained version of itself, typically using a divergence loss between teacher and student predictions.

## Key Points

- In xRAG, the teacher is a full-text RAG model and the student is the compressed xRAG model that only sees the projected retrieval embedding.
- The paper uses a KL-divergence objective `D_KL(p_teacher || p_student)` and combines it with language-modeling loss as `L_nll + α L_kl`, with `α = 2.0`.
- The stated purpose is not just accuracy transfer but robustness: the student should learn when to rely on context and when to avoid overcommitting to noisy retrieval.
- Ablations show self-distillation matters more than plain next-token supervision for resilience, especially under misleading retrieval.
- The paper frames self-distillation as a way to preserve the benefits of full RAG while keeping the compressed, one-token retrieval interface.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cheng-2024-xrag-2405-13792]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cheng-2024-xrag-2405-13792]].
