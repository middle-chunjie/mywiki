---
type: concept
title: Selective Retrieval
slug: selective-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [adaptive retrieval, retrieval abstention, 选择性检索]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Selective Retrieval** (选择性检索) — a retrieval strategy in which a model decides per instance whether external retrieval is likely to help generation, instead of retrieving unconditionally.

## Key Points

- Repoformer frames selective retrieval as a self-assessment problem: after seeing `X_l` and `X_r`, the LM predicts whether cross-file context should be fetched.
- The retrieval action is represented directly by a special token `<cc>`, allowing retrieval control to be learned within the same autoregressive model used for code completion.
- Training labels are self-supervised from the measured gain `ES(Y_hat_RAG, Y) - ES(Y_hat_base, Y) > T`, rather than from manual annotations or a stronger teacher model.
- Threshold-based selection improves both code-completion quality and latency relative to always retrieving on RepoEval and CrossCodeLongEval.
- The paper shows abstention is usually accurate: when retrieval is skipped, the base completion is often already correct or not meaningfully improved by cross-file context.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wu-2024-repoformer-2403-10059]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wu-2024-repoformer-2403-10059]].
