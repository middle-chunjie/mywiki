---
type: concept
title: Hallucination Detection
slug: hallucination-detection
date: 2026-04-20
updated: 2026-04-20
aliases: [factuality detection, 幻觉检测]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Hallucination Detection** (幻觉检测) — the task of estimating whether a model output contains claims unsupported by the available evidence or task context.

## Key Points

- Lumina treats RAG hallucination detection as a balance problem between external-context use and internal-knowledge use rather than as pure uncertainty estimation.
- The method is unsupervised and training-free, using only model forward passes and internal activations.
- The paper evaluates threshold-free metrics such as `AUROC`, `AUPRC`, and `PCC` instead of relying only on a tuned classification threshold.
- Lumina is competitive with a supervised hidden-state classifier while avoiding labeled training data.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[yeh-2026-lumina-2509-21875]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[yeh-2026-lumina-2509-21875]].
