---
type: concept
title: Maximum Marginal Likelihood
slug: maximum-marginal-likelihood
date: 2026-04-20
updated: 2026-04-20
aliases: [MML, maximal marginal likelihood, 最大边缘似然]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Maximum Marginal Likelihood** (最大边缘似然) — an objective that maximizes output likelihood after marginalizing over latent or retrieved intermediate variables instead of conditioning on a single observed choice.

## Key Points

- In this paper, MML is used to update the retriever by summing response likelihood over the top retrieved entities rather than relying only on the generator's NLL.
- The retrieval distribution is defined as a softmax over retriever scores, `q(e_{t,i}|c_t; φ) = exp(s_{t,i}) / Σ_j exp(s_{t,j})`.
- The resulting loss `L_MML = -log Σ_i q(e_{t,i}|c_t; φ) p(r_t | c_t, e_{t,i}; θ)` provides a differentiable path from response quality back to retriever parameters.
- Ablation results show MML improves both Recall@7 and final Entity F1 on large-scale knowledge-base settings.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[shen-2023-retrievalgeneration]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[shen-2023-retrievalgeneration]].
