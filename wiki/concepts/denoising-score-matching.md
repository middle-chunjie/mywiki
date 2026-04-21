---
type: concept
title: Denoising Score Matching
slug: denoising-score-matching
date: 2026-04-20
updated: 2026-04-20
aliases: [去噪评分匹配, DSM]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Denoising Score Matching** (去噪评分匹配) — an objective that trains a model to estimate the score of noisy data, enabling reverse denoising dynamics in score-based generative modeling.

## Key Points

- The paper presents DSM as the main training objective behind continuous-time score-based diffusion models.
- It states that the optimal solution to the weighted DSM objective recovers the true score `\nabla_x \log p_t(x)` for almost all `x,t`.
- The survey also explains that DDPM noise prediction can be interpreted as a reparameterization of score estimation.
- DSM appears again later in the paper when discussing likelihood optimization and hybrid training objectives.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[cao-2023-survey-2209-02646]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[cao-2023-survey-2209-02646]].
