---
type: concept
title: Latent Variable Model
slug: latent-variable-model
date: 2026-04-20
updated: 2026-04-20
aliases: [latent variable models, 潜变量模型]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Latent Variable Model** (潜变量模型) — a probabilistic model that explains observed data using one or more unobserved variables that capture hidden structure.

## Key Points

- [[wang-2024-large-2301-11916]] models in-context learning as inference over a latent task concept `\theta` rather than as direct conditioning on demonstrations alone.
- The latent variable is meant to summarize both task semantics and prompt-format information that the LM infers from examples.
- The paper treats the LM continuation distribution as an integral over latent concepts, connecting prompting to probabilistic latent-variable reasoning.
- Learned concept tokens are used as an operational surrogate for the latent variable so the model can estimate and exploit it in practice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-large-2301-11916]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-large-2301-11916]].
