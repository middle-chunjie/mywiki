---
type: concept
title: Post-Norm
slug: post-norm
date: 2026-04-20
updated: 2026-04-20
aliases: [Post-Layer Normalization, Post-LN, 后归一化]
tags: [neural-architecture, normalization, transformer]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Post-Norm** (后归一化) — the original [[residual-connection]] arrangement in Transformers, where layer normalization is applied after the residual addition: `LayerNorm(x + Sublayer(x))`, as used in the Vaswani et al. 2017 Transformer.

## Key Points

- Post-Norm suppresses representation collapse by reducing the contribution of the residual input to subsequent layers after each normalization, keeping layer outputs more distinct.
- The trade-off is that Post-Norm reintroduces gradient vanishing, as gradients must pass through normalization operations that can attenuate them, making deep Post-Norm models harder to train.
- In the [[hyper-connections]] framework, Post-Norm is expressible as an `n=1` HC matrix whose non-diagonal entries depend on the input/output variances and their covariance (Eq. 16) — non-trainable and data-dependent but not learnable.
- Post-Norm is associated with a local decay pattern in dense connection visualization: contributions to later layers rapidly diminish for early-layer outputs.
- Practical LLM training typically avoids Post-Norm at scale because of training instability, though it was the default in the original Transformer paper.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2025-hyperconnections-2409-19606]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2025-hyperconnections-2409-19606]] as the other pole of the normalization seesaw.
