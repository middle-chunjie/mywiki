---
type: concept
title: Masked Language Modeling
slug: masked-language-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [MLM, 掩码语言建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Masked Language Modeling** (掩码语言建模) — a self-supervised objective that trains an encoder to reconstruct masked tokens from their bidirectional context.

## Key Points

- NeoBERT uses MLM as its only pretraining objective and removes next-sentence prediction entirely.
- The paper raises the masking rate from the historical `15%` used in BERT and RoBERTa to `20%` for a base-width encoder.
- NeoBERT uses a `100%` mask replacement scheme rather than BERT's `80/10/10` corruption recipe.
- The authors argue that encoders are less sample-efficient than decoders because they predict only masked positions, motivating training on `2.1T` theoretical tokens.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[breton-2025-neobert-2502-19587]]
- [[jha-2023-codeattack-2206-00052]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[breton-2025-neobert-2502-19587]].
