---
type: concept
title: Masked Training
slug: masked-training
date: 2026-04-20
updated: 2026-04-20
aliases: [masked identifier training, token masking robustness training, 掩码训练]
tags: [robustness, training, adversarial-defense, source-code]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Masked Training** (掩码训练) — a lightweight adversarial training method for code models that improves robustness by jointly optimizing a standard loss on original code and a loss on masked code (where programmer-defined identifiers are replaced with `<unk>` tokens), reducing model reliance on identifier surface forms.

## Key Points

- Objective: `θ* = argmin_θ (λ·L_origin(p, com) + (1−λ)·L_masked(p', com))`, where `p'` is constructed by randomly masking `k` identifiers with `<unk>` and `λ ∈ [0,1]` is a hyperparameter.
- The masked loss forces the model to generate correct comments without access to specific identifier names, encouraging learning of code structure rather than surface-form shortcuts (non-robust features per Ilyas et al.).
- Unlike full adversarial training (which requires generating adversarial examples at training time — computationally expensive), masked training uses simple random masking and is therefore lightweight.
- On the Transformer model for code comment generation (Java), masked training recovers adversarial BLEU from 13.23 to 40.10 (max=2 attack), versus data augmentation's 18.10 — a ~2× robustness gain — while slightly improving clean BLEU (44.84 vs. 44.58).
- The approach generalizes across LSTM, Transformer, GNN, and CSCG dual models, producing consistent robustness gains with minimal clean-accuracy sacrifice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2022-adversarial]].
