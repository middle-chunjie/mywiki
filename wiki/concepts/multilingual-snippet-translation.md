---
type: concept
title: Multilingual Snippet Translation
slug: multilingual-snippet-translation
date: 2026-04-20
updated: 2026-04-20
aliases: [MuST, 多语言代码片段翻译]
tags: [program-translation, pretraining, multilingual, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Multilingual Snippet Translation** (多语言代码片段翻译) — a pre-training strategy for program translation that jointly trains on bilingual code snippet pairs across all available language pairs, using snippet-level granularity to provide finer supervision and enable multilingual transfer to low-resource language pairs.

## Key Points

- Proposed in [[zhu-2022-multilingual]] as the MuST objective: trains a shared encoder-decoder model on pairwise snippets from all 42 language pairs simultaneously, with language identifiers added to each input token.
- Objective: `L_M = Σ_{l_i, l_j ∈ L} E_{(x,y)~D_bi} [-log p_G(y | E(x, α_{l_i}), α_{l_j})]`; combined with a DAE loss weighted by `λ`.
- Snippet-level training bridges the distribution gap between short code snippets and longer programs, reducing the performance drop seen in baseline models when switching from snippet-level to program-level evaluation.
- MuST is shown to generalize: applying MuST pre-training to Transformer, CodeBERT, and TransCoder baselines consistently improves BLEU scores across all tested language pairs.
- Multilingual joint training enables knowledge transfer from high-resource pairs (e.g., Java-C++) to low-resource pairs (e.g., C-PHP), especially for syntactically similar languages.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhu-2022-multilingual]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhu-2022-multilingual]].
