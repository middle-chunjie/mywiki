---
type: concept
title: BERT
slug: bert
date: 2026-04-20
updated: 2026-04-20
aliases: [Bidirectional Encoder Representations from Transformers, bert-encoder, BERT pre-training]
tags: [pre-trained-model, nlp, transformer, language-model]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**BERT** (Bidirectional Encoder Representations from Transformers) — a pre-trained language model based on the Transformer encoder stack, trained via masked language modelling (MLM) and next sentence prediction (NSP) on large corpora, producing deep bidirectional contextual representations used for downstream NLP tasks via fine-tuning.

## Key Points

- BERT-base uses `N = 12` encoder layers, `n_heads = 12` attention heads, and `n_hidden = 768`; each layer applies multi-head self-attention followed by a position-wise FFN with residual connections and layer normalization.
- Pre-training on MLM and NSP yields strong sentence-level semantics, but these objectives do not condition representations on a specific aspect or query — a known limitation in fine-grained conditional tasks like ABSA.
- Fine-tuning BERT for ABSA by simply appending the aspect sequence (BERT-SPC) or post-training on domain data (BERT-PT) yields meaningful gains over non-pretrained baselines; DR-BERT further improves this by inserting a Dynamic Re-weighting Adapter.
- Adding task-specific structural modules (graph convolutional layers, relational attention) on top of frozen BERT consistently outperforms vanilla fine-tuning in ABSA (T-GCN, RGAT-BERT).
- The canonical citation is Devlin et al. (2019); the architecture is akin to the Transformer encoder of Vaswani et al. (2017).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-incorporating]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-incorporating]].
