---
type: concept
title: Pretrained Language Model
slug: pretrained-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [PLM, pre-trained language model, 预训练语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Pretrained Language Model** (预训练语言模型) — a language model trained on large generic corpora and later adapted as a reusable encoder or generator for downstream tasks.

## Key Points

- [[liu-2023-enhancing]] builds on `bert-base-uncased` and treats the PLM as the semantic backbone of the Knowledge-aware Text Encoder.
- The paper argues recent HTC progress largely comes from PLM-based encoders, but that these encoders still lack external knowledge needed for some inferences.
- K-HTC augments the PLM with concept embeddings from a knowledge graph rather than replacing the PLM architecture outright.
- The empirical comparison shows a knowledge-enhanced PLM can outperform prior hierarchy-aware HTC systems and a knowledge-infused flat classifier baseline.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[liu-2023-enhancing]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[liu-2023-enhancing]].
