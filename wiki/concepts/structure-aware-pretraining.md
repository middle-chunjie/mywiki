---
type: concept
title: Structure-Aware Pretraining
slug: structure-aware-pretraining
date: 2026-04-20
updated: 2026-04-20
aliases: [structure-aware language model pretraining, 结构感知预训练]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Structure-Aware Pretraining** (结构感知预训练) — continued pretraining that explicitly teaches a language model to encode the semantics carried by structural organization, fields, or identifiers in non-plain-text data.

## Key Points

- SANTA implements structure-aware pretraining with two tasks, [[structured-data-alignment]] and [[masked-entity-prediction]], rather than relying only on generic masked language modeling.
- The goal is retrieval-oriented representation learning: the pretrained encoder should place structured documents and relevant text close in a shared embedding space.
- The paper argues that natural alignment signals between structured and unstructured data are a useful supervisory source for this kind of pretraining.
- Experimental gains in both zero-shot and fine-tuned retrieval suggest that structure-aware pretraining is more effective than vanilla LM masking for structured retrieval.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2023-structureaware-2305-19912]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2023-structureaware-2305-19912]].
