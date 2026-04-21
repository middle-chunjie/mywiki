---
type: concept
title: Retrieval-Augmented Language Model
slug: retrieval-augmented-language-model
date: 2026-04-20
updated: 2026-04-20
aliases: [REALM, 检索增强语言模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Retrieval-Augmented Language Model** (检索增强语言模型) — a language model that conditions predictions on documents retrieved from an external corpus instead of relying only on parametric memory.

## Key Points

- REALM trains the retriever and encoder jointly with masked language modeling rather than supervised passage labels.
- The model treats retrieved passages as latent variables and marginalizes over the top-scoring candidates.
- External retrieval makes factual evidence more modular and interpretable than storing all knowledge in parameters.
- The paper shows that this design transfers effectively to open-domain question answering after fine-tuning.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[guu-2020-realm-2002-08909]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[guu-2020-realm-2002-08909]].
