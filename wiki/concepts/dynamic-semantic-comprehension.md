---
type: concept
title: Dynamic Semantic Comprehension
slug: dynamic-semantic-comprehension
date: 2026-04-20
updated: 2026-04-20
aliases: [dynamic semantics, sequential semantic comprehension]
tags: [cognitive-modeling, attention, nlp, psycholinguistics]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Dynamic Semantic Comprehension** — the psycholinguistically-motivated process by which a reader iteratively focuses on a small region of text at each step, shifting attention across important words as comprehension builds, rather than processing all words simultaneously; modelled in NLP by sequential re-reading or re-weighting mechanisms.

## Key Points

- Psycholinguistic research (Kuperberg 2007; Tononi 2008; Koch & Tsuchiya 2007) suggests humans focus on approximately 1.5 words at a time and can retain the meaning of about 7–9 words per attentional cycle; this motivates the `T = 7` re-weighting steps in DR-BERT.
- Eye-tracking studies show that readers dynamically re-read important words during comprehension of long sentences, especially when understanding is conditioned on a specific target or aspect.
- In the DR-BERT framework this is modelled by the Dynamic Re-weighting Adapter: at each step `t`, the model selects the single most aspect-relevant word (soft argmax with `λ = 100`) and updates a GRU hidden state, mimicking sequential focused reading.
- The analogy to human cognition is qualitative and empirical rather than formally derived; the optimal `T = 7` matches the psycholinguistic threshold, providing interpretability support for the design choice.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-incorporating]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-incorporating]].
