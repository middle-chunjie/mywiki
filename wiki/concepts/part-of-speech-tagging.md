---
type: concept
title: Part-of-Speech Tagging
slug: part-of-speech-tagging
date: 2026-04-20
updated: 2026-04-20
aliases: [POS tagging, POS tags, part-of-speech tags, 词性标注]
tags: [nlp, syntax, tagging, linguistic-features]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Part-of-Speech Tagging** (词性标注) — a foundational NLP task that assigns a grammatical category label (noun, verb, adjective, etc.) to each word in a sentence according to its syntactic role and context.

## Key Points

- POS tags such as "JJ" (adjective) are useful for identifying sentiment-bearing words (e.g., "horrible", "interesting") in sentiment analysis tasks, since sentiment words are predominantly adjectives and adverbs.
- Because POS tags reflect grammatical functions, they are largely domain-invariant: the same tag categories and distributions appear across different text domains, making them useful as transferable features for cross-domain tasks.
- In GAST (Zhang et al., 2022), POS tag embeddings (`d_t = 30`) are incorporated into the multi-head attention mechanism as a parallel attention stream, allowing the model to refocus attention toward sentiment-relevant words regardless of domain.
- Standard tagsets (Penn Treebank, Universal Dependencies) are widely adopted, enabling consistency across languages and parsers.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-graph-2205-08772]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-graph-2205-08772]].
