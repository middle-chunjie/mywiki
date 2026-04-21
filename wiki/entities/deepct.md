---
type: entity
title: DeepCT
slug: deepct
date: 2026-04-20
entity_type: model
aliases: [Deep Contextualized Term Weighting]
tags: []
---

## Description

DeepCT is a language-model-augmented lexical retriever referenced in [[gao-2021-coil-2104-07186]] as a strong lexical baseline. It adjusts document term weights with BERT-derived signals but still operates within a lexical inverted-index framework.

## Key Contributions

- Provides a direct comparison showing that contextualized token similarity is stronger than scalar contextual term weighting.
- Represents the class of deep-LM lexical methods that narrow, but do not eliminate, the gap to semantic retrieval.
- Helps motivate COIL's claim that token vectors carry richer matching information than single-value term weights.

## Related Concepts

- [[information-retrieval]]
- [[exact-lexical-match]]
- [[vocabulary-mismatch]]

## Sources

- [[gao-2021-coil-2104-07186]]
