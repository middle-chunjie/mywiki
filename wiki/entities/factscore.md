---
type: entity
title: FACTSCORE
slug: factscore
date: 2026-04-20
entity_type: tool
aliases: [FactScore]
tags: []
---

## Description

FACTSCORE is the factual precision metric used in [[dhuliawala-2023-chainofverification-2309-11495]] to evaluate long-form biography generation.

## Key Contributions

- Measures fine-grained factual precision for generated biographies using a retrieval-augmented fact-checking pipeline.
- Quantifies the long-form gain from `55.9` with few-shot Llama 65B to `71.4` with CoVe factor+revise.

## Related Concepts

- [[long-form-generation]]
- [[fact-verification]]
- [[hallucination-mitigation]]

## Sources

- [[dhuliawala-2023-chainofverification-2309-11495]]
