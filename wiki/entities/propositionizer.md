---
type: entity
title: Propositionizer
slug: propositionizer
date: 2026-04-20
entity_type: tool
aliases: [passage-to-proposition generator]
tags: []
---

## Description

The Propositionizer is the paper's passage-to-proposition generation model, implemented by fine-tuning Flan-T5-large on GPT-4-produced seed data.

## Key Contributions

- Converts Wikipedia passages into proposition lists for building FACTOIDWIKI.
- Uses `42k` GPT-4-generated passage-to-proposition pairs as distillation data.
- Produces mostly faithful and stand-alone propositions with low manual error rates.

## Related Concepts

- [[proposition]]
- [[knowledge-distillation]]
- [[dense-retrieval]]

## Sources

- [[chen-2024-dense-2312-06648]]
