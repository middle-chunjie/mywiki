---
type: entity
title: KeyBERT
slug: keybert
date: 2026-04-20
entity_type: tool
aliases: []
tags: []
---

## Description

KeyBERT is the keyword extraction tool used in [[mao-2022-convtrans]] to derive keyword-style queries from natural-language queries when building NL-T5 training pairs.

## Key Contributions

- Produces the approximate keyword query `q^kw` paired with each natural-language query `q^nl` for NL-T5 training.
- Is configured with `msmarco-bert-base-dot-v5`, `keyphrase_ngram_range = (1, 2)`, and `top_n = 5` in the paper's implementation details.

## Related Concepts

- [[query-rewriting]]
- [[conversational-query-rewriting]]
- [[data-augmentation]]

## Sources

- [[mao-2022-convtrans]]
