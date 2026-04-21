---
type: entity
title: ENWIK8
slug: enwik8
date: 2026-04-20
entity_type: dataset
aliases: [enwiki8, EnWik8, Hutter Prize benchmark]
tags: [dataset, language-modeling, character-level]
---

## Description

ENWIK8 is a character-level language modeling benchmark introduced by Matt Mahoney (2009) as part of the Hutter Prize compression challenge, containing 100M characters of raw Wikipedia XML. Performance is measured in bits-per-character (bpc).

## Key Contributions

- Provides a standard character-level LM benchmark used to evaluate long-range modeling capacity.
- TRIME (TRIMELMlong) achieves 1.05 bpc on a 38M Transformer and 0.98 bpc on a 42M SRU++ model with 24,576 token long-term memory at inference.
- Commonly used alongside WIKITEXT-103 to assess whether long-context methods generalize across word-level and character-level tasks.

## Related Concepts

- [[language-modeling]]
- [[long-term-memory]]
- [[perplexity]]

## Sources

- [[zhong-2022-training-2205-12674]]
