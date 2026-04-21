---
type: entity
title: SentencePiece
slug: sentencepiece
date: 2026-04-20
entity_type: tool
aliases: [Sentence Piece]
tags: []
---

## Description

SentencePiece is the tokenizer used in [[borgeaud-2022-improving-2112-04426]] to segment MassiveText with a vocabulary of `128,000` tokens.

## Key Contributions

- Defines the tokenization granularity for both LM training and chunk-level retrieval.
- Allows RETRO to operate over a multilingual corpus with a shared subword vocabulary.

## Related Concepts

- [[large-language-model]]
- [[multilingual-pretraining]]

## Sources

- [[borgeaud-2022-improving-2112-04426]]
