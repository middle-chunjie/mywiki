---
type: concept
title: Byte-Pair Encoding
slug: byte-pair-encoding
date: 2026-04-17
updated: 2026-04-17
aliases: [Byte-Pair Encoding, BPE, 字节对编码, 子词编码]
tags: [tokenization, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Byte-Pair Encoding (字节对编码, BPE) — a subword tokenization scheme that iteratively merges the most frequent adjacent symbol pairs in a training corpus, yielding a fixed-size vocabulary of subword units that balances coverage with out-of-vocabulary robustness.

## Key Points

- [[vaswani-2017-attention-1706-03762]] uses a shared source-target BPE vocabulary of ~37k tokens for WMT 2014 EN–DE.
- For EN–FR the authors use a word-piece vocabulary of 32k tokens instead (word-piece ≈ BPE variant using a likelihood-based merge criterion).
- Originated in Sennrich, Haddow & Birch (NMT of rare words with subword units, 2015/2016).
- Subword tokenization reduces OOV problems for rare words and morphologically rich languages without relying on full-word vocabularies.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] as the tokenization scheme used for WMT 2014 EN–DE (37k shared vocab).
