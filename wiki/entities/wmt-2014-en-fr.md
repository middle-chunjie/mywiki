---
type: entity
title: WMT 2014 English-French
slug: wmt-2014-en-fr
date: 2026-04-17
entity_type: tool
aliases: [WMT 2014 EN-FR, WMT14 EN-FR, WMT 2014 English-to-French]
tags: [dataset, mt, benchmark]
---

## Description

Standard machine-translation benchmark from WMT 2014 for English→French translation, containing about 36M sentence pairs. Used as a larger-scale complement to WMT EN–DE for Transformer evaluation.

## Key Contributions

- Reference benchmark on which [[vaswani-2017-attention-1706-03762]] established a single-model state-of-the-art BLEU of 41.8 (big model).
- Uses a 32k word-piece vocabulary in the Transformer paper (rather than BPE on the shared vocab).

## Related Concepts

- [[transformer]]

## Sources

- [[vaswani-2017-attention-1706-03762]]
