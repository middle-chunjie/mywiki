---
type: entity
title: XLIR
slug: xlir
date: 2026-04-20
entity_type: tool
aliases: [XLIR]
tags: []
---

## Description

XLIR is the Transformer-based method proposed in [[gui-2022-crosslanguage-2201-07420]] for cross-language binary-source and source-source code matching via LLVM-IR embeddings. It combines IR conversion, IR-BERT pretraining, and triplet-loss alignment in a shared vector space.

## Key Contributions

- Converts both source code and binaries into LLVM-IR before semantic matching.
- Achieves `0.65` F1 on `C/C++ binary -> Java source` and `0.61` F1 on `Java binary -> C/C++ source`.
- Extends to cross-language source-source clone detection with up to `0.89` F1 on `C/C++`.

## Related Concepts

- [[binary-source-code-matching]]
- [[llvm-ir]]
- [[triplet-loss]]

## Sources

- [[gui-2022-crosslanguage-2201-07420]]
