---
type: entity
title: CLCDSA
slug: clcdsa
date: 2026-04-20
entity_type: dataset
aliases: [CLCDSA dataset]
tags: []
---

## Description

CLCDSA is the cross-language source-code clone dataset reused in [[gui-2022-crosslanguage-2201-07420]] as the basis for both source-source evaluation and the curated cross-language binary-source benchmark. It contains solutions in multiple programming languages from AtCoder and Google Code Jam.

## Key Contributions

- Provides the starting point for the paper's curated binary-source dataset.
- After filtering to compilable C, C++, and Java programs, the paper reports `17,683` train, `6,181` validation, and `6,494` test samples.
- Supports both cross-language source-source clone detection and binary-source matching experiments.

## Related Concepts

- [[code-clone-detection]]
- [[binary-source-code-matching]]
- [[intermediate-representation]]

## Sources

- [[gui-2022-crosslanguage-2201-07420]]
