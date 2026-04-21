---
type: entity
title: Tracr
slug: tracr
date: 2026-04-20
entity_type: tool
aliases: [TRACR]
tags: []
---

## Description

Tracr is the compiler cited in [[friedman-2023-transformer-2306-01128]] for converting RASP programs into Transformer networks. In this paper it serves as a key precursor showing how human-written programs can be mapped into Transformer weights.

## Key Contributions

- Supplies the architectural inspiration for keeping Transformer computations close to program-like primitives.
- Motivates the paper's reverse direction: learning a constrained Transformer first and then decompiling it back into code.

## Related Concepts

- [[rasp]]
- [[transformer-programs]]
- [[mechanistic-interpretability]]

## Sources

- [[friedman-2023-transformer-2306-01128]]
