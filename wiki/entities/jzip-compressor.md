---
type: entity
title: jzip-compressor
slug: jzip-compressor
date: 2026-04-20
entity_type: tool
aliases: [jzip compressor, jzip-compressor]
tags: []
---

## Description

`jzip-compressor` is the public implementation released with [[xiao-2026-embedding-2602-00079]] for compressing unit-norm embeddings via spherical coordinates plus entropy coding.

## Key Contributions

- Packages the paper's spherical-transform, transpose, byte-shuffle, and `zstd` pipeline into a reusable tool.
- Demonstrates that the method is intended as a practical systems component rather than only an analytic observation.

## Related Concepts

- [[spherical-coordinate]]
- [[floating-point-compression]]
- [[lossless-compression]]

## Sources

- [[xiao-2026-embedding-2602-00079]]
