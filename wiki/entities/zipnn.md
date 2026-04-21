---
type: entity
title: ZipNN
slug: zipnn
date: 2026-04-20
entity_type: tool
aliases: [ZipNN]
tags: []
---

## Description

ZipNN is the strongest floating-point compression baseline cited in [[xiao-2026-embedding-2602-00079]], using transpose, byte shuffling, and entropy coding without the paper's spherical reparameterization.

## Key Contributions

- Serves as the direct baseline that the proposed method improves from roughly `1.20×` to around `1.5×`.
- Illustrates the ceiling of byte-layout-only compression when mantissa entropy remains near maximum.

## Related Concepts

- [[floating-point-compression]]
- [[byte-shuffling]]
- [[entropy-coding]]

## Sources

- [[xiao-2026-embedding-2602-00079]]
