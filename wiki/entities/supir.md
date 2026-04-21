---
type: entity
title: SUPIR
slug: supir
date: 2026-04-20
entity_type: model
aliases: [SUPIR]
tags: []
---

## Description

SUPIR is a diffusion-based large restoration model used in [[guo-2024-refir-2410-05601]] both as the main mechanistic probe and as a base model for ReFIR augmentation. The paper analyzes its ControlNet and UNet decoder to decide where retrieval-based texture injection should occur.

## Key Contributions

- Serves as the representative LRM for the paper's probing analysis of structure reconstruction versus detail restoration.
- Demonstrates measurable gains from ReFIR, including FID improvement from `168.26` to `148.69` on CUFED5.

## Related Concepts

- [[large-restoration-model]]
- [[diffusion-model]]
- [[image-restoration]]

## Sources

- [[guo-2024-refir-2410-05601]]
