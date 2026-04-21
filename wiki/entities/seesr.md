---
type: entity
title: SeeSR
slug: seesr
date: 2026-04-20
entity_type: model
aliases: [SeeSR]
tags: []
---

## Description

SeeSR is a diffusion-based large restoration model evaluated in [[guo-2024-refir-2410-05601]] as a second base system for ReFIR. It is used to show that the proposed retrieval-augmented framework generalizes beyond a single LRM architecture.

## Key Contributions

- Provides evidence that ReFIR is model-agnostic, improving SeeSR on both RefSR benchmarks and RealPhoto60.
- Shows a CUFED5 PSNR increase from `19.94` to `20.32` and a FID decrease from `142.92` to `134.62` when paired with ReFIR.

## Related Concepts

- [[large-restoration-model]]
- [[super-resolution]]
- [[retrieval-augmented-generation]]

## Sources

- [[guo-2024-refir-2410-05601]]
