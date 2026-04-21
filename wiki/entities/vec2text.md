---
type: entity
title: Vec2Text
slug: vec2text
date: 2026-04-20
entity_type: method
aliases: [Vec2Text]
tags: []
---

## Description

Vec2Text is the iterative embedding inversion system introduced in [[morris-2023-text-2310-06816]]. It alternates text generation and re-embedding to move a hypothesis toward a target embedding in latent space.

## Key Contributions

- Recasts text embedding inversion as controlled generation with discrete correction rounds.
- Achieves `97.3` BLEU and `92.0%` exact recovery on `32`-token GTR-base Wikipedia passages with `50` steps plus sequence beam search.
- Demonstrates strong privacy leakage on MIMIC notes by recovering `89.2%` of full names.

## Related Concepts

- [[embedding-inversion]]
- [[controlled-generation]]
- [[beam-search]]

## Sources

- [[morris-2023-text-2310-06816]]
