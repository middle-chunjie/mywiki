---
type: entity
title: GTR-base
slug: gtr-base
date: 2026-04-20
entity_type: model
aliases: [GTR base, gtr-base]
tags: []
---

## Description

GTR-base is one of the two target text embedding models inverted in [[morris-2023-text-2310-06816]]. The paper treats it as a strong retrieval encoder and uses it for the Wikipedia and MIMIC inversion studies.

## Key Contributions

- Serves as the embedder for the Natural Questions Wikipedia setup where Vec2Text reaches `92.0%` exact reconstruction on `32`-token inputs.
- Underpins the MIMIC clinical-note experiment and the Gaussian-noise defense analysis over `15` BEIR retrieval tasks.

## Related Concepts

- [[text-embedding]]
- [[embedding-inversion]]
- [[cosine-similarity]]

## Sources

- [[morris-2023-text-2310-06816]]
