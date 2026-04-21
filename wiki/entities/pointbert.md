---
type: entity
title: PointBERT
slug: pointbert
date: 2026-04-20
entity_type: tool
aliases: [Point-BERT]
tags: []
---

## Description

PointBERT is the point-cloud encoder family used in [[gao-2024-sculpting-2311-01734]] for zero-shot 3D recognition experiments. MixCon3D scales the PointBERT backbone and reports its strongest headline results with this encoder.

## Key Contributions

- Provides the 3D encoder whose features are aligned with frozen CLIP image and text embeddings.
- Supplies the default medium architecture used in many ablations: `` `25.9M` `` parameters, `` `12` `` layers, width `` `512` ``, and `` `8` `` attention heads.

## Related Concepts

- [[contrastive-learning]]
- [[joint-representation-alignment]]
- [[zero-shot-learning]]

## Sources

- [[gao-2024-sculpting-2311-01734]]
