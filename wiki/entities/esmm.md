---
type: entity
title: ESMM
slug: esmm
date: 2026-04-20
entity_type: tool
aliases:
  - Entire Space Multi-Task Model
tags: []
---

## Description

ESMM is a multi-task conversion-rate prediction architecture with shared embeddings and separate CTR/CVR towers, used in [[ouyang-2023-contrastive]] as the supervised backbone of CL4CVR.

## Key Contributions

- Provides the base supervised objective that CL4CVR augments with contrastive learning.
- Models click and conversion jointly in the entire sample space through shared representations.

## Related Concepts

- [[conversion-rate-prediction]]
- [[contrastive-learning]]
- [[representation-learning]]

## Sources

- [[ouyang-2023-contrastive]]
