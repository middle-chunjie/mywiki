---
type: entity
title: CLEVR
slug: clevr
date: 2026-04-20
entity_type: dataset
aliases: [Compositional Language and Elementary Visual Reasoning]
tags: []
---

## Description

CLEVR is the synthetic compositional dataset used in [[ray-nd-cola]] to evaluate how well adaptation methods generalize to controlled attribute-object combinations. Its exhaustive annotations let the paper compute especially clean hard-distractor retrieval metrics.

## Key Contributions

- Provides a controlled setting with compositional train/test splits for single-object retrieval.
- Shows some of the largest gains from MM-Adapter over baseline CLIP and standard tuning baselines.

## Related Concepts

- [[compositional-generalization]]
- [[attribute-binding]]
- [[text-to-image-retrieval]]

## Sources

- [[ray-nd-cola]]
