---
type: concept
title: Fine-grained object retrieval
slug: fine-grained-object-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [FGOR, fine-grained retrieval, 细粒度目标检索]
tags: [retrieval, vision]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Fine-grained object retrieval** (细粒度目标检索) — retrieving images from the same fine-grained subcategory by learning embeddings that separate highly similar objects within a shared meta-category.

## Key Points

- The task is harder than generic image retrieval because subcategories have small inter-class differences but large intra-class variance.
- FRPT treats the task as retrieving birds, cars, or aircraft instances that share the same subcategory label as the query.
- The paper argues that full fine-tuning on small fine-grained datasets can damage generalization and lead to suboptimal retrieval embeddings.
- The proposed solution preserves a frozen pre-trained backbone and adds lightweight prompting plus feature adaptation instead.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2023-finegrained-2207-14465]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2023-finegrained-2207-14465]].
