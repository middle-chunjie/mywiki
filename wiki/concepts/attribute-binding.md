---
type: concept
title: Attribute Binding
slug: attribute-binding
date: 2026-04-20
updated: 2026-04-20
aliases: [attribute binding, object-attribute binding, 属性绑定]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Attribute Binding** (属性绑定) — the capability to associate each attribute in a query with the correct object instance rather than with a distractor object that shares overlapping words or visual traits.

## Key Points

- [[ray-nd-cola]] frames attribute binding as the central failure mode behind many compositional retrieval errors in current vision-language models.
- The `COLA` benchmark is constructed so that distractor images often contain the same objects and attributes, but with the wrong attachment between them.
- The paper evaluates attribute binding in both single-object and multi-object retrieval settings rather than in simple object-presence classification.
- A lightweight multimodal adapter improves attribute binding more than prompt tuning, linear probing, or standard late-layer fine-tuning.
- Human accuracy remains far above model performance, indicating that attribute binding is still far from solved even for relatively simple compositions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ray-nd-cola]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ray-nd-cola]].
