---
type: concept
title: Mechanistic Interpretability
slug: mechanistic-interpretability
date: 2026-04-20
updated: 2026-04-20
aliases: [Mechanistic Interpretability, 机制可解释性]
tags: [interpretability, transformer, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Mechanistic Interpretability** (机制可解释性) — the study of explaining a model by identifying the concrete computational mechanisms and information-flow circuits that implement its behavior.

## Key Points

- [[friedman-2023-transformer-2306-01128]] frames mechanistic interpretability as an attempt to reverse-engineer Transformer algorithms from weights and activations.
- The paper argues that post-hoc circuit analysis is often incomplete and labor-intensive, motivating models that are interpretable by construction instead of only after training.
- Its proposed Transformer Programs preserve discrete structure so the learned computation can be converted into executable Python rather than only visualized heuristically.
- The paper uses program decompilation, debugging, and feature tracing as practical interpretability tools, not just attention-map inspection.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[friedman-2023-transformer-2306-01128]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[friedman-2023-transformer-2306-01128]].
