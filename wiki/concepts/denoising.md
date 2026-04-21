---
type: concept
title: Denoising
slug: denoising
date: 2026-04-20
updated: 2026-04-20
aliases: [explicit denoising, retrieval denoising, 去噪]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Denoising** (去噪) - the process of identifying, down-weighting, or excluding misleading retrieved content so a model can rely on evidence that actually supports the target output.

## Key Points

- InstructRAG turns denoising from an implicit behavior into an explicit intermediate rationale-generation task.
- The model is asked to explain which retrieved documents are useful and how they support the ground-truth answer, rather than predicting only the final answer.
- The method improves robustness when the number of retrieved documents increases and retrieval precision drops.
- Synthetic denoising rationales can be used both as in-context demonstrations and as supervised fine-tuning targets.
- The paper argues that explicit denoising improves both answer correctness and verifiability in RAG.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-instructrag-2406-13629]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-instructrag-2406-13629]].
