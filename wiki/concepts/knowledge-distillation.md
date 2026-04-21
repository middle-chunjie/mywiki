---
type: concept
title: Knowledge Distillation
slug: knowledge-distillation
date: 2026-04-20
updated: 2026-04-20
aliases:
  - 知识蒸馏
  - model distillation
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Knowledge Distillation** (知识蒸馏) — a training strategy in which a smaller student model learns to match the representations or outputs of a larger teacher model.

## Key Points

- The paper distills Qwen3-Embedding-4B into two smaller multilingual students rather than training them only with contrastive objectives.
- Student embeddings are projected into the teacher space with a linear head `psi(z) = Wz + b`, and cosine-alignment loss is applied to both query and document embeddings.
- Distillation is used not only in the first training stage but also retained inside the retrieval and STS adapter objectives to preserve general-purpose semantic structure.
- Ablations show that embedding-level distillation converges more slowly than InfoNCE early on, but reaches better final retrieval quality.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[akram-2026-jinaembeddingsvtext-2602-15547]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[akram-2026-jinaembeddingsvtext-2602-15547]].
