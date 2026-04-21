---
type: concept
title: Rationale Generation
slug: rationale-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [rationale synthesis, self-synthesized rationales, 理由生成]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Rationale Generation** (理由生成) - the generation of explicit explanatory text that states how an answer follows from available evidence, often to supervise reasoning or make predictions more inspectable.

## Key Points

- InstructRAG uses an instruction-tuned model to synthesize rationales from `q`, `a`, and retrieved documents `D`.
- The rationale states which documents are relevant, which ones are noisy, and how the answer is derived from the evidence.
- The paper converts the original dataset `\mathcal{T} = {\langle q, a \rangle}` into `\mathcal{T}^{+} = {\langle q, r \rangle}` so rationales become reusable supervision.
- A simple template-based substitute underperforms LM-generated rationales, indicating that semantic matching matters for rationale quality.
- Stronger rationale generators improve downstream RAG performance, especially with a `70B` backbone.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-instructrag-2406-13629]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-instructrag-2406-13629]].
