---
type: concept
title: Self-Reflection
slug: self-reflection
date: 2026-04-20
updated: 2026-04-20
aliases: [self critique, reflective generation]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Self-Reflection** (自反思) — a mechanism in which a model explicitly evaluates its own intermediate or final outputs and uses those judgments to guide subsequent generation or selection.

## Key Points

- SELF-RAG operationalizes self-reflection through token-level judgments about retrieval need, passage relevance, support, and response usefulness.
- Reflection is integrated into next-token prediction rather than implemented as a separate inference-only verifier.
- The paper uses self-reflection to re-rank passage-conditioned segment candidates and to expose controllable trade-offs at inference time.
- The approach is intended to improve factuality and attribution without sacrificing the model's ability to answer non-knowledge-intensive prompts.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[asai-2023-selfrag-2310-11511]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[asai-2023-selfrag-2310-11511]].
