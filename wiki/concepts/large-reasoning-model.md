---
type: concept
title: Large Reasoning Model
slug: large-reasoning-model
date: 2026-04-20
updated: 2026-04-20
aliases: [LRM, large reasoning model, 大推理模型]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Large Reasoning Model** (大推理模型) — a language model optimized for long, explicit, inference-time reasoning traces rather than short direct responses.

## Key Points

- Search-o1 treats OpenAI o1, QwQ, and similar models as LRMs because they can sustain long stepwise reasoning with strong test-time performance.
- The paper argues LRMs are still vulnerable to knowledge gaps: a long reasoning chain can fail when one intermediate fact is missing.
- QwQ-32B-Preview is the backbone LRM in this work and already outperforms similarly sized instruction-tuned baselines on reasoning-heavy benchmarks.
- The paper's core claim is that LRMs become more reliable when they can retrieve external evidence on demand instead of relying only on parametric knowledge.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[li-2025-searcho-2501-05366]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[li-2025-searcho-2501-05366]].
