---
type: concept
title: Long-Range Coherence
slug: long-range-coherence
date: 2026-04-20
updated: 2026-04-20
aliases: [long-context coherence, 长程连贯性]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Long-Range Coherence** (长程连贯性) — the ability of a model to keep later outputs logically consistent with distant input evidence and previously generated content over long sequences.

## Key Points

- LONGPROC analyzes coherence by splitting outputs into four segments and measuring correctness in later segments after conditioning on earlier gold segments.
- The paper reports clear degradation in later output segments on long tasks such as Path Traversal and Theory-of-Mind Tracking.
- The authors argue that failure at long procedural generation cannot be explained only by simple error accumulation; per-entry error rates also increase with length.
- Qualitative analysis shows two coherence failures: recall drift from the source context and reasoning drift from prior generated states.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2025-longproc-2501-05414]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2025-longproc-2501-05414]].
