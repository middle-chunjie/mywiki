---
type: concept
title: Faithfulness Hallucination
slug: faithfulness-hallucination
date: 2026-04-20
updated: 2026-04-20
aliases: [context hallucination, 忠实性幻觉]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Faithfulness Hallucination** (忠实性幻觉) — the generation of claims that are not supported by, or contradict, the supplied input context even when the response should be context-grounded.

## Key Points

- The paper distinguishes faithfulness hallucination from factuality hallucination: the former is measured against provided context, while the latter is measured against external world knowledge.
- PIC reformulates faithfulness hallucination as a fine-grained claim-control problem in long-form generation.
- On PIC-Bench, even strong frontier models still produce faithfulness hallucinations in over `70%` of full-PIC generations.
- Manual error analysis of PIC-LM identifies distortion, fabrication, and conflation as major unsupported-claim error types.
- The paper argues that better control over faithfulness can improve downstream factuality when generation is embedded in verified pipelines.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[he-2025-precise-2506-06589]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[he-2025-precise-2506-06589]].
