---
type: concept
title: Program Repair
slug: program-repair
date: 2026-04-20
updated: 2026-04-20
aliases: [automatic program repair]
tags: [software-engineering, generation]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Program Repair** (程序修复) — the task of transforming buggy code into a corrected version that fixes the defect while preserving intended behavior.

## Key Points

- [[ahmad-2021-unified]] treats program repair as same-language sequence generation from buggy Java code to bug-fixed Java code.
- Exact Match is emphasized because repair outputs need to match the gold fix more strictly than free-form code generation.
- PLBART improves over CodeBERT from `16.40` to `19.21` EM on Java-small and from `5.16` to `8.98` EM on Java-medium.
- The paper views these gains as evidence that denoising pretraining transfers not just syntax, but also semantically meaningful edit patterns.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ahmad-2021-unified]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ahmad-2021-unified]].
