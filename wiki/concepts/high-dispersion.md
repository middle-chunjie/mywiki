---
type: concept
title: High Dispersion
slug: high-dispersion
date: 2026-04-20
updated: 2026-04-20
aliases: [information dispersion, 高分散信息]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**High Dispersion** (高分散信息) — a property of long-context tasks in which the evidence needed for correct generation is spread across many distant parts of the input rather than concentrated in a few local snippets.

## Key Points

- LONGPROC is motivated partly by the claim that popular long-context benchmarks are too low-dispersion and therefore too easy.
- The benchmark's tasks require models to repeatedly integrate dispersed evidence while also preserving consistency across long outputs.
- The paper contrasts LONGPROC with benchmarks such as RULER, where strong recall at `64K-128K` tokens does not imply strong long procedural generation.
- High dispersion is especially visible in tasks such as HTML to TSV, Path Traversal, and Travel Planning, where relevant evidence must be repeatedly retrieved from different locations.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ye-2025-longproc-2501-05414]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ye-2025-longproc-2501-05414]].
