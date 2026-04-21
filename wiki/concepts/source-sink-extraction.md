---
type: concept
title: Source-Sink Extraction
slug: source-sink-extraction
date: 2026-04-20
updated: 2026-04-20
aliases: [source and sink extraction, 源汇点提取]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Source-Sink Extraction** (源汇点提取) — the task of identifying the program values that start and terminate the dataflow relations relevant to a downstream analysis objective.

## Key Points

- LLMDFA treats source/sink extraction as a separate subproblem before reasoning about any flows.
- The paper uses LLMs to synthesize reusable extractors over ASTs rather than directly labeling lines with prompts.
- Extraction is conditioned on natural-language task specifications plus example programs and ASTs.
- On both DBZ and XSS benchmarks, the synthesized extractors reach `100.00%` precision and `100.00%` recall.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[wang-2024-when-2402-10754]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[wang-2024-when-2402-10754]].
