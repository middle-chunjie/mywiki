---
type: concept
title: Book-Length Summarization
slug: book-length-summarization
date: 2026-04-20
updated: 2026-04-20
aliases: []
tags: [summarization, llm, long-context]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Book-Length Summarization** — summarizing documents whose length substantially exceeds a model's context window, typically by chunking the source and aggregating partial summaries into one coherent global summary.

## Key Points

- The paper treats books longer than `100K` tokens as the target setting, making one-shot prompting impossible for 2023-era frontier LLMs.
- Practical systems in this setting must combine segmentation, intermediate summarization, and global aggregation rather than direct end-to-end generation.
- Evaluation is unusually difficult because contamination affects public benchmarks such as BookSum and gold plot summaries are often unavailable for recent books.
- Coherence, not just informativeness or overlap, becomes a primary quality axis because chunk-and-combine pipelines can introduce omissions, discontinuities, and inconsistencies.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-booookscorea-2310-00785]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-booookscorea-2310-00785]].
