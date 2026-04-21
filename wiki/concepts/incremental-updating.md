---
type: concept
title: Incremental Updating
slug: incremental-updating
date: 2026-04-20
updated: 2026-04-20
aliases: [refine-style summarization]
tags: [summarization, prompting, long-context]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Incremental Updating** — a sequential summarization strategy that maintains a running global summary and revises it after each new source chunk, optionally compressing the summary when it grows too long.

## Key Points

- The paper models the running summary as `g_i`, updated chunk by chunk with new evidence from `c_i`.
- A dedicated compression prompt is necessary because models tend to keep appending details rather than removing old material from the global summary.
- Incremental updating can preserve more narrative detail than hierarchical merging because it conditions later updates on the accumulated global state.
- In BOOOOKSCORE's experiments, this strategy usually lowers coherence scores even when annotators sometimes prefer its extra detail.
- Claude 2 improves sharply under this workflow when given `88K` context, suggesting that fewer update-compress cycles reduce error accumulation.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[unknown-nd-booookscorea-2310-00785]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[unknown-nd-booookscorea-2310-00785]].
