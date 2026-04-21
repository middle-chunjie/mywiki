---
type: concept
title: Attention Sink
slug: attention-sink
date: 2026-04-20
updated: 2026-04-20
aliases: [attention sinks]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Attention sink** — a token, typically near the beginning of a sequence, that absorbs disproportionate attention mass even when it carries little task-relevant semantic content.

## Key Points

- The paper shows that many decoder-only LLMs allocate large attention scores to the first few tokens across most layers beyond the bottom two.
- Removing those initial tokens from the KV cache causes window attention to collapse once the sequence length exceeds cache size.
- Replacing the first four original tokens with four newline tokens still restores perplexity, indicating that position dominates semantics in the sink effect.
- Keeping four initial sink tokens is usually enough to recover stable perplexity in Llama-2, MPT, Falcon, and Pythia.
- The paper argues that the sink phenomenon arises because softmax attention must distribute probability mass somewhere even when no prior token is strongly relevant.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xiao-2024-efficient-2309-17453]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xiao-2024-efficient-2309-17453]].
