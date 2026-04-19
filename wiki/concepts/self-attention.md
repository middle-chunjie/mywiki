---
type: concept
title: Self-Attention
slug: self-attention
date: 2026-04-17
updated: 2026-04-17
aliases: [Self-Attention, Intra-Attention, 自注意力, 自注意力机制]
tags: [attention, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-17
---

## Definition

Self-Attention (自注意力) — an attention mechanism in which queries, keys, and values all come from the same sequence, computing a representation of each position as a weighted combination of all positions in that sequence.

## Key Points

- Also called intra-attention; enables constant-length dependency paths between any two positions in a sequence.
- Per-layer complexity `O(n²·d)` with `O(1)` sequential operations, versus `O(n·d²)` and `O(n)` for recurrent layers.
- In the Transformer encoder, every position attends to every position in the previous layer; in the decoder, positions are masked to attend only to earlier positions.
- Preceded by uses in reading comprehension, abstractive summarization, and sentence embedding — but [[vaswani-2017-attention-1706-03762]] is the first transduction model built entirely on it.
- Enables interpretable attention maps; heads appear to learn syntactic or semantic roles.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] as constant-path-length mechanism for modeling dependencies within a sequence.
