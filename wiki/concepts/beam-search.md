---
type: concept
title: Beam Search
slug: beam-search
date: 2026-04-17
updated: 2026-04-17
aliases: [Beam Search, 束搜索]
tags: [decoding, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Beam Search (束搜索) — a heuristic decoding algorithm for auto-regressive sequence models that, at each step, keeps the top-`k` partial hypotheses ranked by cumulative log-probability (the "beam"), trading exhaustive search for a tractable approximation of the most probable output sequence.

## Key Points

- [[vaswani-2017-attention-1706-03762]] uses beam size 4 and length penalty `α = 0.6` for machine translation inference; WSJ parsing uses beam 21 and `α = 0.3`.
- Length penalty (Wu et al. 2016) compensates for beam search's bias toward shorter sequences.
- Translation inference stops early when possible; maximum output length is capped at `input_length + 50`.
- Base model outputs come from averaging the last 5 checkpoints (written at 10-minute intervals); the big model averages the last 20.
- Not a Transformer-specific technique, but part of the standard recipe that made the reported BLEU scores reproducible.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] with beam size 4, `α = 0.6` for MT inference.
