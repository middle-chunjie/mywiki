---
type: concept
title: Encoder-Decoder Architecture
slug: encoder-decoder-architecture
date: 2026-04-17
updated: 2026-04-17
aliases: [Encoder-Decoder, Encoder-Decoder Architecture, Seq2Seq, Sequence-to-Sequence, 编码器-解码器, 编解码器架构]
tags: [architecture, nlp]
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-17
---

## Definition

Encoder-Decoder Architecture (编码器-解码器架构) — a neural sequence transduction design in which an encoder maps an input sequence `(x_1, ..., x_n)` to a continuous representation `z = (z_1, ..., z_n)`, and an auto-regressive decoder conditioned on `z` generates the output sequence one symbol at a time.

## Key Points

- The standard architecture for machine translation and other transduction tasks at the time of the [[transformer]]; instantiated previously with RNN/LSTM, GRU, and CNN encoders/decoders.
- Decoder is auto-regressive: consumes previously generated symbols as input when producing the next.
- The [[transformer]] preserves this overall shape but replaces the recurrence/convolution inside encoder and decoder with stacked [[self-attention]] + [[position-wise-feed-forward-network]] layers.
- Decoder-side cross-attention allows every output position to attend over the full encoded input.

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- Disagreements among sources about this concept. -->

## Sources

- [[vaswani-2017-attention-1706-03762]]

## Evolution Log

- 2026-04-17 (1 source): established from [[vaswani-2017-attention-1706-03762]] as the structural pattern the Transformer inherits from prior RNN/CNN seq2seq models.
