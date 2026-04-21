---
type: concept
title: Activation Beacon
slug: activation-beacon
date: 2026-04-20
updated: 2026-04-20
aliases: [Beacon Token, activation condensing, beacon token mechanism]
tags: [long-context, kv-cache-compression, context-extension, efficient-llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Activation Beacon** (激活信标) — A plug-in module for LLMs that condenses raw KV activations from a context interval into compact representations via special beacon tokens, enabling context extension far beyond the original window size without modifying the base model's parameters.

## Key Points

- Introduces special `<bcn>` tokens that prompt the LLM to compress the KV activations of a preceding interval (of length `l`) into `k` condensed activations, giving condensing ratio `α = l/k` (values sampled from `{2, 4, 8, 16, 32, 64, 128}` during training).
- Adds a dedicated set of MHA parameters `{W^b_Q, W^b_K, W^b_V, W^b_O}` (~2B for Llama-2-7B, ~1/3 of base model) while freezing all original LLM weights, preserving short-context capability as a true plug-in.
- Uses stepwise-expansion attention for beacon tokens: beacon `i` attends to spans `1..i` of the interval, giving hierarchical local-to-global coverage; empirically superior to segmentation and full-coverage variants.
- Stream-processes long contexts via a sliding window of fixed size (bounded by the original LLM context limit); condensed activations from past intervals accumulate while raw activations are discarded, achieving linear time and near-constant memory.
- Trained entirely on short sequences (1K–8K) with step-wise randomized condensing ratios, generalizing to 100K–400K+ contexts without any long-sequence training data.

## My Position

<!-- User's stance on this concept. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2024-long-2401-03462]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2024-long-2401-03462]].
