---
type: entity
title: Guanaco
slug: guanaco
date: 2026-04-20
entity_type: model
aliases: [Guanaco 65B, Guanaco 33B, Guanaco 13B, Guanaco 7B]
tags: []
---

## Description

Guanaco is the instruction-tuned model family produced with QLoRA in this paper, built on LLaMA backbones and evaluated as an open chatbot alternative to ChatGPT and Vicuna.

## Key Contributions

- Demonstrates that QLoRA can train competitive chat models up to `65B` scale on a single GPU.
- Reaches `99.3%` of ChatGPT on the paper's GPT-4-scored Vicuna evaluation at `65B`.
- Serves as the main evidence for the claim that dataset quality matters more than raw instruction-tuning set size.

## Related Concepts

- [[large-language-model]]
- [[instruction-tuning]]
- [[quantization]]

## Sources

- [[dettmers-2023-qlora-2305-14314]]
