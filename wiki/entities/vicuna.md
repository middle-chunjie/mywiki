---
type: entity
title: Vicuna
slug: vicuna
date: 2026-04-20
entity_type: model
aliases: [Vicuna-7B, Vicuna-13B]
tags: []
---

## Description

Vicuna is the instruct-tuned LLaMA-based model family evaluated in [[li-2023-compressing]] to test robustness under compressed context. The paper reports that Vicuna handles Selective Context better than base LLaMA in human summary evaluation.

## Key Contributions

- Serves as a main open model family for testing compressed-context summarization and answer generation.
- Shows stronger robustness than LLaMA under Selective Context in the paper's human evaluation.
- Is paired with `LLaMA-7B` as the smaller scorer model for self-information estimation in the experiments.

## Related Concepts

- [[large-language-model]]
- [[summarization]]
- [[context-compression]]

## Sources

- [[li-2023-compressing]]
