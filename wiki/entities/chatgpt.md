---
type: entity
title: ChatGPT
slug: chatgpt
date: 2026-04-20
entity_type: tool
aliases: [GPT-3.5 Turbo, gpt-3.5-turbo]
tags: []
---

## Description

ChatGPT is the strongest API-based LLM baseline evaluated in [[chen-2023-benchmarking-2309-01431]], instantiated through the `gpt-3.5-turbo` API for both English and Chinese RGB experiments.

## Key Contributions

- Achieves the highest English noise-robustness score at noise ratio `0` (`96.33%`) and the best ChatGPT-judged rejection rates (`45.00%` English, `43.33%` Chinese).
- Still shows severe vulnerability to false retrieved evidence, with English counterfactual accuracy dropping from `89%` without documents to `9%` with counterfactual documents.

## Related Concepts

- [[large-language-model]]
- [[retrieval-augmented-generation]]
- [[counterfactual-robustness]]

## Sources

- [[chen-2023-benchmarking-2309-01431]]
