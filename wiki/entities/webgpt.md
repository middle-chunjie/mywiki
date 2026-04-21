---
type: entity
title: WebGPT
slug: webgpt
date: 2026-04-20
entity_type: tool
aliases: [WebGPT model, browser-assisted QA]
tags: [search-agent, reinforcement-learning, openai]
---

## Description

WebGPT is a GPT-3-based system developed by OpenAI that uses reinforcement learning from human feedback to train a language model to autonomously use a web browser — issuing search queries, scrolling through results, and quoting references — to answer long-form questions.

## Key Contributions

- Pioneered the dynamic search agent paradigm in which an LLM freely decides when and what to search, rather than following a fixed pipeline.
- Demonstrated that RL with human feedback can teach an LLM to ground responses in retrieved web documents, reducing hallucinations compared to pure generation.
- Inspired subsequent LLM-as-search-agent research and formed the conceptual foundation for later systems like New Bing.

## Related Concepts

- [[search-agent]]
- [[retrieval-augmented-generation]]
- [[reinforcement-learning-from-human-feedback]]

## Sources

- [[zhu-2024-large-2308-07107]]
