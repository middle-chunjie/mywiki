---
type: entity
title: InstructGPT
slug: instructgpt
date: 2026-04-20
entity_type: tool
aliases: [Instruct GPT, text-davinci-003]
tags: []
---

## Description

InstructGPT is the OpenAI instruction-following model used in [[ma-2023-large-2303-08559]] as both a direct ICL baseline and a reranker in the adaptive SLM+LLM system. In this paper it is instantiated through the `text-davinci-003` API.

## Key Contributions

- Serves as one of the strongest proprietary direct-prompting baselines in the paper.
- Delivers an average `+2.4` F1 gain as the reranker over the non-ensemble SLM baseline across the three adaptive-reranking benchmarks.

## Related Concepts

- [[large-language-model]]
- [[in-context-learning]]
- [[reranking]]

## Sources

- [[ma-2023-large-2303-08559]]
