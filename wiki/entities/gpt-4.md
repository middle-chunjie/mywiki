---
type: entity
title: GPT-4
slug: gpt-4
date: 2026-04-20
entity_type: tool
aliases: [GPT4]
tags: []
---

## Description

GPT-4 is the large language model used in [[anand-2024-understanding-2408-17103]] to generate candidate user intents from each query and its relevant passages.

## Key Contributions

- Generates `5` distinct candidate intents per prompt using temperature `0.6`.
- Provides the first-stage intent proposals before clustering, manual review, and crowdsourced cleanup.

## Related Concepts

- [[large-language-model]]
- [[user-intent]]
- [[query-rewriting]]

## Sources

- [[anand-2024-understanding-2408-17103]]
