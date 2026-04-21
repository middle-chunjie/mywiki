---
type: entity
title: OLMES
slug: olmes
date: 2026-04-20
entity_type: benchmark
aliases: [OLMES suite]
tags: []
---

## Description

OLMES is the downstream evaluation suite used in [[gao-2025-metadata-2501-01956]] to measure how metadata conditioning affects task performance beyond validation loss. The suite aggregates multiple-choice and cloze-style evaluations across ten knowledge and reasoning tasks.

## Key Contributions

- Serves as the paper's main downstream benchmark alongside TruthfulQA.
- Provides the `10`-task average used to compare MeCo against standard pre-training and data selection.
- Supports the paper's argument that downstream averages are more informative than validation perplexity alone.

## Related Concepts

- [[conditional-inference]]
- [[perplexity]]
- [[large-language-model]]

## Sources

- [[gao-2025-metadata-2501-01956]]
