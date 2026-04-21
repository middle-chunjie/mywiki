---
type: entity
title: GTR
slug: gtr
date: 2026-04-20
entity_type: tool
aliases: [Google T5 Retriever]
tags: []
---

## Description

GTR is the strongest supervised dense retriever evaluated in the paper and is used to compare retrieval granularity in both retrieval and downstream QA settings.

## Key Contributions

- Achieves the best average passage retrieval scores in the study, improving Recall@20 from `76.6` to `79.2` with proposition indexing.
- Under retrieval-augmented `LLaMA-2-7B`, proposition prompts improve average `EM@500` from `38.5` to `41.3`.

## Related Concepts

- [[dense-retrieval]]
- [[proposition-retrieval]]
- [[retrieval-augmented-generation]]

## Sources

- [[chen-2024-dense-2312-06648]]
