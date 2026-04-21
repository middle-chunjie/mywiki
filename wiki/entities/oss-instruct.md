---
type: entity
title: OSS-INSTRUCT
slug: oss-instruct
date: 2026-04-20
entity_type: tool
aliases: [OSS-INSTRUCT]
tags: []
---

## Description

OSS-INSTRUCT is the data-generation pipeline proposed in [[wei-2023-magicoder-2312-02120]] for synthesizing coding problems and solutions from randomly sampled open-source code snippets. It is the paper's central mechanism for reducing narrow-task bias in synthetic instruction corpora.

## Key Contributions

- Uses real open-source code as controllable inspiration rather than direct supervision targets.
- Produces about `75K` cleaned instruction-tuning examples that substantially improve `7B` code LLMs.
- Supports cross-language transfer by mixing Python and non-Python seed code.

## Related Concepts

- [[instruction-tuning]]
- [[synthetic-data]]
- [[bias-mitigation]]

## Sources

- [[wei-2023-magicoder-2312-02120]]
