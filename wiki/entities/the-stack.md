---
type: entity
title: The Stack
slug: the-stack
date: 2026-04-20
entity_type: dataset
aliases: [The Stack]
tags: []
---

## Description

The Stack is the permissively licensed code dataset used to train [[starcoder]] in [[li-2023-starcoder-2305-06161]], combining source files, issues, commits, and notebook-derived artifacts under an opt-out-aware governance process.

## Key Contributions

- Supplies the main pretraining corpus for StarCoderBase, ultimately contributing `1T` training tokens after cleaning and sampling.
- Anchors the paper's governance claims through repository-level licensing filters, opt-out handling, and dataset inspection tooling.

## Related Concepts

- [[code-language-model]]
- [[data-deduplication]]
- [[pii-redaction]]

## Sources

- [[li-2023-starcoder-2305-06161]]
