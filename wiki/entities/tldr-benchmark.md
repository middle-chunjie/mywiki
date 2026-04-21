---
type: entity
title: tldr Benchmark
slug: tldr-benchmark
date: 2026-04-20
entity_type: paper
aliases: [tldr, tldr-pages benchmark, tldr NL-to-Bash benchmark]
tags: [benchmark, code-generation, bash, nlp]
---

## Description

tldr-benchmark is a community-curated NL→Bash code generation benchmark introduced in DocPrompting, derived from the [tldr-pages](https://github.com/tldr-pages/tldr) project. It contains `9,187` NL→Bash pairs across `1,879` unique commands, with training/dev/test splits (`1315/376/188` commands) enforcing completely disjoint command sets to test generalization to unseen Bash commands. The documentation pool consists of `400k` paragraphs from `1,879` Bash manuals (sourced from manned.org).

## Key Contributions

- First benchmark to leverage the tldr-pages project for NL→code evaluation with unseen-command generalization.
- Provides oracle documentation annotations (`D*_n`) via command name and flag matching heuristics, enabling supervised retriever training.
- Evaluates four metrics: CMD Acc (command name exact match), exact match (EM), token-level F1, and character-level BLEU (charBLEU).

## Related Concepts

- [[natural-language-to-code]]
- [[retrieval-augmented-code-generation]]
- [[documentation-retrieval]]
- [[bm25]]

## Sources

- [[zhou-2023-docprompting-2207-05987]]
