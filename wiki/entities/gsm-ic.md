---
type: entity
title: GSM-IC
slug: gsm-ic
date: 2026-04-20
entity_type: benchmark
aliases: [GSMIC]
tags: []
---

## Description

GSM-IC is the benchmark in [[weston-2023-system-2311-11829]] that augments math word problems with irrelevant distractor sentences to test reasoning robustness.

## Key Contributions

- Provides `100` math problems sampled from GSM8K with added distractor sentences before the final question.
- Supports both random distractors and in-topic distractors, exposing how irrelevant context reduces baseline LLM accuracy.

## Related Concepts

- [[math-word-problem]]
- [[context-rewriting]]
- [[zero-shot-prompting]]

## Sources

- [[weston-2023-system-2311-11829]]
