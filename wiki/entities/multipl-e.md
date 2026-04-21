---
type: entity
title: MultiPL-E
slug: multipl-e
date: 2026-04-20
entity_type: dataset
aliases: [MultiPL-E, MultiPL E, HumanEval multilingual]
tags: [benchmark, code, evaluation]
---

## Description

MultiPL-E is a multilingual code-generation benchmark that translates HumanEval to 18 programming languages, providing reference solutions and Codex model predictions with their functional correctness labels. Developed by Cassano et al. (2022).

## Key Contributions

- Provides translated HumanEval problems and pre-computed Codex (`code-davinci-002`) predictions for 18 languages, enabling cross-language metric evaluation.
- [[zhou-2023-codebertscore-2302-05527]] uses MultiPL-E for Java, C++, Python, and JavaScript in the functional correctness correlation experiments.
- Enables measurement of how well automated metrics like CodeBERTScore correlate with pass/fail execution outcomes at scale.

## Related Concepts

- [[natural-language-to-code]]
- [[execution-based-evaluation]]
- [[code-generation]]

## Sources

- [[zhou-2023-codebertscore-2302-05527]]
