---
type: entity
title: EvalPlus
slug: evalplus
date: 2026-04-20
entity_type: tool
aliases: [EvalPlus]
tags: []
---

## Description

EvalPlus is the evaluation framework used in [[wei-2023-magicoder-2312-02120]] to obtain the stricter HumanEval+ and MBPP+ results reported for Magicoder. It augments the original benchmarks with substantially more tests.

## Key Contributions

- Supplies HumanEval+ and MBPP+ as more rigorous robustness checks than the original benchmark suites.
- Standardizes the `pass@1` comparisons against ChatGPT, GPT-4 Turbo, WizardCoder, and other baselines.

## Related Concepts

- [[code-generation-benchmark]]
- [[greedy-decoding]]
- [[data-decontamination]]

## Sources

- [[wei-2023-magicoder-2312-02120]]
