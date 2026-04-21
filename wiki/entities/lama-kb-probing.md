---
type: entity
title: LAMA (Knowledge-Base Probing Benchmark)
slug: lama-kb-probing
date: 2026-04-20
entity_type: benchmark
aliases: [LAMA, Language Model Analysis, LAnguage Model Analysis]
tags: [benchmark, fact-retrieval, knowledge-base]
---

## Description

LAMA (LAnguage Model Analysis) is a fact retrieval benchmark introduced by Petroni et al. (EMNLP 2019) that evaluates whether language models encode factual knowledge. It consists of cloze-style queries derived from knowledge-base triples, e.g., "Obama was born in ___". Answers are single tokens, enabling direct probability-based evaluation.

Note: disambiguated from [[lama-planner]], which is a heuristic planning algorithm unrelated to this benchmark.

## Key Contributions

- Provides a standardized suite of cloze templates for probing factual knowledge stored in LM parameters.
- Used in [[zhao-2021-calibrate-2102-09690]] as a fact retrieval evaluation task to demonstrate contextual calibration's effectiveness on generation (as opposed to classification) tasks.

## Related Concepts

- [[in-context-learning]]
- [[few-shot-learning]]
- [[common-token-bias]]

## Sources

- [[zhao-2021-calibrate-2102-09690]]
