---
type: entity
title: CounterFact
slug: counterfact
date: 2026-04-20
entity_type: dataset
aliases: [CounterFact]
tags: []
---

## Description

CounterFact is the factual question-answering benchmark that serves as the paper's main analysis bed for studying LASER on GPT-J. Each item pairs a fact tuple with paraphrased prompts, making it central to both factuality and robustness analyses.

## Key Contributions

- Supplies the main benchmark where GPT-J improves from `13.1%` to `24.0%` under single-step LASER.
- Enables the paper's analysis linking corrected answers to lower training-frequency facts in The Pile.
- Supports the compositional search result that raises GPT-J top-10 accuracy to `29.2%`.

## Related Concepts

- [[factual-recall]]
- [[paraphrase-robustness]]
- [[layer-selective-rank-reduction]]

## Sources

- [[sharma-2023-truth-2312-13558]]
