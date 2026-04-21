---
type: entity
title: WebQuestions
slug: webquestions
date: 2026-04-20
entity_type: dataset
aliases: [Web Questions]
tags: []
---

## Description

WebQuestions is an open-domain QA benchmark collected from Google Suggest API queries and used in [[guu-2020-realm-2002-08909]] to test retrieval-based question answering. The paper follows the standard setting from prior work and evaluates exact match.

## Key Contributions

- Serves as one of the three main benchmarks for comparing REALM with ORQA, T5, and sparse-retrieval systems.
- Contains `3k/2k` train/test examples in the paper's setup.
- REALM reaches `40.2` test EM with Wikipedia pre-training and `40.7` with CC-News pre-training.

## Related Concepts

- [[open-domain-question-answering]]
- [[dense-retrieval]]
- [[retrieval-augmented-language-model]]

## Sources

- [[guu-2020-realm-2002-08909]]
