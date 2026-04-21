---
type: entity
title: NaturalQuestions-Open
slug: naturalquestions-open
date: 2026-04-20
entity_type: dataset
aliases: [Natural Questions Open]
tags: []
---

## Description

NaturalQuestions-Open is the open-domain QA benchmark derived from naturally occurring Google queries that REALM uses as a primary evaluation dataset in [[guu-2020-realm-2002-08909]]. The paper keeps short-answer questions of up to five tokens and evaluates exact-match accuracy.

## Key Contributions

- Provides a realistic open-domain QA testbed with `79k/4k` train/test examples in the paper's setup.
- Shows REALM reaching `39.2` test EM with Wikipedia pre-training and `40.4` with CC-News pre-training.
- Supports the paper's ablation analysis on development-set exact match and zero-shot Recall@5.

## Related Concepts

- [[open-domain-question-answering]]
- [[dense-retrieval]]
- [[retrieval-augmented-language-model]]

## Sources

- [[guu-2020-realm-2002-08909]]
