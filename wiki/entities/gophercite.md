---
type: entity
title: GopherCite
slug: gophercite
date: 2026-04-20
entity_type: model
aliases: []
tags: []
---

## Description

GopherCite is the `280B` self-supported QA system introduced in [[menick-2022-teaching-2203-11147]]. It combines Google-based retrieval, inline evidence generation, reward-model reranking, and abstention.

## Key Contributions

- Demonstrates that a large LM can answer with verbatim supporting quotes rather than unsupported prose.
- Achieves `80.0` S&P on NaturalQuestionsFiltered and `66.9` on ELI5Filtered in its best reported configurations.
- Shows that reward-model thresholding can improve answer quality further via abstention.

## Related Concepts

- [[self-supported-question-answering]]
- [[inline-evidence]]
- [[selective-prediction]]

## Sources

- [[menick-2022-teaching-2203-11147]]
