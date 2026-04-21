---
type: entity
title: ASQA
slug: asqa
date: 2026-04-20
entity_type: benchmark
aliases: [Ambiguous Question Answering, ASQA dataset]
tags: []
---

## Description

ASQA is an ambiguous long-form question-answering benchmark used in [[he-2025-precise-2506-06589]] to test whether PIC-LM improves retrieval-grounded answer generation.

## Key Contributions

- Serves as the main RAG downstream evaluation with `948` QA pairs.
- Shows PIC-LM improving standard exact-match recall from `52.5%` to `61.5%`.
- Tests whether claim-faithful generation generalizes beyond the controlled PIC-Bench setup.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[long-form-generation]]
- [[precise-information-control]]

## Sources

- [[he-2025-precise-2506-06589]]
