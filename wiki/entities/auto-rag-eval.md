---
type: entity
title: auto-rag-eval
slug: auto-rag-eval
date: 2026-04-20
entity_type: tool
aliases: [Auto RAG Eval, auto rag eval]
tags: []
---

## Description

auto-rag-eval is the open-source implementation released with [[guinet-2024-automated-2405-13622]] under the [[amazon-science]] GitHub organization. It packages the paper's workflow for exam generation, evaluation, and iterative exam improvement on task-specific RAG benchmarks.

## Key Contributions

- Implements corpus-grounded synthetic exam generation for RAG evaluation.
- Supports the paper's filtering, grading, and IRT-based exam analysis workflow.
- Makes the framework reusable on new retrieval-augmented QA tasks beyond the four benchmark corpora in the paper.

## Related Concepts

- [[synthetic-exam-generation]]
- [[question-quality-filtering]]
- [[item-response-theory]]

## Sources

- [[guinet-2024-automated-2405-13622]]
