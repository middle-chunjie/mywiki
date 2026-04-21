---
type: entity
title: GenRead
slug: genread
date: 2026-04-20
entity_type: tool
aliases: [Generate Rather Than Retrieve, GenRead prompting]
tags: []
---

## Description

GenRead is the paper's parametric augmentation baseline that asks a language model to generate a contextual document instead of retrieving one from an external corpus. It is used to compare generated context against BM25 and Contriever retrieval.

## Key Contributions

- Provides a retrieval-free augmentation baseline grounded in the LM's own parametric memory.
- Shows little gain for smaller models but sizeable gains for GPT-3, indicating that elicitive prompting can improve factual QA.
- Motivates the paper's hybrid adaptive setup, where generated or retrieved context is used only when it helps.

## Related Concepts

- [[parametric-memory]]
- [[retrieval-augmented-generation]]
- [[adaptive-retrieval]]

## Sources

- [[mallen-2023-when-2212-10511]]
