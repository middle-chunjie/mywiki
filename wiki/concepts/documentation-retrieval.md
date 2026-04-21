---
type: concept
title: Documentation Retrieval
slug: documentation-retrieval
date: 2026-04-20
updated: 2026-04-20
aliases: [code documentation retrieval, API documentation retrieval]
tags: [retrieval, code-generation, documentation, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Documentation Retrieval** (文档检索) — the task of retrieving relevant code documentation entries (API signatures, usage paragraphs, flag descriptions) from a documentation pool given a natural language query or programming intent.

## Key Points

- Documentation pools are structured differently from general text corpora: each entry typically describes a single concept (one function, one argument flag), making fine-grained paragraph-level segmentation important.
- BM25 performs competitively for Bash documentation retrieval because tldr intents are written in command-centric vocabulary that overlaps with manual text; it performs poorly for Python/CoNaLa where queries are StackOverflow-style (low lexical overlap), where dense retrievers like CodeT5-fine-tuned dominate (recall@10: 55.81% vs 9.73% for BM25).
- Dense retriever training benefits from mixing supervised positives (oracle-annotated docs) with weak self-supervised positives (same sentence, different dropout masks from SimCSE), yielding large improvements over either source alone.
- Documentation pool can be updated without model retraining, providing a mechanism for adapting code generation to new libraries at deployment time — a key advantage over purely parametric models.
- The relevance signal for documentation retrieval differs from QA: the relevant doc must contain function signatures and argument names that appear verbatim in the target code, not just semantic answers.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-docprompting-2207-05987]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-docprompting-2207-05987]].
