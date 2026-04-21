---
type: concept
title: DocPrompting
slug: docprompting
date: 2026-04-20
updated: 2026-04-20
aliases: [Doc-Prompting, documentation prompting]
tags: [code-generation, retrieval, documentation, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**DocPrompting** (文档提示) — a retrieve-then-generate framework for NL-to-code generation that retrieves relevant code documentation at inference time and conditions the generator on both the NL intent and retrieved documentation snippets.

## Key Points

- Decomposes code generation as `P(c | D, n) ≈ P(c | D_hat_n, n)` where `D_hat_n` is the top-k retrieved documentation subset; retrieval and generation are separate, independently trainable modules.
- Documentation pool `D` is an external, updatable store — new libraries can be added without retraining, enabling generalization to unseen functions and APIs.
- Two retriever families are supported: sparse (BM25) and dense (contrastive-trained RoBERTa or CodeT5 encoder); for short-context generators (T5, CodeT5), fusion-in-decoder encodes each (intent, doc) pair independently.
- Documentation bridges the "intent terminology" vs "code terminology" gap: retrieving docs increases unigram overlap between NL intent and target code from 12% → 24% (tldr) and 30% → 91% (CoNaLa).
- Consistently improves strong baselines (CodeT5, GPT-Neo, Codex) on both Bash (tldr) and Python (CoNaLa) benchmarks; especially effective for unseen-function generalization (unseen function recall: 9.03% → 18.30% for CodeT5 on CoNaLa).

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-docprompting-2207-05987]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-docprompting-2207-05987]].
