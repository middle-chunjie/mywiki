---
type: concept
title: Code Comment Generation
slug: code-comment-generation
date: 2026-04-20
updated: 2026-04-20
aliases: [code summarization for SE, automatic code documentation, 代码注释生成]
tags: [software-engineering, code, generation, nlp]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Code Comment Generation** (代码注释生成) — an NLP task in software engineering that automatically produces natural-language summaries or descriptions of source code snippets using neural network models, typically framed as sequence-to-sequence translation from code tokens to comment text.

## Key Points

- Encoder-decoder (seq2seq) architectures are the dominant paradigm: the encoder processes the code token sequence (possibly augmented with AST structural information) and the decoder generates the natural-language comment autoregressively.
- Five representative model families have been studied: LSTM-based seq2seq, Transformer-based seq2seq, GNN-based (structure-aware) seq2seq, dual-learning (code-comment co-training), and retrieval-augmented neural models (e.g., Rencos).
- Evaluation uses machine translation metrics (BLEU, METEOR, ROUGE-L); the Java dataset benchmark (≈69k training pairs from Javadoc) and Python dataset (≈50k pairs from docstrings) are standard testbeds.
- Models are fragile to small identifier-level perturbations: renaming a method or variable (preserving functionality) can degrade BLEU by 60–79% in pure seq2seq models, exposing heavy reliance on identifier surface form rather than code structure.
- Retrieval-augmented and structure-aware models (Rencos, GNN) are notably more robust to identifier perturbations than pure contextual models, suggesting that structural priors or similar-code retrieval provides complementary signal.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2022-adversarial]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2022-adversarial]].
