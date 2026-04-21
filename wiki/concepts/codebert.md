---
type: concept
title: CodeBERT
slug: codebert
date: 2026-04-20
aliases: [Code-BERT, codebert-model]
tags: [code-representation, pre-trained-model, transformer, nlp]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**CodeBERT** — a bimodal pre-trained Transformer model (Feng et al., 2020) jointly trained on natural language and programming language, producing general-purpose code representations through masked language modeling and replaced token detection on paired NL-PL data.

## Key Points

- Has approximately 125 million parameters; pre-trained on over 8 million NL-code pairs from CodeSearchNet across 6 programming languages.
- Achieves strong performance on code intelligence benchmarks including code search, code summarization, and code classification, but at significantly larger scale than task-specific models.
- On Project CodeNet code classification, CodeBERT underperforms HiT on C++1000 (+8.92%) and C++1400 (+10.22%), suggesting pre-trained sequence models still struggle with structurally complex C++ code.
- Requires no task-specific architecture; fine-tuned downstream, but demands substantial compute and pre-training data, unlike lightweight structural models like HiT.
- Used as a reference upper-bound in HiT evaluation to show competitive performance at ~27× fewer parameters.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-implant-2303-07826]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-implant-2303-07826]].
