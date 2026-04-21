---
type: concept
title: Natural Language to Code
slug: natural-language-to-code
date: 2026-04-20
updated: 2026-04-20
aliases: [NL-to-Code, NL2Code, text-to-code, NL→Code]
tags: [code-generation, nlp, llm]
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Natural Language to Code** (自然语言到代码生成) — the task of generating source code from a natural language instruction, query, or comment, typically framed as a conditional generation problem `M: X → Y` where `X` is a natural language intent and `Y` is a code snippet.

## Key Points

- [[zhou-2023-codebertscore-2302-05527]] formalizes NL→Code as a metric evaluation problem: given context `x`, generated code `ŷ`, and reference `y*`, a reliable metric `f(ŷ, y*)` should rank functionally equivalent candidates higher.
- LLMs such as Codex, InCoder, and SantaCoder have dramatically raised NL→Code accuracy, making reliable evaluation the new bottleneck.
- Evaluation approaches span token-matching (BLEU, CrystalBLEU), structural (CodeBLEU), execution-based (pass@k), and embedding-based (CodeBERTScore) methods.
- The task includes diverse settings: single-function completion, multi-line generation, API usage, and data science scripting.
- CoNaLa (Stack Overflow queries → Python) and HumanEval (docstring → Python function) are two widely-used benchmarks.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhou-2023-codebertscore-2302-05527]]
- [[zhou-2023-docprompting-2207-05987]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhou-2023-codebertscore-2302-05527]].
