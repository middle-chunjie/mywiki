---
type: concept
title: Abstract Syntax Tree Entity Recognition
slug: abstract-syntax-tree-entity-recognition
date: 2026-04-20
updated: 2026-04-20
aliases: [AST Entity Recognition, AER, 抽象语法树实体识别]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Abstract Syntax Tree Entity Recognition** (抽象语法树实体识别) — a structure-aware pretraining objective that labels code tokens with AST-derived syntactic roles to improve downstream code understanding and translation.

## Key Points

- [[tehranijamsaz-2024-coderosetta-2410-20527]] introduces AER after cross-language MLM to teach CodeRosetta token-level structural roles before sequence-to-sequence translation.
- The method parses source code with [[tree-sitter]] and assigns tags such as identifier/variable, function, type identifier, primitive type, number literal, pointer/reference, and constant.
- Tokens without a matched AST role receive the label `O`, mirroring outside tags in classical named entity recognition.
- The tag set is extensible: for CUDA, the paper highlights adding parallel-programming-specific entities such as `threadIdx`, `blockIdx`, and `gridDim`.
- Ablation shows AER materially helps `C++ → CUDA`: removing it lowers BLEU from `76.90` to `74.98` and CodeBLEU from `78.84` to `75.55`.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[tehranijamsaz-2024-coderosetta-2410-20527]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[tehranijamsaz-2024-coderosetta-2410-20527]].
