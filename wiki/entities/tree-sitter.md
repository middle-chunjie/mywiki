---
type: entity
title: Tree-sitter
slug: tree-sitter
date: 2026-04-20
entity_type: tool
aliases: [Tree Sitter]
tags: []
---

## Description

Tree-sitter is a multi-language parsing library used in [[tehranijamsaz-2024-coderosetta-2410-20527]] to build ASTs for token-level structural supervision. In this paper it is the parsing backend behind CodeRosetta's AER pretraining stage.

## Key Contributions

- Generates ASTs from code snippets so the model can label tokens with syntactic categories during [[abstract-syntax-tree-entity-recognition]].
- Enables language-specific structural tagging, including CUDA-oriented constructs such as `threadIdx`, `blockIdx`, and `gridDim`.

## Related Concepts

- [[abstract-syntax-tree]]
- [[abstract-syntax-tree-entity-recognition]]
- [[code-translation]]

## Sources

- [[tehranijamsaz-2024-coderosetta-2410-20527]]
