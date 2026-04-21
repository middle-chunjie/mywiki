---
type: concept
title: Concrete Syntax Tree
slug: concrete-syntax-tree
date: 2026-04-20
aliases: [CST, Parse Tree]
tags: [program-analysis, code-representation, parsing]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Concrete Syntax Tree** (具体语法树) — a full parse tree of a program that retains every syntactic detail of the source text, including punctuation, braces, and semicolons, with each leaf node corresponding to exactly one source code token.

## Key Points

- Unlike abstract syntax trees (ASTs), CSTs preserve punctuation tokens (braces, semicolons, parentheses), enabling direct alignment between tree leaf nodes and source code token positions.
- In HiT, root-to-leaf paths in the CST are extracted per token and used to compute hierarchical embeddings; the path can be split into a global part (root-to-statement) and a local part (statement-to-leaf).
- CSTs are theoretically convertible to ASTs, but ASTs lose the one-to-one correspondence between leaf nodes and source tokens, making structural alignment harder.
- Tree-sitter is used as a language-agnostic CST parser, enabling HiT to process C++, Java, Python, and Ruby without language-specific preprocessing.
- Average CST root-to-leaf path length is short (6.89–9.26 nodes across datasets), making Transformer encoding of paths computationally cheap.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-implant-2303-07826]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-implant-2303-07826]].
