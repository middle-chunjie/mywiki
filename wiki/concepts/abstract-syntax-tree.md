---
type: concept
title: Abstract Syntax Tree
slug: abstract-syntax-tree
date: 2026-04-20
updated: 2026-04-20
aliases: [AST, 抽象语法树]
tags: [parsing, program-analysis]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Abstract Syntax Tree** (抽象语法树) — a tree-structured representation of program syntax that abstracts away surface formatting while preserving grammatical structure.

## Key Points

- The paper uses `tree-sitter` to obtain ASTs for Python files as part of CCFINDER's preprocessing pipeline.
- AST traversal is used to extract files, classes, functions, global variables, and their local structure.
- Benchmark construction further filters out prompts that cannot be parsed by the AST parser.
- The AST is not the final retrieval object; it is an intermediate representation used to build the project context graph.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[ding-2023-cocomic-2212-10007]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[ding-2023-cocomic-2212-10007]].
