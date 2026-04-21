---
type: concept
title: Source Code Modeling
slug: source-code-modeling
date: 2026-04-20
updated: 2026-04-20
aliases: [code modeling, 源代码建模]
tags: []
source_count: 1
confidence: low
domain_volatility: high
last_reviewed: 2026-04-20
---

## Definition

**Source Code Modeling** (源代码建模) — learning representations of programs that capture lexical form, syntactic structure, and semantic behavior for downstream software-engineering tasks.

## Key Points

- The survey argues that source code is not homogeneous natural language because it combines program tokens, optional natural-language comments, and explicit program structure.
- It frames SE tasks over code as both understanding and generation problems, covering `18` benchmarks across classification, retrieval, repair, summarization, and translation.
- The paper highlights ASTs, data-flow graphs, and identifier patterns as signals that generic NLP modeling often underutilizes.
- In the conclusion, the authors argue that future CodePTMs should learn both code form and code functionality rather than only token-level surface form.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[niu-2022-deep-2205-11739]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[niu-2022-deep-2205-11739]].
