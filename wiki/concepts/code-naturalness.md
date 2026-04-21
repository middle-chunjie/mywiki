---
type: concept
title: Code Naturalness
slug: code-naturalness
date: 2026-04-20
aliases: [Naturalness of Software, software-naturalness]
tags: [code-representation, nlp, program-analysis]
source_count: 1
confidence: low
graph-excluded: false
---

## Definition

**Code Naturalness** (代码自然性) — the empirical observation that real-world source code is highly repetitive and statistically regular, sharing statistical properties with natural language, which justifies applying NLP-inspired sequence models to source code.

## Key Points

- Originated from Hindle et al. (2012, ICSE), who showed that code has low entropy and can be modeled with n-gram language models.
- Motivates sequence-based code representation models: treating token sequences as natural language allows transfer of NLP architectures (RNNs, Transformers) to code tasks.
- The "naturalness" assumption privileges sequential order information; pure sequence models are therefore sub-optimal at capturing structural information such as block nesting and syntactic roles.
- HiT frames itself as *combining* naturalness (sequence model) with hierarchical structure (CST paths), arguing the two are complementary rather than competing.
- Approaches that replace token sequences with flattened AST node sequences sacrifice naturalness: the sequence becomes longer and code tokens are interspersed with non-terminal tree nodes.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2023-implant-2303-07826]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2023-implant-2303-07826]].
