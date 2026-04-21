---
type: concept
title: Error Typology
slug: error-typology
date: 2026-04-20
updated: 2026-04-20
aliases: [error taxonomy, 错误类型学]
tags: []
source_count: 1
confidence: low
domain_volatility: medium
last_reviewed: 2026-04-20
---

## Definition

**Error Typology** (错误类型学) — a structured classification of failure modes that organizes qualitatively different kinds of model errors into interpretable categories.

## Key Points

- Face4RAG proposes an FCE-specific typology with `3` top-level classes and `9` named error types for RAG answers.
- The typology separates knowledge-level mismatches from logical-structure mismatches, which the paper argues prior RAG factuality work often conflated.
- The synthetic dataset is explicitly generated from this typology, allowing per-error evaluation instead of only aggregate accuracy.
- The real-world analysis uses the same typology to show that different LLMs produce different error distributions.

## My Position

<!-- User's stance. Populated by personal writing. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[xu-2024-facerag-2407-01080]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[xu-2024-facerag-2407-01080]].
