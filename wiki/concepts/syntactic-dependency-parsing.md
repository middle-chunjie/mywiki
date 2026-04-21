---
type: concept
title: Syntactic Dependency Parsing
slug: syntactic-dependency-parsing
date: 2026-04-20
updated: 2026-04-20
aliases: [dependency parsing, dependency tree, 依存句法分析, dependency parse]
tags: [nlp, syntax, parsing, graph]
source_count: 1
confidence: low
domain_volatility: low
last_reviewed: 2026-04-20
---

## Definition

**Syntactic Dependency Parsing** (依存句法分析) — an NLP task that analyzes the grammatical structure of a sentence by identifying head-dependent relationships between words, producing a directed dependency tree where each arc is labeled with a syntactic relation (e.g., `nsubj`, `det`, `cop`).

## Key Points

- The resulting dependency tree can be represented as a graph `G = (V, A, R)` where `V` is the word set, `A` is the adjacency matrix, and `R` is the set of labeled relation types — a natural input for graph neural network processing.
- Dependency relations are largely domain-invariant: the distribution of relation types (e.g., `nsubj`, `det`, `amod`) across different text domains is statistically very similar, as empirically shown in GAST (Zhang et al., 2022).
- Popular parsers include the Stanford Parser (rule/neural hybrid) and the Biaffine Parser (Dozat & Manning, 2016); GAST finds the Biaffine Parser yields 0.5–0.8% higher accuracy than Stanford Parser in cross-domain sentiment tasks.
- Graph attention networks (GAT) operating on dependency graphs can leverage labeled arc types to capture the differential influence of syntactic neighbors on word meaning (e.g., "quite" modifying "interesting" via `advmod` is more sentiment-relevant than "the" via `det`).

## My Position

<!-- User's stance on this concept. Fed by personal writing; tag "(personal stance)" on such bullets. -->

## Contradictions

<!-- No known contradictions yet. -->

## Sources

- [[zhang-2022-graph-2205-08772]]

## Evolution Log

- 2026-04-20 (1 source): Introduced via [[zhang-2022-graph-2205-08772]].
