---
type: entity
title: SolutionRAG
slug: solutionrag
date: 2026-04-20
entity_type: tool
aliases: [Solution RAG]
tags: []
---

## Description

SolutionRAG is the RAG system proposed in [[li-2025-deepsolution-2502-20730]] for generating reliable engineering solutions via tree-based exploration, alternating review, and node pruning.

## Key Contributions

- Introduces a bi-point thinking tree that alternates between solution nodes and comment nodes.
- Uses proposal-conditioned retrieval and logit-based node evaluation to refine and prune candidate branches.
- Achieves the strongest reported results on [[solutionbench]] across all eight engineering domains.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[tree-based-exploration]]
- [[bi-point-thinking]]

## Sources

- [[li-2025-deepsolution-2502-20730]]
