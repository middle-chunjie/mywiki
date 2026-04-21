---
type: entity
title: RA-ISF
slug: ra-isf
date: 2026-04-20
entity_type: tool
aliases: [RA-ISF, Retrieval Augmented Iterative Self-Feedback]
tags: []
---

## Description

RA-ISF is the retrieval-augmented reasoning framework proposed in [[liu-2024-raisf-2403-06840]]. It combines self-knowledge assessment, passage relevance filtering, and recursive question decomposition to answer knowledge-intensive questions more robustly.

## Key Contributions

- Improves GPT-3.5 average exact match to `55.0`, outperforming direct prompting, RAG, IRCoT, and SKR on the paper's five-task benchmark.
- Raises Llama-2 13B average exact match to `46.0`, beating REPLUG and Self-RAG on four of five evaluated datasets.
- Uses an iterative fallback structure that returns `"unknown"` once decomposition depth exceeds `D_th = 3`.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[self-knowledge]]
- [[passage-relevance]]
- [[question-decomposition]]

## Sources

- [[liu-2024-raisf-2403-06840]]
