---
type: entity
title: GraphRAGBench-Medical
slug: graphragbench-medical
date: 2026-04-20
entity_type: dataset
aliases: [GraphRAGBench_medical, Medical]
tags: []
---

## Description

GraphRAGBench-Medical is the medical-textbook corpus incorporated into [[wang-2026-ragrouterbench-2602-00296]] as a structured, specialized benchmark environment.

## Key Contributions

- Contributes a single textbook-scale document with average length `221,495.0` tokens and `1,896` pre-annotated questions.
- Shows the strongest overall DeepSeek-V3 average for HybridRAG at `64.7%`, while IterativeRAG is the most cost-efficient at `13.27M` total tokens in appendix accounting.
- Serves as a case where complementary retrieval or iterative refinement is especially valuable for specialized terminology.

## Related Concepts

- [[adaptive-rag-routing]]
- [[query-corpus-compatibility]]
- [[iterative-rag]]

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]
