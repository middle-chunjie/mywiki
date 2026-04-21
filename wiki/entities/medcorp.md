---
type: entity
title: MedCorp
slug: medcorp
date: 2026-04-20
entity_type: tool
aliases: []
tags: [corpus, medical, rag]
---

## Description

MedCorp is a large mixed-domain biomedical retrieval corpus composed of four sub-corpora: PubMed (23.9M biomedical abstracts), StatPearls (301.2k clinical decision support snippets), Textbooks (125.8k medical textbook passages), and Wikipedia (29.9M general knowledge snippets), totaling ~54.2M snippets and ~10 GB of plain text.

## Key Contributions

- Serves as the unified external knowledge base for the MIRAGE benchmark evaluation.
- Combines multiple data formats (abstracts, clinical guides, textbooks, encyclopedic) to stress-test multi-source retrieval methods.
- The Textbooks sub-corpus represents only ~0.2% of MedCorp's total snippets, used to highlight MoGG's efficiency under limited training data.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[medical-question-answering]]
- [[mix-of-granularity]]

## Sources

- [[zhong-2024-mixofgranularity-2406-00456]]
