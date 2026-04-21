---
type: entity
title: QuALITY
slug: quality
date: 2026-04-20
entity_type: dataset
aliases: [QuALITY]
tags: []
---

## Description

QuALITY is the long-form narrative question answering corpus reused in [[wang-2026-ragrouterbench-2602-00296]] as one of the benchmark's four routing environments.

## Key Contributions

- Contributes `265` long documents with average length `5,741.1` tokens in the appendix corpus statistics.
- Exposes a narrative, graph-sparse setting where NaiveRAG outperforms GraphRAG on factual questions (`83.7%` vs. `70.7%` with DeepSeek-V3).
- Serves as a stress test for summary-style questions over long implicit narratives.

## Related Concepts

- [[adaptive-rag-routing]]
- [[query-corpus-compatibility]]
- [[retrieval-augmented-generation]]

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]
