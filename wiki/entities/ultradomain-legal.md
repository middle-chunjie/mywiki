---
type: entity
title: UltraDomain-Legal
slug: ultradomain-legal
date: 2026-04-20
entity_type: dataset
aliases: [UltraDomain_legal, Legal]
tags: []
---

## Description

UltraDomain-Legal is the legal-document corpus used in [[wang-2026-ragrouterbench-2602-00296]] to test routing under long, densely interconnected specialized text.

## Key Contributions

- Contributes `94` documents with average length `50,785.0` tokens in the appendix statistics.
- Exhibits high hubness and low dispersion, which the paper uses to explain limitations of pure dense retrieval.
- Produces the most expensive graph-based retrieval setting, with HybridRAG reaching `230k` average tokens per query in the main table and `293.10M` total tokens in appendix cost accounting.

## Related Concepts

- [[adaptive-rag-routing]]
- [[query-corpus-compatibility]]
- [[hubness]]

## Sources

- [[wang-2026-ragrouterbench-2602-00296]]
