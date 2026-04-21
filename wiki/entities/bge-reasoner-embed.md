---
type: entity
title: BGE-Reasoner-Embed
slug: bge-reasoner-embed
date: 2026-04-20
entity_type: tool
aliases: [BGE Reasoner Embed]
tags: []
---

## Description

BGE-Reasoner-Embed is the first-stage embedding retriever used in the Retro* evaluation on BRIGHT. The paper describes it as the current state-of-the-art embedding model on that benchmark and uses its public top-100 retrieval results.

## Key Contributions

- Supplies the candidate documents that Retro* reranks in the main BRIGHT experiments.
- Provides a strong first-stage retrieval baseline with average BRIGHT nDCG@10 of `32.5`.
- Makes the evaluation more challenging by surfacing more positive candidates for reranking.

## Related Concepts

- [[reasoning-intensive-retrieval]]
- [[reranking]]

## Sources

- [[lan-2026-retro-2509-24869]]
