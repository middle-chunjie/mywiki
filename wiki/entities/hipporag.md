---
type: entity
title: HippoRAG
slug: hipporag
date: 2026-04-20
entity_type: tool
aliases: [HippoRAG]
tags: []
---

## Description

HippoRAG is the retrieval framework introduced in [[guti-rrez-2024-hipporag-2405-14831]] that uses an LLM-built knowledge graph and Personalized PageRank to provide long-term memory for large language models.

## Key Contributions

- Builds a schemaless KG from passages with named entity extraction plus OpenIE, then augments it with synonymy edges from dense retrievers.
- Performs single-step multi-hop retrieval by linking query entities to KG nodes and ranking passages with `p = n' P` after Personalized PageRank.
- Achieves strong gains on MuSiQue and 2WikiMultiHopQA while being substantially cheaper and faster than IRCoT at retrieval time.

## Related Concepts

- [[retrieval-augmented-generation]]
- [[knowledge-graph]]
- [[personalized-pagerank]]

## Sources

- [[guti-rrez-2024-hipporag-2405-14831]]
