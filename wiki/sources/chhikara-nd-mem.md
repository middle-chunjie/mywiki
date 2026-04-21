---
type: source
subtype: paper
title: "Mem0: Building Production-Ready AI Agents with Scalable Long-Term Memory"
slug: chhikara-nd-mem
date: 2026-04-20
language: en
tags: [agents, llm, memory, dialogue, evaluation]
processed: true

raw_file: raw/papers/chhikara-nd-mem/paper.pdf
raw_md: raw/papers/chhikara-nd-mem/paper.md
bibtex_file: raw/papers/chhikara-nd-mem/paper.bib
possibly_outdated: false

authors:
  - Prateek Chhikara
  - Dev Khant
  - Saket Aryan
  - Taranjeet Singh
  - Deshraj Yadav
year: 2025
venue: ECAI 2025
venue_type: conference
arxiv_id:
doi:
url: https://mem0.ai/research
citation_key: chhikarandmem
paper_type: method

read_status: unread
read_date:
rating:

domain: agents
---

## Summary

The paper proposes Mem0, a production-oriented long-term memory architecture for LLM agents, and `Mem0^g`, a graph-memory extension that explicitly stores entity-relation structure. Instead of repeatedly passing full dialogue history, the system incrementally extracts salient facts from each new message pair, updates a persistent memory store with add/update/delete/no-op decisions, and retrieves only relevant memories at query time. Evaluated on the [[locomo]] benchmark, Mem0 delivers the strongest single-hop and multi-hop performance among memory systems, while `Mem0^g` is strongest on temporal reasoning and competitive on open-domain questions. The core empirical claim is that selective persistent memory substantially improves efficiency: compared with full-context inference, Mem0 cuts p95 latency by about 91% while preserving near-competitive answer quality.

## Problem & Motivation

Current LLM agents remain constrained by fixed context windows, so they forget user preferences and previously established facts once those details fall outside the prompt. Simply expanding context windows does not solve the deployment problem because long conversations contain large amounts of irrelevant text, attention degrades over distant tokens, and repeated full-context inference is costly. The paper argues for a memory layer that behaves more like human long-term memory: it should selectively retain salient facts, consolidate related information, resolve contradictions over time, and retrieve only the pieces needed for a new query. This is especially important for multi-session dialogue agents that need coherence, personalization, and temporal consistency over long horizons.

## Method

- **Mem0 extraction pipeline**: each new interaction is processed as a message pair ``(m_{t-1}, m_t)`` together with conversation-level and local context. The extraction prompt is ``P = (S, {m_{t-m}, ..., m_{t-2}}, m_{t-1}, m_t)``, where `S` is an asynchronously refreshed conversation summary.
- **Salient memory extraction**: an LLM extraction function ``\phi(P)`` produces candidate memories ``\Omega = {\omega_1, \omega_2, ..., \omega_n}``, aiming to capture only information worth persisting rather than storing raw dialogue verbatim.
- **Memory update logic**: for each candidate fact ``\omega_i``, the system retrieves the top `s` semantically similar memories from a dense vector store, then uses LLM tool calling to choose one of four operations: `ADD`, `UPDATE`, `DELETE`, or `NOOP`.
- **Configured hyperparameters**: experiments set the local context window to ``m = 10`` previous messages and compare each candidate against ``s = 10`` retrieved memories. All LLM operations use [[gpt-4o-mini]].
- **Graph-memory extension**: `Mem0^g` represents memory as a directed labeled graph ``G = (V, E, L)``, with entity nodes, relation edges, node labels, embeddings ``e_v``, and timestamps ``t_v``. Relations are stored as triples ``(v_s, r, v_d)``.
- **Graph extraction and update**: an entity extractor first identifies typed entities, then a relation generator produces semantic triples such as `prefers`, `lives_in`, or `happened_on`. New nodes are merged or created based on an entity-similarity threshold ``t``; conflicting relations are marked obsolete rather than physically deleted.
- **Retrieval strategy**: `Mem0^g` supports both entity-centric retrieval, which expands neighborhoods around anchor entities, and semantic triplet retrieval, which ranks graph relations against a dense embedding of the query.
- **Infrastructure choices**: the dense-memory variant relies on a vector database for similarity search, while the graph variant uses [[neo4j]] as the backing graph database and keeps the structured extraction/update loop inside the LLM pipeline.

## Key Results

- On [[locomo]], **Mem0** achieves the best memory-system scores for single-hop questions: `F1 = 38.72`, `BLEU-1 = 27.13`, `J = 67.13 ± 0.65`.
- For multi-hop questions, **Mem0** again leads memory methods with `F1 = 28.64` and `J = 51.15 ± 0.31`.
- For temporal questions, **Mem0^g** is strongest with `F1 = 51.55`, `BLEU-1 = 40.28`, and `J = 58.13 ± 0.44`.
- On overall quality, **Mem0^g** reaches `68.44 ± 0.17` J in Table 2, outperforming the best RAG configuration (`60.97 ± 0.20`) by roughly `7.5` absolute points.
- Compared with full-context inference, **Mem0** reduces p95 total latency from `17.117s` to `1.440s` and search latency to `0.200s`, while `Mem0^g` reaches `2.590s` p95 total latency.
- Memory footprint is also reduced: Mem0 stores roughly `7k` tokens per conversation on average, `Mem0^g` about `14k`, versus about `26k` tokens for full raw context and more than `600k` for Zep's graph memory.

## Limitations

- Full-context processing still attains the highest overall J score in Table 2 (`72.90 ± 0.19`), so the proposed memory systems do not dominate the strongest accuracy baseline in every setting.
- The graph variant helps temporal reasoning but underperforms base Mem0 on single-hop and multi-hop questions, indicating that relational structure adds overhead and is not uniformly beneficial.
- Evaluation is centered on one benchmark, [[locomo]], with only 10 long conversations; broader validation across domains, tasks, and user populations is still missing.
- The implementation depends on proprietary components such as [[gpt-4o-mini]] and OpenAI embeddings, so reproducibility and cost may differ for open-weight or self-hosted deployments.
- The paper reports favorable latency and token results but provides limited ablation on extraction quality, threshold sensitivity, and failure cases in memory update decisions.

## Concepts Extracted

- [[large-language-model]]
- [[long-term-memory]]
- [[persistent-memory]]
- [[graph-memory]]
- [[conversation-summary]]
- [[semantic-similarity]]
- [[vector-database]]
- [[retrieval-augmented-generation]]
- [[temporal-reasoning]]

## Entities Extracted

- [[prateek-chhikara]]
- [[dev-khant]]
- [[saket-aryan]]
- [[taranjeet-singh]]
- [[deshraj-yadav]]
- [[mem0-ai]]
- [[locomo]]
- [[gpt-4o-mini]]
- [[neo4j]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
