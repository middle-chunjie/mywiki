---
type: source
subtype: paper
title: "SiReRAG: Indexing Similar and Related Information for Multihop Reasoning"
slug: unknown-nd-sirerag
date: 2026-04-20
language: en
tags: [rag, retrieval, multihop-qa, indexing]
processed: true

raw_file: raw/papers/unknown-nd-sirerag/paper.pdf
raw_md: raw/papers/unknown-nd-sirerag/paper.md
bibtex_file: raw/papers/unknown-nd-sirerag/paper.bib
possibly_outdated: false

authors:
  - Nan Zhang
  - Prafulla Kumar Choubey
  - Alexander Fabbri
  - Gabriel Bernadett-Shapiro
  - Rui Zhang
  - Prasenjit Mitra
  - Caiming Xiong
  - Chien-Sheng Wu
year: 2025
venue: ICLR 2025
venue_type: conference
arxiv_id: 2412.06206
doi: 10.48550/arXiv.2412.06206
url: https://openreview.net/pdf?id=yp95goUAT1
citation_key: unknownndsirerag
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

SiReRAG proposes a dual-tree indexing scheme for [[retrieval-augmented-generation]] aimed at improving [[multihop-question-answering]]. The similarity branch keeps RAPTOR-style recursive summaries over semantically close chunks in a shallow `4`-level tree, while the relatedness branch extracts entities and fine-grained propositions, groups propositions by shared entities into [[proposition-aggregation]], and recursively summarizes those aggregates. The two trees are flattened into one retrieval pool so the system can surface either semantically similar evidence or cross-document bridge facts. On MuSiQue, 2WikiMultiHopQA, and HotpotQA, the method improves average F1 over strong indexing baselines, and the ablations show that aggregated propositions and relatedness-side summaries are the main source of the gain rather than simply adding more raw propositions.

## Problem & Motivation

Existing RAG indexing methods in the paper each emphasize only one organization principle. RAPTOR synthesizes semantically similar chunks, while GraphRAG and HippoRAG focus more on entity-mediated relatedness. The authors argue that multihop QA often needs both: semantically similar chunks help consolidate local evidence, but entity-linked related chunks help bridge hops across documents. If an index models only one side, the retrieval pool may miss the specific synthesized evidence node that best supports the final answer. SiReRAG is motivated by this gap and treats indexing itself as a knowledge-integration stage rather than only a storage step.

## Method

- **Similarity tree**: reuse RAPTOR-style [[rag-indexing]] with [[gaussian-mixture-model]]-based [[soft-clustering]] over chunk representations, producing a shallow tree with `L = 4` total levels.
- **Tree-structure study**: before finalizing the design, the paper compares RAPTOR against a two-level hierarchical chunk placement on QuALITY and finds `78.88` vs `78.76` average accuracy, so it keeps the simpler RAPTOR organization.
- **Entity and proposition extraction**: build the relatedness side by running [[entity-extraction]] and proposition extraction on each chunk/document; a proposition is a factual statement tied to one or more entities.
- **Distillation pipeline**: rewrite `10K` IndustryCorpus documents with `Meta-Llama-3-70B-Instruct`, extract entities/propositions with the same model, then distill the outputs into a fine-tuned `Mistral-7B-Instruct-v0.3` used on MuSiQue, 2WikiMultiHopQA, and HotpotQA.
- **Relatedness tree**: form [[proposition-aggregation]] by exact-match shared entities, preserve original within-document order, discard propositions with `|E| = 0`, and apply [[recursive-summarization]] on top of those aggregates.
- **Unified index**: flatten all nodes from the similarity tree and relatedness tree into one retrieval pool so retrieval can return bottom-level chunks, proposition aggregates, or higher-level summaries from either branch.
- **Retrieval and QA setup**: use `text-embedding-3-small` for embeddings, retrieve `k = 20` candidates per query, and answer with `GPT-4o`; indexing-time LLM calls use either `GPT-3.5-Turbo` or `GPT-4o`.
- **Efficiency metric**: compare against RAPTOR with `TPER = (Time_A / Time_B) / (Pool_A / Pool_B)`, where method `A` is SiReRAG and method `B` is RAPTOR.

## Key Results

- Main QA results: average F1 reaches `65.88` with `GPT-3.5-Turbo` indexing and `65.83` with `GPT-4o`, versus `64.03` for HippoRAG, `61.21` for RAPTOR (`GPT-4o`), and `30.15` for GraphRAG.
- MuSiQue: SiReRAG (`GPT-4o`) reaches `EM 40.50 / F1 53.08`, beating HippoRAG's `43.78` F1 and RAPTOR's `49.09` F1.
- 2WikiMultiHopQA: SiReRAG reaches `EM 59.60 / F1 67.94`, improving over RAPTOR's `61.45` F1 but still trailing HippoRAG's `74.01` F1.
- HotpotQA: SiReRAG reaches `EM 61.70 / F1 76.48`, above RAPTOR's `73.08` F1 and HippoRAG's `74.29` F1.
- Ablations: removing relatedness-side summaries drops average F1 from `65.83` to `64.04`; adding raw propositions gives `64.20`; removing proposition aggregates drops to `60.47`; replacing the design with dual chunk clustering yields only `59.70`.
- Applicability and efficiency: `SiReRAG + BM25` reaches `63.09` average F1 versus `55.26` for BM25 alone; `SiReRAG + ColBERTv2` reaches `63.93` versus `59.90` for ColBERTv2 alone. Average TPQ is `2.315` seconds versus RAPTOR's `1.500`, but average `TPER = 0.539`.

## Limitations

- The method is not uniformly best on every dataset: on 2WikiMultiHopQA, HippoRAG still has the strongest reported F1 (`74.01`).
- Inference is slower than RAPTOR because the retrieval pool is larger and the method maintains two indexing structures.
- The relatedness pipeline depends on multiple external models and prompts, including proprietary APIs and a separate distillation stage, which raises engineering and reproducibility cost.
- Entity grouping uses exact-match shared entities, so aliasing, normalization errors, or noisy extracted entities can propagate directly into proposition aggregates.
- The paper evaluates end-to-end QA rather than standalone retrieval metrics, so the precise retrieval contribution of each component is inferred indirectly from downstream accuracy.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[multihop-question-answering]]
- [[rag-indexing]]
- [[recursive-summarization]]
- [[entity-extraction]]
- [[proposition-aggregation]]
- [[soft-clustering]]
- [[gaussian-mixture-model]]
- [[retrieval-granularity]]
- [[knowledge-integration]]

## Entities Extracted

- [[nan-zhang]]
- [[prafulla-kumar-choubey]]
- [[alexander-fabbri]]
- [[gabriel-bernadett-shapiro]]
- [[rui-zhang]]
- [[prasenjit-mitra]]
- [[caiming-xiong]]
- [[chien-sheng-wu]]
- [[pennsylvania-state-university]]
- [[salesforce-research]]
- [[musique]]
- [[2wiki-multihopqa]]
- [[hotpotqa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
