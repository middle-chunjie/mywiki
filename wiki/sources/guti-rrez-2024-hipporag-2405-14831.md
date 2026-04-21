---
type: source
subtype: paper
title: "HippoRAG: Neurobiologically Inspired Long-Term Memory for Large Language Models"
slug: guti-rrez-2024-hipporag-2405-14831
date: 2026-04-20
language: en
tags: [rag, retrieval, knowledge-graph, multi-hop-qa, long-term-memory]
processed: true

raw_file: raw/papers/guti-rrez-2024-hipporag-2405-14831/paper.pdf
raw_md: raw/papers/guti-rrez-2024-hipporag-2405-14831/paper.md
bibtex_file: raw/papers/guti-rrez-2024-hipporag-2405-14831/paper.bib
possibly_outdated: false

authors:
  - Bernal Jiménez Gutiérrez
  - Yiheng Shu
  - Yu Gu
  - Michihiro Yasunaga
  - Yu Su
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2405.14831
doi:
url: http://arxiv.org/abs/2405.14831
citation_key: gutirrez2024hipporag
paper_type: method

read_status: unread

domain: ir
---

## Summary

HippoRAG proposes a retrieval-augmented long-term memory layer for large language models that is explicitly designed to integrate evidence across passages rather than encoding each passage in isolation. The system builds a schemaless knowledge graph from corpus passages using LLM-based named entity recognition and open information extraction, adds synonymy edges with dense retrievers, and performs query-time graph search with Personalized PageRank. This lets the retriever approximate single-step multi-hop reasoning before generation. On MuSiQue and 2WikiMultiHopQA, HippoRAG substantially improves recall over strong single-step baselines, and when combined with IRCoT it yields further gains. The paper's main claim is that graph-based associative memory gives RAG a more scalable and efficient approximation to continuously updated long-term memory than repeated iterative retrieval alone.

## Problem & Motivation

The paper targets a core weakness of standard retrieval-augmented generation: new passages are typically embedded independently, so the retriever struggles when answering a query requires integrating facts that never co-occur in one passage. This makes multi-hop QA and other knowledge-integration workloads difficult unless the system repeatedly alternates retrieval and LLM reasoning, which is slower and more expensive. HippoRAG is motivated by hippocampal indexing theory and tries to mimic associative long-term memory: keep the parametric model fixed, build an explicit index over new experiences, and use graph-based recall so a query can activate relevant neighborhoods rather than only exact or local lexical matches.

## Method

- **Offline indexing**: given passages `P`, an instruction-tuned LLM `L` first extracts named entities and then performs 1-shot OpenIE to produce noun-phrase nodes `N` and relation edges `E`, yielding a schemaless KG over the corpus.
- **Synonymy augmentation**: a retrieval encoder `M` adds synonymy edges `E'` between nodes when `cosine_similarity(M(n_i), M(n_j)) > tau`, with tuned threshold `tau = 0.8`.
- **Passage incidence matrix**: indexing also constructs a matrix `P in R^{|N| x |P|}` whose entries count how often each KG node appears in each original passage.
- **Query processing**: the same LLM extracts query named entities `C_q = {c_1, ..., c_n}` from a question, and the retriever maps each `c_i` to a query node `r_i = argmax_j cosine_similarity(M(c_i), M(e_j))`.
- **Personalized PageRank retrieval**: HippoRAG defines a seed distribution `n` over KG nodes, assigns equal mass to query nodes, and runs PPR over the graph with damping factor `0.5`, producing updated node probabilities `n'`.
- **Node specificity weighting**: before PPR, each query node is reweighted by node specificity `s_i = |P_i|^-1`, where `P_i` is the set of passages containing node `i`, acting as a local IDF-like prior.
- **Passage scoring**: final retrieval scores are computed by aggregating node activations back to passages via `p = n' P`, so passages connected through relevant graph neighborhoods rise even if they do not literally contain all query cues together.
- **Implementation details**: the default LLM is `GPT-3.5-turbo-1106` with temperature `0.0`; retrievers are Contriever or CoLBERTv2; hyperparameters are tuned on `100` MuSiQue training examples.

## Key Results

- On single-step retrieval with CoLBERTv2, HippoRAG improves MuSiQue from `37.9/49.2` to `40.9/51.9` in `R@2/R@5`, and 2WikiMultiHopQA from `59.2/68.2` to `70.7/89.1`.
- With Contriever, HippoRAG reaches `41.0/52.1` on MuSiQue and `71.5/89.5` on 2WikiMultiHopQA, again outperforming dense and LLM-augmented baselines.
- On HotpotQA, HippoRAG is competitive rather than dominant: `60.5/77.7` with CoLBERTv2 versus `64.7/79.3` for plain CoLBERTv2.
- As a retriever inside IRCoT, HippoRAG raises average retrieval to `62.7/78.2` and reaches `45.3/57.6` on MuSiQue plus `75.8/93.9` on 2WikiMultiHopQA.
- QA improvements track retrieval gains: with CoLBERTv2, HippoRAG improves 2WikiMultiHopQA from `33.4` EM / `43.3` F1 to `46.6` EM / `59.5` F1; IRCoT + HippoRAG reaches `47.7` EM / `62.7` F1.
- The paper reports comparable or better quality than iterative retrieval while online retrieval is `10-30x` cheaper and `6-13x` faster than IRCoT.

## Limitations

- All components are used off the shelf; the authors note that many observed errors come from named entity recognition and OpenIE quality rather than the graph idea itself.
- The graph search is still simple PPR; the paper suggests relation-aware traversal and better graph search as clear next steps.
- OpenIE consistency degrades on longer documents, which can weaken the quality of the synthetic hippocampal index.
- The strongest gains are on benchmarks with stronger knowledge-integration demands; HotpotQA shows smaller or mixed benefits.
- Scalability beyond the benchmark-sized corpora in the paper remains unvalidated, so practical large-scale deployment is still an open question.

## Concepts Extracted

- [[large-language-model]]
- [[retrieval-augmented-generation]]
- [[long-term-memory]]
- [[knowledge-graph]]
- [[open-information-extraction]]
- [[personalized-pagerank]]
- [[dense-retrieval]]
- [[named-entity-recognition]]
- [[multihop-question-answering]]
- [[knowledge-integration]]

## Entities Extracted

- [[bernal-jimenez-gutierrez]]
- [[yiheng-shu]]
- [[yu-gu]]
- [[michihiro-yasunaga]]
- [[yu-su]]
- [[ohio-state-university]]
- [[stanford-university]]
- [[hipporag]]
- [[ircot]]
- [[gpt-3-5]]
- [[contriever]]
- [[colbertv2]]
- [[musique]]
- [[2wiki-multihopqa]]
- [[hotpotqa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
