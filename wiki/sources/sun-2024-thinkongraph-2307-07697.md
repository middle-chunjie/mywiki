---
type: source
subtype: paper
title: "Think-on-Graph: Deep and Responsible Reasoning of Large Language Model on Knowledge Graph"
slug: sun-2024-thinkongraph-2307-07697
date: 2026-04-20
language: en
tags: [llm, knowledge-graph, reasoning, kbqa, beam-search]
processed: true

raw_file: raw/papers/sun-2024-thinkongraph-2307-07697/paper.pdf
raw_md: raw/papers/sun-2024-thinkongraph-2307-07697/paper.md
bibtex_file: raw/papers/sun-2024-thinkongraph-2307-07697/paper.bib
possibly_outdated: false

authors:
  - Jiashuo Sun
  - Chengjin Xu
  - Lumingyuan Tang
  - Saizhuo Wang
  - Chen Lin
  - Yeyun Gong
  - Lionel M. Ni
  - Heung-Yeung Shum
  - Jian Guo
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2307.07697
doi:
url: http://arxiv.org/abs/2307.07697
citation_key: sun2024thinkongraph
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper proposes Think-on-Graph (ToG), a training-free `LLM ⊗ KG` reasoning framework that treats a large language model as an agent which iteratively explores a knowledge graph, prunes candidate relations and entities, and decides when enough evidence has been gathered to answer a question. Instead of only translating questions into a fixed KG query, ToG maintains top-`N` reasoning paths and uses beam search to grow them step by step, aiming to improve deep reasoning while keeping the evidence explicit and editable. Across `9` datasets spanning multi-hop KBQA, open-domain QA, slot filling, and fact checking, the paper reports strong gains over prompting baselines and claims new state of the art on most settings without additional training.

## Problem & Motivation

The paper targets a failure mode of standalone LLM reasoning: answers to knowledge-intensive questions often suffer from hallucination, outdated knowledge, and weak multi-hop reasoning. It argues that prior `LLM + KG` pipelines use the LLM mostly as a translator from natural language into a KG query or retrieval command, so the model does not directly participate in graph exploration. ToG is motivated by a tighter coupling in which the LLM can use its own reasoning ability to expand, rank, and combine knowledge-graph paths, while keeping the evidence chain explicit enough for tracing and correction.

## Method

- **Paradigm**: ToG instantiates an `LLM ⊗ KG` setup in which the LLM acts as an agent during graph reasoning rather than only before retrieval.
- **State representation**: it maintains top-`N` reasoning paths `P = {p_1, p_2, ..., p_N}` for question `x`; at iteration `D`, each path contains `D - 1` triples of the form ``(e^d_s, r^d_j, e^d_o)``.
- **Initialization**: the LLM extracts topic entities from the question and initializes the beam with top-`N` starting entities `E^0 = {e^0_1, ..., e^0_N}`.
- **Relation exploration**: from current tail entities `E^{D-1}`, ToG searches KG neighbors to get candidate relations `R^D_cand`, then asks the LLM to prune them into top relations `R^D` conditioned on the question and partial paths.
- **Entity exploration**: using the retained relations, ToG expands to candidate entities `E^D_cand` and again lets the LLM prune them into the next top-`N` entities `E^D`, extending the beam.
- **Reasoning and stopping**: after each exploration round, the LLM judges whether the current paths are sufficient for answer generation; if yes, it answers from the paths, otherwise it continues until maximum depth `D_max`.
- **Complexity in LLM calls**: ToG needs at most ``2ND + D + 1`` LLM calls; the relation-only variant ToG-R reduces this to ``ND + D + 1`` by replacing entity pruning with random pruning.
- **ToG-R variant**: ToG-R keeps relation chains ``(e^0_n, r^1_n, ..., r^D_n)`` instead of explicit triples, reducing cost but dropping intermediate-entity information.
- **Experimental hyperparameters**: default beam width `N = 3`, maximum depth `D_max = 3`, reasoning prompts use `5` shots, and Llama-2-70B-Chat uses temperature `0.4` for exploration, `0` for reasoning, and maximum generation length `256`.
- **Knowledge sources**: Freebase is used for CWQ, WebQSP, GrailQA, Simple Questions, and WebQuestions; Wikidata is used for QALD10-en, T-REx, Zero-Shot RE, and Creak.

## Key Results

- On the main benchmark table, GPT-4-backed ToG/ToG-R reaches `82.6/81.9` on WebQSP, `81.4/80.3` on GrailQA, `53.8/54.7` on QALD10-en, `88.3/86.9` on Zero-Shot RE, and `95.6/95.4` on Creak.
- The paper claims new SOTA in `6` out of `9` datasets overall and says GPT-4-backed ToG achieves new SOTA in `7` out of `9` datasets when compared against prior methods, despite requiring no additional training.
- Against no-external-knowledge prompting baselines, ToG with ChatGPT improves from `28.1` to `68.7` on GrailQA and from `28.8` to `88.0` on Zero-Shot RE, showing especially large gains on knowledge-intensive multi-hop settings.
- Table `2` reports that ToG with GPT-4 exceeds CoT with GPT-4 by `+26.5` on CWQ and `+15.3` on WebQSP; the paper also states that ToG with Llama-2-70B can outperform CoT with GPT-4 on these two datasets.
- Ablations show that larger search depth and width help until about depth `3`, after which gains diminish while cost keeps growing.
- KG choice matters: on CWQ/WebQSP, Freebase-backed ToG scores `58.8/76.2`, while Wikidata-backed ToG drops to `54.9/68.6`.

## Limitations

- The method depends heavily on KG coverage and quality; the paper explicitly shows weaker results when the underlying KG is less aligned with the benchmark, such as Wikidata on CWQ/WebQSP.
- Search cost grows with beam width and depth, and ToG can require up to ``2ND + D + 1`` LLM calls, so better reasoning comes with higher latency and token cost.
- The gains are stronger on multi-hop KBQA than on single-hop KBQA, suggesting the approach is not uniformly beneficial across all reasoning regimes.
- Longer explored paths can create prompt-length pressure and small formatting failures; the appendix says format-error rate stays below `3%`, but it is not zero.
- The paper contains an internal reporting inconsistency on CWQ: Table `1` and Table `2` appear to swap the GPT-4 scores of ToG and ToG-R (`67.6` vs `72.5`), so the exact headline number should be checked against the original implementation or code release.

## Concepts Extracted

- [[large-language-model]]
- [[knowledge-graph]]
- [[knowledge-graph-question-answering]]
- [[beam-search]]
- [[multi-hop-reasoning]]
- [[hallucination]]
- [[chain-of-thought-prompting]]
- [[few-shot-prompting]]
- [[in-context-learning]]
- [[knowledge-traceability]]
- [[knowledge-correctability]]

## Entities Extracted

- [[jiashuo-sun]]
- [[chengjin-xu]]
- [[lumingyuan-tang]]
- [[saizhuo-wang]]
- [[chen-lin]]
- [[yeyun-gong]]
- [[lionel-m-ni]]
- [[heung-yeung-shum]]
- [[jian-guo]]
- [[microsoft-research-asia]]
- [[freebase]]
- [[wikidata]]
- [[gpt-4]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
