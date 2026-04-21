---
type: source
subtype: paper
title: "Search-in-the-Chain: Interactively Enhancing Large Language Models with Search for Knowledge-intensive Tasks"
slug: xu-2024-searchinthechain-2304-14732
date: 2026-04-20
language: en
tags: [llm, retrieval, reasoning, multi-hop-qa, traceability]
processed: true

raw_file: raw/papers/xu-2024-searchinthechain-2304-14732/paper.pdf
raw_md: raw/papers/xu-2024-searchinthechain-2304-14732/paper.md
bibtex_file: raw/papers/xu-2024-searchinthechain-2304-14732/paper.bib
possibly_outdated: false

authors:
  - Shicheng Xu
  - Liang Pang
  - Huawei Shen
  - Xueqi Cheng
  - Tat-Seng Chua
year: 2024
venue: WWW 2024
venue_type: conference
arxiv_id: "2304.14732"
doi: "10.1145/3589334.3645363"
url: http://arxiv.org/abs/2304.14732
citation_key: xu2024searchinthechain
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

Search-in-the-Chain (SearChain) is a retrieval-augmented reasoning framework for knowledge-intensive tasks that makes a large language model first plan a global Chain-of-Query (CoQ) and then lets information retrieval verify or complete each node in that chain. Each node contains an IR-oriented query, an answer, and optionally an unsolved flag, so retrieval intervenes only when the model is uncertain or inconsistent with evidence. The resulting interaction forms a Tree-of-Reasoning whose correct path is traced back to produce an answer with step-level supporting references. Across multi-hop QA, slot filling, fact checking, and long-form QA, SearChain outperforms strong CoT, tool-use, and retrieval baselines while reducing the rate at which retrieval misleads the model.

## Problem & Motivation

The paper targets complex knowledge-intensive tasks where LLMs need both multi-step reasoning and access to factual, often long-tail or time-sensitive knowledge. The authors argue that prior retrieval-augmented methods inject retrieval too locally, so the reasoning chain becomes fragmented, incorrect retrieved evidence can overwrite correct model knowledge, and the system cannot easily revise its reasoning direction once a branch goes wrong. SearChain is designed to keep reasoning globally coherent while using retrieval selectively for verification, missing-knowledge completion, and citation-style traceability.

## Method

- **Global reasoning chain**: SearChain prompts the LLM with in-context learning to generate a Chain-of-Query `CoQ = (q_1, a_1) -> (q_2, a_2) -> ... -> (q_n, a_n)`, where each node is a query-answer pair aligned to retrieval.
- **Uncertainty marking**: if the model cannot answer a sub-question, it emits `[Unsolved Query]`; this acts as an explicit missing-knowledge flag instead of forcing premature generation.
- **Tree-based interaction**: each newly generated CoQ is a branch in a Tree-of-Reasoning; after retrieval feedback, the LLM regenerates a new CoQ rooted at the corrected or completed node, yielding a node-identify depth-first search process rather than a fixed linear chain.
- **Retriever and reader loop**: for each node `(q_i, a_i)`, IR retrieves the Top-1 document `d_i`, then a Reader extracts answer span `g = d_i[s:e]` and confidence `f = H_[CLS] w_f`, with span boundaries `s = argmax(softmax(H w_s))` and `e = argmax(softmax(H w_e))`.
- **Verification**: for short-form tasks, SearChain checks whether `g` appears in `a_i`; for long-form tasks, it checks whether `ROUGE(a_i, d_i) > alpha`. If the answer is inconsistent and `f > theta`, retrieval sends corrective feedback and the LLM rewrites the chain.
- **Completion**: for an unsolved query `q_i*`, the system feeds back extracted answer `g*` and reference `d_i*` regardless of whether `f > theta`, because the LLM has already declared missing knowledge.
- **Tracing**: after interaction ends, SearChain traces the correct path in the Tree-of-Reasoning and asks the LLM to generate final content with references attached to the supporting document for each reasoning step.
- **Implementation details**: the LLM is `[[gpt-3-5-turbo]]`, the retriever is `[[colbertv2]]`, the maximum interaction rounds are `r_max = 5`, the ROUGE threshold is `alpha = 0.35`, and the confidence threshold is `theta = 1.5`.
- **Evaluation setup**: HotpotQA uses Wikipedia 2017 full-wiki as corpus; the other tasks use a large-scale Wikipedia passage collection, and all retrieval baselines are reproduced in the same setting.

## Key Results

- On multi-hop QA, SearChain reaches `56.91` on HotpotQA, `17.07` on MuSiQue, `46.27` on WikiMultiHopQA, and `76.95` on StrategyQA, outperforming DSP (`51.97`, `15.83`, `43.52`, `72.41`) and Tree-of-Thought w/ IR (`50.65`, `15.61`, `42.49`, `72.55`).
- On slot filling, fact checking, and long-form QA, SearChain scores `57.29` on zsRE, `65.07` on T-REx, `81.15` on FEVER, and `25.57` ROUGE-L on ELI5, again beating the strongest reported baselines.
- Ablations show both retrieval operations matter: removing verification drops HotpotQA from `56.91` to `46.11`, and removing completion yields `53.05`; similar degradations appear on zsRE (`57.29 -> 43.58` without verification) and T-REx (`65.07 -> 56.03` without completion).
- On difficult questions that actually need retrieval support, SearChain improves HotpotQA accuracy from `31.38` without IR to `60.86` with IR, and MuSiQue from `10.20` to `18.49`.
- The method also reduces retrieval-induced harm: the percentage of IR misleading the LLM falls to `6.33` on HotpotQA and `12.71` on WikiMultiHopQA, versus `15.76` and `25.76` for Self-Ask.
- Traceability analysis reports better citation behavior than New Bing, with Scope of Knowledge Coverage `2.882` vs `1.143` and Accuracy of Marking Position `0.80` vs `0.45`.
- Efficiency remains competitive: SearChain averages `2.21` interaction rounds and `8.52s` total runtime, versus DSP at `10.47s` and Tree-of-Thought w/ IR at `13.28s`, while achieving the best average performance (`53.29`).

## Limitations

- The retrieval side uses only the Top-1 document for each query, so verification and completion quality are tightly bounded by first-stage retrieval quality.
- The method depends on a trained reader plus manually selected thresholds `theta = 1.5` and `alpha = 0.35`; these hyperparameters may need re-tuning when the task mix, corpus, or model family changes.
- Experiments are centered on `[[gpt-3-5-turbo]]` and mostly Wikipedia-based corpora, so the paper does not establish how well the framework transfers to stronger LLMs, denser tool ecosystems, or non-Wikipedia knowledge bases.
- Traceability gains are shown through case studies and human evaluation against New Bing, but the paper does not provide a large-scale fully automatic citation benchmark for the generated references.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[chain-of-thought]]
- [[chain-of-query]]
- [[tree-of-reasoning]]
- [[question-decomposition]]
- [[answer-verification]]
- [[answer-completion]]
- [[multihop-question-answering]]
- [[citation-grounding]]
- [[traceability]]

## Entities Extracted

- [[shicheng-xu]]
- [[liang-pang]]
- [[huawei-shen]]
- [[xueqi-cheng]]
- [[tat-seng-chua]]
- [[chinese-academy-of-sciences]]
- [[national-university-of-singapore]]
- [[gpt-3-5-turbo]]
- [[colbertv2]]
- [[hotpotqa]]
- [[musique]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
