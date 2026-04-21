---
type: source
subtype: paper
title: Benchmarking Large Language Models in Retrieval-Augmented Generation
slug: chen-2023-benchmarking-2309-01431
date: 2026-04-20
language: en
tags: [retrieval-augmented-generation, llm-evaluation, benchmark, retrieval, multilingual]
processed: true

raw_file: raw/papers/chen-2023-benchmarking-2309-01431/paper.pdf
raw_md: raw/papers/chen-2023-benchmarking-2309-01431/paper.md
bibtex_file: raw/papers/chen-2023-benchmarking-2309-01431/paper.bib
possibly_outdated: true

authors:
  - Jiawei Chen
  - Hongyu Lin
  - Xianpei Han
  - Le Sun
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2309.01431
doi:
url: http://arxiv.org/abs/2309.01431
citation_key: chen2023benchmarking
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper introduces RGB, a bilingual benchmark for evaluating whether retrieval-augmented generation actually helps large language models under realistic failure modes rather than idealized QA settings. RGB isolates four core abilities: noise robustness, negative rejection, information integration, and counterfactual robustness. The benchmark is constructed from recent news to reduce contamination from model parametric knowledge, then pairs questions with retrieved web evidence produced by search and dense reranking. Across six representative LLMs, retrieval improves simple answer accuracy, but the models remain brittle: they often answer when evidence is missing, fail to combine evidence across multiple documents, and are easily steered by false retrieved content. The paper reframes RAG evaluation around reliability, not just answer correctness.

## Problem & Motivation

The paper argues that standard RAG evaluation overstates progress because it mostly measures answer accuracy on cases where retrieved evidence is assumed useful. In practice, retrieved web context can be noisy, incomplete, or factually wrong, and models may still hallucinate or over-trust it. The authors therefore target four concrete capabilities a reliable RAG system needs: extracting answers despite distractors, refusing to answer when evidence is absent, integrating evidence across multiple documents, and resisting incorrect retrieved facts. They also design RGB around recent news so that model-internal knowledge contributes less spurious advantage to the first three evaluations.

## Method

- **Benchmark scope**: RGB evaluates `4` RAG abilities on bilingual data: noise robustness, negative rejection, information integration, and counterfactual robustness. The corpus contains `n_base = 600` base questions, plus `n_info_extra = 200` information-integration examples and `n_cf_extra = 200` counterfactual examples; half the instances are English and half Chinese.
- **QA generation**: starting from recent news articles, ChatGPT is prompted to generate `(event, question, answer)` tuples. The authors manually verify answers and filter instances that are hard to retrieve with web search, aiming to reduce contamination from model pretraining knowledge.
- **Retrieval pipeline**: for each query, Google's search API fetches `n_pages = 10` web pages. Their text is chunked with `chunk_len_max = 300` tokens, then an English dense retriever (`all-mpnet-base-v2`) or Chinese dense retriever (`m3e-base`) selects the top `k = 30` matching chunks. Search snippets plus reranked chunks become candidate external documents, labeled as positive or negative depending on whether they contain the answer.
- **Noise robustness testbed**: each instance provides `n_docs = 5` retrieved documents, with noise ratios `r ∈ {0, 0.2, 0.4, 0.6, 0.8}` controlling how many negative documents are mixed with positive evidence.
- **Negative rejection testbed**: all `5` retrieved documents are negative documents; the model is instructed to emit a rejection string such as "insufficient information" rather than fabricate an answer.
- **Information integration testbed**: base questions are rewritten into multi-aspect questions whose answers require combining evidence from multiple retrieved documents, e.g. asking for two events or two winners instead of one.
- **Counterfactual robustness testbed**: the authors first generate questions that the model is likely to know internally, then manually corrupt retrieved documents by replacing answer spans with false facts. Models are warned that retrieved content may contain factual errors and must detect or correct them.
- **Models and evaluation**: RGB evaluates `6` LLMs: ChatGPT (`gpt-3.5-turbo` API), ChatGLM-6B, ChatGLM2-6B, Vicuna-7B-v1.3, Qwen-7B-Chat, and BELLE-7B-2M. Metrics are exact-match accuracy for noise robustness and information integration, rejection rate for negative rejection, and error detection / correction rates for counterfactual robustness. Because models often deviate from fixed trigger strings, ChatGPT-assisted judging is also used for rejection and error detection. Experiments run on an NVIDIA GeForce RTX `3090`.

## Key Results

- **Noise robustness is real but brittle**: ChatGPT drops from `96.33%` to `76.00%` accuracy in English as noise ratio rises from `0` to `0.8`; ChatGLM2-6B drops from `91.33%` to `57.33%`. In Chinese, ChatGPT drops from `95.67%` to `70.67%`.
- **Negative rejection is poor across all models**: the best exact rejection rates are only `31.00%` in English and `8.67%` in Chinese (both from Qwen-7B-Chat), while ChatGPT-assisted judging lifts the best rates only to `45.00%` in English and `43.33%` in Chinese.
- **Information integration is much harder than simple noisy QA**: even at noise ratio `0`, the highest accuracy is only `60%` in English (Vicuna-7B-v1.3) and `67%` in Chinese (Qwen-7B-Chat). At noise ratio `0.4`, the best scores fall to `43%` and `55%`, respectively.
- **Counterfactual retrieval severely misleads models**: for ChatGPT-en, direct-answer accuracy without documents is `89%`, but falls to `9%` with counterfactual documents; its error correction rate is only `57.14%`. For ChatGPT-zh, accuracy falls from `91%` to `17%`.
- **Failure modes are diagnostically categorized**: noise robustness errors concentrate in long-distance information, evidence uncertainty, and concept confusion; information integration adds merging errors (`28%` of ChatGLM2-6B integration errors), ignoring errors (`28%`), and misalignment errors (`6%`).

## Limitations

- RGB is built from recent news and short retrieved snippets, so it does not cover all knowledge-intensive settings such as long-form scientific synthesis or domain-specific enterprise retrieval.
- The retrieval setup is fixed to search-engine retrieval plus dense reranking with `5` provided documents, so the benchmark does not separate retriever quality from generator quality as cleanly as a fully modular evaluation would.
- The evaluated model set is small and centered on 2023-era chat models, with only one strong API model and several `6B` to `7B` open models.
- Negative rejection and error detection partly depend on fixed response patterns and ChatGPT-based adjudication, which introduces evaluator sensitivity.
- Counterfactual robustness is only evaluated on models whose direct-answer accuracy exceeds `70%`, so the counterfactual analysis covers a narrower subset of systems than the other three testbeds.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[benchmark]]
- [[dense-retrieval]]
- [[neural-reranking]]
- [[question-generation]]
- [[noise-robustness]]
- [[negative-rejection]]
- [[information-integration]]
- [[counterfactual-robustness]]

## Entities Extracted

- [[jiawei-chen]]
- [[hongyu-lin]]
- [[xianpei-han]]
- [[le-sun]]
- [[chatgpt]]
- [[chatglm-6b]]
- [[chatglm2-6b]]
- [[vicuna-7b-v1-3]]
- [[qwen-7b-chat]]
- [[belle-7b-2m]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
