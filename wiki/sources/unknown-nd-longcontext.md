---
type: source
subtype: paper
title: "Long-Context LLMs Meet RAG: Overcoming Challenges for Long Inputs in RAG"
slug: unknown-nd-longcontext
date: 2026-04-20
language: en
tags: [rag, llm, long-context, retrieval, fine-tuning]
processed: true

raw_file: raw/papers/unknown-nd-longcontext/paper.pdf
raw_md: raw/papers/unknown-nd-longcontext/paper.md
bibtex_file: raw/papers/unknown-nd-longcontext/paper.bib
possibly_outdated: false

authors:
  - Bowen Jin
  - Jinsung Yoon
  - Jiawei Han
  - Sercan O. Arik
year: 2024
venue: OpenReview
venue_type: preprint
arxiv_id:
doi:
url: https://openreview.net/pdf?id=oU3tpaR8fm
citation_key: unknownndlongcontext
paper_type: method

read_status: unread

domain: llm
---

## Summary

This paper studies retrieval-augmented generation with long-context LLMs and shows that adding more retrieved passages is not monotonically helpful, even when stronger retrievers improve recall. On Natural Questions and PopQA, performance follows an inverted-U curve because additional passages introduce hard negatives that distract generation; this degradation can be worse with stronger dense retrievers such as e5 than with BM25. To address the problem, the paper proposes a training-free retrieval reordering strategy that exploits [[lost-in-the-middle]], plus two training-based methods: implicit RAG fine-tuning on noisy retrieved contexts and explicit fine-tuning with intermediate reasoning. Across multiple QA, multi-hop, long-form, and slot-filling benchmarks, these methods improve robustness and generalization for long-context RAG.

## Problem & Motivation

Long-context LLMs make it feasible to pack many retrieved passages into a single RAG prompt, so one might expect higher retrieval recall to translate into better downstream accuracy. The paper shows this intuition is incomplete: once retrieval sets grow, irrelevant but semantically plausible hard negatives begin to dominate the end-to-end failure mode. The core motivation is therefore to understand why more context can hurt long-context RAG, characterize the interaction between retriever strength and generator robustness, and design both inference-time and training-time remedies that let long-context models benefit from larger retrieved context windows instead of being misled by them.

## Method

- **Failure analysis setup**: evaluate long-context RAG on NQ and PopQA with retrievers `BM25` and `e5`, varying the number of retrieved passages up to the model context limit for Gemma, Mistral-Nemo, and Gemini generators.
- **Recall vs. accuracy diagnosis**: compare retrieval recall/precision against final answer accuracy to isolate cases where relevant passages are present but the generator still fails because of distracting hard negatives.
- **Hard-negative stress test**: construct controlled prompts with `1` golden passage plus varying numbers of hard negatives from `e5`, Contriever, BM25, or random sampling to measure robustness independently of recall.
- **Retrieval reordering**: replace the default prompt order `[I, d_1, d_2, ..., d_k, q]` with an edge-biased arrangement that places high-score passages near both ends. The placement rule is ``Order(d_i) = (i + 1) / 2`` for odd `i` and ``Order(d_i) = (k + 1) - i / 2`` for even `i`.
- **Implicit RAG fine-tuning**: train the generator on noisy retrieved context with the mapping ``[I, d_1, ..., d_k, q] -> a`` so the model implicitly learns to ignore irrelevant passages while extracting answer-bearing evidence.
- **Explicit reasoning-augmented fine-tuning**: augment supervision to ``[I, d_1, ..., d_k, q] -> [r, a]``, where `r` is an intermediate reasoning paragraph identifying useful documents before producing the final answer.
- **Training data mixture**: use `50k` samples from NQ, Wizard of Wikipedia, FEVER, and MMLU (`12.5k` each), with top-`40` retrieved chunks from 2018 Wikipedia (`21M` chunks total) for the main tuning setup.
- **Hyperparameters**: fine-tune Gemma-2-9B-Base and Mistral-Nemo-12B-Base for `4` epochs with peak learning rate `1e-6`, cosine schedule, `5%` warmup, batch size `64`, and up to `40` retrieved passages; Gemma uses `8192` input tokens, roughly `40` passages.
- **Reasoning labels**: generate intermediate reasoning supervision with Gemini-1.5-Pro so the model explicitly learns relevance identification rather than only answer generation.

## Key Results

- On NQ, `Recall@40` is `0.90` for `e5` versus `0.73` for `BM25`, yet stronger retrieval leads to sharper downstream degradation as more passages are added.
- On PopQA, `Recall@40` is `0.85` for `e5` versus `0.57` for `BM25`, and the same inverted-U pattern reappears for long-context RAG.
- In the hard-negative stress test, accuracy drops consistently as more hard negatives are added; negatives from stronger retrievers are harder than those from weaker retrievers or random sampling.
- Implicit RAG fine-tuning outperforms both chat-model-with-RAG baselines and direct QA fine-tuning across unseen evaluation sets including TriviaQA, HotpotQA, 2WikiMultiHopQA, ASQA, T-REx, and zsRE.
- Explicit intermediate reasoning improves further over implicit RAG fine-tuning on sampled evaluations, showing that relevance supervision adds value beyond robustness acquired implicitly.
- Mixed training data (`4` sources, `50k` examples) generalizes better than single-source tuning, and mixed-retriever training generalizes better to unseen retrievers than BM25-only or e5-only tuning.
- Training with the maximum retrieval budget (`40` passages, `100%` of available context) yields the strongest performance across different inference-time retrieval sizes.
- Scaling RAG-specific training data from `5k` to `200k` examples improves average NQ accuracy from `0.5805` to `0.6176`.

## Limitations

- The paper relies on a sparse bibliographic record in `paper.bib`; the manuscript text is detailed, but formal publication metadata is incomplete.
- Main empirical analysis centers on QA-style benchmarks, so conclusions may not directly transfer to other retrieval-heavy tasks such as code generation or tool-augmented agents.
- The retrieval reordering method is simple and effective, but it is still heuristic position engineering rather than a learned optimal ordering policy.
- Intermediate-reasoning inference is more expensive, and the paper reports reasoning-augmented results on `1000` sampled examples because of decoding cost.
- The work studies robustness to retrieved hard negatives, but it does not jointly optimize the retriever and generator in a fully end-to-end manner.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[long-context-language-model]]
- [[context-window]]
- [[hard-negative]]
- [[lost-in-the-middle]]
- [[retrieval-reordering]]
- [[supervised-fine-tuning]]
- [[chain-of-thought]]

## Entities Extracted

- [[bowen-jin]]
- [[jinsung-yoon]]
- [[jiawei-han]]
- [[sercan-o-arik]]
- [[google-cloud-ai-research]]
- [[university-of-illinois-urbana-champaign]]
- [[natural-questions]]
- [[popqa]]
- [[bm25]]
- [[e5]]
- [[gemini-1-5-pro]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
