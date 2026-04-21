---
type: source
subtype: paper
title: "INSTRUCTRAG: Instructing Retrieval-Augmented Generation via Self-Synthesized Rationales"
slug: unknown-nd-instructrag-2406-13629
date: 2026-04-20
language: en
tags: [rag, denoising, rationales, retrieval, qa]
processed: true

raw_file: raw/papers/unknown-nd-instructrag-2406-13629/paper.pdf
raw_md: raw/papers/unknown-nd-instructrag-2406-13629/paper.md
bibtex_file: raw/papers/unknown-nd-instructrag-2406-13629/paper.bib
possibly_outdated: false

authors:
  - Zhepei Wei
  - Wei-Lin Chen
  - Yu Meng
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2406.13629
doi:
url: https://arxiv.org/pdf/2406.13629
citation_key: unknownndinstructrag
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

InstructRAG is a retrieval-augmented generation framework that converts denoising from an implicit side effect into an explicit rationale-generation task. Given question-answer training pairs and noisy retrieved documents, an instruction-tuned language model synthesizes rationales explaining which passages support the answer and how the answer should be derived. Those rationales are then reused either as question-rationale demonstrations for in-context learning or as supervision for fine-tuning, so the model learns to denoise retrieval explicitly rather than only memorizing final answers. Across five knowledge-intensive QA benchmarks, the method improves answer accuracy, scales with stronger rationale generators, and remains robust when more distractor documents are retrieved. The paper's central claim is that self-synthesized rationales provide cheap yet effective supervision for making RAG more accurate, interpretable, and trustworthy.

## Problem & Motivation

Standard RAG pipelines usually ask the model to predict the final answer directly from retrieved context even when that context contains irrelevant, misleading, or contradictory passages. This makes denoising implicit, hard to verify, and increasingly brittle as the number of retrieved documents grows. Existing fixes often depend on stronger retrievers, adaptive retrieval, or extra supervision from expensive proprietary models. InstructRAG targets a different bottleneck: it asks whether an instruction-tuned model can synthesize explicit denoising rationales from ordinary QA training data, then use those rationales to teach a generator how to separate useful evidence from retrieval noise without requiring additional human annotations.

## Method

- **Problem setup**: given training data ``\mathcal{T} = {\langle q, a \rangle}`` and retrieved documents ``D = {d_1, \ldots, d_K}``, the generator answers with access to both retrieval and parametric knowledge, modeled as ``p_\theta(a \mid q, D)``. The paper keeps retrieval simple: no filtering or reranking is added before the model reads the retrieved documents.
- **Rationale generation**: an instruction-tuned LM ``\mathcal{M}_\phi`` receives question ``q``, ground-truth answer ``a``, and retrieved documents ``D`` and produces a rationale ``r`` that identifies supportive versus noisy evidence. The resulting rationale-augmented dataset is ``\mathcal{T}^{+} = {\langle q, r \rangle}``. A substring-based sanity check reports `98%` rationale-answer consistency on samples with at least one answer-bearing document.
- **InstructRAG-ICL**: for inference, the model samples `N` rationale demonstrations ``{\langle q_i, r_i \rangle}`` from ``\mathcal{T}^{+}`` and conditions on those exemplars plus the test retrieval context. To save memory, the demonstrations contain only questions and rationales; the default inference setting uses `N = 2`.
- **InstructRAG-FT**: the trainable variant fine-tunes a rationale learner ``\mathcal{M}_\theta`` with the objective ``\max_\theta \mathbb{E}_{(q,r) \sim \mathcal{T}^{+}} \log p_\theta(r \mid q, D)``. Training and inference use the same data format: retrieved documents first, then the question, then rationale generation.
- **Retrieval/evaluation setup**: the study uses Wikipedia as the corpus, `top-K = 5` retrieved documents for PopQA, TriviaQA, Natural Questions, and ASQA, and `K = 10` for 2WikiMultiHopQA. Retrievers are Contriever for PopQA/TriviaQA, DPR for Natural Questions, GTR for ASQA, and BM25 for 2WikiMultiHopQA.
- **Implementation details**: by default, the rationale generator and rationale learner share the same backbone family, instantiated with Llama-3-Instruct `8B` or `70B`. Fine-tuning uses `4 x H100 80GB`, FSDP, FlashAttention, `bf16`, `2` epochs, batch size `128`, learning rate `2.5e-5`, cosine schedule with `3%` warmup, and maximum sequence length `4096`.

## Key Results

- Across five knowledge-intensive benchmarks, InstructRAG reports an average relative improvement of `8.3%` over the best baseline.
- In the training-free setting, InstructRAG-ICL with Llama-3-Instruct `8B` reaches `64.2` on PopQA, `76.8` on TriviaQA, `62.1` on Natural Questions, `50.4` on 2WikiMultiHopQA, and `44.7` EM on ASQA; the `70B` variant further improves to `81.2` on TriviaQA, `66.5` on Natural Questions, and `47.8` EM on ASQA.
- In the trainable setting, InstructRAG-FT with Llama-3-Instruct `8B` scores `66.2` on PopQA, `78.5` on TriviaQA, `65.7` on Natural Questions, `57.2` on 2WikiMultiHopQA, and `47.6` EM / `65.7` precision / `70.5` recall on ASQA, outperforming vanilla SFT on every main task metric.
- Ablations show that removing the ground-truth answer during rationale synthesis drops PopQA / ASQA trainable performance from `66.2` / `47.6` to `65.2` / `46.4`, while removing retrieved documents drops them further to `64.5` / `45.2`.
- Replacing the LM rationale generator with a simple template hurts performance substantially: PopQA trainable accuracy falls from `66.2` to `59.6`, and PopQA training-free accuracy falls from `64.2` to `60.0`.
- Stronger rationale generators help: using Llama-3-Instruct `70B` instead of `8B` raises trainable PopQA from `66.2` to `67.0` and ASQA from `47.6` to `49.1`.
- Under GPT-4o judging on Natural Questions, InstructRAG-ICL reaches `67.6` versus `64.5` for in-context RALM, and InstructRAG-FT reaches `69.7` versus `65.1` for vanilla SFT.

## Limitations

- The empirical study is centered on QA-style knowledge-intensive tasks, so generalization to broader open-ended generation remains uncertain.
- Synthetic rationales are less reliable when retrieval fails completely: the paper reports `98%` rationale-answer consistency on answer-bearing samples but only `89%` overall.
- Standard exact-match or accuracy metrics remain biased toward lexical overlap and longer outputs; the paper therefore treats LLM-as-a-judge as a useful but not definitive complement.
- The method does not add an explicit filtering mechanism before rationale generation, leaving room for future integration with long-context settings, active retrieval, or retrieval-quality control.
- The authors note possible sample bias in training data and do not fully address fairness or bias-mitigation issues.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[instruction-tuning]]
- [[in-context-learning]]
- [[supervised-fine-tuning]]
- [[denoising]]
- [[rationale-generation]]
- [[noise-robustness]]
- [[parametric-knowledge]]
- [[llm-as-a-judge]]
- [[hallucination]]
- [[chain-of-thought]]

## Entities Extracted

- [[zhepei-wei]]
- [[wei-lin-chen]]
- [[yu-meng]]
- [[university-of-virginia]]
- [[popqa]]
- [[triviaqa]]
- [[natural-questions]]
- [[asqa]]
- [[2wiki-multihopqa]]
- [[contriever]]
- [[bm25]]
- [[gtr]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
