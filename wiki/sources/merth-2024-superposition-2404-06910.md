---
type: source
subtype: paper
title: "Superposition Prompting: Improving and Accelerating Retrieval-Augmented Generation"
slug: merth-2024-superposition-2404-06910
date: 2026-04-20
language: en
tags: [rag, llm, prompting, inference-efficiency, question-answering]
processed: true

raw_file: raw/papers/merth-2024-superposition-2404-06910/paper.pdf
raw_md: raw/papers/merth-2024-superposition-2404-06910/paper.md
bibtex_file: raw/papers/merth-2024-superposition-2404-06910/paper.bib
possibly_outdated: false

authors:
  - Thomas Merth
  - Qichen Fu
  - Mohammad Rastegari
  - Mahyar Najibi
year: 2024
venue: arXiv preprint
venue_type: preprint
arxiv_id: 2404.06910
doi:
url: https://arxiv.org/abs/2404.06910
citation_key: merth2024superposition
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper proposes superposition prompting, a training-free retrieval-augmented generation method that represents the prompt as a directed acyclic graph instead of one concatenated token sequence. A shared preamble fans out into per-document query paths, the model scores each path with a Bayesian saliency criterion, prunes low-value paths, and then generates from the retained KV caches. This reorganizes attention so documents are processed independently, which exposes prompt-path caching and parallelization opportunities while shortening the maximum path length the transformer sees. Across OpenELM, BLOOMZ, and MPT models on NaturalQuestions-Open and MuSiQue, the method consistently improves QA accuracy under large retrieved contexts while dramatically reducing user-observed inference cost, with a headline result of `93.7x` theoretical speedup and `43%` relative accuracy gain over naive RAG on `mpt-7b-instruct`.

## Problem & Motivation

Naive LLM-based RAG concatenates the system prompt, all retrieved documents, and the query into one long sequence `x = p ⊕ d_1 ⊕ ... ⊕ d_n ⊕ q`. This makes inference expensive because transformer self-attention scales quadratically with sequence length, and it also worsens the distraction problem: irrelevant documents can degrade answer quality when the retrieved context is long. Existing long-context acceleration methods often require architecture changes, retraining, or fine-tuning, which is unattractive for already deployed pre-trained LLMs. The paper therefore asks whether prompt structure alone can reduce effective context length, enable lossless systems optimizations, and improve RAG quality without touching model weights.

## Method

- **Prompt as DAG**: replace the classical linear prompt with a directed acyclic graph where the preamble `p` branches into document paths and duplicated query paths `q_i`; a token sequence `v` can attend to `u` iff there is a path from `u` to `v`.
- **ForkJoin topology**: for `n_d` retrieved documents, process each `(d_i, q_i)` path independently conditioned on the shared preamble. The query is duplicated per path so the model can score documents independently before response generation.
- **Equilibrium position assignment**: assign document positions by spacing overlapping paths according to the harmonic mean `S(D) = n_d / Σ_{d in D} 1 / ||d||`, avoiding the discontinuities introduced by left-aligned padding and making the method compatible with ALiBi and interpolated RoPE positions.
- **Bayesian path pruning**: compute a saliency score `P(d_i | q_i, p) ∝ P(q_i | d_i, p) P(d_i | p)` from LM logits, normalize by sequence length, and greedily keep the top-`k` paths. The kept KV caches are concatenated before autoregressive decoding.
- **Path caching**: precompute the preamble KV cache and per-document KV caches offline because they do not depend on the query. This generalizes prompt caching from root-only reuse to path-prefix reuse; the paper notes memory cost scales with a model-dependent constant `c_model` per raw tokenized length, e.g. `492 KB` for `bloomz-7b1`.
- **Path parallelization**: compute independent query-path logits in parallel via batching or distributed inference. This mainly reduces wall-clock latency rather than total CPU time.
- **MuSiQue extension**: for multi-hop QA, apply iterative superposition for `t` rounds, each time pruning to top-`k` documents and prepending the retained evidence chain before the next iteration.
- **Evaluation setup**: no additional training or fine-tuning; models are `OpenELM-3B-Instruct`, `bloomz-3b`, `bloomz-7b1`, and `mpt-7b-instruct`, with greedy decoding and randomized document order.

## Key Results

- On NaturalQuestions-Open with `mpt-7b-instruct`, superposition reaches `0.465` accuracy versus naive RAG `0.026`, with `2.31e+11` compute cycles and `93.7x` theoretical speedup.
- On NaturalQuestions-Open with `bloomz-7b1`, it achieves `0.253` accuracy and `93.5x` speedup versus naive RAG `0.022`; with `bloomz-3b`, it reaches `0.223` at `98.3x` speedup.
- On MuSiQue, superposition is the highest-accuracy method for every tested model; for `mpt-7b-instruct`, the best reported setting reaches `F1 = 0.120` and `EM = 0.040` versus naive `0.064/0.008`.
- Equilibrium positioning improves over left alignment on every model; the largest gain is on `mpt-7b-instruct`, from `0.348` to `0.465`.
- Bayesian pruning strongly outperforms attention-based or no-pruning selection; on `mpt-7b-instruct`, accuracy rises from `0.218`/`0.224` to `0.465`.
- The method reduces the model-perceived maximum sequence length on NaturalQuestions-Open from an average of `2923` tokens to `206` tokens.

## Limitations

- The largest speedups are theoretical and assume auxiliary memory plus parallel compute; measured CUDA speedups are much smaller, typically around `5.6x` to `6.6x` for superposition on NaturalQuestions-Open.
- The evaluation scope is narrow: two QA benchmarks and a small set of open LLM families. The paper does not test summarization, dialogue, or non-RAG generation.
- MuSiQue requires iterative superposition, which reduces caching opportunities after the first retrieval step and lowers the achievable speedup relative to single-hop RAG.
- Performance depends on hyperparameters such as top-`k` and superposition factor `γ`; the paper shows accuracy is not monotonic and typically peaks around intermediate values.
- The method is presented as training-free, but the paper leaves open whether RoPE-based models would benefit from fine-tuning for continuous position interpolation and whether the approach generalizes beyond RAG.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[transformer]]
- [[prompt-engineering]]
- [[prompt-caching]]
- [[kv-cache]]
- [[path-pruning]]
- [[positional-encoding]]
- [[length-extrapolation]]
- [[long-context-inference]]
- [[open-domain-question-answering]]
- [[multihop-question-answering]]

## Entities Extracted

- [[thomas-merth]]
- [[qichen-fu]]
- [[mohammad-rastegari]]
- [[mahyar-najibi]]
- [[apple]]
- [[openelm]]
- [[bloomz]]
- [[mpt-7b-instruct]]
- [[naturalquestions-open]]
- [[musique]]
- [[contriever]]
- [[prompt-cache]]
- [[attention-sort]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
