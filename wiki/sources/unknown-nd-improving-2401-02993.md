---
type: source
subtype: paper
title: Improving Natural Language Understanding with Computation-Efficient Retrieval Representation Fusion
slug: unknown-nd-improving-2401-02993
date: 2026-04-20
language: en
tags: [retrieval, nlu, prompt-learning, few-shot-learning, architecture-search]
processed: true
raw_file: raw/papers/unknown-nd-improving-2401-02993/paper.pdf
raw_md: raw/papers/unknown-nd-improving-2401-02993/paper.md
bibtex_file: raw/papers/unknown-nd-improving-2401-02993/paper.bib
possibly_outdated: false
authors:
  - Shangyu Wu
  - Ying Xiong
  - Yufei Cui
  - Xue Liu
  - Buzhou Tang
  - Tei-Wei Kuo
  - Chun Jason Xue
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2401.02993
doi:
url: https://arxiv.org/abs/2401.02993
citation_key: unknownndimproving
paper_type: method
read_status: unread
domain: nlp
---

## Summary

This paper proposes ReFusion, a retrieval-augmented natural language understanding framework that avoids concatenating retrieved texts into the prompt and instead injects retrieved sentence representations directly into transformer hidden states. The system combines an online dense retriever, a retrieval fusion module with a learnable reranker and an ordered-mask mechanism, and a DARTS-style neural architecture search procedure that decides where retrieval fusion should be enabled or disabled across layers. In prompt-based few-shot experiments on 15 non-knowledge-intensive tasks using RoBERTa-large, ReFusion improves both accuracy and robustness over LM-BFF, DART, KPT, and a retrieval-context concatenation baseline, while keeping computation much closer to plain representation fusion than long-context augmentation.

## Problem & Motivation

Retrieval augmentation works naturally for knowledge-intensive tasks, but prior prompt-based approaches for non-knowledge-intensive tasks mostly concatenate retrieved sentences with the original input. That design increases sequence length, quickly hits the model's maximum context window, and drives up FLOPs while still leaving retrieval relevance under-optimized. The paper asks whether retrieval can help NLU tasks such as sentiment classification and entailment without paying the cost of long-context prompting. Its answer is to fuse retrieved sentence embeddings directly into model internals and to learn which fusion scheme and which layers are actually useful.

## Method

- **Task setting**: prompt-based few-shot classification on single-sentence and sentence-pair tasks, optimizing `-log p([MASK] = y_w | X_prompt)` with verbalized labels instead of a standard classifier head.
- **Online retrieval module**: encode each input into a query representation `h_x`, retrieve top-`k` similar sentences `Z = {z_1, ..., z_k}` from an external dense index, and return sentence representations `H_Z = {h_{z_1}, ..., h_{z_k}}`; the implementation uses an in-memory cache to reduce repeated retrieval cost during training.
- **Retriever back-end**: assumes a task-agnostic dense vector store built offline and queried online, with FAISS or ScaNN-like ANN infrastructure and a compressed key-value store for texts plus embeddings.
- **Reranker fusion scheme**: learn a `k`-dimensional weight vector `R = {r_1, ..., r_k}`, normalize it with `r_i = exp(r_i) / sum_j exp(r_j)`, then inject the weighted mean of retrievals into the sentence representation as ``h_y_[CLS] = h_x_[CLS] + (1 / k) * sum_i r_i * h_{z_i}``.
- **Ordered-mask fusion scheme**: treat each retrieval dimension as an ordered dropout process over the ranked retrieval list, sample masks with a Gumbel-Softmax relaxation `c^d ~ Gumbel(beta, tau)`, derive `v_i^d = 1 - cumsum_i(c^d)`, form masked retrieval units ``hat(h)_{z_i}^d = v_i^d * h_{z_i}^d``, and fuse them with the same mean-addition rule.
- **Architecture search module**: for each searchable module, mix the original module, reranker fusion, and ordered-mask fusion through soft architectural weights ``hat(o)(h) = sum_i softmax(alpha_i) * o_i(h)``; optimize model weights `omega` on training loss and architecture weights `alpha` on validation loss in alternating steps, then keep the strongest candidate at inference.
- **Search space**: when replacing key/value linear modules in a transformer with `N` layers, the paper notes at least `9^N` candidate retrieval-augmented architectures; for RoBERTa-large with 24 layers, that becomes a septillion-scale discrete space before DARTS relaxation.
- **Training configuration**: experiments use RoBERTa-large, `k = 64` retrievals, `lr = 1e-5`, batch size `32`, max sequence length `128`, max steps `1000`, `16` examples per class, AdamW, one NVIDIA V100 GPU, and five few-shot seeds `S_seed = {13, 21, 42, 87, 100}`.
- **Analysis variants**: the paper compares querying the retriever with static input-text encodings versus dynamic hidden states; static text queries are much faster and are favored for online settings.

## Key Results

- On 15 tasks overall, ReFusion reaches `74.3` average accuracy versus `71.8` for LM-BFF and `69.5` for the retrieval-context concatenation baseline `CA-512`.
- On single-sentence tasks, ReFusion scores `75.5` average versus `73.4` for LM-BFF and `70.7` for `CA-512`; on TREC it improves from `84.8` to `90.3`.
- On sentence-pair tasks, ReFusion scores `72.9` average versus `69.9` for LM-BFF and `68.1` for `CA-512`; on QNLI it improves from `64.5` to `73.0`, and on SNLI from `77.2` to `80.6`.
- ReFusion attains state-of-the-art results on `5 / 8` single-sentence tasks and `5 / 7` sentence-pair tasks in the reported comparison.
- Ablations on six representative tasks show NAS matters: on QNLI, plain reranker reaches `68.8`, NAS with reranker reaches `73.5`, NAS with ordered mask reaches `73.0`, and full ReFusion reaches `73.0`.
- Querying with input-text embeddings is dramatically faster than querying with hidden states: e.g. on MPQA the per-iteration time is `3.7s` versus `108.2s`, while the hidden-state query only improves accuracy from `87.9` to `88.7`.

## Limitations

- The method is evaluated only in prompt-based NLU settings, mainly classification and entailment benchmarks; it does not test broader generation or structured prediction workloads.
- Retrieval quality is still bottlenecked by a task-agnostic dense index, so irrelevant neighbors can leak noise before the fusion module corrects them.
- NAS improves performance but adds a nontrivial search procedure and validation split dependency on top of already expensive large-model fine-tuning.
- The paper reports comparison against contemporaneous prompt and retrieval baselines, but not against newer retrieval adapters or long-context architectures that emerged later.
- The hidden-state query variant is too slow for practical online use, suggesting that stronger adaptive retrieval comes with a large systems cost.

## Concepts Extracted

- [[transformer]]
- [[dense-retrieval]]
- [[prompt-based-fine-tuning]]
- [[few-shot-learning]]
- [[representation-fusion]]
- [[reranking]]
- [[ordered-mask]]
- [[neural-architecture-search]]
- [[gumbel-softmax]]
- [[text-classification]]
- [[natural-language-inference]]

## Entities Extracted

- [[shangyu-wu]]
- [[ying-xiong]]
- [[yufei-cui]]
- [[xue-liu]]
- [[buzhou-tang]]
- [[tei-wei-kuo]]
- [[chun-jason-xue]]
- [[mila]]
- [[mcgill-university]]
- [[faiss]]
- [[scann]]
- [[glue]]
- [[snli]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
