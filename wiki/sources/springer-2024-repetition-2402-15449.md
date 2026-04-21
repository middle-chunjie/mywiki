---
type: source
subtype: paper
title: Repetition Improves Language Model Embeddings
slug: springer-2024-repetition-2402-15449
date: 2026-04-20
language: en
tags: [text-embeddings, llm, retrieval, mteb, contrastive-learning]
processed: true

raw_file: raw/papers/springer-2024-repetition-2402-15449/paper.pdf
raw_md: raw/papers/springer-2024-repetition-2402-15449/paper.md
bibtex_file: raw/papers/springer-2024-repetition-2402-15449/paper.bib
possibly_outdated: false

authors:
  - Jacob Mitchell Springer
  - Suhas Kotha
  - Daniel Fried
  - Graham Neubig
  - Aditi Raghunathan
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.15449
doi:
url: http://arxiv.org/abs/2402.15449
citation_key: springer2024repetition
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

The paper proposes **echo embeddings**, a way to turn decoder-only language models into stronger text embedders without changing the architecture or adding unsupervised pretraining. The core idea is to prompt the model to reconstruct the input and then pool hidden states from the repeated second occurrence, whose tokens can attend to the full original sequence despite causal masking. This directly targets a failure mode of classical causal embeddings, where early token states cannot encode later evidence. On MTEB with Mistral-7B-Instruct-v0.1, zero-shot echo embeddings reach `48.64` average versus `42.38` for classical embeddings, and even the compute-matched variant reaches `49.02`. After supervised contrastive fine-tuning, echo embeddings still lead with `64.68` average, slightly above both causal (`63.98`) and bidirectionally cast (`64.23`) classical baselines.

## Problem & Motivation

Classical embeddings from autoregressive language models are limited by causal attention: hidden states for early tokens cannot incorporate information from later tokens, so mean pooling dilutes late discriminative evidence and last-token pooling overweights the end of the sequence. Prior work therefore treated bidirectional attention or extra unsupervised training as necessary for high-quality embeddings. This paper argues that the real requirement is not architectural bidirectionality per se, but access to whole-sequence information at the positions being pooled. The motivation is to obtain strong embeddings from the same decoder-only LM used for generation, avoiding architectural conversion and preserving a unified model for retrieval, similarity, and generation.

## Method

- **Embedding formulation**: map text `x` to `phi(x) in R^d`; similarity is mainly cosine similarity `Sim(x, y) = <phi(x), phi(y)> / (||phi(x)|| ||phi(y)||)`. Mean pooling uses `phi_A(x) = (1 / |A|) sum_{t in A} phi_t(x)`, while last-token pooling uses `phi_{-1}(x)`.
- **Classical baseline**: feed the sentence once and pool hidden states over the original span. Because the model is causal, token state `phi_k(x)` cannot encode tokens `x_{k+1}, x_{k+2}, ...`, creating a systematic mismatch for embedding extraction.
- **Echo embeddings**: prompt the LM to repeat or rewrite the input and pool only over the second occurrence, e.g. `Rewrite the following paragraph: S. The rewritten paragraph: S`. Tokens in the second copy can attend to the full first copy, so their contextual states can encode bidirectional evidence without changing the attention mask.
- **Synthetic diagnosis**: the paper builds a toy retrieval dataset with two structures, `S1` (early discriminatory, late redundant) and `S2` (early redundant, late discriminatory), showing that classical mean and last-token pooling fail on mixed data while echo embeddings handle both structures.
- **Zero-shot setup**: main results use `mistralai/Mistral-7B-Instruct-v0.1`; additional backbones are LLaMA-2-7B and S-LLaMA-1.3B. Zero-shot echo embeddings use mean pooling; compute-matched echo halves the input length so total encoded tokens stay comparable to the classical baseline.
- **Evaluation**: primary benchmark is [[mteb]] with `56` English datasets across classification, clustering, pair classification, reranking, retrieval, STS, and summarization. A reduced `28`-dataset MTEB-MINI is used for faster ablations and bidirectionality studies.
- **Fine-tuning objective**: supervised training optimizes SimCSE-style contrastive loss with in-batch and mined hard negatives. The authors use GradCache with batch size `2048`, LoRA with `r = 16` and `alpha = 16`, temperature `tau = 1/50`, and learning rate `8e-4`.
- **Fine-tuning prompts and pooling**: query/document prompts repeat the input for echo mode and use a single occurrence for classical mode. Unlike zero-shot evaluation, fine-tuning works best with last-token pooling plus a trainable EOS embedding.
- **Comparison baseline**: a bidirectional classical variant replaces the causal mask with bidirectional attention and is trained under the same supervised setup, testing whether architecture conversion alone closes the gap.

## Key Results

- **Zero-shot MTEB average (56 datasets, Mistral-7B-Instruct)**: Echo `48.64`, Echo compute-matched `49.02`, Classical `42.38`, PromptEOL `43.69`, LLM2Vec-unsupervised `49.43`.
- **Zero-shot task-level gains**: retrieval improves from `18.26` to `20.82`; STS improves from `57.07` to `73.74`; reranking improves from `43.60` to `47.56`.
- **Pooling ablation**: zero-shot echo with mean pooling scores `48.64`, but last-token pooling drops to `31.55`; classical mean is `42.38`, classical last-token is `31.94`.
- **Bidirectionality study on MTEB-MINI**: for Mistral-7B, Echo `59.78` beats Classical+Uni `49.03` and Classical+Bi `58.24`; for LLaMA-2-7B, Echo `56.99` vs `47.27` and `43.03`; for S-LLaMA-1.3B, Echo `49.55` vs `40.14` and `35.77`.
- **Fine-tuned MTEB average (Mistral-7B backbone)**: Echo `64.68`, Echo FT+IT compute matched `64.66`, Classical `63.98`, Classical + bidirectional attention `64.23`.
- **Model scale in zero-shot mode**: S-LLaMA-1.3B rises from `34.31` to `40.45`; LLaMA-7B rises from `40.29` to `45.84`.
- **Inference budget trend**: zero-shot echo beats bidirectional classical embeddings once the budget exceeds roughly `64` tokens; in fine-tuning, the crossover is roughly `128` tokens.

## Limitations

- Echo embeddings require repeating the input, which roughly doubles sequence length and naive training/inference cost.
- Under very small inference budgets, the repetition overhead dominates; the paper reports bidirectional classical embeddings can be stronger below about `64` tokens in zero-shot mode and below about `128` tokens after fine-tuning.
- The empirical study is centered on English MTEB and a small set of decoder backbones, so multilingual and broader domain generalization are not established.
- The training-data description for supervised fine-tuning is high-level in the main text; reproducibility depends on appendix details and the exact public dataset mixture.

## Concepts Extracted

- [[text-embedding]]
- [[autoregressive-language-model]]
- [[bidirectional-attention]]
- [[causal-attention]]
- [[echo-embedding]]
- [[mean-pooling]]
- [[last-token-pooling]]
- [[zero-shot-embedding]]
- [[contrastive-learning]]
- [[lora]]

## Entities Extracted

- [[jacob-springer]]
- [[suhas-kotha]]
- [[daniel-fried]]
- [[graham-neubig]]
- [[aditi-raghunathan]]
- [[carnegie-mellon-university]]
- [[mistral-7b-instruct-v0-1]]
- [[mteb]]
- [[simcse]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
