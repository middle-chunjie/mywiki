---
type: source
subtype: paper
title: "LLM2Vec: Large Language Models Are Secretly Powerful Text Encoders"
slug: behnamghader-2024-llmvec-2404-05961
date: 2026-04-20
language: en
tags: [llm, embeddings, text-encoders, contrastive-learning, mteb]
processed: true

raw_file: raw/papers/behnamghader-2024-llmvec-2404-05961/paper.pdf
raw_md: raw/papers/behnamghader-2024-llmvec-2404-05961/paper.md
bibtex_file: raw/papers/behnamghader-2024-llmvec-2404-05961/paper.bib
possibly_outdated: false

authors:
  - Parishad BehnamGhader
  - Vaibhav Adlakha
  - Marius Mosbach
  - Dzmitry Bahdanau
  - Nicolas Chapados
  - Siva Reddy
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2404.05961
doi:
url: https://arxiv.org/abs/2404.05961
citation_key: behnamghader2024llmvec
paper_type: method

read_status: unread

domain: llm
---

## Summary

LLM2Vec proposes a simple recipe for converting decoder-only LLMs into strong universal text encoders without labeled pairs or synthetic GPT-4 data. The method has three stages: replace causal masking with bidirectional attention, adapt the model with masked next token prediction (MNTP), and then learn sentence representations with unsupervised SimCSE plus mean pooling. The paper applies this recipe to S-LLaMA-1.3B, Llama-2-7B, Mistral-7B, and Meta-Llama-3-8B, showing that the transformed models become much better at both token-level and sequence-level embedding tasks. The strongest unsupervised system, LLM2Vec on Mistral-7B, reaches `56.80` average on full MTEB, while supervised variants set a new public-data state of the art at the time. The paper also analyzes why Mistral unexpectedly tolerates bidirectional attention even before adaptation.

## Problem & Motivation

Decoder-only large language models are strong general NLP systems, but their causal attention mask is poorly matched to embedding tasks because token representations cannot directly absorb right-context information. That hurts both contextualized word representations and pooled sequence embeddings. The paper argues that this mismatch, rather than an inherent weakness of decoder-only models, is a key reason the community still preferred encoder-style models for text embeddings. If the architectural constraint can be removed and the model lightly adapted, one can reuse the data efficiency, tooling ecosystem, and instruction-following ability of modern LLMs for universal embedding.

## Method

- **Step 1: bidirectional attention.** Replace the causal mask with an all-ones mask, effectively changing self-attention from `M_{j <= i}` to `M = 1_{N x N}` so every token can attend to both past and future tokens.
- **Step 2: masked next token prediction (MNTP).** For an input sequence `x = (x_1, ..., x_N)`, mask a subset of tokens and predict masked token `x_i` from the representation at position `i - 1`, not position `i`, so the objective stays aligned with decoder-style pretraining.
- **Masking choices.** The paper searches masking probabilities `{20%, 40%, 60%, 80%, 90%}` and BERT-style vs. RoBERTa-style masking. Best settings are BERT-style `20%` masking for S-LLaMA-1.3B, LLaMA-2-7B, and Meta-Llama-3-8B, and RoBERTa-style `80%` masking for Mistral-7B on sentence-level tasks.
- **Step 3: unsupervised contrastive learning.** Apply SimCSE by encoding the same sentence twice with different dropout masks, treating the two views as positives and the rest of the batch as in-batch negatives. Sequence embeddings are built with mean pooling, which beats EOS and weighted-mean pooling for LLM2Vec.
- **Parameter-efficient tuning.** Both MNTP and SimCSE use LoRA with `r = 16` and `alpha = 32` for `1000` steps. MNTP uses batch size `32`; SimCSE uses batch size `32` for S-LLaMA-1.3B and `128` for the larger models.
- **Training data.** MNTP uses `Wikitext-103`; SimCSE uses a Wikipedia sentence subset released by the SimCSE authors. Sequence-level evaluation uses task instructions on MTEB, with instruction tokens excluded from mean pooling.
- **Compute profile.** For `7B`/`8B` models, MNTP takes about `100` minutes on one `80GB` A100. SimCSE takes about `3` hours on one `80GB` A100. Larger runs use `bfloat16`, FlashAttention-2, and gradient checkpointing.
- **Analysis protocol.** To measure future-token usage, the paper evaluates `35` sentence triples and checks whether prefix-pooled embeddings assign higher cosine similarity to semantically matched continuations than mismatched ones.

## Key Results

- On full unsupervised MTEB, LLM2Vec reaches `49.42` with S-LLaMA-1.3B, `55.36` with LLaMA-2-7B, `56.80` with Mistral-7B, and `56.23` with Meta-Llama-3-8B; the `56.80` Mistral result is the paper's unsupervised state of the art.
- Relative to the best causal baseline on the 15-task MTEB subset, adding the full recipe improves scores by `49.8%` for S-LLaMA-1.3B, `23.2%` for LLaMA-2-7B, and `37.5%` for Mistral-7B.
- Word-level probing also improves after MNTP: S-LLaMA-1.3B chunking goes from `86.10` to `90.51`, LLaMA-2-7B NER from `96.59` to `97.16`, and Mistral-7B POS from `90.86` to `92.35`.
- Mistral is a special case: simply enabling bidirectional attention already raises its MTEB average from `42.46` to `46.86`, unlike S-LLaMA-1.3B and LLaMA-2-7B where naive bidirectionality hurts.
- In supervised training on public data, Meta-Llama-3-8B plus LLM2Vec (without SimCSE) reaches `65.01` average on MTEB, which the paper reports as the best public-data result at the time.
- LLM2Vec is more efficient than Echo embeddings at inference-time evaluation: on Mistral-7B, full MTEB evaluation is about `44` hours for LLM2Vec versus `64` hours for Echo on `8 x 80GB` A100 GPUs.

## Limitations

- The approach still inherits the latency and memory cost of large decoder-only models; `4096`-dimensional embeddings from 7B-scale models are much heavier than typical encoder embeddings.
- Results are only established on English corpora and benchmarks, so multilingual transfer is left open.
- Benchmark contamination cannot be ruled out because the full pretraining mixtures of LLaMA-2 and Mistral are not public.
- The Mistral explanation is only a hypothesis: the paper infers some form of bidirectional or prefix-style pretraining from representation behavior, but does not verify it directly.
- SimCSE is helpful for sequence embeddings but can slightly hurt supervised end performance relative to MNTP-only initialization on some backbones.

## Concepts Extracted

- [[llm2vec]]
- [[decoder-only-language-model]]
- [[bidirectional-attention]]
- [[masked-next-token-prediction]]
- [[simcse]]
- [[text-embedding]]
- [[mean-pooling]]
- [[low-rank-adaptation]]
- [[large-language-model]]
- [[instruction-tuning]]

## Entities Extracted

- [[parishad-behnamghader]]
- [[vaibhav-adlakha]]
- [[marius-mosbach]]
- [[dzmitry-bahdanau]]
- [[nicolas-chapados]]
- [[siva-reddy]]
- [[mcgill-university]]
- [[mila]]
- [[servicenow-research]]
- [[mteb]]
- [[conll-2003]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
