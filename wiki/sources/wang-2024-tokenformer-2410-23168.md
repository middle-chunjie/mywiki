---
type: source
subtype: paper
title: "TokenFormer: Rethinking Transformer Scaling with Tokenized Model Parameters"
slug: wang-2024-tokenformer-2410-23168
date: 2026-04-20
language: en
tags: [llm, transformer, scaling, attention, architecture]
processed: true

raw_file: raw/papers/wang-2024-tokenformer-2410-23168/paper.pdf
raw_md: raw/papers/wang-2024-tokenformer-2410-23168/paper.md
bibtex_file: raw/papers/wang-2024-tokenformer-2410-23168/paper.bib
possibly_outdated: false

authors:
  - Haiyang Wang
  - Yue Fan
  - Muhammad Ferjad Naeem
  - Yongqin Xian
  - Jan Eric Lenssen
  - Liwei Wang
  - Federico Tombari
  - Bernt Schiele
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2410.23168
doi:
url: https://arxiv.org/abs/2410.23168
citation_key: wang2024tokenformer
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

TokenFormer reformulates a Transformer's parameter-heavy linear projections as attention over learned parameter tokens. Instead of scaling by widening hidden states and retraining from scratch, it appends new key-value parameter pairs to Pattention layers while keeping token-state dimensionality fixed. This decouples parameter growth from token-token interaction cost, supports parameter reuse from smaller checkpoints, and can preserve the old model's output distribution when new keys are zero-initialized. The paper scales a 124M model progressively to 354M, 757M, and 1.4B parameters, reporting language-model and vision performance that is competitive with dense Transformers trained from scratch while requiring substantially less cumulative training compute during scaling.

## Problem & Motivation

Standard Transformers use fixed linear projections for token-parameter interaction, so meaningful model growth usually changes core architectural dimensions and forces retraining from scratch. That makes parameter scaling much more expensive than data scaling and complicates reuse of already trained models. TokenFormer targets a native parameter-axis scaling mechanism: increase model capacity by adding parameter tokens, preserve learned distributions during continued training, and avoid the growing token-token cost that comes from widening hidden states for long-context language modeling.

## Method

- **Pattention**: replace a linear projection with `Pattention(X, K_P, V_P) = Θ(X K_P^T) V_P`, where `K_P in R^{n x d_1}` and `V_P in R^{n x d_2}` are learnable parameter tokens and `Θ` is `L2` normalization followed by `GeLU`, not exponential softmax.
- **Pre-norm layer**: TokenFormer keeps the Transformer residual structure, using `X_inter = X_in + MHA(LN(X_in))` and `X_out = X_inter + FFN(LN(X_inter))`.
- **All projections become token-parameter attention**: `Q = Pattention(X, K_P^Q, V_P^Q)`, `K = Pattention(X, K_P^K, V_P^K)`, `V = Pattention(X, K_P^V, V_P^V)`, then standard token-token attention `softmax(Q K^T / sqrt(d)) V`, followed by `O_att = Pattention(X_att, K_P^O, V_P^O)`.
- **FFN simplification**: the feed-forward block is also a single Pattention layer, `O_ffn = Pattention(X_ffn, K_P^ffn, V_P^ffn)`, so both token-token and token-parameter computation are expressed through attention mechanisms.
- **Configuration rule**: for GPT-2-like settings with `12` layers and hidden size `768`, the attention/output projection token counts mirror hidden size (`n_q = n_k = n_v = n_o = 768`) and FFN uses `n_ffn = 4d = 3072`; the parameter-reuse series keeps `12` layers, hidden size `768`, and `12` heads while increasing attention KV pairs `576 -> 2140 -> 4850 -> 8620` and FFN KV pairs `2304 -> 8560 -> 19400 -> 34480`.
- **Progressive scaling**: larger models concatenate new parameter tokens, `K_P^scale = [K_P^old, K_P^new]` and `V_P^scale = [V_P^old, V_P^new]`, so capacity grows without changing input/output feature dimensions.
- **Distribution-preserving expansion**: with new key parameters zero-initialized and new values randomly initialized, the paper shows the Pattention output can remain unchanged at expansion time, enabling continued training without sharply disturbing the old model.
- **Training details**: scaling experiments use OpenWebText, the GPT-NeoX-20B tokenizer, AdamW with `beta_1 = 0.9`, `beta_2 = 0.95`, learning rate `6e-4`, `2000` warmup steps, cosine decay, batch size `512`, sequence length `1024`, and no dropout.
- **Additional design choices**: the paper removes learnable layer-norm parameters to make normalization non-parametric, and uses the `GeLU + L2Norm` Pattention activation because it trains more stably than exponential softmax.

## Key Results

- **Progressive scaling efficiency**: a TokenFormer grown from `124M` to `1.4B` parameters with parameter reuse and `30B` tokens at the final scaling stage reaches validation perplexity `11.77`, versus `11.63` for a same-size Transformer trained from scratch on `300B` tokens and `13.34` for a Transformer trained from scratch on only `30B` tokens.
- **Best final reused model**: with `60B` tokens at the `1.4B` stage, TokenFormer reaches perplexity `11.60`, slightly better than the scratch Transformer's `11.63`.
- **Zero-shot LM competitiveness**: average zero-shot accuracy is `44.7` for TokenFormer-150M vs `40.6` for Pythia-160M, `52.0` for TokenFormer-450M vs `48.2` for Pythia-410M, `56.4` for TokenFormer-900M vs `51.9` for Pythia-1B, and `59.3` for TokenFormer-1.5B vs `55.2` for Pythia-1.3B.
- **Vision expressiveness**: on ImageNet-1K, TokenFormer-B/16 reaches `82.5` top-1 at `109M` parameters, compared with `82.3` for ViT-B/16 (MAE), and TokenFormer-L/16 reaches `83.1` at `407M`, compared with `82.6` for ViT-L/16 (MAE).
- **Ablations justify Pattention design**: replacing exponential + `L1` normalization with `GeLU + L2` normalization improves ImageNet top-1 from `79.6` to `82.5`; removing learnable layer-norm weights and biases changes top-1 only from `82.6` to `82.5`.

## Limitations

- TokenFormer reduces the need to widen hidden states, but token-parameter computation still scales linearly with the number of parameter tokens, so inference efficiency is not fully solved.
- The strongest scaling claim is about compute reuse and perplexity under matched or reduced token budgets; the paper does not show decisive superiority over the best alternative sequence models at larger scales.
- Long-context benefits are argued mainly through FLOPs analysis and scaling behavior rather than extensive end-task long-context benchmarks.
- Sparse routing, multimodal fusion, stronger interpretability analysis, and MoE-style computation are framed as future work, not demonstrated end-to-end systems in this paper.

## Concepts Extracted

- [[transformer]]
- [[attention-mechanism]]
- [[cross-attention]]
- [[model-scaling]]
- [[token-parameter-attention]]
- [[parameter-token]]
- [[autoregressive-language-model]]
- [[long-context-modeling]]
- [[parameter-efficient-fine-tuning]]
- [[mixture-of-experts]]

## Entities Extracted

- [[haiyang-wang]]
- [[yue-fan]]
- [[muhammad-ferjad-naeem]]
- [[yongqin-xian]]
- [[jan-eric-lenssen]]
- [[liwei-wang]]
- [[federico-tombari]]
- [[bernt-schiele]]
- [[max-planck-institute-for-informatics]]
- [[google]]
- [[peking-university]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
