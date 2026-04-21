---
type: source
subtype: paper
title: "Longtriever: a Pre-trained Long Text Encoder for Dense Document Retrieval"
slug: yang-2023-longtriever
date: 2026-04-20
language: en
tags: [dense-retrieval, long-document-retrieval, pretraining, transformer, emnlp]
processed: true

raw_file: raw/papers/yang-2023-longtriever/paper.pdf
raw_md: raw/papers/yang-2023-longtriever/paper.md
bibtex_file: raw/papers/yang-2023-longtriever/paper.bib
possibly_outdated: true

authors:
  - Junhan Yang
  - Zheng Liu
  - Chaozhuo Li
  - Guangzhong Sun
  - Xing Xie
year: 2023
venue: EMNLP 2023
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.223
url: https://aclanthology.org/2023.emnlp-main.223
citation_key: yang2023longtriever
paper_type: method
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

Longtriever targets dense retrieval over long documents, where standard PLM encoders either truncate aggressively or incur prohibitive `O(L^2 d)` attention cost. The model splits a document into blocks, then alternates an inter-block encoder over `[DOC]` and block `[CLS]` tokens with an intra-block encoder inside each block, yielding a tightly coupled hierarchical architecture that propagates both local and global semantics. It is further continuously pre-trained with standard MLM and a new Local Masked Autoencoder (LMAE) objective that reconstructs block tokens from global document state, local block state, and original token embeddings. On MS MARCO Document Ranking and TREC 2019 Doc, Longtriever outperforms strong sparse, dense, and efficient-transformer baselines while retaining competitive time and memory costs.

## Problem & Motivation

Long-document retrieval is difficult because vanilla Transformer encoders scale quadratically with sequence length, passage truncation discards evidence dispersed across the document, and prior efficient or hierarchical encoders often weaken cross-block interaction. The paper argues that long documents require both efficient computation and more comprehensive semantic integration across distant spans. It also identifies annotation scarcity as a sharper problem for long-document retrieval than passage retrieval, motivating an unsupervised pretraining phase specialized for document-level representations rather than relying only on supervised fine-tuning.

## Method

- **Blockwise input formulation**: tokenize the document with WordPiece into `X = {x_1, ..., x_L}`, split into `N = ceil(L / M)` blocks of size at most `M = 512`, prepend one global `[DOC]` token and one `[CLS]` token per block, and use hidden size `d = 768`.
- **Inter-block encoder**: build `\hat{B}^{(l)} = [s_d^{(l)}, c_1^{(l)}, ..., c_N^{(l)}] \in R^{(N+1) x d}` and apply multi-head self-attention `\widetilde{B}^{(l)} = softmax(Q_c^{(l)} K_c^{(l)\top} / \sqrt{d}) V_c^{(l)}` so each block summary exchanges information through the global document state.
- **Intra-block encoder**: dispatch the updated block summary back to block `i` with `\hat{H}^{(l)} = [\tilde{c}_i^{(l)}, h_1^{(l)}, ..., h_M^{(l)}] \in R^{(M+1) x d}` and compute `\widetilde{H}^{(l)} = softmax(Q_e^{(l)} K_e^{(l)\top} / \sqrt{d}) V_e^{(l)}` to inject cross-block context into token-level modeling.
- **Retrieval scoring and training**: use Longtriever as both query and document encoder in a bi-encoder setup, score pairs by dot product `rel_{q,d} = LT(q) LT(d)^T`, and optimize in-batch negative log-likelihood `L_r = \sum -log( e^{rel_{q,d^+}} / ( e^{rel_{q,d^+}} + \sum_i e^{rel_{q,d_i^-}} ) )`.
- **Complexity**: inter-block attention costs `O((N+1)^2 d)`, all intra-block encoders cost `O((M+1)^2 d N)`, so total complexity is `O(M^2 N d + N^2 d)`, compared with vanilla long-sequence attention `O(M^2 N^2 d)`.
- **Pretraining objectives**: continue pretraining from BERT with MLM and Local Masked Autoencoder. LMAE uses global `[DOC]` state `s_d^{(L')}`, local block state `\tilde{c}_i^{(L')}`, and original token embeddings to decode masked block tokens, optimizing `L_LMAE = \sum_{x_k \in X} CE(x_k | \psi(h_k^{Dec}))`.
- **Decoder construction for LMAE**: form block queries from repeated global state plus positional embeddings, keys/values from `[DOC]`, block `[CLS]`, and original token embeddings, then decode with a masked Transformer layer using a random mask matrix `A`.
- **Implementation hyperparameters**: `24` Transformer layers total (`12` intra-block + `12` inter-block), vocabulary size `30,522`, maximum `8` blocks per document, MLM masking ratio `0.3`, LMAE masking ratio `0.5`, AdamW with peak learning rate `1e-4`, warmup ratio `0.1`, linear decay, weight decay `0.01`.
- **Compute setup**: continuous pretraining starts from a BERT checkpoint on `8 x NVIDIA A100 40GB` GPUs for `8` epochs, batch size `3` per device, taking about `3` days.

## Key Results

- On **MS MARCO Dev Doc**, Longtriever reaches `MRR@100 = 0.434` and `R@100 = 0.940`, beating the strongest listed baselines ADORE (`0.405`, `0.919`) and COSTA (`0.422`, `0.919`).
- On **TREC 2019 Doc**, Longtriever achieves `NDCG@10 = 0.645` and `R@100 = 0.356`, outperforming ADORE (`0.628`, `0.317`) and other sparse/dense retrievers.
- Against other pre-trained long-document encoders, Longtriever records `0.329 / 0.893` on MS MARCO Dev Doc and `0.572 / 0.345` on TREC 2019 Doc, clearly ahead of Longformer (`0.291 / 0.859`, `0.520 / 0.280`) and BigBird (`0.293 / 0.859`, `0.544 / 0.281`).
- Efficiency trade-off is competitive: for a batch of `16` documents of length `2048`, Longtriever uses `696.10 ms` and `4.56 GiB`, close to Hi-Transformer (`674.77 ms`, `4.00 GiB`) and much lighter than BERT-Document (`1083.98 ms`, `8.42 GiB`).
- Ablations show large gains from pretraining and cross-block design: removing LMAE drops MS MARCO Dev Doc from `0.329` to `0.307` MRR@100, removing all continuous pretraining drops it to `0.280`, removing `[DOC]` lowers it to `0.272`, and removing the inter-block encoder further drops it to `0.249`.

## Limitations

The paper's own limitation section is narrow: it does not fully benchmark a `2048`-token BERT document encoder because of GPU memory constraints, so the empirical upper-bound comparison against dense full-attention BERT is incomplete. More broadly, Longtriever is only validated on two document retrieval benchmarks derived from MS MARCO/TREC, so generalization to other long-document domains is not established. Its hierarchical design still depends on fixed block partitioning and a maximum of `8` blocks, which may miss very long-range interactions beyond that window. The method also adds architectural complexity and pretraining cost relative to simpler passage-based baselines.

## Concepts Extracted

- [[dense-retrieval]]
- [[long-document-retrieval]]
- [[bi-encoder]]
- [[hierarchical-transformer]]
- [[inter-block-encoder]]
- [[intra-block-encoder]]
- [[local-masked-autoencoder]]
- [[masked-language-modeling]]
- [[pretraining]]
- [[in-batch-negative-sampling]]
- [[sparse-attention]]

## Entities Extracted

- [[junhan-yang]]
- [[zheng-liu-msra]]
- [[chaozhuo-li-msra]]
- [[guangzhong-sun]]
- [[xing-xie]]
- [[longtriever]]
- [[university-of-science-and-technology-of-china]]
- [[microsoft-research-asia]]
- [[ms-marco]]
- [[bookcorpus]]
- [[wikipedia]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
