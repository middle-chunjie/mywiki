---
type: source
subtype: paper
title: "LightRetriever: A LLM-based Text Retrieval Architecture with Extremely Faster Query Inference"
slug: ma-2026-lightretriever-2505-12260
date: 2026-04-20
language: en
tags: [retrieval, llm, efficiency, hybrid-retrieval, dense-retrieval]
processed: true

raw_file: raw/papers/ma-2026-lightretriever-2505-12260/paper.pdf
raw_md: raw/papers/ma-2026-lightretriever-2505-12260/paper.md
bibtex_file: raw/papers/ma-2026-lightretriever-2505-12260/paper.bib
possibly_outdated: false

authors:
  - Guangyuan Ma
  - Yongliang Ma
  - Xuanrui Gou
  - Zhenpeng Su
  - Ming Zhou
  - Songlin Hu
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2505.12260
doi:
url: https://arxiv.org/abs/2505.12260
citation_key: ma2026lightretriever
paper_type: method
read_status: unread
domain: retrieval
---

## Summary

LightRetriever proposes an explicitly asymmetric LLM retriever that keeps a full document encoder offline but replaces online query encoding with nearly cost-free operations. For dense retrieval, it trains token-level query embeddings with a full encoder during training, caches the resulting vocabulary embeddings, and serves queries by lookup plus averaging. For sparse retrieval, it removes the query encoder entirely and uses token counts as the sparse query vector while learning only document-side term impacts. Across BeIR and CMTEB Retrieval, the method preserves roughly 95% of full symmetric retriever quality while reducing query encoding latency by more than three orders of magnitude and improving end-to-end throughput by over 10x, making it a practical design for latency-sensitive large-scale retrieval.

## Problem & Motivation

Recent LLM-based retrievers typically use symmetric dual encoders, so queries and documents both pass through large Transformer stacks. That symmetry is wasteful in deployment because documents can be pre-encoded offline, while queries must be encoded online under tight latency and resource constraints. The paper asks whether queries truly require the same degree of deep contextual modeling as documents. LightRetriever answers this by keeping full document-side modeling, but collapsing the query side into cached dense token embeddings plus count-based sparse features so that online query processing no longer depends on running a full LLM.

## Method

- **Base objective**: the retriever is trained with listwise contrastive loss `` `\ell^{CL} = -\log \frac{e^{v_q \cdot v_{d^+}/\tau}}{e^{v_q \cdot v_{d^+}/\tau} + \sum_{d^- \in \mathcal{N}(q)} e^{v_q \cdot v_{d^-}/\tau}}` `` using dot-product similarities.
- **Dense query training**: each query token is encoded independently with the shared instruction prompt, then averaged as `` `v_q^{den} = \frac{1}{n}\sum_{i=0}^{n-1} Enc_q(Inst; t_i)` ``. The implementation uses EOS pooling and a customized causal mask to avoid recomputing the prompt for every token.
- **Dense query serving**: after training, all vocabulary token vectors are cached into a lookup matrix `` `E \in \mathbb{R}^{V \times H}` `` and online queries become `` `v_q^{den} = \frac{1}{n}\sum_i E[t_i]` ``, eliminating Transformer inference on the query side.
- **Sparse retrieval**: the sparse query vector is purely lexical, `` `v_q^{spr}[t] = count(t)` `` for tokens present in the query. The document vector is learned by projecting last-layer hidden states through the LM head and applying `` `v_d^{spr} = \max(\log(\max(h_{last}\cdot P, 0) + 1))` `` over sequence positions.
- **Sparsification**: sparse document training adds the FLOPs regularizer `` `\ell_{FLOPS} = \sum_{t=0}^{V-1}\left(\frac{1}{N}\sum_{i=0}^{N-1} w_t^{(d_i)}\right)^2` `` to discourage dense lexical activations.
- **Hybrid scoring**: final retrieval uses a linear sum of normalized dense and sparse scores, combining semantic matching from the dense branch with lexical matching from the sparse branch.
- **Training setup**: `` `batch_size = 128` ``, `` `hard_negatives = 7` ``, `` `max_length = 512` ``, `` `\tau = 0.02` ``, `` `steps = 12000` ``, FLOPs coefficient `` `0.001` `` ramped up over the first `` `4000` `` steps, and LoRA with `` `r = 16` ``, `` `alpha = 32` ``, `` `dropout = 0.1` ``. Backbones include Llama-3.2 `` `1B/3B` ``, Llama-3.1 `` `8B` ``, and Qwen-2.5 `` `1.5B/3B/7B` `` on `` `8` `` H800/A800 GPUs.
- **Speed evaluation protocol**: throughput is measured on `` `65536` `` Bing queries over `` `1M` `` MS MARCO passages using a single A800 GPU, `` `batch_size = 256` ``, Faiss exact search with `` `1000` `` dense dimensions, and Lucene/Anserini sparse search with `` `64` `` threads.

## Key Results

- On the speed benchmark, LightRetriever-Llama8b cuts query encoding time from `` `109.4853 s` `` to `` `0.0412 s` `` relative to Full-Llama8b, while end-to-end time drops from `` `119.3730 s` `` to `` `9.3630 s` `` and QPS rises from `` `549` `` to `` `6999` ``.
- LightRetriever-Qwen7b shows a similar pattern: encoding time drops from `` `100.6716 s` `` to `` `0.0420 s` ``, total time from `` `110.6140 s` `` to `` `9.3870 s` ``, and QPS increases from `` `592` `` to `` `6982` ``.
- Retrieval quality remains close to full symmetric retrievers. LightRetriever-Llama3.1-8b reaches `` `54.4` `` nDCG@10 on BeIR and `` `63.0` `` on CMTEB-R, only `` `2.4` `` and `` `4.6` `` below the corresponding full models. LightRetriever-Qwen2.5-7b reaches `` `53.8` `` and `` `66.5` ``, only `` `2.8` `` and `` `3.6` `` lower than full Qwen2.5-7b.
- Compared with prior baselines, LightRetriever-Llama3.1-8b at `` `54.4` `` BeIR nDCG@10 outperforms BGE-m3 dense+sparse at `` `49.6` `` and approaches LLM2Vec `` `56.6` `` and E5-Mistral `` `56.9` `` while avoiding full query-side LLM inference.
- Ablations show the asymmetry is essential: making both sides lightweight drops BeIR/CMTEB-R by `` `13.8/18.6` `` points, replacing the training-time query encoder with an MLP drops `` `11.2/17.4` `` points, and keeping only one Transformer layer on the query side still loses `` `4.3/8.6` `` points.

## Limitations

- The paper only validates the approach on recent LLM backbones; it does not study whether the same design is worthwhile for smaller non-LLM retrievers such as BERT-based systems.
- LightRetriever is query-lightweight, not fully lightweight: document encoding still relies on full LLMs, and indexing large corpora can still consume substantial storage.
- Performance drops are usually modest but non-zero, and the paper notes somewhat larger degradations on smaller domain-specific datasets such as FiQA and cMedQA.
- Sparse retrieval cannot use task instructions in this design, because the sparse query side has no learnable encoder.
- The study is a preprint evaluated on benchmark suites and controlled throughput tests rather than a production deployment study with real serving costs over time.

## Concepts Extracted

- [[dense-retrieval]]
- [[sparse-retrieval]]
- [[hybrid-retrieval]]
- [[dual-encoder-architecture]]
- [[contrastive-learning]]
- [[instruction-tuning]]
- [[maximum-inner-product-search]]
- [[embedding-lookup]]

## Entities Extracted

- [[guangyuan-ma]]
- [[yongliang-ma]]
- [[xuanrui-gou]]
- [[zhenpeng-su]]
- [[ming-zhou-langboat]]
- [[songlin-hu]]
- [[beir]]
- [[ms-marco]]
- [[faiss]]
- [[lucene]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
