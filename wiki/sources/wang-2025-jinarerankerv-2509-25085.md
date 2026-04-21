---
type: source
subtype: paper
title: "jina-reranker-v3: Last but Not Late Interaction for Listwise Document Reranking"
slug: wang-2025-jinarerankerv-2509-25085
date: 2026-04-20
language: en
tags: [reranking, retrieval, multilingual, listwise-ranking, long-context]
processed: true
raw_file: raw/papers/wang-2025-jinarerankerv-2509-25085/paper.pdf
raw_md: raw/papers/wang-2025-jinarerankerv-2509-25085/paper.md
bibtex_file: raw/papers/wang-2025-jinarerankerv-2509-25085/paper.bib
possibly_outdated: false
authors:
  - Feng Wang
  - Yuqing Li
  - Han Xiao
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2509.25085
doi: 10.48550/arXiv.2509.25085
url: http://arxiv.org/abs/2509.25085
citation_key: wang2025jinarerankerv
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper presents jina-reranker-v3, a `0.6B` multilingual listwise reranker that replaces post-hoc token matching with a new last-but-not-late interaction mechanism. Instead of encoding each candidate independently as in late-interaction rerankers, it places the query and multiple documents in one long context window, lets causal self-attention produce cross-document context during encoding, and then extracts ranking embeddings from designated special-token positions. Built on Qwen3-0.6B with a lightweight projector, the model is designed to sit between cross-encoders and embedding-based rerankers on the efficiency-effectiveness spectrum. Across BEIR, MIRACL, MKQA, and CoIR, it reports strong English, multilingual, and code-retrieval performance, including the best reported BEIR score among compared rerankers in the paper.

## Problem & Motivation

The paper targets the standard retrieval bottleneck between expressive but expensive cross-encoders and efficient but weaker embedding-based rerankers. Late-interaction methods such as ColBERT retain token-level evidence, but they still encode documents separately and only interact after encoding, which prevents query-document interaction during encoding and makes cross-document comparison impossible. The authors argue that reranking is naturally listwise: the model should compare documents within the same candidate set and exploit long-context transformer capacity to capture inter-document evidence, while still producing compact embeddings that keep scoring efficient.

## Method

- **Backbone and scale**: built on Qwen3-0.6B with `28` transformer layers, hidden size `1024`, `16` query heads / `8` KV heads (GQA), context length `131072`, and effective sequence length up to `8192`.
- **Input construction**: the model uses an instruction-style prompt with system/user roles, places `k` passages in a shared context window, and appends special markers `<|doc_emb|>` and `<|query_emb|>` after each document and the final query.
- **Last-but-not-late interaction**: query and candidate documents are encoded jointly under causal self-attention, so documents can observe both the query and other documents before scoring, enabling listwise cross-document comparison during encoding rather than after separate encoding.
- **Embedding extraction**: contextual states are taken at designated token positions, with `` `q~ = H_{t_q}` `` and `` `d~_i = H_{t_i}` ``, then projected as `` `q = P_phi(q~)` `` and `` `d_i = P_phi(d~_i)` ``.
- **Projection and scoring**: the projector is reported as `` `1024 -> 512 -> 512` `` with ReLU in Appendix Table 5, while the main text says it maps to a `256`-dimensional space; relevance is scored by cosine similarity `` `s_i = cos(q, d_i)` ``.
- **Long-list handling**: when candidate sets exceed the window budget, the system reranks in batches of up to `64` documents per forward pass while keeping the query representation consistent across batches.
- **Training objective**: the total loss is `` `L = L_rank + 0.45 * L_disperse + 0.85 * L_dual + 0.85 * L_similar` ``, where `` `L_rank` `` is InfoNCE over one positive and `K` negatives with temperature `` `tau` ``, `` `L_disperse` `` penalizes embedding collapse, `` `L_dual` `` enforces consistency using the query representation at the sequence start, and `` `L_similar` `` aligns augmented document pairs.
- **Stage 1 training**: LoRA fine-tuning with rank `16` and `` `alpha = 32` `` on attention and FFN layers; backbone frozen; `16` documents per query (`1` positive + `15` negatives); per-document length `768`; maximum total sequence length `12288`; learning rate `5e-5`; batch size per device `60`; temperature `0.25`.
- **Stage 2 training**: context is extended to document lengths up to `8192`, negatives scale from `15` to `45`, hard negatives are mined across multiple retrieval systems, learning rates range from `5e-5` to `6e-6`, batch sizes can drop to `6`, and temperatures go as low as `0.05`.
- **Stage 3 training**: multiple specialized models are linearly merged with reported merge weights ranging from `0.25` to `0.65` to combine domain-specific strengths.

## Key Results

- On BEIR, jina-reranker-v3 reports `61.85` nDCG@10, beating jina-reranker-v2 (`57.06`), bge-reranker-v2-m3 (`56.51`), and slightly surpassing mxbai-rerank-large-v2 (`61.44`) while using `2.5x` fewer parameters (`0.6B` vs. `1.5B`).
- On MIRACL, it reports `66.83` nDCG@10 across `18` languages, close to Qwen3-Reranker-4B (`67.52`) and slightly above jina-reranker-m0 (`66.75`), though below bge-reranker-v2-m3 (`69.32`).
- On MKQA, it reaches `67.92` Recall@10; on CoIR, it reaches `70.64`, improving substantially over jina-reranker-v2 on code retrieval (`58.35`).
- On BEIR subsets, it reports `78.58` on HotpotQA and `94.01` on FEVER, suggesting particular strength on multi-hop retrieval and fact verification.
- The listwise ordering ablation is fairly stable: descending order gives `61.85`, ascending `61.45`, and random ordering `62.24` average nDCG@10 on BEIR.

## Limitations

- The method depends on very long context windows and still requires batching when candidate lists exceed the `131K` token limit, so inference cost is not as cheap as pure bi-encoder reranking.
- Cross-document interaction makes ranking partially sensitive to candidate-set composition and ordering; the paper reports only modest ordering studies and does not fully characterize robustness.
- The evaluation is benchmark-heavy but does not provide serving latency, memory, or throughput measurements against comparable rerankers, so the efficiency claim is only partially substantiated.
- The paper explicitly leaves prompt-injection robustness and in-window deduplication for future work.
- The architecture description contains a reporting inconsistency: the main text says the projector yields `256`-dimensional embeddings, while Appendix Table 5 lists a `` `1024 -> 512 -> 512` `` projector.

## Concepts Extracted

- [[document-reranking]]
- [[listwise-ranking]]
- [[late-interaction]]
- [[cross-encoder]]
- [[bi-encoder]]
- [[dense-retrieval]]
- [[hard-negative-mining]]
- [[contrastive-loss]]
- [[multilingual-retrieval]]
- [[last-but-not-late-interaction]]
- [[cross-document-interaction]]

## Entities Extracted

- [[feng-wang-jina-ai]]
- [[yuqing-li]]
- [[han-xiao]]
- [[jina-ai]]
- [[university-of-pittsburgh]]
- [[colbert]]
- [[beir]]
- [[miracl]]
- [[mkqa]]
- [[coir]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
