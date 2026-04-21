---
type: source
subtype: paper
title: "ToolWeaver: Weaving Collaborative Semantics for Scalable Tool Use in Large Language Models"
slug: fang-2026-toolweaver-2601-21947
date: 2026-04-20
language: en
tags: [agents, tool-use, llm, retrieval, quantization]
processed: true

raw_file: raw/papers/fang-2026-toolweaver-2601-21947/paper.pdf
raw_md: raw/papers/fang-2026-toolweaver-2601-21947/paper.md
bibtex_file: raw/papers/fang-2026-toolweaver-2601-21947/paper.bib
possibly_outdated: false

authors:
  - Bowen Fang
  - Wen Ye
  - Yunyue Su
  - Jinghao Zhang
  - Qiang Liu
  - Yesheng Liu
  - Jiabing Yang
  - Xin Sun
  - Shu Wu
  - Baole Wei
  - Liang Wang
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2601.21947
doi:
url: https://arxiv.org/abs/2601.21947
citation_key: fang2026toolweaver
paper_type: method

read_status: unread

domain: agents
---

## Summary

ToolWeaver proposes a generative tool-use framework that replaces one-token-per-tool identifiers with hierarchical code sequences learned from both tool documentation and tool co-usage trajectories. Tools are encoded by a collaborative-aware residual-quantization pipeline over dense text embeddings, then aligned with Llama-3-8B through two stages of fine-tuning for retrieval and full tool-use trajectories. The design expands the vocabulary by only `L x K` new tokens instead of one token per API, making growth logarithmic in the tool library size while letting related tools share parent codes. On ToolBench with `46,985` APIs, ToolWeaver improves multi-domain retrieval over ToolGen on the hardest I3 split (`NDCG@1 = 88.00` vs. `81.00`) and substantially improves end-to-end multi-tool completion (`SoPR = 52.19`, `SoWR = 59.02`) while preserving general language ability better than atomic tool tokens.

## Problem & Motivation

The paper targets the scalability and semantic weaknesses of generative tool-use systems that assign each API a unique special token. That flat design scales linearly with the number of tools, creates heavy vocabulary expansion, and forces the model to infer tool collaboration only from sparse co-occurrences of isolated IDs. ToolWeaver instead aims to build structured tool identifiers that preserve intrinsic tool semantics, encode extrinsic co-usage patterns, and remain efficient when the tool inventory approaches tens of thousands of APIs.

## Method

- **Problem setup**: a tool-augmented agent executes trajectories `Traj = [q, (p_1, d_1, alpha_1, f_1), ..., (p_t, d_t, alpha_t, f_t), a]`, where tool selection is reframed as autoregressive generation over structured tool codes instead of atomic IDs.
- **Semantic initialization**: each tool document is encoded with `all-mpnet-base-v2` into `e_d in R^768`, then projected by an MLP with hidden sizes `1024 -> 512 -> 256 -> 128` to `z_d in R^64`.
- **Hierarchical coding**: ToolWeaver uses `L = 2` codebooks with `K = 1024` vectors each, so the identifier capacity is `K^L` while the added vocabulary is only `L x K = 2048` tokens for a library of nearly `47k` tools.
- **Residual quantization**: for level `l`, ToolWeaver assigns `iota_{d,l} = argmin_k ||r_{d,l} - v_{l,k}||_2^2` and updates `r_{d,l+1} = r_{d,l} - v_{l,iota_{d,l}}`; reconstruction uses `hat{z}_d = sum_l v_{l,iota_{d,l}}` with `L_recon = ||z_d - hat{z}_d||_2^2` and commitment weight `beta = 0.25` in `L_quant`.
- **Collaborative guidance**: a tool-tool co-usage matrix `C` from training trajectories induces similarity `A_uv = C_uv / sqrt(C_uu C_vv)`, and the graph regularizer `L_collab = sum_{u,v} A_uv ||hat{z}_u - hat{z}_v||_2^2` is added with `lambda = 1.0`.
- **Collision mitigation**: the final codebook is constrained by uniform assignment via optimal transport, solving a Sinkhorn-Knopp objective for the last-layer assignment matrix with `50` iterations so that tools distribute evenly across final centroids.
- **Stage 1 alignment**: retrieval tuning optimizes `L_retrieval = -E_{(q,d)} log P(iota_d | q)` on `489,702` query-tool pairs for `5` epochs.
- **Stage 2 alignment**: trajectory tuning fine-tunes on `183,336` full tool-use trajectories for `2` more epochs, supervising assistant-side reasoning, code-token generation, tool arguments, and final answers.
- **Training and inference details**: both alignment stages use cosine decay, warmup ratio `3%`, peak learning rate `4e-5`, context length `6144`, DeepSpeed ZeRO-3, FlashAttention-2, and constrained beam search over a precomputed trie of valid code sequences.

## Key Results

- **Multi-domain retrieval**: ToolWeaver reaches `NDCG@1 / @3 / @5 = 91.16 / 91.14 / 93.48` on I1, `89.76 / 89.70 / 91.80` on I2, and `88.00 / 85.80 / 90.12` on I3, outperforming ToolGen by `+7.00` absolute `NDCG@1` on I3.
- **In-domain retrieval**: it also leads in most easier settings, e.g. I1 `93.76 / 94.80 / 95.69` and I2 `91.91 / 93.07 / 95.63`, while remaining competitive on I3 (`86.00 / 86.13 / 90.39`).
- **End-to-end tool use**: on ToolBench multi-domain evaluation, ToolWeaver obtains `SoPR = 53.17 / 44.03 / 52.19` for I1/I2/I3 and `54.85 / 57.41 / 46.24` on unseen-tool or unseen-category splits, with `SoWR = 59.02` on I3 versus ToolGen's `49.18`.
- **General language preservation**: compared with ToolGen, ToolWeaver lowers WikiText-2 perplexity from `104.54` to `25.36` and reduces summarization average drop from `2.47` to `0.57`, with CNN/DailyMail BERTScore `85.07` versus the Llama-3-8B base `85.35`.
- **Cross-model robustness**: on Qwen-2.5, ToolWeaver consistently beats ToolGen across `1.5B`, `3B`, `7B`, and `14B`, with especially large gains on complex I3 retrieval (for example, `87.00` vs. `69.00` `NDCG@1` on Qwen-2.5-1.5B).

## Limitations

- The empirical study is centered on ToolBench, so claims about broader tool ecosystems or different API distributions still require external validation.
- Error analysis shows that wrong-tool selection remains the dominant failure mode, growing from about `71%` of failures on I1 to about `95%` on I3; better codes do not remove the hard retrieval bottleneck in large tool corpora.
- ToolWeaver still increases decoding length because each tool identifier is a sequence of `L` codes; the paper reports a latency increase from `108.16 ms` (atomic) to `128.21 ms` for the default `L = 2` setting.
- The method depends on historical co-usage trajectories to compute collaborative structure, which may be weak or noisy for newly added tools with sparse interaction data.
- Although deeper code hierarchies can improve retrieval (`L = 4` peaks in the appendix), the main system uses `L = 2` to balance accuracy against inference cost, so the deployed configuration is not the absolute best-performing one.
- The paper explicitly notes safety and data-quality risks in real-world tool deployment because ToolBench APIs were not audited for privacy, bias, or malicious behavior.

## Concepts Extracted

- [[large-language-model]]
- [[tool-augmented-language-model]]
- [[tool-selection]]
- [[tool-retrieval]]
- [[generative-tool-use]]
- [[vector-quantization]]
- [[residual-quantization]]
- [[collaborative-semantics]]
- [[hierarchical-tool-representation]]
- [[constrained-beam-search]]

## Entities Extracted

- [[bowen-fang]]
- [[wen-ye]]
- [[yunyue-su]]
- [[jinghao-zhang]]
- [[qiang-liu]]
- [[yesheng-liu]]
- [[jiabing-yang]]
- [[xin-sun]]
- [[shu-wu]]
- [[baole-wei]]
- [[liang-wang]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
