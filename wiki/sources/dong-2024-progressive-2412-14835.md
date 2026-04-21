---
type: source
subtype: paper
title: Progressive Multimodal Reasoning via Active Retrieval
slug: dong-2024-progressive-2412-14835
date: 2026-04-20
language: en
tags: [multimodal-reasoning, retrieval, mcts, process-reward-model, multimodal-retrieval]
processed: true

raw_file: raw/papers/dong-2024-progressive-2412-14835/paper.pdf
raw_md: raw/papers/dong-2024-progressive-2412-14835/paper.md
bibtex_file: raw/papers/dong-2024-progressive-2412-14835/paper.bib
possibly_outdated: false

authors:
  - Guanting Dong
  - Chenghao Zhang
  - Mengjie Deng
  - Yutao Zhu
  - Zhicheng Dou
  - Ji-Rong Wen
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2412.14835
doi: 10.48550/arXiv.2412.14835
url: http://arxiv.org/abs/2412.14835
citation_key: dong2024progressive
paper_type: method

read_status: unread

domain: llm
---

## Summary

This paper proposes AR-MCTS, a multimodal reasoning framework that combines hybrid-modal retrieval, active retrieval during search, Monte Carlo Tree Search, and progressive process reward modeling to improve multi-step reasoning in multimodal large language models. The system first builds a hybrid corpus from math-focused multimodal datasets plus Wikipedia and COIG, then retrieves step-relevant evidence with dense text and cross-modal retrievers, filters it by concept consistency, and injects it into MCTS expansion. The resulting search traces are used to create step-wise supervision for a process reward model via step-wise DPO and point-wise fine-tuning. Across MathVista, We-Math, and GAOKAO-MM, the framework improves both proprietary and open MLLMs, with especially strong gains on weaker models and harder multi-step settings.

## Problem & Motivation

Multimodal large language models remain brittle on multi-step reasoning tasks because early cross-modal misunderstandings compound over subsequent steps, and existing MCTS-style reasoning systems usually leave expansion to beam-search sampling from the model's internal knowledge. That assumption is much weaker in multimodal settings than in text-only LLMs, where pretraining covers the dominant modality more uniformly.

The paper argues that reliable multimodal reasoning requires two coupled fixes: better step-level expansion and better step-level verification. AR-MCTS addresses the first by retrieving different problem-solving insights at each reasoning step rather than reusing a static evidence bundle, and addresses the second by turning MCTS exploration traces into automatic process supervision for a multimodal process reward model.

## Method

- **Hybrid-modal retrieval corpus**: constructs `D_H = D_M ∪ D_G`, where `D_M` contains 22K text-only math QA pairs plus 12.5K multimodal samples from GSM8K, MATH, MathVista, MathVerse, MathVision, and We-Math, while `D_G` adds Wikipedia and COIG for general reasoning.
- **Text retrieval**: uses multilingual Contriever to rank text-only documents by dense similarity, `D_q = argtop_k E_d(d_i)^T E_q(q)`.
- **Cross-modal retrieval**: uses CLIP `ViT-L/14@336px` to encode image-text pairs with `E_x(x,t) = (E_I(x) + E_T(t)) / 2` when both modalities exist, then retrieves `D_cross = argtop_k E_x(Q^m)^T E_x(x_j,t_j)`. FAISS is used for dense indexing.
- **Knowledge concept filtering**: retains only retrieved items satisfying both the original retrieval threshold `T_r` and the concept-consistency threshold `T_kc`, yielding `D_ins = {r in D_H | Sim(r,Q^m) >= T_r and Sim(r,L_kc) >= T_kc}`.
- **MCTS selection**: uses Upper Confidence Bound `UCB(i) = w_i + C * sqrt(2 * ln(N_i / n_i))` to traverse the tree over sentence-level reasoning states.
- **Active-retrieval expansion**: at state `s_i`, concatenates the current multimodal query with prior steps, retrieves new step-specific insights `r_i`, and samples candidate next steps from `p_theta(y | x) = ∏_{i=1}^k p_theta({y_i^j}_{j=1}^k | Q_i^m, r_i)`, replacing static beam-search expansion.
- **Simulation and backup**: evaluates partial paths with one-step rollout using `V(s_i) = (sum_{j=1}^k I(y_j = y_hat_i)) / k`, then updates counts and Q-values by `N(s,a) <- N(s,a)+1` and `Q(s,a) <- Q(s,a) + (V(s)-Q(s,a))/N(s,a)`.
- **Progressive PRM alignment**: first forms step-level preference pairs with `v_j > 0.8` as positive and `v_j = 0` as negative, optimizing a step-wise DPO objective with `beta = 0.3`; then applies point-wise fine-tuning with cross-entropy over sigmoid scores.
- **Training / inference details**: DPO pre-alignment uses learning rate `5e-7`, cosine schedule, warm-up ratio `0.1`, global batch size `64`, `2` epochs, and max context length `4096`; point-wise fine-tuning uses learning rate `7e-6`, global batch size `128`, weight decay `0.1`, `3` epochs, max context length `8192`, and training on `8 x NVIDIA A800` GPUs. Inference uses PRM soft scores and early stopping after round `4`.

## Key Results

- On **MathVista**, AR-MCTS improves GPT-4o from `59.0` to `62.6` overall accuracy and Qwen2-VL-7B from `58.8` to `64.1`.
- On **We-Math**, AR-MCTS improves GPT-4o from `40.8` to `46.8` on strict AVG and from `46.1` to `56.4` on S3; for Qwen2-VL-7B it improves AVG from `19.8` to `28.1` and S3 from `33.9` to `40.6`.
- On **GAOKAO-MM**, AR-MCTS raises GPT-4o from `45.6` to `52.2` overall and Qwen2-VL-7B from `30.2` to `37.4`.
- Compared with ORM, AR-MCTS gains are strongest on hard multi-step reasoning: on We-Math S3, GPT-4o improves from `50.3` to `56.4`, and Qwen2-VL-7B from `34.6` to `40.6`.
- Ablations on Qwen2-VL-7B show every major component matters: removing PRM drops MathVista from `64.1` to `61.0`; removing active retrieval drops it to `61.9`; removing concept filtering drops it to `62.8`.
- Diversity analysis shows AR-MCTS covers a larger reasoning space than beam search, yielding `46` semantic clusters versus `38` under the same sampling budget on 1,000 sampled solutions.

## Limitations

- The framework is computationally expensive because MCTS-based annotation and verifier training still require repeated search, rollout, and scoring over many reasoning paths.
- Knowledge concept filtering depends on high-quality concept labels or offline concept annotation; this assumption may not hold as cleanly outside curated math-style benchmarks.
- The PRM is aligned using a text backbone rather than a jointly trained multimodal verifier, so image-text interaction is only indirectly reflected in the reward signal.
- The main evaluation is concentrated on mathematical and exam-style multimodal reasoning, leaving broader multimodal domains less tested.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[multimodal-reasoning]]
- [[monte-carlo-tree-search]]
- [[active-retrieval]]
- [[process-reward-model]]
- [[outcome-reward-model]]
- [[multimodal-retrieval]]
- [[knowledge-concept-filtering]]
- [[direct-preference-optimization]]
- [[curriculum-learning]]

## Entities Extracted

- [[guanting-dong]]
- [[chenghao-zhang]]
- [[mengjie-deng]]
- [[yutao-zhu]]
- [[zhicheng-dou]]
- [[ji-rong-wen]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
