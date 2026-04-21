---
type: source
subtype: paper
title: Generalizing Conversational Dense Retrieval via LLM-Cognition Data Augmentation
slug: chen-2024-generalizing-2402-07092
date: 2026-04-20
language: en
tags: [conversational-search, dense-retrieval, data-augmentation, llm, contrastive-learning]
processed: true

raw_file: raw/papers/chen-2024-generalizing-2402-07092/paper.pdf
raw_md: raw/papers/chen-2024-generalizing-2402-07092/paper.md
bibtex_file: raw/papers/chen-2024-generalizing-2402-07092/paper.bib
possibly_outdated: false

authors:
  - Haonan Chen
  - Zhicheng Dou
  - Kelong Mao
  - Jiongnan Liu
  - Ziliang Zhao
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.07092
doi:
url: http://arxiv.org/abs/2402.07092
citation_key: chen2024generalizing
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

This paper proposes ConvAug, a data augmentation and training framework for [[conversational-dense-retrieval]] that aims to improve robustness to diverse multi-turn search interactions. The core idea is to use an LLM to generate both positive conversation variants that preserve intent and hard negatives that keep surface form similar while shifting key entities or intent. To control augmentation quality, the authors introduce a three-step [[cognition-aware-prompting]] procedure and a difficulty-adaptive sampler that matches harder augmented samples to more complex conversations. Built on ANCE with multi-task contrastive training, ConvAug improves normal evaluation on QReCC and TopiOCQA, transfers better in zero-shot tests on CAsT-20/21, and also boosts other retrievers such as Conv-SPLADE and LeCoRE.

## Problem & Motivation

Existing conversational dense retrieval systems typically treat each training conversation as a fixed multi-turn text sequence. That assumption is brittle because real users can express the same search need through many alternative conversational paths, while available conversational retrieval datasets record only a tiny subset of those possibilities. The paper argues that this data sparsity hurts generalization, especially for longer and more complex conversations. Prior augmentation methods for conversational search are often rule-based or annotation-heavy and do not model turn dependencies well. ConvAug is motivated by the need to expose the context encoder to diverse yet semantically controlled conversation variants so that it can better recover search intent under distribution shift.

## Method

- **Task setup**: given conversation context `C_n = {q_1, r_1, ..., q_{n-1}, r_{n-1}, q_n}`, retrieve the relevant passage `d^+` from collection `D` using a conversational context encoder.
- **Multi-level positive and negative augmentation**: ConvAug generates positives `C^+` and hard negatives `C^-` across token, turn, and conversation levels.
- **Token-level operations**: token masking replaces a proportion `r_w` of tokens with `[token_mask]` to form positives; entity replacement generates hard negatives by preserving flow but changing critical entities.
- **Turn-level operations**: turn masking, turn reordering, and noisy-turn insertion create positives while respecting a turn dependency DAG so the altered conversation still preserves logical dependencies.
- **Conversation-level operations**: paraphrasing generates semantically equivalent positives, while intent shifting creates negatives with similar phrasing but different search goals.
- **Cognition-aware prompting**: the LLM follows three stages, `Comprehension Synthesis -> Associative Expansion -> Conclusion`, to reduce false positives, false negatives, and hallucinations during generation.
- **Difficulty-adaptive filter**: conversation difficulty is defined as `` `Diff(C) = |T_h| + (|Topic(C)| * avg_PPL(C))` ``. Positive-pair difficulty is `` `Diff^+(C_i^+, C_j^+) = 1 - BERTSim(C_i^+, C_j^+)` ``, and hard negatives are selected according to similarity to the chosen positives.
- **Training objective**: the model combines passage-ranking contrastive loss with augmentation-based contrastive learning, optimized as `` `L = L_rank + alpha * L_CL` `` over selected positive pairs, in-batch negatives, and `k` hard negatives.
- **Base models and encoders**: ConvAug uses [[ance]] as the base retriever and [[llama-2-chat]] `7B` for augmentation; similarity filtering uses `all-MiniLM-L6-v2`.
- **Key hyperparameters**: `k = 1` hard negative; batch size `12`; QReCC uses `tau = 0.0012`, `r_w = 0.5`, `r_t = 0.5`, learning rate `1e-5`, `alpha = 1.0`; TopiOCQA uses `tau = 0.001`, `r_w = 0.9`, `r_t = 0.5`, learning rate `1.5e-5`, `alpha = 0.1`.

## Key Results

- On QReCC normal evaluation, ConvAug reaches `MRR 52.7`, `NDCG@3 50.4`, and `Recall@10 75.6`, outperforming LeCoRE (`51.1/48.5/73.9`) and Conv-SPLADE (`50.0/46.6/69.9`).
- On TopiOCQA normal evaluation, ConvAug reaches `MRR 35.0`, `NDCG@3 33.3`, and `Recall@10 57.9`, again exceeding the strongest baseline LeCoRE (`32.0/31.4/54.3`).
- In zero-shot transfer, training on QReCC and testing on CAsT yields `45.0/30.7` (`MRR/NDCG@3`) on CAsT-20 and `54.8/36.8` on CAsT-21, beating InstructoR-ANCE (`43.7/29.6`, `53.0/34.9`).
- Ablations show the largest QReCC drop comes from removing entity replacement: full ConvAug `52.7/50.4` vs. `50.8/48.5` without `C^-_ent`; removing cognition-aware prompting drops MRR from `52.7` to `51.1`.
- Hard-negative ratio matters: `k = 1` is best on QReCC (`52.7/50.4`) and CAsT-21 (`54.8/36.8`), while `k = 0` or `k = 2` performs worse.
- The framework transfers to other bases: Conv-SPLADE improves from `50.0/46.6` to `52.4/49.8`, and LeCoRE improves from `51.1/48.5` to `53.1/50.7`.

## Limitations

- The difficulty score `` `Diff(C)` `` is deliberately simple and may not capture conversational complexity as accurately as a stronger learned estimator.
- Augmentation is performed offline and is computationally expensive because the method generates millions of conversations; the paper reports limited resources of `4` NVIDIA A100 GPUs.
- Only one augmentation LLM, Llama 2-Chat `7B`, is tested, so cross-LLM robustness remains unclear.
- The method assumes source conversations do not contain sensitive private information; otherwise LLM-based generation may propagate risky content.
- Improvements are demonstrated on conversational retrieval benchmarks, but the framework is still tied to the quality of the chosen base retriever, prompts, and automatic filtering heuristics.

## Concepts Extracted

- [[conversational-search]]
- [[conversational-dense-retrieval]]
- [[data-augmentation]]
- [[cognition-aware-prompting]]
- [[large-language-model]]
- [[contrastive-learning]]
- [[hard-negative-sampling]]
- [[dependency-graph]]
- [[query-rewriting]]
- [[multi-task-learning]]
- [[zero-shot-generalization]]

## Entities Extracted

- [[haonan-chen]]
- [[zhicheng-dou]]
- [[kelong-mao]]
- [[jiongnan-liu]]
- [[ziliang-zhao]]
- [[renmin-university-of-china]]
- [[convaug]]
- [[ance]]
- [[llama-2-chat]]
- [[qrecc]]
- [[topiocqa]]
- [[cast-20]]
- [[cast-21]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
