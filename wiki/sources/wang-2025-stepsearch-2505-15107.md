---
type: source
subtype: paper
title: "StepSearch: Igniting LLMs Search Ability via Step-Wise Proximal Policy Optimization"
slug: wang-2025-stepsearch-2505-15107
date: 2026-04-20
language: en
tags: [agents, search, reinforcement-learning, retrieval, multi-hop-qa]
processed: true

raw_file: raw/papers/wang-2025-stepsearch-2505-15107/paper.pdf
raw_md: raw/papers/wang-2025-stepsearch-2505-15107/paper.md
bibtex_file: raw/papers/wang-2025-stepsearch-2505-15107/paper.bib
possibly_outdated: false

authors:
  - Ziliang Wang
  - Xuhui Zheng
  - Kang An
  - Cijun Ouyang
  - Jialu Cai
  - Yuhang Wang
  - Yichao Wu
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2505.15107
doi: 10.48550/arXiv.2505.15107
url: https://arxiv.org/abs/2505.15107
citation_key: wang2025stepsearch
paper_type: method

read_status: unread

domain: agents
---

## Summary

StepSearch trains search-capable LLM agents with step-wise reinforcement learning instead of relying only on final-answer rewards. The core idea is to augment PPO with process supervision over each retrieval round: the model receives a global answer reward plus token-level search rewards that encourage newly useful evidence and penalize redundant retrieval. The paper also builds a MuSiQue-derived training set with about `60k` filtered sub-question search keywords and trains on only `19k` examples. Across Qwen2.5 `3B` and `7B` models, StepSearch consistently improves multi-hop QA on HotpotQA, 2WikiMultiHopQA, MuSiQue, and Bamboogle, with especially large gains over standard PPO / GRPO baselines in smaller models.

## Problem & Motivation

Existing search-RL systems for LLM agents mostly optimize coarse global rewards such as final answer correctness or output format. That signal is too sparse for multi-hop question answering, where success depends on issuing a sequence of useful intermediate queries and avoiding repeated low-value retrieval. The paper targets this credit-assignment problem by supervising the search path itself: query formulation, retrieved evidence novelty, and per-step progress toward the gold evidence set.

## Method

- **Training data pipeline**: starts from [[musique]] and expands each decomposed question into subquestion-answer pairs plus candidate search queries using `GPT-4o`; queries are kept only if they return valid results from at least `ceil(M / 2)` of `M` sources such as Google, Bing, and Wiki-18.
- **Interaction format**: the agent rolls out iterative `<think> ... </think>`, `<search> ... </search>`, `<information> ... </information>` blocks until `<answer> ... </answer>` or a search budget limit; retrieved `<information>` spans are masked out of the loss so gradients update only model-generated reasoning and search tokens.
- **Global reward**: final reward combines answer F1 and keyword-match reward, `r_overall = r_answer + gamma_key * r_key`, where `r_answer` is word-level F1 against the ground truth and `r_key` matches emitted queries against gold sub-question keywords.
- **StePPO objective**: extends [[proximal-policy-optimization]] with masked token optimization and step rewards; advantages `A_t` are computed with GAE, policy clipping uses `epsilon = 0.2`, and KL regularization uses `beta = 1e-3`.
- **Step-wise reward**: each retrieval turn gets `r_step^t = G^t - P^t`, where [[information-gain]] is `G^t = (1 / n) * sum_i max(c_i^t - m_i^t, 0)` and `c_i^t` is cosine similarity between retrieved evidence and gold documents; memory updates as `m_i^t = max(m_i^(t-1), c_i^t)`.
- **Redundancy modeling**: [[redundancy-penalty]] is `P^t = (1 / k) * sum_j 1(d_j^{r(t)} in H^(t-1))`, penalizing documents repeated from earlier search rounds.
- **Retriever and evaluation setup**: uses [[e5]] as the retriever during training, augments evaluation with the 2018 Wikipedia dump, and retrieves `k = 3` documents per search step.
- **Models and optimization**: trains Qwen2.5 `3B` / `7B` Base and Instruct models for `500` steps on `16` H800 GPUs with policy LR `7e-7`, value LR `7e-6`, warm-up ratios `0.285` / `0.015`, batch sizes `256 / 64 / 32` (total / mini / micro), `temperature = 1.0`, `top_p = 1.0`, and FSDP + CPU offloading with GPU memory utilization `0.7`.

## Key Results

- On Qwen2.5-`3B`-Base, StepSearch reaches HotpotQA `0.329 EM / 0.434 F1` vs PPO `0.223 / 0.315` and GRPO `0.256 / 0.366`.
- On Qwen2.5-`3B`-Base, StepSearch reaches MuSiQue `0.181 EM / 0.273 F1` vs Search-R1 `0.081 / 0.146`; this is an absolute F1 gain of `12.7` points.
- On Qwen2.5-`7B`-Base, StepSearch reaches 2WikiMultiHopQA `0.385 EM / 0.450 F1` vs PPO `0.282 / 0.329` and GRPO `0.266 / 0.345`.
- On Qwen2.5-`7B`-Base, StepSearch reaches Bamboogle `0.467 EM / 0.573 F1`, outperforming Search-R1 `0.430 / 0.545`.
- Across ablations, removing redundancy penalty drops 2Wiki F1 on `7B` from `0.450` to `0.367`, and removing full step rewards drops Bamboogle F1 from `0.573` to `0.485`.
- The paper claims absolute gains of `11.2%` and `4.2%` over search-RL baselines for `3B` and `7B` models respectively while using only `19k` training examples.

## Limitations

- Evaluation is limited to text-only multi-hop QA; the method is not validated for multimodal retrieval or broader agent tasks.
- Training is only reported for relatively modest model scales (`3B` and `7B`); the paper explicitly notes that `14B` and `32B` settings may worsen reward collapse and training instability.
- The training corpus is comparatively small (`19k` MuSiQue-derived examples), so it is unclear whether gains persist when matched against larger-scale baselines such as Search-R1 trained on much more data.
- The step reward depends on gold sub-question trajectories and gold-support documents, which may be costly or brittle to construct outside benchmark-style settings.

## Concepts Extracted

- [[proximal-policy-optimization]]
- [[process-supervision]]
- [[multihop-question-answering]]
- [[retrieval-augmented-generation]]
- [[chain-of-thought]]
- [[token-level-reward]]
- [[information-gain]]
- [[redundancy-penalty]]

## Entities Extracted

- [[ziliang-wang]]
- [[xuhui-zheng]]
- [[kang-an]]
- [[cijun-ouyang]]
- [[jialu-cai]]
- [[yuhang-wang]]
- [[yichao-wu]]
- [[sensetime]]
- [[musique]]
- [[hotpotqa]]
- [[2wikimultihopqa]]
- [[bamboogle]]
- [[search-r1]]
- [[e5]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
