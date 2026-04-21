---
type: source
subtype: paper
title: "Math-Shepherd: Verify and Reinforce LLMs Step-by-step without Human Annotations"
slug: wang-2024-mathshepherd-2312-08935
date: 2026-04-20
language: en
tags: [llm, reasoning, math, verification, reinforcement-learning]
processed: true

raw_file: raw/papers/wang-2024-mathshepherd-2312-08935/paper.pdf
raw_md: raw/papers/wang-2024-mathshepherd-2312-08935/paper.md
bibtex_file: raw/papers/wang-2024-mathshepherd-2312-08935/paper.bib
possibly_outdated: false

authors:
  - Peiyi Wang
  - Lei Li
  - Zhihong Shao
  - R.X. Xu
  - Damai Dai
  - Yifei Li
  - Deli Chen
  - Y. Wu
  - Zhifang Sui
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2312.08935
doi: 10.48550/arXiv.2312.08935
url: http://arxiv.org/abs/2312.08935
citation_key: wang2024mathshepherd
paper_type: method

read_status: unread

domain: llm
---

## Summary

Math-Shepherd is a process-oriented verifier for mathematical reasoning that scores each intermediate reasoning step instead of only the final answer. The paper's main contribution is an automatic process-annotation pipeline that replaces expensive human step labels by asking a fine-tuned completer to continue from an intermediate step and estimating that step's quality from the fraction of continuations that reach the gold answer. The resulting process reward model is used both for best-of-256 verification and for step-by-step PPO. Across GSM8K and MATH, the method consistently beats self-consistency and outcome reward models, and it improves open-source generators from 7B to 70B. The strongest reported result reaches `93.3%` on GSM8K and `48.1%` on MATH with DeepSeek-67B plus verification.

## Problem & Motivation

Existing math verifiers for large language models often score only the final outcome, which makes their feedback too coarse for multi-step reasoning and weak for reinforcement learning. Process reward models are more attractive because they can identify where a solution first goes wrong and can therefore support reranking, correction, and step-level policy optimization. The bottleneck is supervision: prior high-quality PRMs rely on costly human annotations over reasoning traces. This paper targets that bottleneck by defining step quality through future solvability and using automatic rollouts from intermediate steps to construct process-level labels at scale, aiming to make PRM training practical for open-source math LLMs.

## Method

- **Task setup**: the paper evaluates reward models in two scenarios, verification via best-of-`N` reranking and reinforcement learning via step-by-step PPO over math solutions.
- **ORM baseline**: an [[outcome-reward-model]] assigns one scalar to the whole solution and is trained with binary cross-entropy `` `L_ORM = y_s log r_s + (1 - y_s) log (1 - r_s)` ``.
- **PRM objective**: a [[process-reward-model]] scores every step `` `s_i` `` and is trained with `` `L_PRM = sum_i [y_{s_i} log r_{s_i} + (1 - y_{s_i}) log (1 - r_{s_i})]` ``, treating step supervision as binary classification.
- **Automatic process annotation**: for an intermediate step `` `s_i` ``, a completer decodes `` `N = 8` `` continuations; hard estimation sets `` `y_{s_i}^{HE} = 1` `` iff any continuation reaches the gold answer, while soft estimation uses the success frequency `` `y_{s_i}^{SE} = (1/N) sum_j I(a_j = a*)` ``.
- **Ranking rule**: for PRM-based verification, a solution score is the minimum score across its steps; the paper also combines reward models with [[self-consistency]] using `` `a_{sc+rm} = argmax_a sum_i I(a_i = a) * RM(p, S_i)` ``.
- **Data construction**: 7B and 13B generators are trained for `1` epoch on GSM8K and MATH, `15` solutions are sampled per problem, duplicates are removed, and the resulting training corpora contain about `170k` GSM8K solutions and `270k` MATH solutions.
- **Training configuration**: generators and completers are trained for `3` epochs on [[metamath]]; the completer is LLemma-7B; reward models are trained for `1` epoch with learning rate `` `1e-6` `` and maximum sequence length `` `512` ``.
- **Step-by-step PPO**: the policy receives rewards at the end of each reasoning step instead of only the final answer; RL uses learning rates `` `4e-7` `` for LLaMA2-7B and `` `1e-7` `` for Mistral-7B, KL coefficient `` `0.04` ``, and a cosine scheduler with minimum learning rate `` `1e-8` ``.
- **Evaluation protocol**: verification uses `256` sampled candidates on GSM8K and MATH500, while reinforcement learning is evaluated by greedy decoding accuracy on the full GSM8K and MATH test sets.

## Key Results

- As a verifier, [[math-shepherd]] beats self-consistency and ORM across all reported generators: with DeepSeek-67B it reaches `93.3%` on GSM8K and `48.1%` on MATH, versus `92.6%` / `45.3%` for ORM and `88.2%` / `45.4%` for self-consistency.
- With LLaMA2-70B on GSM8K/MATH, MATH-SHEPHERD achieves `93.2%` / `44.5%`, improving over ORM at `91.8%` / `40.4%`.
- With LLemma-34B, MATH-SHEPHERD reaches `90.9%` on GSM8K and `46.0%` on MATH; adding self-consistency raises MATH further to `47.3%`.
- In reinforcement learning, step-by-step PPO lifts Mistral-7B from `77.9%` to `84.1%` on GSM8K and from `28.6%` to `33.0%` on MATH; the same model with verification after PPO reaches `89.1%` and `43.5%`.
- For LLaMA2-7B, step-by-step PPO improves GSM8K from `66.6%` to `73.2%` and MATH from `19.2%` to `21.6%`, outperforming both RFT and ORM-PPO.
- Manual analysis of `160` GSM8K steps reports that automatic hard-estimation labels reach `86%` accuracy at `N = 4`, and Table 4 shows the proposed annotation method outperforming NLI- and rule-based alternatives (`85.0%` vs `75.6%` / `75.0%` step-label accuracy).

## Limitations

- Automatic process annotation is still noisy because step quality is inferred from sampled completions rather than directly observed human judgments.
- The completion stage is computationally expensive: label quality improves with larger `` `N` ``, but each extra rollout increases annotation cost substantially.
- The study is concentrated on mathematical reasoning, mainly GSM8K and MATH, so transfer to broader reasoning domains is only lightly tested.
- Verification quality depends on reward-model strength; the paper shows weaker reward models can hurt performance when validating stronger post-PPO generators.
- The strongest verification setting relies on `256` candidates, which improves accuracy but raises inference cost relative to top-1 decoding.

## Concepts Extracted

- [[large-language-model]]
- [[mathematical-reasoning]]
- [[process-reward-model]]
- [[outcome-reward-model]]
- [[process-supervision]]
- [[automatic-process-annotation]]
- [[best-of-n-selection]]
- [[self-consistency]]
- [[step-by-step-ppo]]
- [[monte-carlo-tree-search]]

## Entities Extracted

- [[math-shepherd]]
- [[peiyi-wang]]
- [[lei-li-hku]]
- [[zhihong-shao]]
- [[r-x-xu]]
- [[damai-dai]]
- [[yifei-li]]
- [[deli-chen-deepseek]]
- [[y-wu]]
- [[zhifang-sui]]
- [[gsm8k]]
- [[math]]
- [[metamath]]
- [[mistral-7b]]
- [[deepseek-67b]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
