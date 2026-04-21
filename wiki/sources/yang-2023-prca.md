---
type: source
subtype: paper
title: "PRCA: Fitting Black-Box Large Language Models for Retrieval Question Answering via Pluggable Reward-Driven Contextual Adapter"
slug: yang-2023-prca
date: 2026-04-20
language: en
tags: [rag, retrieval, question-answering, reinforcement-learning, adapter]
processed: true

raw_file: raw/papers/yang-2023-prca/paper.pdf
raw_md: raw/papers/yang-2023-prca/paper.md
bibtex_file: raw/papers/yang-2023-prca/paper.bib
possibly_outdated: true

authors:
  - Haoyan Yang
  - Zhitao Li
  - Yong Zhang
  - Jianzong Wang
  - Ning Cheng
  - Ming Li
  - Jing Xiao
year: 2023
venue: EMNLP 2023
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.326
url: https://aclanthology.org/2023.emnlp-main.326
citation_key: yang2023prca
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

PRCA inserts a trainable adapter between a frozen retriever and a frozen black-box generator for retrieval question answering. The adapter is initialized from BART-Large, first trained to extract context from retrieved documents, then optimized with reward-driven reinforcement learning so that its distilled context improves downstream answer quality without tuning the LLM itself. The reward is derived from answer-level `ROUGE-L` and used in a PPO-style objective with a token-weighted return estimator, so the black-box generator is called once per generated context instead of once per token. Across `75` retriever-generator configurations on SQuAD, HotpotQA, and TopiOCQA, PRCA improves `71` settings and helps most on harder multi-hop and conversational QA cases.

## Problem & Motivation

The paper targets retrieval question answering settings where a retriever supplies documents and a generator answers from that evidence, but the generator is increasingly a large language model that is either too expensive to fine-tune or only exposed through an API. Prior retrieval-augmented methods often assume white-box access to logits or frequent interaction with the generator, which is impractical for closed models. The authors therefore aim to keep both retriever and generator frozen, learn a lightweight module that rewrites or distills the retrieved context, and use the black-box generator's answer quality itself as the training signal so that the adapter can fit generator-specific preferences while reducing hallucination and input length.

## Method

- **Placement and input**: PRCA is a pluggable adapter inserted between a frozen retriever and a frozen generator. It takes the question plus retrieved `Top-K` documents as input sequence `S_input` and emits a shorter context `C_extracted` for answer generation.
- **Stage 1: contextual extraction**: the adapter is initialized from `BART-Large` pretrained on CNN/DailyMail and optimized with supervised extraction loss `min_theta L(theta) = -(1/N) sum_i C_truth^(i) log(f_PRCA(S_input^(i); theta))`, where the target is context-rich text useful for answering.
- **Stage 2: reward-driven alignment**: because the generator is a black box, PRCA is further trained with reinforcement learning so that the extracted context maximizes downstream answer quality while staying close to the stage-1 policy.
- **PPO-style objective**: the paper uses `J(theta) = E_t[min(r_t(theta) A_t^GAE, clip(r_t(theta), 1-epsilon, 1+epsilon) A_t^GAE)] - beta (V(s_t) - R_t)^2`, with `r_t(theta) = pi_theta(a_t|s_t) / pi_theta_ori(a_t|s_t)` and generalized advantage estimation `A_t^GAE(gamma, lambda) = sum_l (gamma lambda)^l delta_(t+l)^V`.
- **Black-box return estimator**: instead of token-level reward queries, the final context receives `R_EOS = ROUGE-L(O, O*) - beta * D_KL(pi_theta || pi_theta_ori)`, and token-level returns are approximated by `R_t = R_EOS * exp(pi_theta(a_t|s_t)) / sum_(t=1)^K exp(pi_theta(a_t|s_t))`.
- **Reward efficiency**: this estimator reduces reward-model invocations from once per token to roughly `1/K` of standard PPO, which is important when the generator is an API-served LLM.
- **Experimental setup**: the paper evaluates `5` retrievers (`BM25`, SentenceBert, `DPR`, `SimCSE`, Contriver) with `5` generators (`T5-large`, Phoenix-7B, Vicuna-7B, ChatGLM, GPT-3.5) over `3` datasets, always keeping retriever and generator frozen.
- **Hyperparameters**: reported settings include learning rate `5 x 10^-5`, batch size `1/2/4`, beam size `3`, temperature `1`, `topk = 0.0`, `topp = 1.0`, and `early_stopping = true`.

## Key Results

- PRCA improves `71/75` retriever-generator configurations, with average gains of `+3%` on SQuAD, `+6%` on HotpotQA, and `+9%` on TopiOCQA.
- The strongest reported single improvement is on TopiOCQA with SimCSE plus Vicuna, where the table reports `0.10 + 0.20`, i.e. a `20`-point absolute gain after adding PRCA.
- Vicuna gains an average of `14%` on TopiOCQA across the five retrievers, suggesting the adapter helps most when the QA setting is conversational and distractor-heavy.
- Parameter-efficiency analysis reports that a roughly `0.4B`-parameter PRCA can boost the best baseline generators enough to yield improvements of `12.0%`, `27.1%`, and `64.5%` across the three datasets while remaining far smaller than GPT-3.5.
- Inference-speed measurements on A100 show `126`, `231`, and `492` tokens/s for batch sizes `1`, `2`, and `4`, respectively.
- The case-study and reward-trajectory analysis indicate that PRCA can let the generator reach the correct answer with about `4x` fewer tokens by filtering distracting retrieved text.
- Negative results remain: on SQuAD, T5 is hurt in some settings, including `-0.03` with BM25, `-0.06` with SentenceBert, and `-0.08` with Contriver.

## Limitations

- The reward comes from the downstream generator, so PRCA must be retrained for different generators instead of transferring cleanly across all black-box LLMs.
- The reward-driven stage can suffer from convergence instability, which the paper flags as a practical issue for stable training.
- Because PRCA is only a pluggable middle module, it cannot jointly repair poor retriever behavior; weak retrieval quality still bottlenecks the full system.
- Evaluation is limited to three QA datasets and a fixed `Top-5` retrieval setting, so generalization to newer LLMs, longer contexts, and broader domains remains untested in this paper.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[retrieval-question-answering]]
- [[large-language-model]]
- [[black-box-model]]
- [[pluggable-contextual-adapter]]
- [[context-distillation]]
- [[reinforcement-learning]]
- [[proximal-policy-optimization]]
- [[reward-model]]
- [[hallucination]]

## Entities Extracted

- [[haoyan-yang]]
- [[zhitao-li]]
- [[yong-zhang]]
- [[jianzong-wang]]
- [[ning-cheng]]
- [[ming-li]]
- [[jing-xiao]]
- [[ping-an-technology-shenzhen]]
- [[new-york-university]]
- [[university-of-maryland]]
- [[squad]]
- [[hotpotqa]]
- [[topiocqa]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
