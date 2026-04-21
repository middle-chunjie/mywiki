---
type: source
subtype: paper
title: "WizardCoder: Empowering Code Large Language Models with Evol-Instruct"
slug: luo-2023-wizardcoder-2306-08568
date: 2026-04-20
language: en
tags: [llm, code-generation, instruction-tuning, synthetic-data, benchmark]
processed: true

raw_file: raw/papers/luo-2023-wizardcoder-2306-08568/paper.pdf
raw_md: raw/papers/luo-2023-wizardcoder-2306-08568/paper.md
bibtex_file: raw/papers/luo-2023-wizardcoder-2306-08568/paper.bib
possibly_outdated: true

authors:
  - Ziyang Luo
  - Can Xu
  - Pu Zhao
  - Qingfeng Sun
  - Xiubo Geng
  - Wenxiang Hu
  - Chongyang Tao
  - Jing Ma
  - Qingwei Lin
  - Daxin Jiang
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2306.08568
doi:
url: http://arxiv.org/abs/2306.08568
citation_key: luo2023wizardcoder
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature. The paper presents WizardCoder, a code large language model obtained by adapting WizardLM's Evol-Instruct pipeline to the code domain and fine-tuning [[starcoder]] 15B on evolved [[code-alpaca]] data. The core idea is that more complex, code-specific instructions can teach a strong pretrained model to follow programming requests better than raw code pretraining alone. The resulting model is evaluated on HumanEval, HumanEval+, MBPP, and DS-1000, where it substantially improves over open-source baselines and, on some single-attempt settings, even exceeds larger closed models such as Claude and Bard. The work is an early but influential demonstration that instruction-data engineering can materially shift code-LLM capability.

## Problem & Motivation

Open-source code LLMs had strong pretraining but comparatively weak instruction-following behavior, leaving a large gap to systems such as ChatGPT on user-facing coding tasks. The paper argues that the missing ingredient is not only more raw code, but higher-quality instruction data tailored to the code domain. It therefore asks whether code-specific evolution of simple seed instructions can make a pretrained model better at code synthesis, debugging-oriented requests, and constraint-sensitive generation without changing the underlying base architecture.

## Method

- **Base model and seed data**: start from [[starcoder]] `15B` and initialize the instruction corpus with `20,000` examples from [[code-alpaca]].
- **Code-specific Evol-Instruct**: adapt [[evol-instruct]] to code by simplifying the prompt template to placeholders like `` `${question}` `` and `` `${method}` ``, removing general-domain transformations such as deepening, complicating input, and in-breadth evolving, and adding code-oriented transformations for [[code-debugging]] and time-space complexity constraints.
- **Iterative data growth**: evolve the seed set across multiple rounds, merging new samples with all prior rounds; the paper reports dataset sizes of about `38k`, `58k`, `78k`, and `98k` after rounds 1 to 4 respectively, and selects the round-3 `78k` corpus as the final training set.
- **Training recipe**: fine-tune with batch size `512`, sequence length `2048`, `200` optimization steps, `30` warmup steps, learning rate `2e-5`, cosine scheduler, and `fp16` mixed precision.
- **Model selection**: after each evolution round, evaluate on HumanEval and keep the checkpoint with the best `pass@1`; stop adding evolution rounds once performance drops.
- **Evaluation protocol**: use greedy decoding for the single-attempt HumanEval leaderboard comparison; for `pass@1` estimation on HumanEval and MBPP, sample with `temperature = 0.2` and `top_p = 0.95`; on DS-1000, use `temperature = 0.2`, `top_p = 0.5`, `max_length = 1024`, and average over `n = 40` samples.
- **Benchmarks and oracles**: evaluate under [[execution-based-evaluation]] on HumanEval (`164` problems, `9.6` tests/problem on average), HumanEval+ (`774.8` tests/problem), MBPP (`500` problems), and DS-1000 (`1,000` workflows across seven libraries).
- **Source caveat**: the local `paper.md` extraction states that five evolution types were used, but the rendered list itself is missing from the markdown dump, so only the visible prompt changes and hyperparameters are recorded here.

## Key Results

- HumanEval `pass@1 = 57.3`, beating StarCoder's `33.6` by `+22.3`.
- MBPP `pass@1 = 51.8`, beating StarCoder's `43.6` by `+8.2`.
- On the single-attempt HumanEval leaderboard, WizardCoder reaches `59.8`, exceeding Claude-Plus `53.0` and Bard `44.5`, while still trailing GPT-4 `67.0`.
- On DS-1000, WizardCoder improves over StarCoder from `26.0` to `29.2` in completion mode and from `25.4` to `32.8` in insertion mode.
- Ablation over evolution rounds peaks at the third round with a `78k` evolved dataset; the fourth round expands to `98k` but no longer improves HumanEval.

## Limitations

- The paper explicitly notes that WizardCoder still remains well below GPT-4 on code generation.
- Comparisons to several closed models reuse external benchmark reports rather than running every baseline under a single controlled setup.
- The evaluation suite is dominated by benchmarked code-generation tasks and does not test broader software-engineering workflows such as long-horizon editing, repository reasoning, or interactive debugging loops.
- Although the data synthesis introduces complexity constraints, the paper does not directly benchmark runtime or memory efficiency as a first-class target.
- The broader-impact section acknowledges that the model can still generate harmful or misleading outputs.

## Concepts Extracted

- [[large-language-model]]
- [[code-generation]]
- [[fine-tuning]]
- [[instruction-tuning]]
- [[evol-instruct]]
- [[self-instruct]]
- [[synthetic-data]]
- [[code-debugging]]
- [[code-efficiency]]
- [[execution-based-evaluation]]
- [[functional-correctness]]
- [[pass-at-k]]

## Entities Extracted

- [[ziyang-luo]]
- [[can-xu]]
- [[pu-zhao]]
- [[qingfeng-sun]]
- [[xiubo-geng]]
- [[wenxiang-hu]]
- [[chongyang-tao-microsoft]]
- [[jing-ma-hkbu]]
- [[qingwei-lin]]
- [[daxin-jiang]]
- [[microsoft]]
- [[hong-kong-baptist-university]]
- [[starcoder]]
- [[code-alpaca]]
- [[humaneval]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
