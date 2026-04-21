---
type: source
subtype: paper
title: Composer 2 Technical Report
slug: research-2026-composer-2603-24477
date: 2026-04-20
language: en
tags: [agents, coding, reinforcement-learning, software-engineering, benchmarks]
processed: true

raw_file: raw/papers/research-2026-composer-2603-24477/paper.pdf
raw_md: raw/papers/research-2026-composer-2603-24477/paper.md
bibtex_file: raw/papers/research-2026-composer-2603-24477/paper.bib
possibly_outdated: false

authors:
  - Aaron Chan
  - Ahmed Shalaby
  - Alexander Wettig
  - Aman Sanger
  - Andrew Zhai
  - Anurag Ajay
  - Ashvin Nair
  - Charlie Snell
  - Chen Lu
  - Chen Shen
  - Emily Jia
  - Federico Cassano
  - Hanpeng Liu
  - Haoyu Chen
  - Henry Wildermuth
  - Jacob Jackson
  - Janet Li
  - Jediah Katz
  - Jiajun Yao
  - Joey Hejna
  - Josh Warner
  - Julius Vering
  - Kevin Frans
  - Lee Danilek
  - Less Wright
  - Lujing Cen
  - Luke Melas-Kyriazi
  - Michael Truell
  - Michiel de Jong
  - Naman Jain
  - Nate Schmidt
  - Nathan Wang
  - Niklas Muennighoff
  - Oleg Rybkin
  - Paul Loh
  - Phillip Kravtsov
  - Rishabh Yadav
  - Sahil Shah
  - Sam Kottler
  - Alexander M. Rush
  - Shengtong Zhang
  - Shomil Jain
  - Sriram Sankar
  - Stefan Heule
  - Stuart H. Sul
  - Sualeh Asif
  - Victor Rong
  - Wanqi Zhu
  - William Lin
  - Yuchen Wu
  - Yuri Volkov
  - Yury Zemlyanskiy
  - Zack Holbrook
  - Zhiyuan Zhang
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2603.24477
doi: 10.48550/arXiv.2603.24477
url: http://arxiv.org/abs/2603.24477
citation_key: research2026composer
paper_type: method

read_status: unread

domain: agents
---

## Summary

Composer 2 Technical Report describes a coding-specialized model built by combining domain-focused continued pretraining with large-scale asynchronous reinforcement learning on realistic software-engineering tasks. Starting from Kimi K2.5, a `1.04T`-parameter Mixture-of-Experts model with `32B` active parameters, the system adds a code-heavy `32k -> 256k` training pipeline, Multi-Token Prediction layers for speculative decoding, self-summarization for long-horizon work, and production-like training environments that mirror the deployed Cursor harness. The paper argues that close train-deploy alignment is as important as raw scale: the agent is optimized with the same class of tools and environment abstractions used at inference time. Composer 2 reaches `61.3%` on CursorBench, `73.7%` on SWE-bench Multilingual, and `61.7%` on Terminal-Bench.

## Problem & Motivation

The paper targets agentic software engineering, where a model must plan, inspect repositories, call tools, run commands, and make coherent edits over long horizons rather than merely emit short code snippets. The authors argue that public coding benchmarks only partially reflect this setting because they are often over-specified, narrow in task scope, and vulnerable to contamination. Composer 2 is motivated by the hypothesis that coding performance improves most reliably when training matches deployment conditions: realistic environments, the same harness structure as production, and rewards that capture correctness, efficiency, and usable interactive behavior.

## Method

- **Base model and specialization target**: start from Kimi K2.5, a `1.04T`-parameter / `32B`-active-parameter Mixture-of-Experts model, and specialize it for agentic coding rather than general chat.
- **Continued pretraining**: train on a large code-dominated mixture in three phases: bulk training at `32k` context length, a long-context extension to `256k`, and a short targeted SFT phase. Training uses MXFP8 on NVIDIA B300s with AdamW.
- **Predictive scaling study**: on Qwen3-Coder-30B-A3B, the authors test small/medium/large continued-pretraining budgets and report that lower post-SFT cross-entropy is predictive of stronger downstream RL reward.
- **Multi-Token Prediction**: add MTP heads for speculative decoding, initialize them from scratch, and train them with self-distillation to match the main LM head's logit distribution. MTP is trained jointly during the long-context and SFT phases.
- **RL objective**: use group-based policy-gradient training with multiple rollouts per prompt, fixed group size, single-epoch data usage, Adam optimization, and full-parameter updates in a highly asynchronous rollout/training regime.
- **Advantage and KL handling**: remove GRPO-style length standardization, do not normalize group advantages by their standard deviation, define `r = p(x) / q(x)`, and use the KL estimator `k1 = -log r` instead of `k3 = (r - 1) - log r` to avoid variance blow-up when policies diverge.
- **Overlength handling**: do not mask rollouts that exceed the maximum sequence length; instead, rely partly on self-summarization to control long trajectories.
- **Self-summarization**: chain multiple generations through model-written summaries and assign the final reward to all tokens in the chain so that both successful actions and useful summaries are reinforced.
- **Behavior shaping**: add auxiliary rewards for communication quality, coding style, and tool-use behavior, plus a nonlinear length penalty `C_length,k,q(x) = ((1 + kx)^(1-q) - 1) / (k(1-q))` over token and tool-use costs to encourage short solutions on easy tasks without suppressing long reasoning on hard tasks.
- **Parallel training stack**: use Context Parallelism as the main long-context axis and decouple Expert Parallelism from Tensor Parallelism. Report `EP = 8, CP = 2` in continued pretraining and `EP = 8, CP = 8` in RL, with DeepEP for token dispatch/combine.
- **Numerics and router alignment**: use per-token-scaled NVFP4 in the MoE forward pass, MXFP8 in the backward pass, and router replay so the training forward pass matches the experts selected during inference.
- **Environment fidelity**: train in production-like codebase environments running on Anyrun with the same class of tools used by deployed Cursor agents, including file editing, shell, search, and controlled egress.

## Key Results

- Composer 2 reaches `61.3%` on CursorBench-3, versus `44.2%` for Composer 1.5 and `38.0%` for Composer 1, a `37%` relative improvement over Composer 1.5 and `61%` over Composer 1.
- On public benchmarks, Composer 2 scores `73.7%` on SWE-bench Multilingual and `61.7%` on Terminal-Bench, improving over Composer 1.5 by `7.8%` and `13.8%`, respectively.
- Against its base model Kimi K2.5, Composer 2 improves from `36.0%` to `61.3%` on CursorBench and from `47.3%` to `61.7%` on Terminal-Bench, while SWE-bench Multilingual moves from `65.1%` in the authors' harness to `73.7%`.
- CursorBench is materially harder than public issue-fixing suites: the paper reports a median of `181` changed lines per task versus `7-10` for SWE-bench variants, and a median prompt length of `390` characters versus `1,185-3,055`.
- The RL infrastructure is built for scale, spanning `3` GPU regions and `4` CPU regions; each Anyrun cluster can schedule more than `500` pods per second.

## Limitations

- Much of the empirical evidence depends on internal assets such as CursorBench, FreshBench, private codebase perplexity, and production-like infrastructure, which limits external reproducibility.
- The report discloses many system choices but does not provide controlled ablations isolating how much each ingredient, such as self-summarization, router replay, behavior rewards, or CP/EP design, contributes to the final gains.
- Training data composition, exact reward mixes, and the full task distribution are only partially specified, so reproducing the recipe from the paper alone would be difficult.
- Evaluation is tightly focused on coding agents; the paper does not establish whether the specialization preserves or improves broader non-coding capabilities.

## Concepts Extracted

- [[continual-pretraining]]
- [[reinforcement-learning]]
- [[policy-gradient]]
- [[self-summarization]]
- [[multi-token-prediction]]
- [[speculative-decoding]]
- [[mixture-of-experts]]
- [[context-parallelism]]
- [[expert-parallelism]]
- [[benchmark-evaluation]]
- [[long-context-training]]

## Entities Extracted

- [[aaron-chan]]
- [[ahmed-shalaby]]
- [[alexander-wettig-cursor]]
- [[aman-sanger]]
- [[andrew-zhai]]
- [[anurag-ajay]]
- [[ashvin-nair]]
- [[charlie-snell-cursor]]
- [[chen-lu]]
- [[chen-shen]]
- [[emily-jia]]
- [[federico-cassano]]
- [[hanpeng-liu]]
- [[haoyu-chen]]
- [[henry-wildermuth]]
- [[jacob-jackson]]
- [[janet-li]]
- [[jediah-katz]]
- [[jiajun-yao]]
- [[joey-hejna]]
- [[josh-warner]]
- [[julius-vering]]
- [[kevin-frans]]
- [[lee-danilek]]
- [[less-wright]]
- [[lujing-cen]]
- [[luke-melas-kyriazi]]
- [[michael-truell]]
- [[michiel-de-jong]]
- [[naman-jain]]
- [[nate-schmidt]]
- [[nathan-wang]]
- [[niklas-muennighoff-cursor]]
- [[oleg-rybkin]]
- [[paul-loh]]
- [[phillip-kravtsov]]
- [[rishabh-yadav]]
- [[sahil-shah]]
- [[sam-kottler]]
- [[alexander-rush]]
- [[shengtong-zhang]]
- [[shomil-jain]]
- [[sriram-sankar]]
- [[stefan-heule]]
- [[stuart-sul]]
- [[sualeh-asif]]
- [[victor-rong]]
- [[wanqi-zhu]]
- [[william-lin]]
- [[yuchen-wu]]
- [[yuri-volkov]]
- [[yury-zemlyanskiy]]
- [[zack-holbrook]]
- [[zhiyuan-zhang]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
