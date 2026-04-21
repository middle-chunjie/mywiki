---
type: source
subtype: paper
title: "IterResearch: Rethinking Long-Horizon Agents with Interaction Scaling"
slug: chen-2025-iterresearch-2511-07327
date: 2026-04-20
language: en
tags: [agents, long-horizon, reasoning, reinforcement-learning, web-research]
processed: true

raw_file: raw/papers/chen-2025-iterresearch-2511-07327/paper.pdf
raw_md: raw/papers/chen-2025-iterresearch-2511-07327/paper.md
bibtex_file: raw/papers/chen-2025-iterresearch-2511-07327/paper.bib
possibly_outdated: false

authors:
  - Guoxin Chen
  - Zile Qiao
  - Xuanzhong Chen
  - Donglei Yu
  - Haotian Xu
  - Wayne Xin Zhao
  - Ruihua Song
  - Wenbiao Yin
  - Huifeng Yin
  - Liwen Zhang
  - Kuan Li
  - Minpeng Liao
  - Yong Jiang
  - Pengjun Xie
  - Fei Huang
  - Jingren Zhou
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2511.07327
doi: 10.48550/ARXIV.2511.07327
url: https://arxiv.org/abs/2511.07327
citation_key: chen2025iterresearch
paper_type: method

read_status: unread

domain: agents
---

## Summary

The paper proposes IterResearch, a long-horizon research-agent paradigm that replaces a single ever-growing context with iterative workspace reconstruction. Each round keeps only the question, an evolving report, and the latest tool interaction, casting deep research as a [[markov-decision-process]] and preserving bounded working memory. On top of this design, the authors introduce [[efficiency-aware-policy-optimization]] (EAPO), which applies geometrically discounted rewards to prefer shorter successful trajectories and uses adaptive downsampling for distributed RL. Across six deep-research benchmarks, the resulting IterResearch-30B-A3B model reports an average gain of 14.5 percentage points over prior open-source agents, strong interaction scaling up to 2048 turns, and prompt-level improvements over ReAct even without additional training.

## Problem & Motivation

Existing open-source deep-research systems mostly follow a mono-contextual design: they append every search result, observation, and reasoning step into one continuously growing prompt. The paper argues that this creates two structural failures on long-horizon tasks: context suffocation, where remaining room for fresh reasoning keeps shrinking, and noise contamination, where early mistakes and irrelevant evidence remain permanently in context. IterResearch is motivated by the claim that long-horizon research needs periodic synthesis and strategic forgetting rather than unlimited accumulation. The goal is to maintain stable reasoning capacity at arbitrary interaction depth while still preserving task-relevant information.

## Method

- **Iterative state design**: model the agent as an extended [[markov-decision-process]] `⟨S, D, E, T⟩` with state `s_t = (q, M_t, {a_(t-1), TR_(t-1)})`, where `q` is the question, `M_t` is an evolving report, and `{a_(t-1), TR_(t-1)}` is the latest action-response pair.
- **Structured decision output**: at each round the policy emits `d_t = (Think_t, M_(t+1), a_t)`, so reasoning, memory update, and next action are generated jointly rather than through a separate memory module.
- **Workspace reconstruction**: the transition reconstructs the next state as `s_(t+1) = (q, M_(t+1), {a_t, TR_t})`, contrasting mono-contextual growth `O(t)` with an iterative workspace that remains `O(1)` in structure.
- **Strategic forgetting**: full historical trajectories are discarded after each round; only synthesized findings survive through `M_t`, which is intended to filter noise and preserve decision-relevant evidence.
- **Efficiency-aware reward shaping**: EAPO assigns each round reward `r_t = γ^(T-t) · R_T`, where `R_T ∈ {0, 1}` is terminal correctness and `γ = 0.995` in the analysis example, biasing learning toward shorter successful trajectories.
- **Multi-round RL corpus**: one rollout with `T_i` rounds yields `T_i` training samples `(s_(i,t), d_(i,t), r_(i,t))`, producing a corpus `C` much richer than trajectory-level supervision.
- **Adaptive downsampling for distributed training**: the usable corpus is truncated to `|C_train| = floor(|C| / DP_size) × DP_size` so all workers receive balanced sample counts with typically `<1%` data loss.
- **Optimization objective**: EAPO is implemented on top of [[group-sequence-policy-optimization]] with group-normalized advantages across all rounds from the `G = 16` rollouts of the same question.
- **Backbone and training pipeline**: the main agent is built on [[qwen3-30b-a3b]] with `max context length = 40960`, first via [[rejection-sampling-fine-tuning]] on 30K QA pairs synthesized into 110K trajectories, then RL on 4,096 questions selected from the `20%-60%` success band.
- **Key hyperparameters**: SFT uses `lr = 1e-5`, `batch size = 512`, `epochs = 3`, `warmup ratio = 0.03`; RL uses `lr = 1e-6`, `batch size = 16`, `temperature = 1.0`, `top-p = 0.95`, `KL coeff = 0`, `entropy coeff = 0`, and training `T_max = 32`.
- **Inference budget scaling**: benchmark-specific maximum rounds are `32` for GAIA, `64` for HLE and BrowseComp-zh, and `256` for BrowseComp, while the scaling study extends the cap to `2048` turns.

## Key Results

- Across HLE, BrowseComp, BrowseComp-zh, GAIA, Xbench-DS, and SEAL-0, IterResearch-30B-A3B reports `28.8 / 37.3 / 45.2 / 72.8 / 71.0 / 39.6`, beating the strongest open-source baseline by `8.8 / 20.1 / 15.8 / 8.7 / 15.0 / 18.9` points respectively.
- Averaged over the six benchmarks, the paper reports a `+14.5pp` improvement over prior open-source deep-research agents.
- On proprietary-system comparisons, IterResearch exceeds OpenAI DeepResearch on HLE (`28.8` vs `26.6`) and BrowseComp-zh (`45.2` vs `42.9`), while remaining competitive on BrowseComp (`37.3` vs `51.5`) and GAIA (`72.8` vs `67.4`).
- In methodology ablations, EAPO reaches average score `49.1`, versus `48.3` for GSPO and `45.5` for SFT, while reducing average interactions from `19.13` to `18.04` relative to GSPO.
- In paradigm ablations, the iterative agent scores `49.1` average versus `36.5` for the mono-contextual agent, a `+12.6pp` gap despite the mono-contextual baseline using a larger `64K` context window against IterResearch's `40K`.
- Cross-paradigm transfer remains useful: adding iterative trajectories to mono-agent training lifts average score from `36.5` to `41.9` (`+5.4pp`).
- The interaction-scaling study on BrowseComp-200 reports accuracy rising from `5.5%` at `2` turns to `50.1%` at `2048` turns, while average actual usage remains only `80.1` turns under the `2048` budget.
- As a prompting strategy, IterResearch beats ReAct on long-horizon tasks, with gains up to `+12.7pp` for `o3` and `+19.2pp` for DeepSeek-V3.1 on BrowseComp.

## Limitations

- The paper evaluates only six benchmarks concentrated on deep-research and web information-seeking tasks; it does not test whether the same paradigm transfers to other agent settings such as coding or embodied interaction.
- Intermediate actions are not directly rewarded or verified; the RL signal remains terminal correctness only, so the method still depends on a coarse LLM-as-judge objective.
- The implementation is tightly coupled to a Qwen3-based stack and synthesized training trajectories, so the cost and stability of reproducing the method on other backbones are not established.
- Workspace reconstruction depends on the model's ability to compress evidence into `M_t`; the paper does not provide an independent faithfulness audit of whether important evidence is dropped during report updates.
- The paper is internally inconsistent on interaction-scaling numbers: the abstract/conclusion cite `3.5% → 42.5%`, while Section 4.4 reports `5.5% → 50.1%` on BrowseComp-200.

## Concepts Extracted

- [[markov-decision-process]]
- [[workspace-reconstruction]]
- [[long-horizon-reasoning]]
- [[deep-research-agent]]
- [[efficiency-aware-policy-optimization]]
- [[reward-shaping]]
- [[interaction-scaling]]
- [[retrieval-augmented-generation]]
- [[rejection-sampling-fine-tuning]]
- [[group-sequence-policy-optimization]]

## Entities Extracted

- [[guoxin-chen]]
- [[zile-qiao]]
- [[xuanzhong-chen]]
- [[donglei-yu]]
- [[haotian-xu]]
- [[wayne-xin-zhao]]
- [[ruihua-song]]
- [[wenbiao-yin]]
- [[huifeng-yin]]
- [[liwen-zhang]]
- [[kuan-li]]
- [[minpeng-liao]]
- [[yong-jiang]]
- [[pengjun-xie]]
- [[fei-huang]]
- [[jingren-zhou]]
- [[renmin-university-of-china]]
- [[tongyi-lab]]
- [[alibaba-group]]
- [[openrlhf]]
- [[qwen3-30b-a3b]]
- [[qwen3-235b-a22b]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
