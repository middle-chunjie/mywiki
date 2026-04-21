---
type: source
subtype: paper
title: Generating Code World Models with Large Language Models Guided by Monte Carlo Tree Search
slug: dainese-2024-generating-2405-15383
date: 2026-04-20
language: en
tags: [llm, reinforcement-learning, world-models, planning, code-generation]
processed: true
raw_file: raw/papers/dainese-2024-generating-2405-15383/paper.pdf
raw_md: raw/papers/dainese-2024-generating-2405-15383/paper.md
bibtex_file: raw/papers/dainese-2024-generating-2405-15383/paper.bib
possibly_outdated: false
authors:
  - Nicola Dainese
  - Matteo Merler
  - Minttu Alakuijala
  - Pekka Marttinen
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2405.15383
doi:
url: http://arxiv.org/abs/2405.15383
citation_key: dainese2024generating
paper_type: method
read_status: unread
domain: agents
---

## Summary

The paper proposes Code World Models (CWMs): executable Python environment models synthesized by a large language model from textual environment descriptions plus offline trajectories, then used for model-based planning. To produce these programs, the authors introduce GIF-MCTS, a Monte Carlo Tree Search procedure with generate, improve, and fix actions that expands partial programs, scores them by transition-prediction accuracy, and feeds back failures to revise logic or repair bugs. They also introduce the Code World Models Benchmark (CWMB), spanning 18 reinforcement-learning environments with curated trajectories and documentation-derived descriptions. Across CWMB, APPS, and a simplified RTFM setting, GIF-MCTS consistently outperforms WorldCoder and shows that planning with code world models can be dramatically faster than repeatedly querying an LLM during inference.

## Problem & Motivation

The paper asks how to combine the language understanding and coding ability of LLMs with the efficiency and controllability required by model-based reinforcement learning. Directly using an LLM as a world model inside a planning loop is slow, expensive, and unreliable, while classical learned world models do not naturally exploit rich textual descriptions of environments. The authors therefore target a middle ground: generate an explicit Python world model that can ingest natural-language specifications, be verified against observed transitions, and then support fast downstream planning. This requires solving a difficult long-form code synthesis problem under sparse but structured feedback.

## Method

- **Code World Model formulation**: a synthesized environment implements `(\hat{s}', \hat{r}, \hat{d}) = code_environment.step(s, a)`, jointly predicting next state, reward, and done signal from the current state and action.
- **Evaluation objective**: CWM accuracy is `A = (1/N) Σ_i [ (1/3) 1[s'_i = \hat{s}'_i] + (1/3) 1[r_i = \hat{r}_i] + (1/3) 1[d_i = \hat{d}_i] ]`, computed on an offline dataset of transitions.
- **Problem setting**: the framework assumes deterministic, fully observable MDPs and uses an offline dataset `D = {(s, a, r, s', d)_i}` plus a textual environment description. In CWMB, most environments use `10` trajectories, typically `5` random and `5` higher-return demonstrations.
- **Tree representation**: each GIF-MCTS node stores a partial program state plus a rollout completion. Expanding a node appends `L = 2` newly generated code lines to the parent state and evaluates the completed program.
- **Search loop**: the method follows selection, expansion, evaluation, and backpropagation, using a modified UCT score `v_i + C * sqrt(ln N_i / (n_{a=a_i} + ε))` with `C = 0.1` and `ε = 1.0` so action types that are overused from the same parent are penalized.
- **Three action types**: `generate` samples alternative continuations, `improve` rewrites a full program using one failing example plus self-reflective error analysis, and `fix` repairs syntax/runtime failures from execution feedback.
- **Value estimation**: unexplored actions are scored by a learned interpolation `v_i = (w_G v_G + w_L v_L) / (w_G + w_L)` over global and local action statistics, with priors `v_g = 0.5` and `v_i = 0.55`, each initialized with count `2`.
- **Bug handling**: buggy nodes receive a temporary value starting at `0.99` so the search will attempt repair; up to `f = 3` sequential fixes are allowed before the branch is strongly discouraged.
- **Planning with synthesized models**: downstream planning uses vanilla MCTS for discrete-action environments with `I_mcts = 25`, `max_actions = 100`, `γ = 0.99`, and CEM for continuous actions with `T_cem = 100`, `I_cem = 20`, `N_cem = 1000`, `K_cem = 100`.

## Key Results

- On APPS Competition with Llama 3 70B, GIF-MCTS reaches `28.3 ± 1.4` strict accuracy, outperforming WorldCoder at `25.1 ± 1.4` and Zero-shot CoT at `23.2 ± 1.3`.
- On CWMB with Llama 3 70B and budget `50`, GIF-MCTS achieves discrete-environment accuracy/normalized return of `0.84 ± 0.03` / `0.76 ± 0.03`, versus WorldCoder's `0.79 ± 0.04` / `0.60 ± 0.04`.
- On the same CWMB split for continuous environments, Llama 3 + GIF-MCTS reaches `0.35 ± 0.03` accuracy and `0.22 ± 0.01` normalized return, versus `0.32 ± 0.03` and `0.19 ± 0.01` for WorldCoder.
- With GPT-4 Turbo and budget `10`, GIF-MCTS attains CWMB discrete accuracy/return `0.91 ± 0.08` / `0.81 ± 0.06` and continuous accuracy/return `0.40 ± 0.03` / `0.26 ± 0.01`, improving over WorldCoder in both splits.
- On the simplified RTFM task, GPT-4 Turbo + GIF-MCTS reaches a perfect `1.00 ± 0.00` accuracy and `1.00 ± 0.00` normalized return at budget `50`; WorldCoder remains at `0.64 ± 0.02` accuracy and `-0.06 ± 0.12` return.
- Planning with CWMs is vastly faster than using GPT-4 directly as a world model: for `CartPole-v1`, `2.2 s` versus `0.00005 s`, and for `Humanoid-v4`, `146.7 s` versus `0.0001 s`.
- Ablation results show bug fixing is crucial: removing the fix action drops CWMB continuous-action performance from `0.38 ± 0.04` to `0.21 ± 0.08` in accuracy and from `0.23 ± 0.02` to `0.10 ± 0.01` in normalized return.

## Limitations

- The framework assumes deterministic and fully observable environments; stochasticity and partial observability are explicitly left unresolved.
- It relies on having an environment description that can be translated into code, which is difficult for settings defined mainly by raw images or poorly documented dynamics.
- Continuous control and physics-heavy tasks remain challenging: Llama 3 could not synthesize executable CWMs for `Humanoid-v4` or `HumanoidStandup-v4`.
- GIF-MCTS depends on executable feedback and validation trajectories; when high-quality tests are unavailable, the method has no equally strong external reward signal.
- Code world models can be too rigid under changing dynamics, and the paper suggests modular rewriting or external simulators as future extensions rather than solving this directly.
- Compared with offline RL baselines such as CQL, CWMs are not uniformly stronger, especially on harder continuous-control environments.

## Concepts Extracted

- [[code-world-model]]
- [[model-based-reinforcement-learning]]
- [[offline-reinforcement-learning]]
- [[monte-carlo-tree-search]]
- [[cross-entropy-method]]
- [[reinforcement-learning]]
- [[markov-decision-process]]
- [[large-language-model]]
- [[code-synthesis]]
- [[program-repair]]
- [[chain-of-thought]]

## Entities Extracted

- [[nicola-dainese]]
- [[matteo-merler]]
- [[minttu-alakuijala]]
- [[pekka-marttinen]]
- [[aalto-university]]
- [[gpt-4-turbo]]
- [[apps-benchmark]]
- [[gymnasium]]
- [[worldcoder]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
