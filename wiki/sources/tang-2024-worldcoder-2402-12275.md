---
type: source
subtype: paper
title: "WorldCoder, a Model-Based LLM Agent: Building World Models by Writing Code and Interacting with the Environment"
slug: tang-2024-worldcoder-2402-12275
date: 2026-04-20
language: en
tags: [agents, llm, world-model, planning, program-synthesis]
processed: true

raw_file: raw/papers/tang-2024-worldcoder-2402-12275/paper.pdf
raw_md: raw/papers/tang-2024-worldcoder-2402-12275/paper.md
bibtex_file: raw/papers/tang-2024-worldcoder-2402-12275/paper.bib
possibly_outdated: false

authors:
  - Hao Tang
  - Darren Key
  - Kevin Ellis
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.12275
doi:
url: http://arxiv.org/abs/2402.12275
citation_key: tang2024worldcoder
paper_type: method

read_status: unread

domain: agents
---

## Summary

WorldCoder proposes a model-based LLM agent that learns an explicit symbolic world model as Python code from environment interaction traces rather than using the LLM itself as the simulator or planner. The learned program jointly represents a transition function and a goal-conditioned reward model, and is synthesized under both data-consistency and optimism constraints so that it explains observed trajectories while still supporting plans to positive reward. The system combines iterative code refinement, transfer across related environments, and planner-driven acting with value iteration or MCTS depending on task complexity. Across Sokoban, MiniGrid, and AlfWorld, the paper argues that code-based world models yield better sample efficiency, stronger transfer, and lower amortized LLM-query cost than deep RL or ReAct-style agents.

## Problem & Motivation

The paper targets a gap between deep model-based reinforcement learning and contemporary LLM agents. Deep RL can eventually learn competent world models, but usually needs large amounts of interaction data and offers limited interpretability. ReAct-style agents exploit pretrained priors, yet they keep the world model implicit in network weights and repeatedly query the LLM at action time. The authors want an alternative that learns quickly from sparse symbolic interaction, exposes its learned knowledge in human-auditable code, transfers that knowledge across related tasks, and can adapt to new goals by modifying the reward side of the world model rather than relearning the environment dynamics.

## Method

- **Problem formulation**: model the environment as a contextual MDP `M(c) = (S, A, T, R^c, \gamma)` where goals are natural-language contexts `c`, dynamics `T` are shared across goals, and episodes terminate when the goal is reached.
- **Symbolic state representation**: represent each state as a set of typed objects with string fields and coordinates such as `x` and `y`, effectively using an object-oriented state space amenable to code synthesis.
- **World model as code**: learn Python subroutines for a transition function `\hat{T}: S \times A \to S` and a reward model `\hat{R}: C \to (S \times A \times S \to \mathbb{R} \times \{0,1\})`, separating environment dynamics from goal-conditioned reward logic.
- **Learning objectives**: require both data fit `\phi_1`, meaning the program exactly replays all observed tuples `(s, a, r, s', c, d)`, and optimism `\phi_2`, meaning there exists a plan from an initial state that reaches positive reward under the current learned model.
- **Theoretical guarantee**: under realizability, Appendix A proves the number of actions needed to reach reward is bounded by `D_{S,A,T} \times (K_{\mathcal{T} \times \mathcal{R}} + 1)`, where `D_{S,A,T}` is MDP diameter and `K` is logical dimensionality of the program class.
- **Agent architecture**: maintain replay buffers `\mathcal{D}` and `\mathcal{D}_{sc}`; start synthesis after `MINDATASIZE = 10` actions; keep random exploration with probability `\epsilon = 5%`; otherwise act by planning with the current world model.
- **Program synthesis by refinement**: prompt an LLM with subsets of replay data to draft candidate code, then backprompt it with counterexamples whenever the program violates `\phi_1` or `\phi_2`, iteratively debugging rather than synthesizing from scratch each time.
- **REx-guided search**: choose which candidate program to refine using heuristic `h(\hat{T}, \hat{R})`, which scores the fraction of replay data satisfying `\phi_1` plus the fraction of initial states/contexts satisfying `\phi_2`, while balancing exploration and exploitation over partially successful programs.
- **Planning**: use depth-limited value iteration in simpler domains and `[[monte-carlo-tree-search]]` in harder sparse-reward domains; for AlfWorld, replace rollout estimates with a BM25-based heuristic and UCT score `Q(v') / N(v') + c \sqrt{2 \ln N(v) / N(v')}`.
- **Transfer mechanism**: reuse and refine previously synthesized code across curricula of related environments so that new dynamics require local edits rather than full resynthesis, and new goals trigger reward-model extension instead of relearning `\hat{T}`.

## Key Results

- In Sokoban, WorldCoder induces a usable world model from the first `50` actions and generalizes to levels with more boxes, whereas the paper reports deep model-free and model-based RL need `>1` million interactions to learn basic `2`-box Sokoban.
- Against a ReAct baseline on basic Sokoban, the paper reports only `15% ± 8%` success for ReAct, while WorldCoder amortizes LLM usage by front-loading about `400k` tokens of synthesis cost, roughly `$15`, and then acting without per-step LLM calls.
- In MiniGrid UnlockPickup, the optimism objective enables learning the correct world model from scratch in `<=100` actions, while PPO still fails to obtain sufficient positive-reward signal after `3 × 10^8` actions; the paper also states WorldCoder without optimism fails there as well.
- In AlfWorld, the system synthesizes world models with `250+` lines of Python, typically reaches reward in the first episode after around `20` exploratory steps, and the optimism objective is reported as necessary for any non-zero performance on the studied tasks.

## Limitations

- The method assumes deterministic dynamics; stochastic environments would require probabilistic program synthesis rather than plain deterministic code generation.
- It depends on symbolic, discrete state descriptions rather than raw perceptual inputs, so direct application to messy continuous environments is out of scope without extra perception modules or neurosymbolic machinery.
- Planning, not world-model learning alone, remains a bottleneck: the paper explicitly notes that hard Sokoban levels with `5+` boxes are not solved by the current setup.
- The transfer mechanism is still rudimentary, relying on iterative debugging of monolithic code instead of learned reusable libraries, abstractions, or subroutines.
- The paper likely benefits from pretrained prior knowledge in the underlying LLM, especially on domains like Sokoban, so gains are not purely from the symbolic learning objective.

## Concepts Extracted

- [[world-model]]
- [[large-language-model]]
- [[program-synthesis]]
- [[model-based-reinforcement-learning]]
- [[transfer-learning]]
- [[curriculum-learning]]
- [[monte-carlo-tree-search]]
- [[reward-model]]
- [[optimism-under-uncertainty]]
- [[transition-function]]
- [[goal-directed-exploration]]
- [[contextual-markov-decision-process]]

## Entities Extracted

- [[hao-tang]]
- [[darren-key]]
- [[kevin-ellis]]
- [[cornell-university]]
- [[worldcoder]]
- [[gpt-4]]
- [[sokoban]]
- [[minigrid]]
- [[alfworld]]
- [[bm25]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
