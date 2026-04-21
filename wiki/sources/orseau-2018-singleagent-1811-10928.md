---
type: source
subtype: paper
title: Single-Agent Policy Tree Search With Guarantees
slug: orseau-2018-singleagent-1811-10928
date: 2026-04-20
language: en
tags: [agents, planning, tree-search, reinforcement-learning, sokoban]
processed: true

raw_file: raw/papers/orseau-2018-singleagent-1811-10928/paper.pdf
raw_md: raw/papers/orseau-2018-singleagent-1811-10928/paper.md
bibtex_file: raw/papers/orseau-2018-singleagent-1811-10928/paper.bib
possibly_outdated: true

authors:
  - Laurent Orseau
  - Levi H. S. Lelis
  - Tor Lattimore
  - Theophane Weber
year: 2018
venue: arXiv
venue_type: preprint
arxiv_id: 1811.10928
doi:
url: http://arxiv.org/abs/1811.10928
citation_key: orseau2018singleagent
paper_type: method

read_status: unread

domain: agents
---

## Summary

⚠ Possibly outdated: published 2018; re-verify against recent literature.

This paper introduces two policy-guided tree-search algorithms for deterministic single-agent problems: LevinTS, which expands nodes by minimizing `d_0(n) / pi(n)`, and LubyTS, which samples trajectories under a universal restart schedule. The key claim is that a policy over action sequences gives stronger search-time guarantees than a heuristic value estimate: LevinTS gets a strict upper bound on node expansions to the first goal, while LubyTS gets an expectation bound that benefits when many solution paths exist. On 1,000 procedurally generated Sokoban levels, a policy trained with A3C lets LevinTS solve all instances, and a simple `1%` uniform-noise mixture further reduces total expansions from `6.60e6` to `5.03e6`. The work connects reinforcement-learned policies with classical search guarantees.

## Problem & Motivation

The paper studies deterministic single-agent search problems where the objective is to find any goal-reaching trajectory, not to optimize an adversarial value function. The authors argue that Monte Carlo tree search is poorly matched to sparse-reward settings of this kind because it can collapse toward uniform exploration when rewards appear only at the end of a successful trajectory.

Their alternative is to treat a policy `pi` as the search prior. Because `pi` assigns probability mass to action sequences, the search cost can be tied directly to the probability of solving trajectories instead of to a heuristic estimate whose quality is hard to analyze theoretically. The aim is to obtain algorithms with formal guarantees while still exploiting learned guidance from reinforcement learning.

## Method

- **Problem setup**: nodes are action sequences `n in A*`, depth is `d_0(n)`, and the target set is `N^g = {n : T(n) in S^g}` in a deterministic transition system. A policy satisfies `pi(n_0) = 1` and `pi(n) = sum_{n' in C(n)} pi(n')`.
- **LevinTS**: expand the fringe node with minimum cost `d_0(n) / pi(n)`. For a Markovian policy, apply a state cut when the same state was already expanded with probability at least as large, which preserves best-first ordering over states.
- **LevinTS guarantee**: the first goal is found after at most `N(LevinTS, N^g) <= min_{n in N^g} d_0(n) / pi(n)` node expansions, adapting Levin search from program search to tree search.
- **Sampling baselines**: `multiTS` samples trajectories to fixed depth `d_max`, giving `E[N] <= d_max / pi^+_{d_max}` where `pi^+_{d_max}` is the cumulative probability of goal nodes up to depth `d_max`.
- **LubyTS**: sample trajectories with restart depths following `A6519`, yielding `E[N] <= min_d d + (d / pi^+_d) * (log_2(d / pi^+_d) + 6.1)`. This favors cases with many shallow-but-individually-unlikely solutions.
- **Policy robustness**: the paper analyzes Bayes mixtures and local mixing to avoid near-zero probabilities on crucial actions; experiments additionally test a noisy policy `tilde(pi)(a|n) = (1 - epsilon) pi(a|n) + epsilon / 4` with `epsilon = 0.01`.
- **Learned policy and evaluation setup**: the Sokoban policy network takes a `10 x 10 x 4` grid, uses conv layers `4 x 4 x 64` and `3 x 3 x 64`, then a `512`-unit ReLU layer and `4` action logits. Training uses A3C with reward `-0.1` per step, `+1` per box on goal, `-1` for reversing that event, terminal reward `+10`, discount `0.99`, RMSProp with epsilon `0.1` and decay `0.99`, entropy regularization `0.005`, four learning rates (`4e-4`, `2e-4`, `1e-4`, `5e-5`), and `3.5e9` training steps.

## Key Results

- On `1,000` generated Sokoban levels, `LevinTS` solved `1,000/1,000` with average solution length `39.8`, maximum length `106`, and `6,602,666` total node expansions.
- Adding `1%` uniform noise to the policy preserved `1,000/1,000` solved while reducing total expansions to `5,026,200` and slightly lowering average solution length to `39.5`.
- `LAMA` also solved `1,000/1,000`, but with longer solutions on average (`51.6` vs. `39.8`) despite fewer total expansions (`3,151,325`).
- The best sampling variants did not match LevinTS completeness: `LubyTS(512, 32) (*)` solved `896 +/- 2`, while `multiTS(200, 200) (*)` solved `895 +/- 3`.
- A uniform-policy searcher solved only `88/1,000` levels and needed `94,423,278` total expansions, showing that learned policy guidance is essential.
- On the easiest `998` instances, LevinTS used about `2.7e6` cumulative expansions versus LAMA's `3.1e6`, but a few hard levels dominated LevinTS's final total.

## Limitations

- The analysis and experiments focus on deterministic single-agent problems; stochastic extensions are discussed only at a high level.
- Empirical validation is limited to one domain, Sokoban, so generalization to other planning tasks is not established.
- LevinTS can require substantial memory for the priority queue and visited-state tracking, while LubyTS often returns much longer solutions.
- Both algorithms remain sensitive to policy miscalibration; if `pi` assigns near-zero mass to the right branch, the theoretical guarantees can still become impractically loose.

## Concepts Extracted

- [[single-agent-search]]
- [[policy-guided-search]]
- [[levin-search]]
- [[monte-carlo-tree-search]]
- [[best-first-search]]
- [[markovian-policy]]
- [[state-cut]]
- [[policy-mixing]]
- [[universal-restart-strategy]]

## Entities Extracted

- [[laurent-orseau]]
- [[levi-h-s-lelis]]
- [[tor-lattimore]]
- [[theophane-weber]]
- [[deepmind]]
- [[universidade-federal-de-vicosa]]
- [[a3c]]
- [[lama-planner]]
- [[fast-downward]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
