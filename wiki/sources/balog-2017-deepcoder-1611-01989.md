---
type: source
subtype: paper
title: "DeepCoder: Learning to Write Programs"
slug: balog-2017-deepcoder-1611-01989
date: 2026-04-20
language: en
tags: [program-synthesis, inductive-program-synthesis, neural-guided-search, domain-specific-language, deep-learning]
processed: true

raw_file: raw/papers/balog-2017-deepcoder-1611-01989/paper.pdf
raw_md: raw/papers/balog-2017-deepcoder-1611-01989/paper.md
bibtex_file: raw/papers/balog-2017-deepcoder-1611-01989/paper.bib
possibly_outdated: false

authors:
  - Matej Balog
  - Alexander L. Gaunt
  - Marc Brockschmidt
  - Sebastian Nowozin
  - Daniel Tarlow
year: 2017
venue: arXiv
venue_type: preprint
arxiv_id: 1611.01989
doi:
url: https://arxiv.org/abs/1611.01989
citation_key: balog2017deepcoder
paper_type: method

read_status: unread

domain: program-synthesis
---

## Summary

DeepCoder proposes a neural-guided inductive program synthesis pipeline for recovering small programs from input-output examples. The core idea is not to predict source code directly, but to train a neural model on synthetically generated DSL programs and use its predictions about which high-level functions appear in the target program to guide symbolic search. The system combines a compact list-processing DSL, a feed-forward encoder over sets of examples, and search procedures such as DFS, sort-and-add enumeration, Sketch, and Lambda^2. Across multiple settings, the learned guidance yields substantial runtime reductions over unguided baselines, often by an order of magnitude or more, while solving tasks comparable to the simplest programming-competition problems.

## Problem & Motivation

The paper studies inductive program synthesis from input-output examples, where the system must recover a human-readable program consistent with observed behavior. Prior differentiable-interpreter approaches used gradient descent over relaxed program spaces, but they did not transfer knowledge effectively across synthesis tasks and underperformed strong discrete search methods. DeepCoder argues that the main opportunity for machine learning is to learn reusable search guidance from a large synthetic corpus, transforming program synthesis from isolated per-task optimization into supervised prediction of search-relevant program attributes.

## Method

- **LIPS formulation**: the paper instantiates Learning Inductive Program Synthesis by combining a DSL, a data-generation process, a model for `q(a | E)`, and a search procedure guided by predicted attributes `a = A(P)`.
- **DSL design**: programs are straight-line sequences of list/integer operations over fresh variables. The DSL contains first-order functions such as `Head`, `Last`, `Take`, `Drop`, `Access`, `Minimum`, `Maximum`, `Reverse`, `Sort`, `Sum`, and higher-order functions such as `Map`, `Filter`, `Count`, `ZipWith`, `Scanl1`.
- **Attribute choice**: the predicted attribute vector is binary presence/absence of DSL functions, including higher-order functions and lambdas. The full DSL exposes `C = 34` function labels.
- **Data generation**: the training corpus is built by enumerating DSL programs, pruning redundant or shorter-equivalent ones, then generating valid input-output pairs by backward-propagating value constraints and executing the retained programs.
- **Encoder**: each input-output example is padded to `L = 20`; integers are embedded with size `E = 20`; type indicators are one-hot encoded; the concatenated representation is passed through `H = 3` hidden layers with `K = 256` sigmoid units, then pooled across examples by arithmetic averaging.
- **Decoder**: a learned `C x K` projection maps the pooled representation to `C` logits, interpreted as marginal probabilities that each DSL function appears in the target program.
- **Training setup**: the model is trained with negative cross entropy over independent binary labels, treating function prediction as multi-label classification over observed DSL attributes.
- **Search integration**: predicted marginals are used to order DFS expansions and to drive sort-and-add search for enumeration, [[sketch]], and [[lambda-squared]], where the active function set grows in decreasing probability order.
- **Inference regime**: experiments use `M = 5` input-output examples per task with integer magnitudes bounded by `256`; the appendix reports an optimized DFS implementation exploring roughly `3 x 10^6` programs per second via prefix-result caching.

## Key Results

- On `T = 3` programs with a search space around `2 x 10^6`, DeepCoder reduces timeout-to-solve for DFS from `41 ms` to `2.7 ms` at 20% solved (`15.2x`) and for sort-and-add enumeration from `80 ms` to `1.3 ms` (`62.2x`).
- On the same `T = 3` setting, guidance accelerates `Lambda^2` from `18.9 s` to `0.23 s` at 20% solved (`80.4x`) and brings Sketch from over `10^3 s` to `2.13 s` (`>467x`).
- In the main `T = 5` experiment with search space on the order of `10^10`, a network trained on `T = 4` programs cuts DFS timeout at 20% solved from `163 s` to `24 s` (`6.8x`).
- In that `T = 5` setting, sort-and-add enumeration improves from `8181 s` to `9 s` at 20% solved (`907x`), while `Lambda^2` improves from `463 s` to `48 s` (`9.6x`).
- The method generalizes across program lengths: models trained on shorter programs still provide useful guidance for longer ones, whereas the paper argues a fixed-length [[sequence-to-sequence]] decoder would not be expected to generalize in the same way.
- A purely neural decoder with [[beam-search]] is much weaker: it is competitive only for very small search spaces with `T <= 2`, and still needs `292 s` to solve 20% of `T = 3` tasks.

## Limitations

- The DSL only supports straight-line compositions of predefined primitives, so it cannot express many algorithmic patterns needed for harder programming-competition tasks, including richer looping and search procedures.
- The approach still depends on symbolic search; the neural model only reorders search and does not directly solve synthesis when the space becomes too expressive.
- Experiments rely on `5` relatively informative input-output examples with values up to magnitude `256`, which may be easier than real user-provided specifications.
- The decoder predicts independent marginals and ignores correlations among functions; the paper provides justification mainly for sort-and-add style procedures, not for all search algorithms.
- The fully neural token-level decoder is hard to train and underperforms hybrid neural-symbolic search, suggesting limited end-to-end generation capability in this setup.

## Concepts Extracted

- [[inductive-program-synthesis]]
- [[domain-specific-language]]
- [[neural-guided-search]]
- [[enumerative-search]]
- [[satisfiability-modulo-theories]]
- [[multi-label-classification]]
- [[sequence-to-sequence]]
- [[beam-search]]

## Entities Extracted

- [[matej-balog]]
- [[alexander-gaunt]]
- [[marc-brockschmidt]]
- [[sebastian-nowozin]]
- [[daniel-tarlow]]
- [[microsoft-research]]
- [[university-of-cambridge]]
- [[deepcoder]]
- [[sketch]]
- [[lambda-squared]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
