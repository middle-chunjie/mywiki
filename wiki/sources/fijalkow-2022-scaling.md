---
type: source
subtype: paper
title: Scaling Neural Program Synthesis with Distribution-Based Search
slug: fijalkow-2022-scaling
date: 2026-04-20
language: en
tags: [program-synthesis, neural-program-synthesis, symbolic-search, pcfg, sygus]
processed: true

raw_file: raw/papers/fijalkow-2022-scaling/paper.pdf
raw_md: raw/papers/fijalkow-2022-scaling/paper.md
bibtex_file: raw/papers/fijalkow-2022-scaling/paper.bib
possibly_outdated: false

authors:
  - Nathanaël Fijalkow
  - Guillaume Lagarde
  - Théo Matricon
  - Kevin Ellis
  - Pierre Ohlmann
  - Akarsh Nayan Potta
year: 2022
venue: AAAI 2022
venue_type: conference
arxiv_id:
doi: 10.1609/aaai.v36i6.20616
url: https://ojs.aaai.org/index.php/AAAI/article/view/20616
citation_key: fijalkow2022scaling
paper_type: method

read_status: unread

domain: program-synthesis
---

## Summary

The paper studies the search stage of neural program synthesis rather than the neural predictor itself. It formalizes a setting where input-output examples are first mapped to a probabilistic context-free grammar (PCFG), then a symbolic algorithm searches the induced program distribution. Within this framework, the authors define loss as the expected number of programs emitted before finding a target drawn from the distribution, derive optimality criteria for both enumeration and sampling, and introduce two concrete algorithms: Heap Search and SQRT Sampling. They also propose a grammar splitter for parallel execution. Across random and machine-learned PCFGs, Heap Search is consistently the strongest practical method, while SQRT Sampling provides a theoretically grounded, lightweight alternative with strong parallel scaling.

## Problem & Motivation

Neural program synthesis systems often spend far more time in combinatorial search than in neural inference, so better learned models alone do not solve the scaling problem. The paper targets this bottleneck directly: given a neural network that scores grammar productions from input-output examples, how should one search the resulting program space as efficiently as possible? The authors want methods that remain sound and complete even when predictions are imperfect, integrate naturally with syntax-guided synthesis pipelines, and exploit parallel hardware without redundant work.

## Method

- **Two-stage pipeline**: compile a typed [[domain-specific-language]] with syntactic constraints into a CFG, then label productions with neural probabilities to obtain a [[probabilistic-context-free-grammar]] over candidate programs.
- **Distribution-based objective**: define a search algorithm's loss as `L(A, D) = E_{x ~ D}[inf { n in N : A(n) = x }]`, i.e. the expected number of emitted programs before recovering a target sampled from `D`.
- **Loss-optimal enumeration criterion**: any algorithm that outputs each program exactly once and in non-increasing probability order is optimal under this loss, motivating explicit probability-ordered enumeration.
- **Heap Search**: a bottom-up [[enumerative-search]] algorithm that maintains, for each non-terminal `T`, a heap `Heap_T`, a successor table `Succ_T`, and a seen set `Seen_T`; `Query(T, x)` returns the successor of `x` by popping the best candidate and recursively composing successor subprograms.
- **Heap Search complexity**: after the first `i` programs from a non-terminal have been generated, querying the successor runs in `O(log i)` time; the method is proven to enumerate all programs exactly once in non-increasing probability order.
- **SQRT Sampling**: a [[sqrt-sampling]] strategy that samples from `sqrt(D)(x) = sqrt(D(x)) / sum_y sqrt(D(y))`; when `sum_x sqrt(D(x)) < infinity`, this is loss-optimal among all memoryless sampling algorithms.
- **PCFG construction for SQRT Sampling**: for acyclic PCFGs, build a grammar for `sqrt(D)` by taking square roots of transition probabilities and globally renormalizing via the partition function `Z`.
- **Parallel grammar splitter**: partition a PCFG into `k` disjoint sub-PCFGs with near-balanced probability mass; partition quality is `alpha = max_s mass(s) / min_s mass(s)`, and the experiments target `alpha_desired = 1.05`.
- **Experimental setup**: evaluate on DeepCoder and DreamCoder list-processing DSLs with program depth `6`; the learned-PCFG experiments use a one-layer GRU with `s_GRU = 10`, a 3-layer MLP with hidden size `s_MLP = 64`, `L_max = 10`, `L_in = [-30, 30]`, Adam with `lr = 0.001`, batch size `128`, and `10,000` synthetic training problems.

## Key Results

- On random PCFGs, Heap Search generates `2.35x` more programs than `A*` while preserving the same optimal probability ordering.
- With grammar splitting, scaling from `2` to `6` CPUs yields `2.8x` more generated programs for SQRT Sampling and `7.6x` for Heap Search.
- On `137` filtered DreamCoder tasks with machine-learned PCFGs, Heap Search solves `97` tasks before timeout, compared with `85` for `A*` and `83` for Threshold.
- Throughput on the learned-PCFG benchmark is `38,735` programs/s for Heap Search versus `6,071` for `A*`, roughly a `6.4x` speedup.
- SQRT Sampling reaches `14,020` programs/s and performs only slightly worse than `A*` on solved-task count despite being memoryless and allowing repeated samples.

## Limitations

- The empirical evaluation is restricted to synthesis from input-output examples in list-processing DSLs, especially a filtered DreamCoder subset with only `int list -> int list` tasks.
- The PCFG construction used for SQRT Sampling assumes acyclic grammars for tractable normalization, so the implementation story is narrower than the general theorem statement.
- The paper focuses on search-time behavior and uses a relatively simple predictor (`1`-layer GRU + `3`-layer MLP trained for `1` epoch), so it does not test interaction with stronger modern neural code models.
- The authors explicitly note that the algorithms do not straightforwardly extend to neural autoregressive program models such as RobustFill-style decoders.
- Heap Search benefits from cached partial evaluations but can therefore become memory hungry in large search spaces.

## Concepts Extracted

- [[program-synthesis]]
- [[neural-program-synthesis]]
- [[domain-specific-language]]
- [[syntax-guided-synthesis]]
- [[probabilistic-context-free-grammar]]
- [[distribution-based-search]]
- [[enumerative-search]]
- [[heap-search]]
- [[sqrt-sampling]]
- [[grammar-splitter]]

## Entities Extracted

- [[nathanael-fijalkow]]
- [[guillaume-lagarde]]
- [[theo-matricon]]
- [[kevin-ellis]]
- [[pierre-ohlmann]]
- [[akarsh-nayan-potta]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
