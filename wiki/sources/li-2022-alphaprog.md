---
type: source
subtype: paper
title: "ALPHAPROG: Reinforcement Generation of Valid Programs for Compiler Fuzzing"
slug: li-2022-alphaprog
date: 2026-04-20
language: en
tags: [compiler-fuzzing, reinforcement-learning, program-generation, software-testing, deep-learning]
processed: true

raw_file: raw/papers/li-2022-alphaprog/paper.pdf
raw_md: raw/papers/li-2022-alphaprog/paper.md
bibtex_file: raw/papers/li-2022-alphaprog/paper.bib
possibly_outdated: false

authors:
  - Xiaoting Li
  - Xiao Liu
  - Lingwei Chen
  - Rupesh Prajapati
  - Dinghao Wu
year: 2022
venue: AAAI 2022
venue_type: conference
arxiv_id:
doi: 10.1609/aaai.v36i11.21527
url: https://ojs.aaai.org/index.php/AAAI/article/view/21527
citation_key: li2022alphaprog
paper_type: method

read_status: unread
read_date:
rating:

domain: software-engineering
---

## Summary

The paper presents ALPHAPROG, a reinforcement-learning framework for compiler fuzzing that generates Brainfuck programs from scratch instead of relying on manually written grammars or seed corpora. Program generation is cast as an MDP in which a deep Q-network appends one of the eight Brainfuck tokens to the current prefix and receives rewards derived from compiler feedback, execution-trace coverage, and optionally cyclomatic complexity. The implementation uses an LSTM-based Q-network and compares four reward functions that trade off validity against diversity and control-flow complexity. On the BFC Brainfuck compiler, the best configuration reaches around 80% valid programs, exercises more than 100,000 basic blocks with 30,000 generated tests, outperforms AFL on both validity and coverage, and helps uncover two confirmed compiler bugs.

## Problem & Motivation

Compiler fuzzing is difficult because compilers consume highly structured inputs, so naively generated programs are often rejected before deeper compilation stages are exercised. Prior grammar-based generators require substantial expert effort and remain incomplete, while supervised neural generators depend on seed corpora that already encode valid language patterns. The paper targets a stricter setting: learning to synthesize valid and diverse programs from scratch, using only compiler messages and runtime traces as feedback, so that the generator can both maintain semantic/syntactic validity and improve testing efficacy.

## Method

- **Formulation**: model program generation as a Markov decision process over prefixes, where state `s_t` is the current Brainfuck string and an action appends one of 8 language tokens to obtain the next state.
- **Q-network**: use deep Q-learning rather than tabular Q-learning; the network embeds the current prefix with an LSTM of `128` neurons, followed by two fully connected layers of sizes `100` and `512` with ReLU activations, and an output layer of size `8`.
- **Learning target**: update the action-state estimate by minimizing the deviation between `Q(s_t, a_t)` and `r_t + gamma * max_a Q(s_{t+1}, a)`, with discounted rate `gamma = 1`.
- **Exploration**: apply an epsilon-greedy policy with `epsilon_max = 1`, `epsilon_min = 0.01`, and decay `(epsilon_max - epsilon_min) / 100000` after each prediction, so exploration effectively ends around episode `20,000`.
- **Optimization details**: initialize weights uniformly in `w in [0, 0.1]` and train with learning rate `alpha = 0.0001`.
- **Validity reward**: define `R_1` as `0` before the length cap, `-1` for compilation error, and `1` for compilation success; warnings and internal compiler errors are treated as valid outcomes because they reach deeper compilation stages.
- **Coverage reward**: define `R_2 = B(T_p) / sum_{rho in I'} B(T_rho)`, where `B(T_p)` is the number of unique basic blocks on the current execution trace and `I'` is the accumulated generated suite.
- **Combined rewards**: use `R_3` to add validity and coverage, and `R_4 = R_3 + C(p) / max(C(rho : rho in I'))` to further reward cyclomatic complexity for more complex control flow.
- **Instrumentation**: collect execution traces with Pin and compute accumulated basic-block coverage in a custom analysis tool while compiling generated Brainfuck programs against BFC.

## Key Results

- Under `Reward 1`, the valid-program rate reaches `100%` by about `20,000` generated programs, but coverage stops improving soon after convergence.
- Under `Reward 3`, the final valid rate rises to about `90%` while achieving the second-highest accumulated coverage among the four reward designs.
- Under `Reward 4`, ALPHAPROG reaches around `80%` valid programs, achieves the highest testing coverage, and exceeds `100,000` tested basic blocks with `30,000` generated programs.
- Generated program complexity increases substantially: the paper reports sample cyclomatic complexities of `2`, `11`, `32`, and `39`, and average complexity across reward settings increases from `4` to `18`.
- Against AFL with `30,000` programs from a single empty seed, AFL reaches only `35%` peak validity, `43,135` basic blocks, and `162` paths, while ALPHAPROG finds `2` confirmed bugs in BFC and covers far more compiler code.

## Limitations

- The evaluation is narrow: the main experiments target a single esoteric language (Brainfuck) and one industrial-grade Brainfuck compiler (BFC), so external validity to mainstream languages is not demonstrated.
- Scalability is limited for complex languages such as C; the paper explicitly notes that the search space becomes too large for the prototype to find valid programs efficiently without additional grammar guidance.
- Reward design remains heuristic: validity, coverage, and control-flow complexity are proxies for fuzzing effectiveness, and the paper says it is still hard to determine a principled stopping point for training.
- The framework depends on dynamic instrumentation and compiler execution traces, which adds engineering overhead and may complicate deployment on other compiler toolchains.

## Concepts Extracted

- [[compiler-fuzzing]]
- [[reinforcement-learning]]
- [[q-learning]]
- [[deep-q-learning]]
- [[markov-decision-process]]
- [[long-short-term-memory]]
- [[basic-block-coverage]]
- [[cyclomatic-complexity]]
- [[epsilon-greedy-exploration]]

## Entities Extracted

- [[xiaoting-li]]
- [[xiao-liu-facebook]]
- [[lingwei-chen]]
- [[rupesh-prajapati]]
- [[dinghao-wu]]
- [[alphaprog]]
- [[bfc]]
- [[afl]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
