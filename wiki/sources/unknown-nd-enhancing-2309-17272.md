---
type: source
subtype: paper
title: Enhancing Large Language Models in Coding Through Multi-Perspective Self-Consistency
slug: unknown-nd-enhancing-2309-17272
date: 2026-04-20
language: en
tags: [llm, code-generation, self-consistency, reranking, verification]
processed: true

raw_file: raw/papers/unknown-nd-enhancing-2309-17272/paper.pdf
raw_md: raw/papers/unknown-nd-enhancing-2309-17272/paper.md
bibtex_file: raw/papers/unknown-nd-enhancing-2309-17272/paper.bib
possibly_outdated: true

authors:
  - Baizhou Huang
  - Shuai Lu
  - Weizhu Chen
  - Xiaojun Wan
  - Nan Duan
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2309.17272
doi:
url: https://arxiv.org/abs/2309.17272
citation_key: unknownndenhancing
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper proposes Multi-Perspective Self-Consistency (MPSC), a post-hoc code-generation reranking framework that treats candidate solutions, generated specifications, and generated test cases as equally informative views of the same latent program semantics. Instead of trusting tests or oracles as superior verifiers, MPSC builds a graph over these three perspectives, measures cross-perspective agreement by deterministic code execution, injects within-perspective consistency as a prior, and optimizes a vertex scoring function to select the most reliable solution. Using ChatGPT / GPT-3.5-Turbo as the base model, MPSC improves pass@1 by `+15.91` on HumanEval, `+15.64` on HumanEval+, `+6.43` on MBPP, and `+9.37` on CodeContests, and the reported ChatGPT-based system surpasses GPT-4 on pass@1 across all four benchmarks.

## Problem & Motivation

Single-shot code generation from large language models remains unreliable even when the model is strong. Prior post-hoc methods such as CodeT and oracle-based verification typically assume that generated verifiers are more trustworthy than generated solutions, then use those verifiers to rank candidate code by majority voting. The paper argues this assumption is fragile because solutions, specifications, and test cases are usually produced by the same underlying model under the same prompt context, so none should automatically dominate. The authors instead frame them as different reasoning perspectives over one latent functionality and ask whether agreement patterns across these perspectives can identify a better final program than any single perspective alone.

## Method

- **Three perspectives**: for each natural-language intent, the model generates candidate solutions `g: X -> Y`, specifications as pre-/post-condition functions `h_pre: X -> {False, True}` and `h_post: X x Y -> {False, True}`, and test cases `(x, y) in X x Y`.
- **Graph construction**: MPSC builds an undirected 3-partite graph `G = (V, E)` with vertex sets `V = V_func U V_spec U V_test`, where solutions, specifications, and test cases are distinct partitions.
- **Inter-consistency weights**: edges connect only different perspectives, with weights `omega(v_i, v_j)` computed by deterministic execution. Representative cases are `1[g_i(x_j) = y_j]` for solution-test, `1[h_pre(x_j) and h_post(x_j, y_j)]` for specification-test, and expectation-style execution checks for solution-specification agreement.
- **Inter-consistency objective**: the graph smoothness term is `L_inter = sum_(v_i,v_j in E) W_ij (f(v_i) - f(v_j))^2 = f^T L f`, where `L = D - W` is the graph Laplacian. This enforces similar scores for outputs that agree functionally.
- **Intra-consistency priors**: lexical intra-consistency uses Minimum Bayes Risk style similarity, `phi(v_i) = C * sum_j BLEU(v_i, v_j)`, while semantic intra-consistency groups structurally equivalent vertices and scores them as `phi(v_i) = C * |S(v_i)| * prod_t |N_t(v_i)|`.
- **Joint optimization**: the final score function solves `min_f alpha * L_inter + (1 - alpha) * L_intra`, where `L_intra = 1/2 ||f - y||^2` and `y_i = phi(v_i)`. The iterative update is `f^(i+1) = alpha T f^(i) + (1 - alpha) y` with `T = D^(-1/2) W D^(-1/2)`.
- **Sampling / inference setup**: with ChatGPT / GPT-3.5-Turbo as the foundation model, the system samples `200` solutions, `50` specifications, and `100` test cases per problem; strong baselines such as CodeT and MBR-EXEC use the same `200` sampled solutions and up to `500` generated tests.
- **Adaptive hyperparameter**: MPSC sets `alpha = 0.01` when the mean edge weight is `< 0.16` and `alpha = 0.95` otherwise, so sparse graphs rely more on intra-consistency while denser graphs emphasize cross-perspective agreement.
- **Computational profile**: the reported inter-consistency stage evaluates `(200 x 50) + (50 x 100) + (100 x 200) = 35000` edges with a `0.001`-second limit per edge; the iterative solver usually converges in about `50` rounds and under `0.1` seconds.

## Key Results

- HumanEval pass@1 rises from `68.38` with GPT-3.5-Turbo to `84.29` with MPSC-Label (`+15.91`); MPSC-Semantic reaches `83.38`, and GPT-4 is reported at `81.48`.
- HumanEval+ pass@1 rises from `58.75` to `74.39` with MPSC-Lexical (`+15.64`); GPT-4 is `70.52`, so the ChatGPT-based MPSC variants exceed the reported GPT-4 baseline.
- MBPP pass@1 improves from `66.80` to `73.23` with MPSC-Semantic (`+6.43`), slightly above GPT-4's `71.26`; CodeT reaches `71.90`.
- CodeContests pass@1 improves from `2.57` to `11.94` with MPSC-Label (`+9.37`), compared with GPT-4 at `6.10` and CodeT at `9.92`.
- Ablations show both auxiliary perspectives matter: on HumanEval pass@1, MPSC-Semantic is `83.38`, dropping to `82.32` without specifications, `78.30` without test cases, and back to the base `68.38` without both.
- The generated verifier perspectives are not intrinsically stronger than solutions: on HumanEval, solution / specification / test-case accuracy is `68.38 / 45.93 / 63.82`; on MBPP it is `66.80 / 53.70 / 34.64`, supporting the paper's claim that naive verifier-majority assumptions are unreliable.
- MPSC generalizes across base models: for GPT-4 on HumanEval, MPSC-Semantic reaches `92.15` pass@1 (`+10.67`), and for DeepSeekCoder-6.7B it reaches `82.38` (`+10.65`).

## Limitations

- The evaluation is still benchmark-centric; the paper explicitly notes that real-world coding tasks involve noisier intents and more complex functional requirements than HumanEval, MBPP, or CodeContests.
- Inter-consistency is easy to compute here because code execution provides a deterministic agreement signal; the method does not directly solve how to measure agreement for open-ended natural-language tasks.
- Generated specifications and test cases are themselves noisy, with standalone accuracies sometimes well below the solution accuracy, so MPSC helps by aggregating weak signals rather than by obtaining strong verifiers.
- The top-ranked candidates can collapse into semantically similar clusters, which explains the weak gains at larger `k` and the need for round-robin selection to recover diversity.
- The framework adds nontrivial inference overhead from pairwise execution over `35000` cross-perspective edges, even if the authors argue that the cost is parallelizable and still lower than some baselines.

## Concepts Extracted

- [[large-language-model]]
- [[code-generation]]
- [[self-consistency]]
- [[reranking]]
- [[test-case-generation]]
- [[minimum-bayes-risk-decoding]]
- [[pass-at-k]]
- [[formal-specification]]
- [[inter-consistency]]
- [[intra-consistency]]

## Entities Extracted

- [[baizhou-huang]]
- [[shuai-lu]]
- [[weizhu-chen]]
- [[xiaojun-wan]]
- [[nan-duan]]
- [[peking-university]]
- [[microsoft-research-asia]]
- [[microsoft]]
- [[chatgpt]]
- [[gpt-4]]
- [[humaneval]]
- [[mbpp]]
- [[codecontests]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
