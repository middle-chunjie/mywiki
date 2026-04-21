---
type: source
subtype: paper
title: Execution-Based Evaluation for Open-Domain Code Generation
slug: wang-2023-executionbased-2212-10481
date: 2026-04-20
language: en
tags: [code-generation, benchmark, execution, multilingual, open-domain]
processed: true

raw_file: raw/papers/wang-2023-executionbased-2212-10481/paper.pdf
raw_md: raw/papers/wang-2023-executionbased-2212-10481/paper.md
bibtex_file: raw/papers/wang-2023-executionbased-2212-10481/paper.bib
possibly_outdated: true

authors:
  - Zhiruo Wang
  - Shuyan Zhou
  - Daniel Fried
  - Graham Neubig
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2212.10481
doi:
url: https://arxiv.org/abs/2212.10481
citation_key: wang2023executionbased
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper introduces ODEX, the first execution-based benchmark for open-domain natural-language-to-Python code generation. Built from CoNaLa and MCoNaLa samples harvested from Stack Overflow, ODEX contains `945` NL-code pairs and `1,707` human-written tests spanning `79` libraries and four intent languages. The benchmark operationalizes open-domain evaluation by wrapping snippets into executable functions, specifying library prerequisites, and handling failure modes such as irreproducible API calls, randomized outputs, and library-specific equality checks. Evaluations of Codex and CodeGen show that Codex is stronger overall, while CodeGen scales more parameter-efficiently and exhibits smaller open-versus-closed-domain gaps. The paper also argues that execution-based metrics better capture functional correctness than lexical overlap metrics, making ODEX a more realistic stress test for practical code generation.

## Problem & Motivation

Prior execution-based code-generation benchmarks were largely closed-domain: they either restricted programs to Python built-ins or focused on narrow library families such as data-science stacks. That mismatch makes them poor proxies for real developer workflows, where Stack Overflow-style requests routinely depend on diverse external libraries and practical execution context. The paper therefore targets open-domain code generation, where evaluation must verify functional correctness even when code requires imports, mocking, approximate equivalence checks, or multilingual natural-language intents.

## Method

- **Dataset source**: build ODEX from [[conala]] and [[mconala]], yielding `945` NL-code pairs with `1,707` human-written tests over `79` libraries and four languages: `439` English, `90` Spanish, `164` Japanese, and `252` Russian samples.
- **Annotation pipeline**: each example goes through four steps: wrapping snippets into standalone functions, specifying library prerequisites, annotating executable tests, and self-verification. A test is accepted only if the canonical solution passes it, leading to a reported `100%` canonical pass rate.
- **Open-domain execution handling**: irreproducible runs are replaced with mocks such as `mock`-based HTTP responses; randomized programs use bounded assertions instead of exact values; library objects use custom equality checks such as `np.array_equal(a, b)` rather than raw `a == b`.
- **Dataset analysis**: complexity is quantified with `len(NL)`, `len(Code)`, AST depth, and the counts `N_in^var` and `N_out^var`, using spaCy tokenizers and Python `ast` parsing.
- **Model evaluation**: compare [[codex]] and [[codegen]] under zero-shot prompting, where the prompt concatenates function context and a docstring, optionally with test cases. Main evaluation uses `pass@k`.
- **Decoding setup**: generation follows nucleus sampling with `top-p = 0.95`, `temperature = 0.8`, and a maximum output length of `512` tokens.

## Key Results

- ODEX is the first executable open-domain benchmark for NL-to-code generation, covering `79` libraries; `505 / 945 = 53.4%` of samples are open-domain and the average annotation density is `1.8` test cases per example.
- Best Codex model `code-davinci-002` reaches pass@1 of `47.15` (en), `47.44` (es), `41.46` (ja), and `51.87` (ru); its pass@10 rises to `73.12`, `71.11`, `64.02`, and `78.17`.
- CodeGen `6.1B` reaches pass@1 of `34.49` (en), `28.56` (es), `35.55` (ja), and `44.64` (ru), making it competitive with `12B` Codex on some settings despite a much smaller parameter budget.
- Open-domain performance is substantially worse than closed-domain performance for both families; CodeGen gaps are smaller than Codex gaps by an average of about `6.0` points, and scaling CodeGen from `2.7B` to `6.1B` reduces the gap by about `6.3` points in English and `1.7` in Spanish.
- Adding one test case to the prompt significantly improves execution accuracy, while using one random evaluation test often preserves nearly the same ranking as using all tests.
- Execution-free metrics such as BLEU, ROUGE, METEOR, ChrF, and CodeBLEU do not reliably reproduce execution-based model rankings; BLEU and ROUGE correlate better than the others, but still only imperfectly.

## Limitations

- The paper is a 2023 benchmark study centered on Codex and CodeGen, so the empirical landscape predates newer code LLMs and should not be treated as current capability evidence without re-evaluation.
- Safe execution remains a core concern: open-domain code and tests can conceal malicious behavior or unsafe dependencies, so the benchmark does not eliminate execution-security risk.
- Language coverage is limited to four languages inherited from Stack Overflow and MCoNaLa resources; broader multilingual generalization is not validated.
- ODEX is still modest in size because human test authoring is expensive; `945` samples and `1.8` tests per example cannot exhaustively cover semantic corner cases.
- Experiments are prompt-only and zero-shot centric, so the paper does not study finetuning or stronger adaptation strategies for closing the open-domain gap.

## Concepts Extracted

- [[execution-based-evaluation]]
- [[open-domain-code-generation]]
- [[multilingual-code-generation]]
- [[test-case-generation]]
- [[code-execution]]
- [[self-verification]]
- [[pass-at-k]]
- [[execution-free-metrics]]
- [[prompting-with-test-cases]]
- [[code-generation-benchmark]]

## Entities Extracted

- [[zhiruo-wang]]
- [[shuyan-zhou-cmu]]
- [[daniel-fried]]
- [[graham-neubig]]
- [[odex]]
- [[conala]]
- [[mconala]]
- [[stack-overflow]]
- [[codex]]
- [[codegen]]
- [[carnegie-mellon-university]]
- [[inspired-cognition]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
