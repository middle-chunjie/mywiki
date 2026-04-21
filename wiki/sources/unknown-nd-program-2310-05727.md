---
type: source
subtype: paper
title: The Program Testing Ability of Large Language Models for Code
slug: unknown-nd-program-2310-05727
date: 2026-04-20
language: en
tags: [code-llm, program-testing, test-case-generation, program-synthesis, software-engineering]
processed: true

raw_file: raw/papers/unknown-nd-program-2310-05727/paper.pdf
raw_md: raw/papers/unknown-nd-program-2310-05727/paper.md
bibtex_file: raw/papers/unknown-nd-program-2310-05727/paper.bib
possibly_outdated: true

authors:
  - Weimin Xiong
  - Yiwen Guo
  - Hao Chen
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.05727
doi:
url: https://arxiv.org/abs/2310.05727
citation_key: unknownndprogram
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper studies whether code-oriented large language models can generate useful tests, not only synthesize programs. Across `11` models, `4` prompting settings, and two Python benchmarks, it evaluates generated tests by deduplicated pass rate and branch coverage, showing that stronger code models also tend to be stronger test generators. The analysis further isolates two practical levers for downstream synthesis: generate tests conditioned on self-generated code rather than a blank placeholder, and discount later test cases because their correctness decays with generation order. Applied to CodeT-style reranking on HumanEval+, these ideas raise GPT-3.5-turbo pass@1 by `+11.77` points over the plain synthesis baseline and by `+4.22` over the prior state of the art, positioning program testing as a measurable capability of code LLMs.

## Problem & Motivation

Evaluation of code LLMs had largely centered on program synthesis benchmarks such as HumanEval and MBPP, even though software engineering practice also depends on writing tests that expose bugs and specify intended behavior. The paper argues that test generation is both an intrinsic capability worth measuring and an instrumental capability that can improve synthesis by filtering or reranking candidate programs. The main motivation is therefore twofold: understand how well modern code LLMs test programs under different prompt conditions, and extract actionable findings that can improve synthesis methods that already rely on generated tests.

## Method

- **Benchmarks and scale**: evaluate on HumanEval+ with `164` problems and sanitized MBPP with `427` problems, using `11` code-capable LLMs spanning roughly `770M` to `16B` parameters plus GPT-3.5-turbo.
- **Four testing settings**: compare `self-generated`, `all-generated`, `oracle`, and `placeholder` prompts. In the first two settings, each problem uses `N = 100` imperfect implementations; in the oracle setting, the corrected oracle is copied `100×`; the placeholder setting removes implementation details entirely.
- **Prompting protocol**: append an instruction equivalent to `# Check the correctness of this function with three test cases` and force continuation with `assert <function_name>`, then parse at most the first `3` generated assert statements as test cases.
- **Correctness metric**: define pass rate as ``P = (1 / MN) Σ_i Σ_j (p_ij / n_ij)`` where `p_ij` counts generated tests that the oracle passes and `n_ij ≤ 3`. To reduce duplication bias, also compute the deduplicated variant `P'` over unique correct tests.
- **Diversity metric**: define coverage rate as ``C = (1 / MN) Σ_i Σ_j c_ij`` where `c_ij` is branch coverage from executing the three retained tests with `pytest`.
- **Generation setup**: synthesize candidate programs with temperature `0.2`, generate tests with temperature `0.2`, and in the all-generated setting pool implementations from InCoder `1.3B`, CodeGen2 `1B`, CodeT5+ `770M`, and SantaCoder `1.1B`.
- **Analysis slices**: compare performance against code-synthesis strength, split self-generated prompts into correct vs. incorrect code, and measure how correctness changes with the order of the `1st`, `2nd`, and `3rd` generated tests.
- **Synthesis improvement recipe**: modify CodeT by switching from placeholder prompts to self-generated code (`SG`) and weighting tests by order with ``w_i = p^{i-1}``, using `p = 0.8` and `i ∈ {1,2,3,4,5}` while keeping the same `100 × 5` test budget on HumanEval+.

## Key Results

- On HumanEval+, GPT-3.5-turbo is the strongest tester among evaluated large models, reaching `71.03%` pass rate in the oracle setting and `72.45%` in the self-generated setting, with coverage around `77%`.
- On MBPP, GPT-3.5-turbo reaches `74.30%` pass rate in the oracle setting, `66.14%` with self-generated code, and `63.34%` with placeholders, showing a consistent gap between richer prompt context and no-implementation prompting.
- Among small models on HumanEval+, CodeT5+ achieves the highest oracle pass rate at `35.43%`, while SantaCoder attains the strongest code-synthesis baseline in Table 1 at `15.21%` / `29.42%` pass@1 on HumanEval+ / MBPP.
- Conditioning on correct self-generated code improves test quality: on HumanEval+, GPT-3.5-turbo rises from `68.52%` with incorrect code to `75.39%` with correct code, WizardCoder from `45.12%` to `48.02%`, and CodeGeeX2 from `48.63%` to `52.84%`.
- The paper reports that earlier generated tests are more reliable than later ones, motivating rank-based weighting instead of treating all tests as equally trustworthy.
- In synthesis reranking on HumanEval+, GPT-3.5-turbo improves from a plain baseline of `61.70%` pass@1 to `69.25%` with CodeT and to `73.47%` with `SG + RW`; WizardCoder improves from `46.23%` to `61.45%`, and StarCoder from `27.90%` to `43.15%`.

## Limitations

- The evaluation is restricted to HumanEval+ and sanitized MBPP, both short-form Python benchmarks that are much simpler than real software testing workflows or multi-file repositories.
- Test quality is measured mainly through oracle pass rate and branch coverage, which miss qualities such as readability, maintainability, mutation score, or fault localization utility.
- The conclusions are tied to the 2023 model set and prompt recipes; stronger post-2023 code LLMs, tool-augmented systems, and reasoning models may change the ranking substantially.
- The synthesis improvement section validates only a lightweight modification of CodeT on HumanEval+, so it does not establish that the same gains persist on broader benchmarks, other languages, or industrial codebases.

## Concepts Extracted

- [[large-language-model]]
- [[code-language-model]]
- [[test-case-generation]]
- [[pass-rate]]
- [[code-generation]]
- [[benchmark-dataset]]
- [[chain-of-thought-prompting]]

## Entities Extracted

- [[weimin-xiong]]
- [[yiwen-guo]]
- [[hao-chen]]
- [[tencent]]
- [[peking-university]]
- [[uc-davis]]
- [[mbpp]]
- [[gpt-3-5-turbo]]
- [[starcoder]]
- [[wizardcoder]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
