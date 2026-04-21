---
type: source
subtype: paper
title: "Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding"
slug: suzgun-2024-metaprompting-2401-12954
date: 2026-04-20
language: en
tags: [llm, prompting, scaffolding, reasoning, tool-use]
processed: true

raw_file: raw/papers/suzgun-2024-metaprompting-2401-12954/paper.pdf
raw_md: raw/papers/suzgun-2024-metaprompting-2401-12954/paper.md
bibtex_file: raw/papers/suzgun-2024-metaprompting-2401-12954/paper.bib
possibly_outdated: false

authors:
  - Mirac Suzgun
  - Adam Tauman Kalai
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2401.12954
doi:
url: http://arxiv.org/abs/2401.12954
citation_key: suzgun2024metaprompting
paper_type: method

read_status: unread

domain: llm
---

## Summary

The paper proposes meta-prompting, a task-agnostic inference scaffold in which one fixed language model acts both as a high-level conductor and as multiple temporary experts. Instead of giving task-specific exemplars, the user supplies a query and a reusable meta prompt; the model then decomposes the task, issues expert instructions, integrates intermediate outputs, and optionally invokes a Python interpreter. The framework is evaluated with GPT-4 across arithmetic, symbolic reasoning, code, multilingual math, and constrained writing tasks. In macro average accuracy, meta-prompting with Python reaches `72.9`, outperforming standard prompting (`54.8`), dynamic expert prompting (`54.6`), and multi-persona prompting (`57.7`). The paper positions this as a zero-shot way to improve robustness without finetuning or bespoke task pipelines.

## Problem & Motivation

Large language models can solve many tasks, but single-pass prompting often produces brittle answers, especially when a problem requires decomposition, verification, or external computation. Existing scaffolding methods such as zero-shot chain-of-thought, expert prompting, and multi-persona prompting each capture only part of this workflow and often require more task-specific steering than a user wants to provide. The paper asks whether a single black-box LM can be turned into a reusable orchestration layer that plans, delegates, cross-checks, and recombines intermediate results under one task-agnostic prompt.

## Method

- **Core setup**: use one fixed language model `LM: S -> S` as both the Meta Model and every expert model; role differences come only from prompt templates, not from different parameters.
- **History initialization**: given input query `x`, the system constructs the first prompt history as `` `H_1 <- t_init(x)` ``, where `t_init` injects the reusable meta instructions plus the user query.
- **Meta loop**: at step `t`, the Meta Model outputs `` `y_t <- LM(H_t)` ``. If expert instructions can be extracted, the system wraps them with `` `t_exp(e_exp(y_t))` `` and queries the same LM again as an isolated expert.
- **Expert return path**: the expert output `` `z_t <- LM(prompt)` `` is appended back into the running history via `` `H_{t+1} <- H_t ⊕ t_mid(z_t)` ``, so the conductor can decide whether to consult another expert, revise, or finish.
- **Termination and robustness**: if the Meta Model emits a marked final response, the answer extractor returns `` `e_ret(y_t)` ``; otherwise an explicit error string is appended to history so malformed generations do not silently terminate the procedure.
- **Communication structure**: experts can only be called by the Meta Model and do not directly talk to one another; the Meta Model selectively exposes prior context, preserving centralized orchestration.
- **Tool integration**: the framework can treat a Python interpreter as another callable expert/tool, enabling external computation in a task-agnostic way for arithmetic and programming-style tasks.
- **Answer formatting**: the final answer is required to appear after the marker `` `>>FINAL ANSWER:` `` and inside triple quotes to support deterministic extraction across heterogeneous tasks.
- **Inference settings**: main experiments use `GPT-4` (`gpt-4-32k`) through Azure OpenAI Service; supplementary runs use `GPT-3.5` (`gpt-35-turbo`) with `` `temperature = 0` ``, `` `top_p = 0.95` ``, and `` `max_tokens = 1024` ``.
- **Evaluation protocol**: tasks are scored with Exact Match, Soft Match, or Functionally Correct metrics depending on the task family, while the high-level meta instructions remain fixed across all benchmarks.

## Key Results

- Macro average accuracy: meta-prompting `+ Python = 72.9`, versus standard prompting `54.8`, zero-shot CoT `59.1`, dynamic expert prompting `54.6`, multi-persona prompting `57.7`, and meta-prompting without Python `61.4`.
- Aggregate deltas reported by the paper: meta-prompting with Python beats standard prompting by `17.1` points, dynamic expert prompting by `17.3`, and multi-persona prompting by `15.2`.
- `Game of 24`: meta-prompting with Python reaches `67.0`, versus `3.0` for standard prompting, `11.0` for zero-shot CoT, and `25.0` for multi-persona prompting.
- `Checkmate-in-One`: meta-prompting reaches `57.2`, clearly above standard prompting (`36.4`) and dynamic expert prompting (`33.2`).
- `Python Programming Puzzles`: meta-prompting with Python scores `45.8`, improving over standard prompting (`31.1`) and zero-shot CoT (`36.3`).
- `Word Sorting`: meta-prompting with Python reaches `99.6`, versus `80.4` for standard prompting and `85.2` for dynamic expert prompting.
- `Sonnet Writing`: meta-prompting with Python scores `79.6`, versus `62.0` for standard prompting and `73.2` for multi-persona prompting.
- Gains are not uniform: on `MGSM`, meta-prompting with Python is `84.8`, close to standard prompting (`84.4`) and below multi-persona prompting (`85.7`); on `Geometric Shapes`, the gain over standard prompting is only `+2.4`.

## Limitations

- The empirical study is centered on proprietary OpenAI models, mainly `GPT-4`, so transfer to smaller or open-weight models is not established in this paper.
- The conductor and experts are instantiated from the same underlying LM, so diversity comes from role prompting rather than from genuinely different models or independently trained specialists.
- Improvements are task-dependent: `MGSM` is roughly flat and `Geometric Shapes` improves only marginally, so the scaffold is not uniformly beneficial.
- Some strong gains depend on external tooling: for example, `Game of 24` jumps from `11.0` without Python to `67.0` with Python, indicating that plain text-only orchestration is not the whole story.
- Reproducibility remains imperfect because the authors note that even with `` `temperature = 0` ``, GPT-3.5 and GPT-4 can still vary across runs.

## Concepts Extracted

- [[meta-prompting]]
- [[task-agnostic-scaffolding]]
- [[zero-shot-prompting]]
- [[chain-of-thought-prompting]]
- [[expert-prompting]]
- [[multi-persona-prompting]]
- [[task-decomposition]]
- [[self-reflection]]
- [[tool-augmented-language-model]]
- [[black-box-language-model]]

## Entities Extracted

- [[mirac-suzgun-stanford]]
- [[adam-tauman-kalai-openai]]
- [[stanford-university]]
- [[openai]]
- [[microsoft-research]]
- [[gpt-4]]
- [[gpt-3-5]]
- [[big-bench-hard]]
- [[game-of-24]]
- [[checkmate-in-one]]
- [[python-programming-puzzles]]
- [[mgsm]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
