---
type: source
subtype: paper
title: BIG-Bench Extra Hard
slug: kazemi-2025-bigbench-2502-19187
date: 2026-04-20
language: en
tags: [benchmark, reasoning, llm, evaluation, robustness]
processed: true

raw_file: raw/papers/kazemi-2025-bigbench-2502-19187/paper.pdf
raw_md: raw/papers/kazemi-2025-bigbench-2502-19187/paper.md
bibtex_file: raw/papers/kazemi-2025-bigbench-2502-19187/paper.bib
possibly_outdated: false

authors:
  - Mehran Kazemi
  - Bahare Fatemi
  - Hritik Bansal
  - John Palowitch
  - Chrysovalantis Anastasiou
  - Sanket Vaibhav Mehta
  - Lalit K. Jain
  - Virginia Aglietti
  - Disha Jindal
  - Peter Chen
  - Nishanth Dikkala
  - Gladys Tyen
  - Xin Liu
  - Uri Shalit
  - Silvia Chiappa
  - Kate Olszewska
  - Yi Tay
  - Vinh Q. Tran
  - Quoc V. Le
  - Orhan Firat
year: 2025
venue: arXiv
venue_type: preprint
arxiv_id: 2502.19187
doi: 10.48550/arXiv.2502.19187
url: http://arxiv.org/abs/2502.19187
citation_key: kazemi2025bigbench
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

This paper introduces BIG-Bench Extra Hard (BBEH), a new benchmark intended to replace the now-saturated [[bbh]] as a stress test for broad LLM reasoning. The authors keep BBH's cross-domain coverage but replace each of its `23` tasks with a harder counterpart that increases reasoning depth, context length, distractors, output compositionality, or distribution shift while preserving the underlying skill family. BBEH remains far from saturated: the best general-purpose model reaches only `9.8%` adjusted harmonic-mean accuracy (`23.9%` micro average), while the best reasoning-specialized model reaches `44.8%` (`54.2%` micro average). The paper also argues that reasoning-specialized models gain most on formal, algorithmic, and long-context tasks, but still lag on softer skills such as humour, sarcasm, commonsense, and causal judgment.

## Problem & Motivation

The paper starts from a concrete failure of existing evaluation: frontier LLMs already score above `90%` on [[bbh]], so BBH no longer discriminates among strong models. The authors identify several reasons why BBH became too easy: many tasks have small answer spaces, some contain exploitable shortcuts, inputs are often short (macro average around `700` characters), and many problems require only a few reasoning hops. Their goal is therefore not merely to make tasks harder, but to preserve BBH's breadth while restoring measurement sensitivity for robust general reasoning. BBEH is framed as a broader benchmark for capabilities that matter outside math and coding, including temporal, spatial, causal, commonsense, humour, long-context, and algorithmic reasoning.

## Method

- **Benchmark construction**: BBEH contains `23` tasks, each replacing one BBH task with a harder task in the same reasoning domain so the benchmark preserves diversity while raising difficulty.
- **Dataset size**: each task contains `200` examples except `DisambiguationQA`, which contains `120`; the smaller `BBEH Mini` subset samples `20` examples per task for `460` total examples.
- **Difficulty targets**: the authors use a semi-adversarial loop with Gemini `1.5 Flash` as a general-purpose reference model and Gemini Thinking Experimental as a reasoning-specialized reference model, iterating until both score below `70%` on each task.
- **Task-design levers**: the benchmark raises difficulty through larger label spaces, longer contexts, more reasoning hops, distractors, stronger priors to override, compositional outputs, and replacement of shortcut-friendly formulations.
- **Representative transformations**: examples include converting BBH Boolean Expressions into mixed textual/logical expressions to prevent trivial code execution, extending object and calendar tasks into long-context tracking problems, and turning caption, sarcasm, and disambiguation tasks into higher-arity or longer-context variants.
- **Scoring**: the main aggregate metric is an adjusted harmonic mean over per-task accuracies, with smoothed accuracies `a'_i = a_i + 1` to avoid zero-value collapse and `HM = n / Σ_i (1 / a'_i)`.
- **Secondary metrics**: the paper also reports micro average, especially for `BBEH Mini`, and compares BBEH directly against BBH on task-by-task matched replacements.
- **Answer extraction**: evaluation uses deterministic extraction from final-answer prefixes such as `The answer is:` with light normalization for quotes, brackets, bare option letters, and comma-separated outputs.

## Key Results

- **Overall difficulty**: the best general-purpose model reaches only `9.8%` adjusted harmonic mean on BBEH, while the best reasoning-specialized model reaches `44.8%`; the random baseline is `2.4%`.
- **Micro average**: the best general-purpose model reaches `23.9%` micro average and the best reasoning-specialized model reaches `54.2%`, versus a random baseline of `8.4%`.
- **Harder than BBH**: for Gemini `2.0 Flash`, overall accuracy drops from `85.2%` on BBH counterparts to `23.9%` on BBEH; large single-task drops include Hyperbaton `94.8% -> 4.5%`, Temporal Sequences `98.8% -> 0.5%`, and Object Properties `96.8% -> 1.5%`.
- **Longer and harder inputs**: the macro average context length is about `6x` BBH, and the macro average output length generated by Gemini `2.0 Flash` is about `7x` BBH.
- **Task leaders differ by skill**: DeepSeek R1 leads BoardgameQA at `75.5%`, GPT-4o leads NYCC at `23.0%`, and o3-mini (high) dominates Temporal Sequences at `68.5%`, Object Counting at `90.0%`, and Object Properties at `56.5%`.
- **Reasoning-model gains are uneven**: reasoning-specialized models gain most on counting, planning, arithmetic, and algorithmic/data-structure tasks, but show much smaller gains on commonsense, humour, sarcasm, and causation.

## Limitations

- The benchmark is constructed semi-adversarially against specific reference models, so the failure modes emphasized by BBEH are partly shaped by those models and may underrepresent other weaknesses.
- This construction also makes direct fairness comparisons with the reference models imperfect, a limitation the authors acknowledge explicitly.
- Some extremely low scores reflect answer-extraction failure or output-token degeneration rather than pure reasoning failure, so benchmark accuracy conflates reasoning with completion format robustness.
- Although BBEH is broad, it is still a text-only benchmark of `23` task families and cannot fully cover multimodal, interactive, or real-world tool-using reasoning.

## Concepts Extracted

- [[benchmark-dataset]]
- [[benchmark-evaluation]]
- [[benchmark-saturation]]
- [[multi-hop-reasoning]]
- [[long-context-reasoning]]
- [[algorithmic-reasoning]]
- [[commonsense-reasoning]]
- [[temporal-reasoning]]
- [[harmonic-mean-evaluation]]
- [[reasoning-specialized-model]]
- [[semi-adversarial-benchmark-construction]]

## Entities Extracted

- [[mehran-kazemi]]
- [[bahare-fatemi]]
- [[hritik-bansal]]
- [[john-palowitch]]
- [[chrysovalantis-anastasiou]]
- [[sanket-vaibhav-mehta]]
- [[lalit-k-jain]]
- [[virginia-aglietti]]
- [[disha-jindal]]
- [[peter-chen]]
- [[nishanth-dikkala]]
- [[gladys-tyen]]
- [[xin-liu]]
- [[uri-shalit]]
- [[silvia-chiappa]]
- [[kate-olszewska]]
- [[yi-tay]]
- [[vinh-q-tran]]
- [[quoc-le]]
- [[orhan-firat]]
- [[google-deepmind]]
- [[google-research]]
- [[ucla]]
- [[big-bench]]
- [[bbh]]
- [[bbeh]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
