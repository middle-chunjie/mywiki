---
type: source
subtype: paper
title: The False Promise of Imitating Proprietary LLMs
slug: gudibande-2023-false-2305-15717
date: 2026-04-20
language: en
tags: [llm, model-imitation, evaluation, factuality, distillation]
processed: true
raw_file: raw/papers/gudibande-2023-false-2305-15717/paper.pdf
raw_md: raw/papers/gudibande-2023-false-2305-15717/paper.md
bibtex_file: raw/papers/gudibande-2023-false-2305-15717/paper.bib
possibly_outdated: true
authors:
  - Arnav Gudibande
  - Eric Wallace
  - Charlie Snell
  - Xinyang Geng
  - Hao Liu
  - Pieter Abbeel
  - Sergey Levine
  - Dawn Song
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2305.15717
doi:
url: http://arxiv.org/abs/2305.15717
citation_key: gudibande2023false
paper_type: benchmark
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper argues that supervised imitation of proprietary assistants such as ChatGPT mostly transfers surface behavior rather than underlying capabilities. The authors fine-tune GPT-2 1.5B, LLaMA 7B, and LLaMA 13B models on both broad imitation data (ShareGPT-Mix) and task-specific imitation data (NQ-Synthetic), scaling imitation corpora from `0.3M` to `150M` tokens. Human raters and GPT-4 often judge the strongest imitators as competitive with ChatGPT because the models learn instruction-following style, structure, and tone. However, targeted evaluations on MMLU, Natural Questions, and HumanEval show that broad imitation closes little or none of the capability gap and can even hurt factual QA. The main conclusion is that stronger base pretraining is higher leverage than collecting more imitation traces.

## Problem & Motivation

The paper studies whether open models can cheaply catch up to proprietary large language models by fine-tuning on their outputs. This matters because many open-model efforts assume that API access plus enough imitation data can erase the advantage of closed systems without reproducing their pretraining scale or internal alignment data. The authors specifically challenge optimistic claims around near-parity copycats such as Alpaca- and Vicuna-style systems. Their motivation is to separate stylistic similarity from actual capability transfer and to test whether human preference judgments can be trusted as the main evidence that imitation works.

## Method

- **Target setting**: imitate ChatGPT using decoder-only base models `GPT-2 1.5B`, `LLaMA 7B`, and `LLaMA 13B`, varying both base capability and imitation corpus size.
- **Broad imitation data**: build `ShareGPT-Mix` from three public sources: roughly `50K` deduplicated English ShareGPT conversations (down from `~90K` raw), `~27K` HC3 ChatGPT responses for `~24K` questions, and `10K` Discord ChatGPT bot examples.
- **Local imitation data**: build `NQ-Synthetic` by seeding with `10` Natural Questions validation QA pairs and asking ChatGPT to iteratively generate `6,000` similar but distinct examples.
- **Tokenization and packing**: chunk conversations into blocks of `2048` tokens and add special markers delimiting user queries and model outputs.
- **Objective**: fine-tune with standard causal LM loss on output tokens only, treating imitation as supervised next-token prediction rather than reinforcement learning or preference optimization.
- **Optimization**: train for `1` epoch with `AdamW`, learning rate `2e-3`, linear warmup `1000` steps from `0`, and batch size `32`; gradients are rescaled by weight magnitude following PaLM/FLAN-style recipes.
- **Systems**: run training in JAX with shared data parallelism plus tensor parallelism on Google Cloud TPUs or a single DGX with `8 x A100` GPUs.
- **Automatic evaluation**: measure `5-shot` MMLU, `3-shot` Natural Questions, and `0-shot` HumanEval using each benchmark's canonical metric.
- **Human evaluation**: collect blind pairwise preferences between ChatGPT and imitation models on `255` held-out prompts, with about `70` crowd workers and `3` ratings per example; also run GPT-4 pairwise judging on the same prompts.
- **Analysis focus**: compare scaling imitation data versus scaling the base LM, and contrast broad imitation with task-specific imitation to isolate when distillation-like transfer actually works.

## Key Results

- Human raters initially find imitation outputs highly competitive: roughly `~70%` of outputs from the best imitation models are rated equal to or better than ChatGPT in the paper's headline comparison.
- Broad imitation fails on factual QA: Natural Questions exact match drops from `17` to `10` for `7B` and from `20` to `15` for `13B` when training on ShareGPT-Mix, while ChatGPT reaches `31`.
- Local imitation is much more effective: NQ-Synthetic improves Natural Questions from `17 -> 22` for `7B` and from `20 -> 27` for `13B`, substantially narrowing the gap to ChatGPT's `31`.
- Style matching scales strongly with imitation data: list-format agreement with ChatGPT rises from `13%` in base LLaMA to `50%`, `67%`, and `81%` at `20M`, `80M`, and `150M` imitation tokens.
- Other stylistic metrics also converge: summary-paragraph agreement rises from `2%` to `48%`, unigram overlap from `19.5` to `42.5`, and output-length correlation from `-0.11` to `0.62`.
- GPT-4 labels outputs as authoritative for `57%` of base LLaMA generations and `98-99%` of imitation-model generations, showing that style becomes nearly indistinguishable before capability does.
- Scaling the base model helps more than scaling imitation data: moving to stronger base LMs consistently improves benchmark scores, while adding more broad imitation data yields flat or negative returns.
- Safety style transfers: on RealToxicityPrompts, imitation models become measurably less toxic than their untuned baselines by inheriting ChatGPT-like refusal behavior.

## Limitations

- The study targets ChatGPT only, so the conclusions may not transfer unchanged to imitation of stronger or differently aligned proprietary models.
- Base models are limited to `1.5B-13B` parameters and relatively standard supervised fine-tuning; the paper does not test RLHF, constitutional AI, active learning, or more sophisticated imitation objectives.
- Some of ChatGPT's measured advantage could reflect unknown benchmark contamination in proprietary pretraining data rather than purely better generalization.
- Broad imitation data is gathered from public traces, which may be noisy, distribution-shifted, or unrepresentative of the full API interaction space.
- The paper establishes that broad imitation is weak under current methods, but it does not quantify the exact data scale at which imitation would become competitive.

## Concepts Extracted

- [[large-language-model]]
- [[model-imitation]]
- [[knowledge-distillation]]
- [[instruction-tuning]]
- [[instruction-following]]
- [[factuality]]
- [[human-evaluation]]
- [[benchmark-evaluation]]
- [[model-scaling]]
- [[pretraining]]
- [[distribution-shift]]
- [[toxicity]]

## Entities Extracted

- [[arnav-gudibande]]
- [[eric-wallace]]
- [[charlie-snell]]
- [[xinyang-geng]]
- [[hao-liu]]
- [[pieter-abbeel]]
- [[sergey-levine]]
- [[dawn-song]]
- [[uc-berkeley]]
- [[chatgpt]]
- [[llama]]
- [[sharegpt]]
- [[natural-questions]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
