---
type: source
subtype: paper
title: Textbooks Are All You Need
slug: gunasekar-2023-textbooks-2306-11644
date: 2026-04-20
language: en
tags: [llm, code-generation, synthetic-data, data-curation, fine-tuning]
processed: true
raw_file: raw/papers/gunasekar-2023-textbooks-2306-11644/paper.pdf
raw_md: raw/papers/gunasekar-2023-textbooks-2306-11644/paper.md
bibtex_file: raw/papers/gunasekar-2023-textbooks-2306-11644/paper.bib
possibly_outdated: true
authors:
  - Suriya Gunasekar
  - Yi Zhang
  - Jyoti Aneja
  - Caio César Teodoro Mendes
  - Allie Del Giorno
  - Sivakanth Gopi
  - Mojan Javaheripi
  - Piero Kauffmann
  - Gustavo Rosa
  - Olli Saarikivi
  - Adil Salim
  - Shital Shah
  - Harkirat Singh Behl
  - Xin Wang
  - Sébastien Bubeck
  - Ronen Eldan
  - Adam Tauman Kalai
  - Yin Tat Lee
  - Yuanzhi Li
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2306.11644
doi:
url: http://arxiv.org/abs/2306.11644
citation_key: gunasekar2023textbooks
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature. The paper argues that high-quality, textbook-style training data can beat brute-force scale for code LLMs. It introduces `phi-1`, a `1.3B` decoder-only Transformer trained on fewer than `7B` source tokens: a filtered `~6B`-token web code corpus plus `<1B` synthetic textbooks, followed by finetuning on `~180M` synthetic coding exercises. Despite the small model and data budget, `phi-1` reaches `50.6%` pass@1 on HumanEval and `55.5%` on MBPP. The authors further claim that finetuning induces broader capability jumps, including better instruction following and external-library use, and they support the main benchmark results with unconventional GPT-4-graded tasks and aggressive decontamination-style pruning experiments.

## Problem & Motivation

The paper targets a central question in code-language modeling: whether data quality can substitute for massive parameter counts and trillion-token corpora. The authors argue that standard web code sources such as The Stack and StackOverflow contain too much boilerplate, missing context, and poorly explained logic to teach algorithmic reasoning efficiently. Their motivating hypothesis is that a model trained on clear, self-contained, balanced, and instruction-rich "textbook quality" data should learn code generation far more efficiently than one trained on raw web code alone, and that a small post-training dataset of synthetic exercises can unlock capabilities not obvious from pretraining metrics alone.

## Method

- **Data recipe**: build `CodeTextbook = filtered web code-language data + synthetic textbooks`, then finetune on `CodeExercises`; total source data is `< 7B` tokens, with the exercise stage adding only `~180M` tokens.
- **Quality filtering**: start from the Python subset of deduplicated The Stack plus StackOverflow, `>35M` files and `>35B` tokens; label about `100k` samples with GPT-4 for "educational value", then train a random-forest classifier on pretrained CodeGen embeddings to keep higher-value examples, yielding a filtered corpus of about `6B` tokens.
- **Synthetic textbooks**: generate `< 1B` GPT-3.5 tokens of Python textbook-style material that interleaves exposition with code; diversity is induced by varying topics and target audiences while emphasizing reasoning and basic algorithmic skills.
- **Synthetic exercises**: generate `~180M` GPT-3.5 tokens of docstring-to-function-completion tasks; diversity is encouraged by constraining function names, and the dataset is explicitly used as a supervised finetuning stage rather than pretraining continuation.
- **Backbone**: `phi-1` is a [[decoder-only-transformer]] with `24` layers, hidden size `2048`, MLP size `8192`, `32` attention heads of width `64`, and `1.3B` parameters; `phi-1-small` uses `20` layers, hidden size `1024`, MLP size `4096`, and `16` heads.
- **Attention and position encoding**: use [[flash-attention]] for MHA, parallel MHA/MLP block configuration, and [[rotary-positional-embedding]] with rotary dimension `32`; tokenizer is shared with `codegen-350M-mono`.
- **Training objective and optimization**: concatenate files with separator token `"{|endoft|}"`, slice sequences at length `2048`, and train with [[next-token-prediction]] using fp16, AdamW, linear-warmup-linear-decay schedule, attention/residual dropout `0.1`, and `8` Nvidia A100 GPUs via DeepSpeed.
- **Pretraining hyperparameters**: effective batch size `1024`, max learning rate `1e-3`, warmup `750` steps, weight decay `0.1`, total `36,000` steps; the `24,000`-step checkpoint is used as `phi-1-base`, corresponding to `~8` epochs and a little over `50B` seen tokens.
- **Finetuning hyperparameters**: effective batch size `256`, max learning rate `1e-4`, warmup `50` steps, weight decay `0.01`, total `6,000` steps with checkpoint selection every `1,000` steps.
- **Evaluation extensions**: besides standard pass@1 on HumanEval/MBPP, the paper adds `50` unconventional coding problems graded by GPT-4 and a pruning-based [[data-decontamination]] study using embedding distance plus AST edit-distance matching with thresholds `τ ∈ {0.95, 0.9, 0.85, 0.8}`.

## Key Results

- `phi-1` (`1.3B`, `7B` source tokens) reaches `50.6%` pass@1 on [[humaneval]] and `55.5%` on [[mbpp]], outperforming substantially larger open models such as StarCoder on HumanEval (`33.6%`) and matching or beating many models trained on `52B` to `1T` tokens.
- `phi-1-base` already reaches `29%` HumanEval after pretraining on CodeTextbook alone, showing that the curated pretraining mixture contributes most of the gain before the exercise finetuning stage.
- For `350M` models, HumanEval rises from `12.19%` on unfiltered The Stack/StackOverflow after `96k` steps (`~200B` seen tokens) to `17.68%` with the filtered subset, and then to `20.12%` when synthetic textbooks are added.
- On `50` unconventional GPT-4-graded problems, `phi-1` scores `52`, above StarCoder's `51`, `phi-1-base`'s `37`, and CodeGen-Mono-16.1B's `38`; `phi-1-small` reaches `45`.
- The decontamination-style pruning study removes between `42.5K` and `354K` of `879.5K` CodeExercises problems; even after pruning more than `40%` of the finetuning set, retrained `phi-1` remains at `45.1%` to `50.6%` HumanEval, still above StarCoder-Prompted's `41.5%`.
- Efficiency claim: `phi-1-base` is obtained in under `4` days on `8` A100s, and the final finetuning stage takes only `7` more hours on the same hardware.

## Limitations

- The model is specialized for Python and underperforms larger systems on multi-language coding, rare packages, and domain-specific APIs.
- Robustness is limited: the authors note sensitivity to stylistic variation and grammatical errors in prompts.
- Some synthetic-data generation details are intentionally omitted for proprietary reasons, which makes the recipe harder to reproduce exactly.
- Part of the auxiliary evaluation relies on GPT-4 grading rather than purely executable tests, introducing judge-model dependence.
- The paper studies contamination carefully for HumanEval-style overlap, but the analysis is benchmark-specific rather than a universal guarantee against leakage.
- The authors explicitly acknowledge that GPT-3.5 synthetic data contains many errors and that they still lack a principled way to measure diversity and redundancy in the generated corpora.

## Concepts Extracted

- [[large-language-model]]
- [[code-generation]]
- [[decoder-only-transformer]]
- [[flash-attention]]
- [[rotary-positional-embedding]]
- [[next-token-prediction]]
- [[synthetic-data]]
- [[data-curation]]
- [[data-selection]]
- [[fine-tuning]]
- [[data-decontamination]]
- [[pass-at-k]]
- [[cross-task-generalization]]

## Entities Extracted

- [[suriya-gunasekar]]
- [[yi-zhang]]
- [[jyoti-aneja]]
- [[caio-cesar-teodoro-mendes]]
- [[allie-del-giorno]]
- [[sivakanth-gopi]]
- [[mojan-javaheripi]]
- [[piero-kauffmann]]
- [[gustavo-rosa]]
- [[olli-saarikivi]]
- [[adil-salim]]
- [[shital-shah]]
- [[harkirat-singh-behl]]
- [[xin-wang]]
- [[sebastien-bubeck]]
- [[ronen-eldan]]
- [[adam-tauman-kalai]]
- [[yin-tat-lee]]
- [[yuanzhi-li]]
- [[microsoft-research]]
- [[humaneval]]
- [[mbpp]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
