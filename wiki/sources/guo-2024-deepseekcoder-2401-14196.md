---
type: source
subtype: paper
title: "DeepSeek-Coder: When the Large Language Model Meets Programming -- The Rise of Code Intelligence"
slug: guo-2024-deepseekcoder-2401-14196
date: 2026-04-20
language: en
tags: [code-llm, pretraining, code-generation, code-completion, long-context]
processed: true

raw_file: raw/papers/guo-2024-deepseekcoder-2401-14196/paper.pdf
raw_md: raw/papers/guo-2024-deepseekcoder-2401-14196/paper.md
bibtex_file: raw/papers/guo-2024-deepseekcoder-2401-14196/paper.bib
possibly_outdated: false

authors:
  - Daya Guo
  - Qihao Zhu
  - Dejian Yang
  - Zhenda Xie
  - Kai Dong
  - Wentao Zhang
  - Guanting Chen
  - Xiao Bi
  - Y. Wu
  - Y. K. Li
  - Fuli Luo
  - Yingfei Xiong
  - Wenfeng Liang
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2401.14196
doi:
url: http://arxiv.org/abs/2401.14196
citation_key: guo2024deepseekcoder
paper_type: method

read_status: unread

domain: llm
---

## Summary

DeepSeek-Coder presents an open code-specialized large language model family with `1.3B`, `6.7B`, and `33B` parameters, trained from scratch on `2T` tokens drawn from `87` programming languages plus code-related natural language. The paper's main technical claim is that repository-level data construction, document-level Fill-in-the-Middle (FIM) pretraining, and `16K` context extension together materially improve practical coding ability, especially for cross-file completion. Architecturally the models are decoder-only Transformers with RoPE, while the `33B` variant adds grouped-query attention. Across HumanEval, MBPP, DS-1000, FIM, cross-file completion, and program-aided math reasoning, the strongest base model sets open-weight SOTA, and the `33B` instruct model exceeds GPT-3.5 Turbo on some coding benchmarks.

## Problem & Motivation

Open code LLMs had lagged substantially behind closed models such as GPT-3.5 and GPT-4, especially on realistic programming workloads that require cross-file context, code infilling, and strong natural-language instruction following. Prior open models were also commonly trained on file-level corpora, which weakens their ability to model repository structure and inter-file dependencies. DeepSeek-Coder is motivated by closing that gap with an openly released model family trained on a curated project-level corpus, with explicit support for infilling and long-context repository reasoning.

## Method

- **Training corpus**: pretraining data mixes `87%` source code, `10%` English code-related natural language, and `3%` Chinese natural language; the repository crawl keeps `87` programming languages and reduces raw code to `32.8%` of original volume after filtering.
- **Repository-level construction**: files are ordered with a dependency-analysis procedure that approximates topological sorting by repeatedly selecting the minimal in-degree node inside each disconnected subgraph, so dependency context appears before the target file in packed training samples.
- **Deduplication and filtering**: near-deduplication is applied at the repository level rather than the file level to preserve project structure; decontamination removes files matching benchmark n-grams, using `10`-gram filtering or exact match for shorter `3`- to `9`-gram snippets.
- **FIM objective**: document-level prefix-suffix-middle training uses three sentinel tokens and the template ``<|fim_start|> f_pre <|fim_hole|> f_suf <|fim_end|> f_middle <|eos_token|>``. Ablations compare `0%`, `50%`, and `100%` PSM FIM plus `50%` MSP; the final policy uses `0.5` FIM rate in PSM mode.
- **Tokenizer**: a BPE tokenizer is trained with vocabulary size ``32,000``.
- **Architecture**: all models are [[decoder-only-transformer]]s with SwiGLU activations and [[rotary-positional-embedding]]. The `1.3B` model uses hidden size ``2048``, intermediate size ``5504``, `24` layers, and `16` heads; the `6.7B` model uses ``4096 / 11008 / 32 / 32``; the `33B` model uses ``7168 / 19200 / 62 / 56`` and [[grouped-query-attention]] with group size ``8``.
- **Optimization**: AdamW uses ``beta1 = 0.9`` and ``beta2 = 0.95``. Learning-rate scheduling has `3` stages with `2000` warm-up steps, final LR at `10%` of initial LR, and each stage scaled by ``sqrt(1/10)`` from the preceding stage. Max learning rates are ``5.3e-4``, ``4.2e-4``, and ``3.5e-4`` for `1.3B`, `6.7B`, and `33B`.
- **Systems stack**: training runs on HAI-LLM with tensor parallelism, ZeRO data parallelism, and PipeDream pipeline parallelism on NVIDIA A100/H800 clusters; FlashAttention v2 is used for faster attention computation.
- **Long-context adaptation**: RoPE scaling factor is increased from `1` to `4`, base frequency from ``10000`` to ``100000``, then the model is trained for an additional `1000` steps with batch size ``512`` and sequence length ``16K``. The paper argues this should support up to `64K` tokens theoretically, though it reports best reliability within `16K`.
- **Instruction tuning**: DeepSeek-Coder-Instruct uses Alpaca-style data, delimiter token ``<|EOT|>``, cosine LR schedule with `100` warm-up steps, initial LR ``1e-5``, batch size ``4M`` tokens, and `2B` instruction-tuning tokens in total.

## Key Results

- On multilingual HumanEval, DeepSeek-Coder-Base `33B` reaches `50.3%` average pass@1 and `66.0%` on MBPP, improving over CodeLlama-Base `34B` by roughly `9` and `11` points respectively.
- DeepSeek-Coder-Base `6.7B` already surpasses CodeLlama-Base `34B` on the same setup, scoring `44.7%` average HumanEval and `60.6%` MBPP.
- On DS-1000, DeepSeek-Coder-Base `33B` achieves `40.2%` average, ahead of CodeLlama-Base `34B` at `34.3%`, with strong library-specific scores such as `56.1%` on Matplotlib and `49.6%` on NumPy.
- On the LeetCode Contest benchmark, DeepSeek-Coder-Instruct `33B` scores `27.8%` pass@1 and `28.9%` with CoT prompting, exceeding GPT-3.5-Turbo's `23.3%` while still trailing GPT-4-Turbo's `40.6%` to `41.8%`.
- On single-line infilling, DeepSeek-Coder-Base `33B` reaches `81.2%` mean exact match, outperforming CodeLlama-Base `13B` at `75.5%` and StarCoder `16B` at `69.7%`.
- On CrossCodeEval with retrieval, DeepSeek-Coder-Base improves exact match over other open models across Python, Java, JavaScript, and C#; for example Python exact match rises to `16.14%` with retrieval, versus `13.02%` for CodeLlama-Base.
- On program-aided math reasoning, DeepSeek-Coder-Base `33B` averages `65.8%` across seven benchmarks, including `60.7%` on GSM8K and `29.1%` on MATH.

## Limitations

- The paper is a technical report rather than a peer-reviewed conference paper and omits several training-cost details such as total GPU-hours and optimizer-state memory budget.
- The authors acknowledge that contamination cannot be fully ruled out for their LeetCode Contest benchmark, even though they collect recent problems.
- Cross-file evaluation depends on an external BM25 retriever and a fixed retrieval budget, so the reported gains mix model quality with a particular retrieval setup.
- The long-context extension is only empirically claimed to be reliable within `16K`, despite theoretical discussion of `64K`, so robustness beyond `16K` remains weakly validated.
- The paper compares against a selected benchmark suite focused on coding and program-aided reasoning; safety, instruction robustness, and real software-engineering workflows are much less deeply studied.

## Concepts Extracted

- [[code-language-model]]
- [[decoder-only-transformer]]
- [[rotary-positional-embedding]]
- [[grouped-query-attention]]
- [[fill-in-the-middle]]
- [[code-completion]]
- [[code-generation]]
- [[cross-file-context]]
- [[dependency-graph]]
- [[data-deduplication]]
- [[data-decontamination]]
- [[instruction-tuning]]

## Entities Extracted

- [[daya-guo]]
- [[qihao-zhu]]
- [[dejian-yang]]
- [[zhenda-xie]]
- [[kai-dong]]
- [[wentao-zhang]]
- [[guanting-chen]]
- [[xiao-bi]]
- [[y-wu]]
- [[y-k-li]]
- [[fuli-luo]]
- [[yingfei-xiong]]
- [[wenfeng-liang]]
- [[deepseek-ai]]
- [[peking-university]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
