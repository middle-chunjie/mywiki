---
type: source
subtype: paper
title: "LLMLingua: Compressing Prompts for Accelerated Inference of Large Language Models"
slug: jiang-2023-llmlingua-2310-05736
date: 2026-04-20
language: en
tags: [llm, prompt-compression, inference-efficiency, prompting]
processed: true

raw_file: raw/papers/jiang-2023-llmlingua-2310-05736/paper.pdf
raw_md: raw/papers/jiang-2023-llmlingua-2310-05736/paper.md
bibtex_file: raw/papers/jiang-2023-llmlingua-2310-05736/paper.bib
possibly_outdated: true

authors:
  - Huiqiang Jiang
  - Qianhui Wu
  - Chin-Yew Lin
  - Yuqing Yang
  - Lili Qiu
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.05736
doi:
url: http://arxiv.org/abs/2310.05736
citation_key: jiang2023llmlingua
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

LLMLingua studies prompt compression for black-box large language models where weights and KV-cache internals are inaccessible, so model-side pruning or quantization is not applicable. The paper proposes a coarse-to-fine pipeline that first allocates different compression budgets to instructions, demonstrations, and the query, then keeps high-value tokens with an iterative perplexity-based compressor, and finally instruction-tunes the small compression model to better match the target LLM distribution. Across GSM8K, BBH, ShareGPT, and Arxiv-March23, the method consistently beats Selective-Context and generation-based compression baselines, reaching up to `20x` prompt compression with limited quality loss and producing end-to-end latency speedups up to `5.7x`.

## Problem & Motivation

Long prompts are increasingly required for chain-of-thought prompting, in-context learning, dialogue history, and retrieved context, but they directly increase API latency and inference cost for closed LLMs. Prior prompt-compression approaches either depend on task-specific tuning, multiple expensive LLM calls, or one-shot self-information filtering that ignores dependencies among kept tokens and mismatches between the small compressor model and the target black-box LLM. LLMLingua is motivated by the need to compress prompts aggressively while preserving reasoning traces, semantic integrity, and downstream task behavior under an explicit token budget.

## Method

- **Problem setup**: given prompt `x = (x^ins, x^dems, x^que)` and compressed prompt `x~`, optimize similarity between generation distributions by minimizing `KL(P(x~_G | x~), P(x_G | x))` under compression rate `τ = L~/L`.
- **Budget controller**: allocate separate budgets to instruction, demonstrations, and question because instructions/questions are more sensitive than demonstrations. Demonstration compression rate is `τ_dems = (τL - (τ_ins L_ins + τ_que L_que)) / L_dems`.
- **Coarse-grained selection**: compute each demonstration's perplexity with a small LM `M_s`, rank demonstrations by descending perplexity, and keep them until selected length exceeds `k · τ_dems · L_dems`. The paper uses granular control coefficient `k = 2`.
- **Budget reallocation**: after coarse demonstration filtering, redistribute unused budget to instruction and question with `Δτ = (k · τ_dems · L_dems - L_D~) / (L_ins + L_que)`.
- **Iterative Token-level Prompt Compression**: segment the remaining prompt into chunks `S = {s_1, ..., s_m}` and compress each segment conditioned on previously preserved compressed segments, rather than scoring all tokens independently once.
- **Conditional scoring**: estimate segment probabilities with `p(s~_j) ≈ ∏ p(s_{j,i} | s_{j,<i}, s~_{<j})`, then set segment-specific thresholds `γ_j` according to whether the segment comes from instruction, demonstrations, or question.
- **Token retention rule**: keep tokens satisfying `p(s_{j,i}) > γ_j`; this targets high-perplexity, information-bearing tokens while modeling interdependence across retained content.
- **Distribution alignment**: instruction-tune the small compressor model on LLM-generated instruction-response pairs to reduce mismatch between the small model distribution and the target black-box LLM, optimizing `E[(1/N) Σ L(x_i, y_i^LLM; θ_M_s)]`.
- **Implementation hyperparameters**: target LLMs are GPT-3.5-Turbo-0301 and Claude-v1.3 with greedy decoding and temperature `0`; small models are Alpaca-7B or GPT2-Alpaca; pre-defined rates are `τ_ins = 0.85`, `τ_que = 0.9`; ITPC segment size is `100`.

## Key Results

- On GSM8K under the `1-shot` constraint, LLMLingua reaches `79.08` EM with `446` prompt tokens at `5x` compression, slightly exceeding the full-shot baseline (`78.85` EM, `2366` tokens).
- On GSM8K under the `quarter-shot` constraint, it still attains `77.33` EM with only `117` tokens at `20x` compression, versus `44.20` for Selective-Context and `56.33` for GPT-4 Generation.
- On BBH, the method reaches `70.11` EM at `3x` compression in `1-shot`, and `56.85` EM at `7x` compression in `quarter-shot`, outperforming Selective-Context (`54.27` and `47.37` respectively).
- On ShareGPT, LLMLingua achieves BERTScore F1 `89.52` with `304` tokens at `1.9x` compression and `87.70` with `177` tokens at `3.3x` compression.
- On Arxiv-March23 summarization, it reaches BERTScore F1 `90.33` at `345` tokens (`4x`) and `89.03` at `176` tokens (`9x`), beating sentence selection and Selective-Context on all reported overlap metrics.
- Ablation on GSM8K `1-shot`: removing ITPC drops EM from `79.08` to `72.93`; removing the budget controller drops to `73.62`; random selection in the controller drops to `72.78`; removing distribution alignment still costs `0.46` EM.
- Efficiency results show end-to-end latency on GSM8K falls from `8.6s` without compression to `2.3s` at `5x` and `1.3s` at `10x`, while LLMLingua's own overhead is only `0.3s` and `0.2s`.

## Limitations

- Performance still degrades sharply at very high compression ratios such as `25x-30x`, even though the degradation point is shifted later than competing methods.
- The maximum safe compression ratio depends on prompt length, task type, and sentence structure, so the method does not offer a universal operating point.
- Tokenizer mismatch between the small compressor LM and the target black-box LLM can underestimate actual target-model token length.
- Distribution alignment narrows but does not eliminate the gap between weaker small models and stronger target LLMs; GPT2-Alpaca remains noticeably worse than Alpaca-7B.

## Concepts Extracted

- [[prompt-compression]]
- [[coarse-to-fine-prompt-compression]]
- [[budget-control]]
- [[iterative-token-level-prompt-compression]]
- [[distribution-alignment]]
- [[instruction-tuning]]
- [[perplexity]]
- [[in-context-learning]]
- [[chain-of-thought]]
- [[demonstration-selection]]
- [[large-language-model]]

## Entities Extracted

- [[huiqiang-jiang]]
- [[qianhui-wu]]
- [[chin-yew-lin]]
- [[yuqing-yang]]
- [[lili-qiu]]
- [[microsoft]]
- [[gpt-3-5-turbo-0301]]
- [[claude-v1-3]]
- [[alpaca-7b]]
- [[gpt2-alpaca]]
- [[gsm8k]]
- [[bbh]]
- [[sharegpt]]
- [[arxiv-march23]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
