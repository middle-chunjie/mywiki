---
type: source
subtype: paper
title: Shall We Pretrain Autoregressive Language Models with Retrieval? A Comprehensive Study
slug: wang-2023-shall
date: 2026-04-20
language: en
tags: [llm, retrieval, pretraining, open-domain-qa, factuality]
processed: true
raw_file: raw/papers/wang-2023-shall/paper.pdf
raw_md: raw/papers/wang-2023-shall/paper.md
bibtex_file: raw/papers/wang-2023-shall/paper.bib
possibly_outdated: true
authors:
  - Boxin Wang
  - Wei Ping
  - Peng Xu
  - Lawrence McAfee
  - Zihan Liu
  - Mohammad Shoeybi
  - Yi Dong
  - Oleksii Kuchaiev
  - Bo Li
  - Chaowei Xiao
  - Anima Anandkumar
  - Bryan Catanzaro
year: 2023
venue: EMNLP 2023
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.emnlp-main.482
url: https://aclanthology.org/2023.emnlp-main.482
citation_key: wang2023shall
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This EMNLP 2023 paper asks whether decoder-only language models should be pretrained with retrieval instead of adding retrieval only at fine-tuning or inference time. The authors reproduce RETRO from scratch at `148M` to `9.5B` parameters on a matched `330B`-token corpus, document the chunkwise retrieval and left-padding inference recipe, and compare against size-matched GPT baselines across perplexity, open-ended generation, factuality, toxicity, zero-shot downstream tasks, and open-domain QA. They also introduce RETRO++, which puts the top retrieved passage into the decoder context while leaving additional evidence in the encoder. Overall, retrieval-aware pretraining lowers perplexity, reduces repetition, improves several knowledge-intensive tasks, and strengthens QA, while exposing costs around datastore quality, toxicity control, and compute overhead.

## Problem & Motivation

Large autoregressive LMs can memorize a great deal of knowledge, but doing so purely in parameters is expensive, hard to update, and fragile with respect to factual accuracy. Prior retrieval-augmented work had shown better perplexity for RETRO-like models, yet it remained unclear whether retrieval pretraining actually helps open-ended generation quality and downstream task performance relative to standard GPT or retrieval added only later. This paper therefore asks whether retrieval should be part of decoder-only LM pretraining by default, and tries to answer it under controlled comparisons that match corpus, backbone family, and optimization schedule.

## Method

- **Matched backbone comparison**: both GPT and RETRO use the same Transformer decoder family, with sizes `12/24/24/40` layers, hidden sizes `768/1024/2048/4096`, attention heads `12/16/32/64`, and RETRO parameter counts `148M/410M/1.5B/9.5B`.
- **Chunkwise retrieval architecture**: the input `X = (x_1, ..., x_n)` is split into chunks `(C_1, ..., C_l)` with chunk size `m = 64`; generation of chunk `C_i` conditions on retrieved neighbors `N(C_{i-1})` from the previous chunk to preserve causality, fused through chunk-wise cross-attention.
- **Datastore and ANN index**: the `330B`-token pretraining corpus yields `5.3B` chunks; keys are frozen BERT embeddings and values are raw text chunks. Retrieval uses Faiss with `2^22` IVF centroids accelerated by HNSW, plus optimized product quantization and `64`-bit PQ codes; the paper reports about `4 ms` average per-query latency in batch mode and under `400 GB` memory at max throughput.
- **Pretraining recipe**: GPT and RETRO are trained from scratch on the same corpus with Adam, `β1 = 0.9`, `β2 = 0.95`, cosine LR decay, warmup samples `162761`, and decay samples `166400000`. Per-size schedules are Small `lr = 6e-4`, batch `256`, `750k` steps; Medium `3e-4`, batch `256`, `750k`; XL `2e-4`, batch `512`, `375k`; XXL `1e-4`, batch `512`, `375k`.
- **Generation-time retrieval**: RETRO requires a "left padding" rule so short contexts still have a previous chunk to retrieve from. Retrieval frequency is configurable from `1` token per refresh to `64` tokens per refresh, trading neighbor freshness against compute cost.
- **Text generation setup**: automatic generation evaluation uses nucleus sampling with `top-p = 0.9`, up to `200` generated tokens, and `top-k = 2` retrieved neighbors for RETRO.
- **Downstream QA finetuning**: RETRO preserves causality by left-padding context/question chunks and right-padding answer chunks for batching. For open-domain QA, RETRO++ feeds the top-`1` retrieved passage directly into the decoder prompt while additional evidence remains on the encoder side.
- **QA and instruction-tuning optimization**: task finetuning runs for `3` epochs with sequence length `2048`, batch size `512`, weight decay `0.1`, and size-dependent learning rates `1e-5` (Medium), `3e-6` (XL), `1e-6` (XXL). Separate instruction tuning on `128K` examples uses batch `128`, learning rate `5e-6`, `1000` steps, and Adam with `β1 = 0.9`, `β2 = 0.98`.

## Key Results

- **Perplexity**: RETRO beats matched GPT at every reported size, improving validation perplexity from `17.76 -> 12.99` (Small), `13.18 -> 10.06` (Medium), `10.18 -> 8.10` (XL), and `7.86 -> 6.72` (XXL).
- **Text degeneration**: repetition drops across all sizes, e.g. `2.86% -> 2.26%` (Small), `1.70% -> 1.50%` (Medium), `1.44% -> 0.96%` (XL), and `1.40% -> 1.12%` (XXL); the paper summarizes this as about `21%` average repetition reduction.
- **Human text quality**: RETRO remains essentially on par with GPT in human judgments while slightly improving both relevance and fluency, from `3.715 -> 3.726` for relevance and `3.818 -> 3.826` for fluency.
- **Factuality and toxicity**: on TruthfulQA null-format evaluation, RETRO improves MC1 from `0.234 -> 0.248` and MC2 from `0.435 -> 0.439`. Toxicity depends strongly on datastore quality: with the pretraining corpus, full-set toxicity probability worsens from `37% -> 40%`, but with a Wikipedia database it improves to `35%`.
- **Zero-shot downstream tasks**: average LM Evaluation Harness accuracy improves from `46.7 -> 47.4` (Small), `50.0 -> 50.4` (Medium), `54.3 -> 55.0` (XL), and `60.0 -> 60.5` (XXL). Knowledge-intensive tasks show the clearest gains, including HellaSwag `31.3 -> 36.2` at Small and BoolQ `67.3 -> 70.7` at XXL.
- **Open-domain QA**: on Natural Questions, RETRO reaches `40.9` EM and RETRO++ raises this to `54.1`, exceeding `RAG_GPT` at `50.9` and the original RETRO report at `45.5`. On TriviaQA, RETRO++ reaches `66.7` versus `59.9` for the authors' RETRO model.

## Limitations

- Retrieval quality is a first-order dependency: toxic or low-quality neighbors can amplify harmful output, and the paper explicitly shows worse toxicity with the raw pretraining datastore than with a curated Wikipedia database.
- The system is resource-intensive: pretraining uses a `330B`-token corpus, a `5.3B`-chunk datastore, and less than `400 GB` retrieval memory at peak throughput; RETRO adds roughly `12%` to `25.8%` GPU-hour overhead versus GPT depending on model size.
- Gains are uneven across tasks. Some non-knowledge-intensive benchmarks are flat or worse than GPT, such as RACE `34.6 -> 32.5` at Small and HellaSwag `72.3 -> 70.6` at XXL.
- Their reproduced RETRO still trails the original RETRO paper on NQ (`40.9` vs. `45.5`) because it uses `330B` rather than `600B` training tokens, so some conclusions depend on scale and data budget.

## Concepts Extracted

- [[retrieval-augmented-language-model]]
- [[retrieval-augmented-generation]]
- [[dense-retrieval]]
- [[approximate-nearest-neighbor-search]]
- [[nearest-neighbor-search]]
- [[open-domain-question-answering]]
- [[factuality]]
- [[hallucination]]
- [[toxicity]]
- [[in-context-learning]]
- [[decoder-only-language-model]]
- [[autoregressive-language-model]]

## Entities Extracted

- [[boxin-wang]]
- [[wei-ping]]
- [[peng-xu]]
- [[lawrence-mcafee]]
- [[zihan-liu]]
- [[mohammad-shoeybi]]
- [[yi-dong]]
- [[oleksii-kuchaiev]]
- [[bo-li]]
- [[chaowei-xiao]]
- [[anima-anandkumar]]
- [[bryan-catanzaro]]
- [[nvidia]]
- [[faiss]]
- [[natural-questions]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
