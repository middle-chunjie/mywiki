---
type: source
subtype: paper
title: "OLMo: Accelerating the Science of Language Models"
slug: groeneveld-2024-olmo-2402-00838
date: 2026-04-20
language: en
tags: [llm, open-models, pretraining, evaluation, alignment]
processed: true

raw_file: raw/papers/groeneveld-2024-olmo-2402-00838/paper.pdf
raw_md: raw/papers/groeneveld-2024-olmo-2402-00838/paper.md
bibtex_file: raw/papers/groeneveld-2024-olmo-2402-00838/paper.bib
possibly_outdated: false

authors:
  - Dirk Groeneveld
  - Iz Beltagy
  - Pete Walsh
  - Akshita Bhagia
  - Rodney Kinney
  - Oyvind Tafjord
  - Ananya Harsh Jha
  - Hamish Ivison
  - Ian Magnusson
  - Yizhong Wang
  - Shane Arora
  - David Atkinson
  - Russell Authur
  - Khyathi Raghavi Chandu
  - Arman Cohan
  - Jennifer Dumas
  - Yanai Elazar
  - Yuling Gu
  - Jack Hessel
  - Tushar Khot
  - William Merrill
  - Jacob Morrison
  - Niklas Muennighoff
  - Aakanksha Naik
  - Crystal Nam
  - Matthew E. Peters
  - Valentina Pyatkin
  - Abhilasha Ravichander
  - Dustin Schwenk
  - Saurabh Shah
  - Will Smith
  - Emma Strubell
  - Nishant Subramani
  - Mitchell Wortsman
  - Pradeep Dasigi
  - Nathan Lambert
  - Kyle Richardson
  - Luke Zettlemoyer
  - Jesse Dodge
  - Kyle Lo
  - Luca Soldaini
  - Noah A. Smith
  - Hannaneh Hajishirzi
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.00838
doi: 10.48550/arXiv.2402.00838
url: http://arxiv.org/abs/2402.00838
citation_key: groeneveld2024olmo
paper_type: method

read_status: unread
domain: llm
---

## Summary

OLMo is a technical report on a fully open large language model release aimed at making language-model science reproducible rather than merely consumable through APIs or weights-only drops. The project releases 1B and 7B decoder-only models together with Dolma pretraining data, training code, evaluation code, adaptation code, intermediate checkpoints, and logs. Architecturally, OLMo follows a LLaMA-style decoder with no biases, non-parametric layer normalization, SwiGLU activations, RoPE, and a modified GPT-NeoX tokenizer, and trains the 7B model to `2.46T` tokens with AdamW plus FSDP/ZeRO on both AMD and NVIDIA clusters. Results show competitive zero-shot performance, strong post-training gains from SFT and DPO, explicit perplexity decontamination, and unusually transparent reporting of energy use and released artifacts.

## Problem & Motivation

The paper argues that research on large language models has become bottlenecked by partial releases: many high-capability systems expose only weights or APIs, while withholding crucial details about training data, architecture decisions, optimization, checkpoints, and evaluation procedures. That opacity makes it difficult to study model capabilities, biases, safety risks, data effects, and reproducibility. OLMo is positioned as an answer to that problem: a state-of-the-art open LM framework whose scientific value comes from releasing the full stack, not just a runnable checkpoint.

## Method

- **Release scope**: OLMo is designed as an end-to-end open framework spanning pretrained checkpoints, intermediate checkpoints, training logs, data artifacts, evaluation code, and adaptation code instead of a weights-only release.
- **Base architecture**: decoder-only Transformer with `16` layers / `d_model = 2048` / `16` heads at `1B`, and `32` layers / `d_model = 4096` / `32` heads at `7B`; context length is `2048`.
- **Architectural updates over vanilla Transformer**: removes all bias terms, uses non-parametric layer normalization, adopts SwiGLU with activation hidden size `≈ 8d/3` (`11008` for `7B`), and replaces absolute positions with [[rotary-positional-embedding]].
- **Tokenizer and vocabulary**: modified GPT-NeoX BPE tokenizer with additional PII-masking tokens; vocabulary size is `50280`, while the embedding matrix is padded to `50304` for throughput alignment.
- **Pretraining data**: trains on a `2T`-token sample from Dolma, itself a `3T`-token, `5B`-document corpus from `7` public sources; documents receive EOS markers, are concatenated, chunked into `2048`-token sequences, and shuffled deterministically so batches are reconstructible.
- **Optimization**: AdamW with `lr_peak = 3e-4`, `lr_min = 3e-5`, `β = (0.9, 0.95)`, `ε = 1e-5`, weight decay `0.1`, warmup `5000` steps, linear decay, and global gradient clipping at `1.0` for `7B`.
- **Distributed training**: PyTorch [[fully-sharded-data-parallel]] plus ZeRO sharding; `7B` uses global batch `~4M` tokens and micro-batch `4096` tokens per GPU.
- **Precision strategy**: [[mixed-precision-training]] with BF16 for most operations, but softmax, optimizer state, and gradient reduction remain in full precision for stability.
- **Evaluation loop**: online evaluation runs every `1000` steps (`~4B` tokens); offline evaluation uses Catwalk zero-shot tasks plus Paloma bits-per-byte analysis with explicit [[data-decontamination]] against leaked evaluation paragraphs.
- **Post-training**: adaptation uses Open Instruct / Tulu-style supervised fine-tuning with `lr = 2e-6`, then DPO with `lr = 5e-7`, `β = 0.1`, `3` epochs, and maximum sequence length `2048`.
- **Training hardware**: the authors train on LUMI with up to `256` nodes of `4x` AMD MI250X each, and on MosaicML with `27` nodes of `8x` NVIDIA A100-40GB each, and compare parity across both platforms.

## Key Results

- **Release coverage**: the first release includes one `1B` model, four `7B` variants trained on at least `2T` tokens, a `7B` evaluation checkpoint at `2.46T` tokens, and `500+` intermediate checkpoints.
- **Zero-shot downstream average**: OLMo-7B scores `69.3` average across `8` core tasks, including `48.5` ARC-Challenge, `65.4` ARC-Easy, `73.4` BoolQ, `76.4` HellaSwag, `50.4` OpenBookQA, `78.4` PIQA, `93.8` SciQ, and `67.9` WinoGrande.
- **Adaptation gains**: MMLU improves from `28.3` (base) to `47.3` (+SFT) and `46.2` (+SFT+DPO); AlpacaEval win rate reaches `57.0%` after SFT and `69.3%` after DPO.
- **Safety/truthfulness gains**: ToxiGen toxic rate drops from `81.4%` (base) to `14.4%` (+SFT) and `1.7%` (+DPO), while TruthfulQA informative-and-true rises from `31.6` to `41.2` to `52.0`.
- **Energy and emissions**: the authors estimate total pretraining energy at `239 MWh`; OLMo-7B is reported as `0 tCO2eq` on MI250X under the official LUMI assumption, versus about `69.78 tCO2eq` on A100-40GB in Australia.

## Limitations

- The main release and quantitative comparisons are concentrated at `7B`; the `65B` model was still training at the time of writing, so the report does not yet establish competitiveness at larger scales.
- Context length remains `2048`, so the work does not address long-context scaling or retrieval-augmented serving as part of the base model design.
- The paper reports that OLMo is less sample-efficient on some Paloma sources less aligned with Common Crawl, such as WikiText-103, M2D2 S2ORC, and M2D2 Wikipedia.
- Even after SFT and DPO, OLMo still trails Tulu 2 on some chat-oriented evaluations, and the authors leave the gap partly unresolved.
- Carbon estimates are explicitly lower bounds because they omit embodied emissions, debugging, tuning overhead, downtime, water use, and other lifecycle effects.

## Concepts Extracted

- [[large-language-model]]
- [[open-language-model]]
- [[decoder-only-transformer]]
- [[rotary-positional-embedding]]
- [[swiglu]]
- [[byte-pair-encoding]]
- [[fully-sharded-data-parallel]]
- [[mixed-precision-training]]
- [[instruction-tuning]]
- [[direct-preference-optimization]]
- [[data-decontamination]]
- [[data-deduplication]]

## Entities Extracted

- [[dirk-groeneveld]]
- [[iz-beltagy]]
- [[pete-walsh]]
- [[akshita-bhagia]]
- [[rodney-kinney]]
- [[oyvind-tafjord]]
- [[ananya-harsh-jha]]
- [[hamish-ivison]]
- [[ian-magnusson]]
- [[yizhong-wang]]
- [[shane-arora]]
- [[david-atkinson]]
- [[russell-authur]]
- [[khyathi-raghavi-chandu]]
- [[arman-cohan]]
- [[jennifer-dumas]]
- [[yanai-elazar]]
- [[yuling-gu]]
- [[jack-hessel]]
- [[tushar-khot]]
- [[william-merrill]]
- [[jacob-morrison]]
- [[niklas-muennighoff]]
- [[aakanksha-naik]]
- [[crystal-nam]]
- [[matthew-e-peters]]
- [[valentina-pyatkin]]
- [[abhilasha-ravichander]]
- [[dustin-schwenk]]
- [[saurabh-shah]]
- [[will-smith]]
- [[emma-strubell]]
- [[nishant-subramani]]
- [[mitchell-wortsman]]
- [[pradeep-dasigi]]
- [[nathan-lambert]]
- [[kyle-richardson]]
- [[luke-zettlemoyer]]
- [[jesse-dodge]]
- [[kyle-lo]]
- [[luca-soldaini]]
- [[noah-a-smith]]
- [[hannaneh-hajishirzi]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
