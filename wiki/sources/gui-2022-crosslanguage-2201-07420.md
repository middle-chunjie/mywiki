---
type: source
subtype: paper
title: Cross-Language Binary-Source Code Matching with Intermediate Representations
slug: gui-2022-crosslanguage-2201-07420
date: 2026-04-20
language: en
tags: [code-matching, clone-detection, llvm-ir, transformer, binary-analysis]
processed: true

raw_file: raw/papers/gui-2022-crosslanguage-2201-07420/paper.pdf
raw_md: raw/papers/gui-2022-crosslanguage-2201-07420/paper.md
bibtex_file: raw/papers/gui-2022-crosslanguage-2201-07420/paper.bib
possibly_outdated: false

authors:
  - Yi Gui
  - Yao Wan
  - Hongyu Zhang
  - Huifang Huang
  - Yulei Sui
  - Guandong Xu
  - Zhiyuan Shao
  - Hai Jin
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2201.07420
doi:
url: http://arxiv.org/abs/2201.07420
citation_key: gui2022crosslanguage
paper_type: method

read_status: unread
read_date:
rating:

domain: software-engineering
---

## Summary

The paper introduces XLIR, a Transformer-based method for cross-language binary-source code matching that first converts both source code and compiled binaries into LLVM-IR, then learns a shared embedding space for similarity search. Instead of aligning raw syntax or hand-crafted binary features, the model uses a pre-trained IR-BERT encoder, Byte Pair Encoding tokenization, and triplet-loss fine-tuning with cosine similarity. The authors also curate a new benchmark derived from CLCDSA by compiling each source file with multiple compilers, optimization levels, and architectures. Across cross-language binary-source matching, single-language binary-source matching, and cross-language source-source clone detection, XLIR consistently outperforms prior systems and an LSTM ablation, supporting the claim that IR-level normalization reduces both language and modality gaps.

## Problem & Motivation

Existing binary-source matching methods mainly assume a single programming language and try to align binary code with source code directly in their native representations. That setting is restrictive for real systems, where source implementations may span C, C++, Java, and other languages while deployed artifacts remain binaries. The paper argues that the key obstacle is the semantic gap between high-level source code and low-level machine code, especially across languages. Its core motivation is that an intermediate representation can normalize away part of that gap because both source code and binaries can be translated into semantically closer LLVM-IR with overlapping vocabulary and structure.

## Method

- **Problem reformulation**: instead of directly learning `S -> V_S` and `B -> V_B`, XLIR parses source and binary code into IRs first, i.e. `S -> S_r -> V_S` and `B -> B_r -> V_B`, then compares the resulting embeddings in a shared space.
- **IR choice**: the method uses [[llvm-ir]] because it is source-independent and target-independent; source code is translated with [[llvm-clang]] / JLang / Polyglot, while binaries are decompiled with [[retdec]].
- **Leakage control**: compilation strips debug symbols with `-s` so function and variable names do not trivially leak into binaries.
- **IR embedding**: XLIR uses a `K`-layer [[transformer]] encoder with self-attention, following BERT-style settings; the paper reports hidden size `256` and word embedding size `256`.
- **Pretraining**: the encoder is initialized from [[ir-bert]], a masked language model over LLVM-IR. Inputs are tokenized with [[byte-pair-encoding]], `15%` of tokens are selected for corruption, and the selected subset follows an `80% [MASK] / 10% random / 10% unchanged` policy.
- **Representation learning**: for a triplet `<b, s+, s->`, XLIR minimizes ranking loss `` `L = sum max(0, alpha - sim(b, s+) + sim(b, s-))` `` with margin `` `alpha = 0.06` `` and cosine similarity `` `sim(b, s) = (b^T s) / (||b|| ||s||)` ``.
- **Training setup**: implementation uses PyTorch `1.9`, the paper reports an `ADM` optimizer with learning rate `` `1e-3` ``, dropout `` `0.4` ``, and distributed training on `4 x Tesla V100 32GB` GPUs.
- **Inference rule**: clone matching uses cosine similarity with default threshold `` `0.8` ``; pairs above threshold are treated as matched.
- **Data construction**: the new binary-source benchmark is built from [[clcdsa]] by compiling each source file with `2` compilers (`GCC`, `LLVM Clang`), `4` optimization levels (`-O0` to `-O3`), and `4` architectures (`x86-32`, `x86-64`, `arm-32`, `arm-64`), yielding `32` object files per source program.

## Key Results

- On cross-language binary-source matching, XLIR (Transformer) reaches `0.73 / 0.59 / 0.65` Precision/Recall/F1 for `C/C++ binary -> Java source`, versus `0.62 / 0.53 / 0.57` for the LSTM ablation.
- On `Java binary -> C/C++ source`, XLIR (Transformer) achieves `0.68 / 0.55 / 0.61`, improving over prior baselines B2SFinder (`0.35 / 0.41 / 0.38`) and BinPro (`0.36 / 0.37 / 0.36`).
- On single-language `C++ binary -> C++ source` matching over [[poj-104]], XLIR (Transformer) obtains `0.85 / 0.86 / 0.85`, compared with B2SFinder's `0.43 / 0.46 / 0.44`.
- On cross-language source-source clone detection, XLIR (Transformer) achieves F1 scores of `0.89` on `C/C++`, `0.63` on `C/Java`, and `0.66` on `C++/Java`, all above LICCA.
- Removing pretraining reduces average Precision/Recall/F1 on cross-language clone detection by about `0.04 / 0.03 / 0.03`, indicating that IR-level MLM pretraining is materially useful.
- Across POJ-104 compilation settings, average Precision/Recall/F1 stays around `0.85`, with reported variances `0.0006 / 0.0004 / 0.0002`, suggesting robustness to compiler, optimization, and architecture changes.

## Limitations

- The approach assumes source code is compilable and binaries are decompilable into LLVM-IR; incomplete code and heavily obfuscated binaries fall outside the intended operating regime.
- Language coverage depends on the availability of static LLVM-based compilation pipelines, which excludes or complicates some dynamic languages such as Python.
- Evaluation is dominated by function-level clone retrieval and binary-source matching; finer-grained snippet-level matching is left to future work.
- The method inherits the computational and tooling dependencies of the LLVM ecosystem, including decompilation quality and IR conversion fidelity.

## Concepts Extracted

- [[intermediate-representation]]
- [[llvm-ir]]
- [[binary-source-code-matching]]
- [[code-clone-detection]]
- [[triplet-loss]]
- [[masked-language-modeling]]
- [[transformer]]
- [[self-attention]]
- [[byte-pair-encoding]]

## Entities Extracted

- [[yi-gui]]
- [[yao-wan]]
- [[hongyu-zhang]]
- [[huifang-huang]]
- [[yulei-sui]]
- [[guandong-xu]]
- [[zhiyuan-shao]]
- [[hai-jin]]
- [[xlir]]
- [[ir-bert]]
- [[llvm-clang]]
- [[retdec]]
- [[clcdsa]]
- [[poj-104]]
- [[naturalcc]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
