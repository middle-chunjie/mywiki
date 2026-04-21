---
type: source
subtype: paper
title: Multilingual Code Snippets Training for Program Translation
slug: zhu-2022-multilingual
date: 2026-04-20
language: en
tags: [program-translation, code-generation, multilingual, pretraining, nlp]
processed: true
raw_file: raw/papers/zhu-2022-multilingual/paper.pdf
raw_md: raw/papers/zhu-2022-multilingual/paper.md
bibtex_file: raw/papers/zhu-2022-multilingual/paper.bib
possibly_outdated: true
authors:
  - Ming Zhu
  - Karthik Suresh
  - Chandan K. Reddy
year: 2022
venue: AAAI 2022
venue_type: conference
arxiv_id:
doi: 10.1609/aaai.v36i10.21434
url: https://ojs.aaai.org/index.php/AAAI/article/view/21434
citation_key: zhu2022multilingual
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

This paper addresses the scarcity of high-quality parallel data for program translation by introducing CoST, a multilingual code snippet dataset covering 7 programming languages (C, C++, C#, Java, Python, JavaScript, PHP) with snippet-level alignment across 1,625 programming problems and up to 42 language pairs. Snippet-level alignment provides finer supervision than prior method-level or program-level datasets. The paper also proposes MuST-PT, a Transformer-based model pre-trained with multilingual snippet denoising auto-encoding (DAE) and multilingual snippet translation (MuST) objectives, initialized from DOBF weights. MuST-PT achieves state-of-the-art BLEU on the CodeXGLUE translation task and shows strong gains on low-resource language pairs; MuST pre-training is shown to generalize, consistently improving several baseline models.

## Problem & Motivation

Existing code translation datasets (e.g., CodeXGLUE Java-C#) contain only two languages aligned at the method level via matching function names from open-source projects, limiting both language coverage and alignment granularity. Program-level datasets such as Google Code Jam and Project CodeNet align by task semantics rather than code structure, introducing high variance in variable names, logic flow, and method design. This variance is especially harmful for low-resource languages with limited parallel data. The absence of snippet-level parallel data across many languages blocks fine-grained supervised training and multilingual transfer.

## Method

- **Dataset — CoST**: Scraped from GeeksForGeeks where contributors follow a comment-based template, enabling one-to-one correspondence between code snippets across languages. Includes 132,046 pairwise snippet samples and ~14K pairwise programs per major language pair. Snippet boundaries are defined by matching code comments; misalignments were manually verified and corrected.
- **Model architecture**: Encoder-decoder Transformer; encoder has `12` layers, decoder `6` layers, `d_model = 768`, `12` attention heads. A language identifier `α_{l_i}` is added to each input token embedding: `(x_1 + α_{l_i}, ..., x_n + α_{l_i})`, so the encoder represents all languages in a shared latent space and the decoder is conditioned on the target language.
- **Initialization**: Weights loaded from `dobf_plus_denoising.pth` (DOBF model), which was pre-trained on Python and Java with masked language modeling and code deobfuscation objectives.
- **Stage 1 — Multilingual Snippet DAE**: Trains the model on all 7 languages using the denoising auto-encoding objective. Noise functions (random word shuffle, word dropout, span masking) are drawn from the TransCoder recipe. Objective: `L_DAE(θ_E, θ_G) = Σ_{l_i ∈ L} E_{x~D_mono, x̃~C(x)} [-log p_G(x | E(x̃, α_{l_i}), α_{l_i})]`. DAE weight `λ` starts at `1.0`, decays linearly to `0.1` at `30K` steps, then to `0` at `100K` steps. Uses only monolingual data.
- **Stage 2 — Multilingual Snippet Translation (MuST)**: Fine-tunes on bilingual snippet pairs across all language pairs simultaneously. Objective: `L_M(θ_E, θ_G) = Σ_{l_i, l_j ∈ L} E_{(x,y)~D_bi} [-log p_G(y | E(x, α_{l_i}), α_{l_j})]`. Combined loss: `L = L_M + λ L_DAE`.
- **Stage 3 — Program-level fine-tuning**: Fine-tunes on program-level pairwise data from CoST to bridge the snippet-to-program distribution gap, using the same multilingual strategy.
- **Optimizer**: Adam `lr=0.0001`, same learning rate scheduler as Transformer (Vaswani et al. 2017). Batch size `128`; trained on `4×RTX 8000` (48 GB each). FP16 used for speed.
- **Evaluation metrics**: BLEU (n-gram overlap) and CodeBLEU (extends BLEU with AST syntax and data-flow graph matching).

## Key Results

- MuST-PT achieves **87.37 BLEU** on Java→C# and **85.25 BLEU** on C#→Java (CodeXGLUE), surpassing PLBART (83.02 / 78.35) and all prior models.
- On CoST snippet-level translation, MuST-PT outperforms DOBF across all 42 language pairs; largest gains on low-resource pairs: e.g., C++→PHP BLEU `83.29` vs. DOBF `77.91`, C→C++ `88.58` vs. DOBF `76.85`.
- On CoST program-level translation, most baselines degrade sharply from snippet-level (e.g., DOBF C++→Java drops from `79.83` to `29.06`), while MuST-PT retains near-snippet performance (C++→Java `79.15`).
- Generalizability of MuST: Transformer+MuST on Java→C# improves from `47.34` to `73.70` BLEU; CodeBERT+MuST improves from `85.46` to `90.47`; TransCoder+MuST improves from `44.85` to `91.74` on Java→C#.
- Snippet-level vs program-level gap analysis shows MuST pre-training specifically helps bridge this distribution mismatch.

## Limitations

- Coverage limited to 7 programming languages; languages without GeeksForGeeks template compliance are excluded.
- Data collection relies on a single platform (GeeksForGeeks), which may introduce domain and style biases.
- TransCoder+MuST shows inconsistent gains (C#→Java drops from `29.4` to `27.7`), suggesting MuST does not universally benefit all model types.
- DOBF was only pre-trained on Python and Java; DAE is used to cover remaining 5 languages, which may not be optimal for very different languages (e.g., PHP).
- Evaluation uses BLEU and CodeBLEU only; functional correctness (e.g., test-case pass rates) is not measured.
- Program-level fine-tuning data is smaller and noisier than snippet data, leaving room for improvement on longer programs.

## Concepts Extracted

- [[code-translation]]
- [[denoising-autoencoding]]
- [[back-translation]]
- [[multilingual-machine-translation]]
- [[neural-machine-translation]]
- [[masked-language-modeling]]
- [[sequence-to-sequence]]
- [[bleu]]
- [[codebleu]]
- [[low-resource-language]]
- [[abstract-syntax-tree]]
- [[program-translation]]
- [[multilingual-snippet-translation]]

## Entities Extracted

- [[ming-zhu]]
- [[karthik-suresh]]
- [[chandan-k-reddy]]
- [[virginia-tech]]
- [[cost-dataset]]
- [[must-pt]]
- [[codexglue]]
- [[codebert]]
- [[graphcodebert]]
- [[plbart]]
- [[transcoder]]
- [[baptiste-roziere]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
