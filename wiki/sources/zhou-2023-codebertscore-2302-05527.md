---
type: source
subtype: paper
title: "CodeBERTScore: Evaluating Code Generation with Pretrained Models of Code"
slug: zhou-2023-codebertscore-2302-05527
date: 2026-04-20
language: en
tags: [code-generation, evaluation-metric, bertscore, nlp, pretraining]
processed: true
raw_file: raw/papers/zhou-2023-codebertscore-2302-05527/paper.pdf
raw_md: raw/papers/zhou-2023-codebertscore-2302-05527/paper.md
bibtex_file: raw/papers/zhou-2023-codebertscore-2302-05527/paper.bib
possibly_outdated: true
authors:
  - Shuyan Zhou
  - Uri Alon
  - Sumit Agarwal
  - Graham Neubig
year: 2023
venue: arXiv preprint arXiv:2302.05527
venue_type: preprint
arxiv_id: 2302.05527
doi:
url: http://arxiv.org/abs/2302.05527
citation_key: zhou2023codebertscore
paper_type: method
read_status: unread
domain: nlp
---

## Summary

> ⚠️ Possibly outdated: published 2023 in the rapidly evolving NL→Code / LLM evaluation domain.

CodeBERTScore is a reference-based evaluation metric for natural-language-to-code generation that extends [[bertscore]] to code by (a) encoding natural language context alongside both candidate and reference code sequences using language-specific CodeBERT encoders, and (b) computing contextual token-level cosine similarity with alphanumeric masking and optional IDF weighting. The paper trains and releases five programming-language-specific CodeBERT models (Python, Java, C, C++, JavaScript) via continued MLM pretraining on CodeParrot. Extensive evaluation across four languages on CoNaLa (human preference) and MultiPL-E HumanEval (functional correctness) demonstrates that CodeBERTScore outperforms all prior metrics — BLEU, CrystalBLEU, CodeBLEU, METEOR, chrF, ROUGE — in correlation with both human judgment and execution outcomes. The language-specific models have been downloaded over 1,000,000 times from the Hugging Face Hub.

## Problem & Motivation

Existing code generation metrics are inadequate for evaluating modern LLMs that produce long, syntactically diverse code. Token-matching metrics (BLEU, CrystalBLEU) fail when semantically equivalent code differs in variable names or style. CodeBLEU relies on AST/data-flow matching, which breaks on partial code and still does not correlate well with execution correctness. Execution-based evaluation is reliable but requires hand-crafted test suites that are expensive to produce and introduces security risks from running untrusted code. The gap demands a metric that is unsupervised, captures semantic similarity, and handles implementation diversity.

## Method

**Core idea:** encode both NL context and code with a pretrained code model; compare only the code token embeddings via cosine similarity, masking context and punctuation tokens out.

**Encoding:**
- Concatenate context `x` with reference `y*` and candidate `ŷ`, tokenize with model `B`'s tokenizer.
- Run forward pass: `B(⟨x₁,…,xₖ, y₁*,…,yₘ*⟩)` and `B(⟨x₁,…,xₖ, ŷ₁,…,ŷₙ⟩)`.
- Mask out context vectors `x₁…xₖ` and all non-alphanumeric tokens (except arithmetic operators); retain only `y*[m*]` and `ŷ[m̂]`.

**Similarity computation:**
- `sim(yᵢ*, ŷⱼ) = (yᵢ*ᵀ · ŷⱼ) / (‖yᵢ*‖ · ‖ŷⱼ‖)` (cosine similarity, Eq. 3)

**Precision and Recall** (with IDF token weighting following Zhang et al. 2020):
- `CodeBERTScoreₚ = (1/|ŷ[m̂]|) Σⱼ maxᵢ sim(yᵢ*, ŷⱼ)` (Eq. 4)
- `CodeBERTScoreᵣ = (1/|y*[m*]|) Σᵢ maxⱼ sim(yᵢ*, ŷⱼ)` (Eq. 5)
- `CodeBERTScoreF₁ = 2·P·R / (P+R)` (Eq. 6)
- `CodeBERTScoreF₃ = 10·P·R / (9·P+R)` (Eq. 7, recall-weighted for functional correctness)

**Score scaling:** linear rescaling `(score - b)/(1 - b)` where `b` is the mean similarity of random unrelated code pairs (empirical baseline, e.g., `b_Java = 0.78`, `b_C++ = 0.76`).

**Language-specific models:** base model is CodeBERT; continued MLM pretraining on CodeParrot subsets (115M files from GitHub), `1,000,000` steps per language, batch size `32`, initial LR `5e-5` decayed linearly to `3e-5`.

**Layer selection:** embedding layer is chosen per task via cross-validation — layer `7` for CoNaLa; layers `7` (Java), `10` (C++), `11` (JavaScript), `9` (Python) for HumanEval.

**Score variant:** F₁ used for human preference experiments; F₃ (recall-weighted) used for functional correctness experiments.

## Key Results

- **Human preference (CoNaLa):** CodeBERTScore achieves Kendall-τ `0.517`, Pearson `0.674`, Spearman `0.662` — best across all metrics. Closest competitor: chrF (τ `0.470`), METEOR (τ `0.366`).
- **Functional correctness (HumanEval, avg. across 4 languages):** CodeBERTScore achieves highest or comparable Kendall-τ and Spearman across Java, C++, Python, JavaScript. On C++ it achieves τ `0.327` vs. METEOR `0.301`; on Python τ `0.422` vs. METEOR `0.418`.
- **Distinguishability (Java/C++):** CodeBERTScore scores `9.56` / `9.13` vs. CrystalBLEU `5.96` / `6.94` — but the paper itself argues this metric is gameable via exponentiation and not a reliable comparison tool.
- **Context ablation:** adding NL context increases Kendall-τ from `0.50` to `0.52`.
- **Base vs. language-specific model:** CodeBERT-base is competitive but language-specific models are typically better on HumanEval.
- **Download count:** 5 released models downloaded >1,000,000 times from Hugging Face Hub as of submission.

## Limitations

- Requires a GPU for inference (unlike BLEU/ROUGE); though authors note this is already assumed for training/testing neural models.
- Relies on a strong underlying code model — performance ceiling is tied to base model quality; improvements in base models could change relative rankings.
- Hyperparameters (layer index, F₁ vs. F₃) require cross-validation per dataset/language; not a parameter-free plug-in.
- Distinguishability meta-metric is shown to be gameable by the authors themselves, casting doubt on one of the paper's supporting evaluations.
- Evaluated primarily on CoNaLa (Python) and HumanEval (4 languages); generalization to newer benchmarks (e.g., DS-1000, SWE-bench) is unknown.
- NL context is only available in NL→Code settings; pure code-to-code evaluation reduces to BERTScore without the context benefit.

## Concepts Extracted

- [[bertscore]]
- [[bleu]]
- [[codebleu]]
- [[natural-language-to-code]]
- [[execution-based-evaluation]]
- [[masked-language-modeling]]
- [[cosine-similarity]]
- [[domain-adaptive-pretraining]]
- [[inverse-document-frequency]]
- [[code-generation]]
- [[abstract-syntax-tree]]
- [[program-synthesis]]

## Entities Extracted

- [[shuyan-zhou-cmu]]
- [[uri-alon]]
- [[sumit-agarwal-cmu]]
- [[graham-neubig]]
- [[carnegie-mellon-university]]
- [[neulab]]
- [[codebert]]
- [[humaneval]]
- [[conala]]
- [[codeparrot]]
- [[multipl-e]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
