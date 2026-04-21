---
type: source
subtype: paper
title: A Survey of Multilingual Neural Machine Translation
slug: dabre-2021-survey
date: 2026-04-20
language: en
tags: [mnmt, machine-translation, multilinguality, transfer-learning, zero-shot, survey]
processed: true

raw_file: raw/papers/dabre-2021-survey/paper.pdf
raw_md: raw/papers/dabre-2021-survey/paper.md
bibtex_file: raw/papers/dabre-2021-survey/paper.bib
possibly_outdated: true

authors:
  - Raj Dabre
  - Chenhui Chu
  - Anoop Kunchukuttan
year: 2021
venue: ACM Computing Surveys
venue_type: journal
arxiv_id:
doi: 10.1145/3406095
url: https://dl.acm.org/doi/10.1145/3406095
citation_key: dabre2021survey
paper_type: survey

read_status: unread
read_date:
rating:

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2021; re-verify against recent literature.

This survey organizes multilingual neural machine translation (MNMT) as the problem of learning one model over multiple language pairs and reviews the field along three major settings: multiway translation, low- or zero-resource translation, and multi-source translation. It contrasts complete parameter sharing with language-specific architectures, summarizes training strategies such as joint training and distillation, and highlights how transfer learning, pivoting, and zero-shot methods exploit cross-lingual knowledge transfer. The paper also connects MNMT to older RBMT/SMT paradigms, catalogs common multilingual corpora and shared tasks, and surfaces open questions around language-agnostic representations, target-side diversity, and scaling to many low-resource languages.

## Problem & Motivation

Neural MT had mostly been studied in bilingual settings, even though practical translation systems and many research questions are inherently multilingual. The paper argues that MNMT matters because a single model can reduce deployment footprint, leverage knowledge transfer from high-resource to low-resource language pairs, and expose what it means to learn language-agnostic distributed representations. The survey's motivation is to replace a fragmented literature with a structured map of resource scenarios, modeling choices, datasets, and unresolved research questions.

## Method

- **Formalization**: multiway MNMT is written as learning one model over `l` language pairs `(s_i, t_i) \in L`, where `L \subset S \times T` and the source/target language sets `S` and `T` need not be disjoint.
- **Survey taxonomy**: the review is organized around `3` scenarios: multiway translation, low/zero-resource translation, and multi-source translation. The low/zero-resource branch is further decomposed into `3` major strategies: transfer learning, pivoting, and zero-shot translation.
- **Parameter-sharing spectrum**: the paper contrasts complete sharing (shared embeddings, encoder, decoder, attention, common subword vocabulary, and target-language tags) with minimal sharing (language-specific embeddings/encoders/decoders plus shared attention). For full pair coverage, bilingual systems scale roughly with the number of pairs, while the minimally shared multilingual design grows only linearly with the number of languages instead of quadratically.
- **Controlled sharing mechanisms**: surveyed methods vary sharing at the encoder, decoder, attention, and representation levels, including routing networks, language-embedding-conditioned parameters, fixed-size attention bridges for language-agnostic representations, and target-language-specific decoder or positional components.
- **Training protocols**: joint training minimizes mean negative log-likelihood across language pairs under balanced sampling; knowledge distillation adds a teacher-student loss term when bilingual teachers outperform the multilingual student on validation data.
- **Low-resource and multi-source recipes**: the paper synthesizes parent-to-child fine-tuning, lexical transfer through shared subword vocabularies or aligned embeddings, run-time or pre-training pivoting with pseudo-parallel corpora, and multi-source architectures that either use separate encoders/attentions or concatenate sources for `N > 3` multilingual inputs.

## Key Results

- The survey yields a compact taxonomy with `3` top-level MNMT scenarios and `3` principal low/zero-resource strategies, which is its main organizing contribution rather than a new benchmark.
- It highlights that relatedness-aware adaptation has already been pushed to `1,095` languages to English in cited work, indicating that multilingual transfer can scale far beyond classic bilingual settings.
- It reports that zero-shot translation is generally ineffective when training corpora are below `100k` sentence pairs for many Indian-language settings, and cites similar evidence on small European corpora.
- It notes that, in pivot-based training, a small clean source-target corpus can have an effect comparable to a pseudo-parallel corpus roughly `2` orders of magnitude larger.
- It summarizes the benchmark landscape with concrete scales: XNLI spans `15` languages, the ILCI corpus is `11`-way parallel, and the Asian Language Treebank is `9`-way parallel.
- Across the surveyed literature, self-attention and appropriately shared multilingual models often match or outperform bilingual baselines while using a more compact deployment footprint, though the survey itself does not provide a single unified reimplementation.

## Limitations

- This is a survey paper, so many claims are synthesized from heterogeneous datasets, model families, and evaluation setups rather than a controlled apples-to-apples experiment.
- The coverage is effectively pre-2020 despite publication in 2021, so it predates multilingual denoising pretraining, large-scale many-to-many systems such as NLLB, and the current LLM-centric MT landscape.
- Several central issues remain unresolved in the surveyed literature, including language-agnostic representations, multiple low-resource target languages, and robust code-mixed output generation.
- Dataset coverage remains uneven and often Eurocentric; truly low-resource languages still suffer from sparse parallel data and weak evaluation infrastructure.

## Concepts Extracted

- [[multilingual-machine-translation]]
- [[knowledge-transfer]]
- [[parameter-sharing]]
- [[transfer-learning]]
- [[pivot-translation]]
- [[zero-shot-translation]]
- [[multi-source-translation]]
- [[domain-adaptation]]
- [[byte-pair-encoding]]

## Entities Extracted

- [[raj-dabre]]
- [[chenhui-chu]]
- [[anoop-kunchukuttan]]
- [[nict]]
- [[osaka-university]]
- [[microsoft-ai-and-research]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
