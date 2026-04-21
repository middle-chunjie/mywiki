---
type: source
subtype: paper
title: Contrastive Learning of Sentence Embeddings from Scratch
slug: zhang-2023-contrastive-2305-15077
date: 2026-04-20
language: en
tags: [sentence-embedding, contrastive-learning, synthetic-data, nlp, text-similarity]
processed: true
raw_file: raw/papers/zhang-2023-contrastive-2305-15077/paper.pdf
raw_md: raw/papers/zhang-2023-contrastive-2305-15077/paper.md
bibtex_file: raw/papers/zhang-2023-contrastive-2305-15077/paper.bib
possibly_outdated: true
authors:
  - Junlei Zhang
  - Zhenzhong Lan
  - Junxian He
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2305.15077
doi:
url: http://arxiv.org/abs/2305.15077
citation_key: zhang2023contrastive
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature in sentence embedding and LLM-based data synthesis.

SynCSE proposes to train sentence embeddings entirely with ChatGPT-synthesized contrastive data, removing the need for manually annotated NLI datasets. Two variants are introduced: SynCSE-partial, which generates positive and hard-negative annotations for existing unlabeled sentences, and SynCSE-scratch, which synthesizes both unlabeled sentences and their annotations from scratch using specified genres and topics. Built on the SimCSE objective, SynCSE-partial achieves performance comparable to supervised SimCSE on standard STS benchmarks (+5.37 over unsupervised baseline), and SynCSE-scratch generalizes well to specialized domains where unlabeled data is unavailable. Diversity is enforced through randomized prompt and exemplar pools derived from GPT-4.

## Problem & Motivation

Supervised sentence embedding methods rely on labeled NLI datasets (e.g., MNLI+SNLI) that are costly to create and rarely available outside general domains. Unsupervised methods like SimCSE avoid labeled data but lag behind supervised counterparts by several STS points. For specialized domains (biomedical, programming), even unlabeled text can be hard to acquire due to copyright, format, or distribution issues. The authors ask: can LLMs synthesize the contrastive training signal entirely, bridging the supervised/unsupervised performance gap without any human-labeled or domain-specific data?

## Method

- **Backbone**: SimCSE supervised loss. For a triplet `(x_i, x_i+, x_i-)`, the loss is:
  `−log exp(sim(h_i, h_i+)/τ) / Σ_j [exp(sim(h_i, h_j+)/τ) + exp(sim(h_i, h_j-)/τ)]`
  where `τ` is a temperature hyperparameter and `sim` is cosine similarity.
- **SynCSE-partial**: given existing unlabeled sentences `x_i` (taken from SimCSE_NLI's unlabeled pool for fair comparison), ChatGPT is prompted in a few-shot multi-turn chat format to produce a positive paraphrase `x_i+` and a hard negative `x_i-`. Four distinct hard-negative prompt variants are pooled; one is sampled per generation. For each prompt, 5 exemplars are sampled from a pool of 18 (generated offline by GPT-4).
- **SynCSE-scratch**: ChatGPT first generates raw unlabeled sentences (one-shot prompting specifying genre + 6 random topics from a pre-defined list to encourage diversity), then applies the SynCSE-partial annotation pipeline to them. The "etc." qualifier in prompts keeps generated sentences from being strictly topic-bound.
- **Diversity control**: randomized prompt pools (4 positive, 4 hard-negative variants) and exemplar pools (18 per prompt type) are sampled independently each call, mimicking inter-annotator variance. Ablation shows this over "Naive Generation" yields +8.96 average STS points.
- **Combination setting**: SynCSE-scratch synthetic data is merged with the SimCSE_NLI human-annotated set as a data augmentation experiment.
- **Backbones evaluated**: RoBERTa-base and RoBERTa-large; embeddings extracted from `[CLS]` token.
- **Hyperparameters**: same as SimCSE for STS/transfer tasks; MTEB defaults for reranking (MAP metric).

## Key Results

- STS benchmark (Spearman's correlation, RoBERTa-base): SynCSE-partial `81.94` avg (+5.37 over unsup-SimCSE `76.57`); SynCSE-scratch `80.75` avg (+4.18). Supervised SimCSE-base baseline: `82.04`.
- STS benchmark (RoBERTa-large): SynCSE-partial `82.10`, SynCSE-scratch `82.33` vs unsup-SimCSE `78.90`, sup-SimCSE `83.40`.
- Combination (SynCSE-scratch + SimCSE_NLI, RoBERTa-large): `84.37` avg vs sup-SimCSE `83.40`, surpassing the supervised baseline.
- Reranking (MAP, RoBERTa-large): SynCSE-scratch `49.15` avg vs unsup-SimCSE `48.86`; SynCSE-partial underperforms baselines on reranking, indicating domain mismatch when annotations are conditioned on Wikipedia-domain sentences.
- Domain adaptation (BIOSSES biomedical STS): SynCSE-scratch RoBERTa-base `80.12` vs Wikipedia-domain unsup-SimCSE `68.86` (+11.26 Spearman).
- Domain adaptation (StackOverflow reranking MAP): `43.22` vs `39.25` (+3.97).
- vs. ZeroGen (scratch setting): SynCSE-scratch `80.75` vs ZeroGen `64.48` (+16.27) on STS avg with RoBERTa-large.
- Transfer tasks (RoBERTa-base, avg accuracy): SynCSE-partial `87.52`, SynCSE-scratch `86.82` vs unsup-SimCSE `86.90`, supervised SimCSE `88.08`.
- Ethical safety check: only `0.4%` of 100 sampled sentences flagged as unsafe by human annotators.

## Limitations

- Synthesizing data with ChatGPT incurs API costs and rate limits; scalability beyond the SimCSE_NLI dataset size is untested.
- SynCSE-partial underperforms other unsupervised methods on reranking tasks (Table 3), suggesting that conditioning on NLI-domain unlabeled sentences hurts out-of-domain tasks.
- Quality of synthesis is directly dependent on ChatGPT capability; the approach implicitly inherits ChatGPT's biases and knowledge cutoff.
- Diversity via prompt/exemplar pools is a heuristic — the paper does not provide a principled diversity measure or ablation across pool sizes beyond noting +8.96 STS improvement over naive generation.
- Evaluation confined to English; multilingual or cross-lingual settings are not explored.

## Concepts Extracted

- [[contrastive-learning]]
- [[sentence-embedding]]
- [[semantic-textual-similarity]]
- [[natural-language-inference]]
- [[hard-negative-mining]]
- [[synthetic-data-generation]]
- [[in-batch-negatives]]
- [[few-shot-prompting]]
- [[data-augmentation]]
- [[dropout]]

## Entities Extracted

- [[junlei-zhang]]
- [[zhenzhong-lan]]
- [[junxian-he-hkust]]
- [[zhejiang-university]]
- [[westlake-university]]
- [[hkust]]
- [[chatgpt]]
- [[roberta]]
- [[simcse]]
- [[mteb]]

## Contradictions

<!-- None yet; first source on these concepts from this paper's perspective. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
