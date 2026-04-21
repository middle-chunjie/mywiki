---
type: source
subtype: paper
title: Leveraging LLMs for Synthesizing Training Data Across Many Languages in Multilingual Dense Retrieval
slug: thakur-2024-leveraging-2311-05800
date: 2026-04-20
language: en
tags: [multilingual-retrieval, dense-retrieval, synthetic-data, prompting, ir]
processed: true

raw_file: raw/papers/thakur-2024-leveraging-2311-05800/paper.pdf
raw_md: raw/papers/thakur-2024-leveraging-2311-05800/paper.md
bibtex_file: raw/papers/thakur-2024-leveraging-2311-05800/paper.bib
possibly_outdated: false

authors:
  - Nandan Thakur
  - Jianmo Ni
  - Gustavo Hernández Ábrego
  - John Wieting
  - Jimmy Lin
  - Daniel Cer
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2311.05800
doi:
url: http://arxiv.org/abs/2311.05800
citation_key: thakur2024leveraging
paper_type: method

read_status: unread

domain: ir
---

## Summary

The paper proposes SWIM-IR, a synthetic multilingual retrieval training corpus with `28,265,848` query-passage pairs spanning `33` languages, and uses it to train SWIM-X dense retrievers without human-labeled relevance pairs. Its main technical idea is SAP, a two-stage prompting method that first extracts an informative summary from a passage and then generates a target-language query conditioned on both the summary and the passage. Using PaLM 2 Small for query generation and mT5-based dual encoders for retrieval, the authors show that synthetic supervision can outperform strong supervised baselines on cross-lingual retrieval while remaining competitive on monolingual retrieval. The work is notable because it scales multilingual dense retrieval training beyond the small, unevenly distributed human datasets available for low-resource languages.

## Problem & Motivation

Multilingual dense retrieval lags behind English retrieval because relevance-labeled training pairs are scarce, unevenly distributed across languages, and expensive to annotate with native speakers. Prior synthetic query-generation work had mostly focused on English, leaving open whether large language models can generate multilingual training signals that are good enough to substitute for human supervision. The paper targets this gap by constructing a large multilingual synthetic dataset and testing whether it can support both cross-lingual and monolingual retrieval at a competitive level.

## Method

- **SAP prompting**: decompose multilingual query generation into summary extraction and query generation, with `e_s = LLM(p_s; θ^1, ..., θ^k)` and `q_t = LLM(e_s, p_s; θ^1, ..., θ^k)`, where `p_s` is the source-language passage, `e_s` is an extractive summary, and `q_t` is the target-language query.
- **Cross-lingual generation**: sample up to `1M` English Wikipedia passages per target language, use `k = 5` English exemplars, translate exemplar queries with Google Translate, and generate target-language queries with PaLM 2 Small.
- **Monolingual generation**: sample up to `1M` passages from each MIRACL language-specific Wikipedia, use `k = 3` exemplars for `16/18` languages with training data, and manually prepare exemplars for German and Yoruba.
- **Dataset scale**: SWIM-IR contains `15,532,876` cross-lingual pairs and `12,732,972` monolingual pairs, for `28,265,848` total training pairs across `33` languages.
- **Sampling strategy**: use stratified passage sampling with inclusion threshold `I_th = D_sample / D_total`; for each passage, sample `p_i ~ U(0,1)` and keep it when `p_i <= I_th` to balance coverage over alphabetically ordered Wikipedia entities.
- **Filtering and validation**: remove `6%` to `10%` of generated pairs flagged as `/Adult` or `/Sensitive Subjects`; evaluate human quality on five languages over fluency, adequacy, and language correctness.
- **Retriever backbone**: fine-tune SWIM-X with mT5 Base (`580M` parameters) using contrastive learning with in-batch negatives; XOR-Retrieve and MIRACL use batch size `4096`, XTREME-UP uses `1024`, and learning rate is `1e-3`.
- **Training schedule**: mContriever pretraining uses mC4 for `600K` steps with batch size `8192`; synthetic fine-tuning runs about `5K` to `50K` steps depending on data volume, with language mixing for XOR/XTREME-UP and language unmixing for MIRACL.
- **Evaluation setup**: test on XOR-Retrieve (`Recall@5kt`, `Recall@2kt`), MIRACL (`nDCG@10`, `Recall@100`), and XTREME-UP (`MRR@10`) against zero-shot, supervised, MT-based, and late-interaction baselines.

## Key Results

- SWIM-IR reaches `28,265,848` total synthetic pairs across `33` languages, much larger than prior multilingual retrieval datasets such as MIRACL (`726K`) or Mr.TyDi (`49K`).
- On XOR-Retrieve, `SWIM-X (7M)` with mC4 pretraining achieves `66.7` `Recall@5kt`, beating the best supervised mContriever-X baseline at `59.6` by `+7.1`.
- On XOR-Retrieve, even `SWIM-X (500K)` with mC4 pretraining reaches `63.0` `Recall@5kt`, still outperforming mContriever-X by `+3.4`; the paper text highlights a `+3.6` gain over a different supervised comparison.
- On MIRACL, `SWIM-X (180K)` obtains `46.4` average `nDCG@10`, beating the best zero-shot baseline (`39.8`) by `+6.6` but trailing supervised mContriever-X (`55.4`) by `-9.0`.
- On XTREME-UP, `SWIM-X (120K)MT` reaches `26.1` average `MRR@10`, outperforming the best supervised baseline without MS MARCO (`13.5`) by `+12.6`; `SWIM-X (120K)` reaches `25.2`.
- SAP improves downstream retrieval over standard prompting by at least `+0.6` `Recall@5kt` across PaLM 2 sizes, with gains up to `+3.2` for PaLM 2 Small or smaller.
- With only `2K` synthetic pairs, SWIM-X already reaches `49.1` `Recall@5kt` on XOR-Retrieve; around `250K` pairs it hits `60.5`, surpassing the best supervised baseline.
- Human validation shows `99%` to `100%` language correctness across five tested languages; adequacy and fluency rated `1` or `2` stay at or above roughly `86%` and `88%`, respectively.
- Cost analysis estimates that generating `200K` synthetic pairs costs roughly `$1K`, versus about `$14.1K` to annotate `15.2K` human pairs under the paper's wage assumptions.

## Limitations

- The strongest gains are on cross-lingual retrieval; on MIRACL, synthetic-only training still underperforms supervised training with hard negatives, so SWIM-IR is not yet a full replacement in monolingual settings.
- Query quality is not uniformly strong across languages: Chinese adequacy is weaker in human validation, and extremely low-resource languages such as Boro and Manipuri remain difficult.
- SAP depends on few-shot exemplars and commercial LLM/translation tooling such as PaLM 2, Google Translate, and Bard, which affects reproducibility and operational dependence.
- Generated queries are not human-verified and may contain decontextualization, code-switching, or factual inconsistencies inherited from LLM generation.
- The dataset is built primarily from Wikipedia, so domain transfer beyond encyclopedic passages is not directly established.

## Concepts Extracted

- [[large-language-model]]
- [[few-shot-prompting]]
- [[synthetic-query-generation]]
- [[summarize-then-ask-prompting]]
- [[multilingual-question-generation]]
- [[dense-retrieval]]
- [[multilingual-dense-retrieval]]
- [[cross-lingual-retrieval]]
- [[monolingual-retrieval]]
- [[contrastive-learning]]
- [[in-batch-negatives]]
- [[hard-negative-mining]]

## Entities Extracted

- [[nandan-thakur]]
- [[jianmo-ni]]
- [[gustavo-hernandez-abrego]]
- [[john-wieting]]
- [[jimmy-lin]]
- [[daniel-cer]]
- [[google-research]]
- [[google-deepmind]]
- [[university-of-waterloo]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
