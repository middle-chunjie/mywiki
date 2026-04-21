---
type: source
subtype: paper
title: Exploring efficient zero-shot synthetic dataset generation for Information Retrieval
slug: almeida-2024-exploring
date: 2026-04-20
language: en
tags: [information-retrieval, synthetic-data, question-generation, reranking, zero-shot]
processed: true
raw_file: raw/papers/almeida-2024-exploring/paper.pdf
raw_md: raw/papers/almeida-2024-exploring/paper.md
bibtex_file: raw/papers/almeida-2024-exploring/paper.bib
possibly_outdated: false
authors:
  - Tiago Almeida
  - Sérgio Matos
year: 2024
venue: "Findings of the Association for Computational Linguistics: EACL 2024"
venue_type: conference
arxiv_id:
doi:
url: "https://aclanthology.org/2024.findings-eacl.81"
citation_key: almeida2024exploring
paper_type: method
read_status: unread
domain: ir
---

## Summary

The paper studies whether small language models can generate useful zero-shot synthetic training data for information retrieval without the cost profile of large proprietary models. The pipeline first removes unrepresentative documents using a normalized-information score derived from language-model likelihoods, then prompts small generative models to produce questions from selected documents, and finally filters synthetic pairs with retrieval-based quality checks before training a BERT reranker. Across five datasets, the resulting rerankers consistently beat BM25 and even outperform pretrained monoT5 on some collections. The main practical claim is that careful document selection and question filtering can compensate for weaker generators, making silver-standard retrieval training data substantially cheaper to produce.

## Problem & Motivation

Neural retrieval and reranking systems usually need labeled question-document pairs, but manual relevance annotation is expensive and unavailable for many target collections. Prior synthetic-data pipelines either depend on supervised question generators or on large language models with high inference cost. The paper targets a cheaper unlabeled setting: select representative documents, generate questions in a zero-shot manner with small language models, filter low-quality pairs, and use the accepted synthetic data to train a competitive neural reranker.

## Method

- **Document selection via normalized information**: estimate document information with `I(x) = -\sum_{i=1}^{N} \log P(w_i \mid w_1, \dots, w_{i-1})`, then normalize by a uniform model as `NI(x) = I(x) / (|x| \times \log |V|)` to reduce document-length bias and identify low-/high-NI outliers.
- **Probabilistic sources for NI**: compare pretrained small transformer LMs with finite-context models; the paper reports better practicality for transformer LMs because they can be used directly without corpus-specific training.
- **Zero-shot question generation**: build a prompt of the form `Article: {selected_article}` followed by `Question: {initiator}` and sample a question `\hat{y} \sim P(y \mid p_1,\dots,p_M,y_1)`. Initiators include `What`, `How`, `Where`, `Is`, and `Why`; only outputs ending with `?` are kept.
- **Generator families and scale**: benchmark 10 small language models spanning `70M` to `1.3B` parameters, including Pythia, GPT-Neo, OPT, and BLOOMZ variants.
- **Generation strategies**: compare `random sampling (RS)`, `contrastive search (CS)`, and `beam search (BS)` to trade off efficiency against output quality.
- **Question quality filter**: define a binary acceptance function `f_k(x; m)` that accepts a synthetic pair `(q, d)` when either a relevance model score exceeds threshold `k` or a rank-based retriever places `d` within rank cutoff `k`.
- **Filter models**: test both `BM25` and `monoT5`; BM25 is preferred because it directly encodes retrieval suitability and is cheaper to run at scale.
- **Benchmark protocol**: for each dataset, select `16K` representative documents, generate `5` questions per document, and evaluate synthetic datasets with `hitsR_k(D_s) = (\sum f_k(x;m)) / |D_s|` and `hits-per-sec_k(D_s) = (\sum f_k(x;m)) / \Delta t`.
- **Downstream training**: train a top-100 reranker with `bert-base-uncased` (`110M`) using the accepted synthetic pairs and simple random negative sampling; training uses Hugging Face defaults rather than aggressive hyperparameter tuning.
- **Infrastructure**: all main experiments run on a single `RTX 2070 8GB` GPU, with BM25 implemented through PyTerrier/Terrier and neural models through Hugging Face.

## Key Results

- Across `150` benchmark configurations, `BS` gives the best average question quality at `hitsR(k = 100) = 0.68`, versus `0.48` for `RS` and `0.47` for `CS`.
- `RS` is the efficiency leader, producing accepted questions almost `5x` faster than `BS` and almost `6x` faster than `CS` on average.
- The downstream reranker trained on `BS + GPT-Neo-1.3B` improves over BM25 on all reported datasets: BioASQ `0.436 vs 0.353` nDCG@10, MSMARCO `0.275 vs 0.184` MRR@10, NQ `0.416 vs 0.281` nDCG@10, HotpotQA `0.681 vs 0.585` nDCG@10, SciDocs `0.228 vs 0.157` nDCG@10.
- The cheaper `RS + Pythia-70M` setup is still strong: BioASQ `0.438`, MSMARCO `0.246`, NQ `0.407`, HotpotQA `0.730`, SciDocs `0.187`, all better than BM25.
- Compared with prior synthetic-data methods, the authors report better results than `GenQ (TAS-B)` on NQ/HotpotQA/SciDocs and competitive or better results than `InPars` with a similarly sized reranker.
- Quality filtering matters: on GPT-Neo-1.3B BS, using only accepted questions yields BioASQ `0.436` versus `0.331` for only rejected questions; on Pythia-70M RS, accepted questions yield HotpotQA `0.730` versus `0.237` for rejected ones.
- NI-based document selection helps: for GPT-Neo-1.3B BS on MSMARCO, synthetic-dataset HitsR is `0.714`, versus `0.371` on lowest-NI documents and `0.447` on highest-NI documents.

## Limitations

- The study excludes dense retrieval as a downstream target, so the conclusions are limited to reranking and BM25-based retrieval pipelines.
- Many components use default settings rather than tuned hyperparameters, including BM25, prompt design, and reranker training.
- The approach is not evaluated in a Doc2Query-style document expansion setting, so it does not establish whether the same synthetic questions help lexical indexing directly.
- Even with small language models, the setup still assumes GPU access of roughly `8GB` VRAM and may scale poorly to long documents because transformer generation cost grows with context length.
- Some evaluation criteria are circular by design: question quality is judged by retrieval models, which makes the first-stage benchmark partly dependent on the filter model's inductive bias.

## Concepts Extracted

- [[information-retrieval]]
- [[synthetic-data]]
- [[small-language-model]]
- [[zero-shot-prompting]]
- [[question-generation]]
- [[normalized-information]]
- [[outlier-detection]]
- [[question-quality-filtering]]
- [[neural-reranking]]
- [[random-sampling]]
- [[contrastive-search]]
- [[beam-search]]

## Entities Extracted

- [[tiago-almeida]]
- [[sergio-matos]]
- [[university-of-aveiro]]
- [[synqgen]]
- [[bm25]]
- [[mono-t5]]
- [[gpt-neo-1-3b]]
- [[pythia-70m]]
- [[bioasq]]
- [[ms-marco]]
- [[natural-questions]]
- [[hotpotqa]]
- [[scidocs]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
