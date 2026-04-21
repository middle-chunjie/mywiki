---
type: source
subtype: paper
title: "The RefinedWeb Dataset for Falcon LLM: Outperforming Curated Corpora with Web Data, and Web Data Only"
slug: penedo-2023-refinedweb-2306-01116
date: 2026-04-20
language: en
tags: [llm, pretraining, web-data, deduplication, commoncrawl]
processed: true
raw_file: raw/papers/penedo-2023-refinedweb-2306-01116/paper.pdf
raw_md: raw/papers/penedo-2023-refinedweb-2306-01116/paper.md
bibtex_file: raw/papers/penedo-2023-refinedweb-2306-01116/paper.bib
possibly_outdated: true
authors:
  - Guilherme Penedo
  - Quentin Malartic
  - Daniel Hesslow
  - Ruxandra Cojocaru
  - Alessandro Cappelli
  - Hamza Alobeidli
  - Baptiste Pannier
  - Ebtesam Almazrouei
  - Julien Launay
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2306.01116
doi:
url: https://arxiv.org/abs/2306.01116
citation_key: penedo2023refinedweb
paper_type: dataset
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper introduces REFINEDWEB, a five-trillion-token English web-only pretraining corpus derived from CommonCrawl through MacroData Refinement (MDR), a pipeline that combines URL filtering, trafilatura-based text extraction, fastText language identification, heuristic quality filtering, and aggressive fuzzy plus exact deduplication. The core claim is that web data does not need curated corpora to produce strong large language models if refinement is stringent enough. In matched small-scale studies, models trained on RefinedWeb outperform counterparts trained on C4, OSCAR, and The Pile, and the authors report that full-scale Falcon-RW models trained on 350B tokens are competitive with GPT-3-class baselines in their evaluation setup. The paper matters because it reframes scalable data refinement, rather than source curation, as the key bottleneck for trillion-token LLM pretraining.

## Problem & Motivation

Scaling-law recommendations imply that GPT-3-class models should consume trillions of tokens, but curated corpora such as books, technical papers, or social media mixes are expensive to assemble, hard to scale, and potentially data-limited. The paper asks whether carefully filtered and deduplicated web data alone can replace such curated mixtures. The authors argue that previous negative judgments about web-only corpora were partly caused by weak extraction, insufficient filtering, and incomplete deduplication rather than by an intrinsic ceiling on web data quality.

## Method

- **Pipeline goal**: build an English-only CommonCrawl corpus at `3-6T` target scale; the released dataset reaches about `5T` tokens with a public `600B`-token extract and roughly `10B` documents.
- **Document preparation**: read raw WARC files with `warcio`, filter URLs using an aggregated blocklist of `4.6M` domains plus URL scoring heuristics, and exclude common curated-source domains such as Wikipedia, arXiv, StackOverflow, and Reddit.
- **Text extraction**: extract main page content with `trafilatura`, then normalize formatting by capping runs of newlines to at most `2` and stripping literal URLs from the recovered text.
- **Language identification**: use the CCNet `fastText` document classifier; keep English pages whose top language confidence exceeds `0.65`, treating lower scores as mostly non-natural-text or low-confidence cases.
- **Quality filtering**: apply document-level repetition and quality heuristics inspired by MassiveWeb, then line-wise corrections to remove navigation buttons, counters, boilerplate, or call-to-action fragments; if removed lines exceed `5%` of a document's words, drop the whole document.
- **Fuzzy deduplication**: run MinHash on normalized GPT-2-tokenized `5`-grams, with `b = 20` hashes per bucket, `r = 450` buckets, and `9000` hashes total per document. Duplicate-match probability follows `P = 1 - (1 - s_{i,j}^b)^r`, where similarity is estimated from MinHash overlap.
- **Similarity measure**: use Jaccard overlap on unique `n`-gram sets, `J(d_i, d_j) = |d_i ∩ d_j| / |d_i ∪ d_j|`, to interpret approximate duplicate detection.
- **Exact deduplication**: after MinHash, apply EXACTSUBSTR with suffix arrays to remove exact repeated spans of at least `50` tokens; choose `EXACTSUBSTR-CUT`, and discard documents with fewer than `20` remaining non-duplicated characters.
- **Cross-dump control**: split CommonCrawl into `100` shards for tractability and maintain kept-URL lists so that URLs revisited across dumps can be removed in later shards.
- **Model evaluation**: train autoregressive decoder-only models roughly GPT-3-like but with ALiBi and FlashAttention; use `1B` and `3B` models for small-scale studies on `27GT` and `60GT`, then `1.3B` and `7.5B` Falcon-RW models on `350GT`, evaluated zero-shot over `18` tasks with the Eleuther AI evaluation harness.

## Key Results

- REFINEDWEB reaches about `5,000GT` and `~10B` documents; the public release is a random `600GT` extract distributed through Hugging Face.
- After URL filtering, extraction, and language ID, only `48%` of original CommonCrawl documents remain; after subsequent filtering, only `23%` remain; the full pipeline removes nearly `90%` of documents overall.
- On the small aggregate, `1B@27GT` trained on REFINEDWEB scores `56.2%`, beating C4 (`55.7%`), OSCAR-21.09 (`55.0%`), OSCAR-22.01 (`52.7%`), and The Pile (`53.4%`).
- On the same benchmark, `3B@60GT` trained on REFINEDWEB reaches `59.8%`, slightly above C4 (`59.6%`) and clearly above The Pile (`57.9%`).
- Within the paper's ablations, the RefinedWeb pipeline improves from `52.7% -> 54.3% -> 56.2%` for `1B` models and `57.4% -> 58.2% -> 59.8%` for `3B` models when moving from RW-Raw to RW-Filtered to final REFINEDWEB.
- Applying MDR to other corpora yields gains such as The Pile `53.4% -> 55.2%` and OSCAR-22.01 `52.7% -> 55.4%` after filtering plus deduplication, supporting the claim that stringent deduplication transfers across datasets.

## Limitations

- The dataset and released experiments are English-only, so the paper does not establish that the same heuristics transfer cleanly to multilingual settings.
- Toxicity analysis is narrow: it uses Perspective API's notion of "rude or disrespectful" text and does not directly quantify broader harmfulness or social bias.
- Several filtering heuristics are source- and language-sensitive; the authors explicitly note that naive transfer can overfilter corpora such as books or code and required adjustments on The Pile.
- Cross-paper model comparisons are only partially controlled because prompts, precision, codebases, and evaluation suites differ across external baselines.
- The deduplication pipeline is operationally expensive, requiring `100-250` AWS `c5.18xlarge` instances for large stages and up to `2 TiB` RAM instances for exact substring deduplication.
- The public artifact is only a `600GT` subset of the `5T` internal corpus, and the preprocessing software itself is not released in the paper.

## Concepts Extracted

- [[data-curation]]
- [[data-quality]]
- [[text-extraction]]
- [[language-identification]]
- [[data-filtering]]
- [[data-deduplication]]
- [[minhash]]
- [[exact-substring-deduplication]]
- [[locality-sensitive-hashing]]
- [[jaccard-similarity]]
- [[zero-shot-generalization]]
- [[scaling-law]]

## Entities Extracted

- [[guilherme-penedo]]
- [[quentin-malartic]]
- [[daniel-hesslow]]
- [[ruxandra-cojocaru]]
- [[alessandro-cappelli]]
- [[hamza-alobeidli]]
- [[baptiste-pannier]]
- [[ebtesam-almazrouei]]
- [[julien-launay]]
- [[refinedweb]]
- [[commoncrawl]]
- [[the-pile]]
- [[c4]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
