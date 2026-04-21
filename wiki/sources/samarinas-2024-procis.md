---
type: source
subtype: paper
title: "ProCIS: A Benchmark for Proactive Retrieval in Conversations"
slug: samarinas-2024-procis
date: 2026-04-20
language: en
tags: [conversational-search, proactive-retrieval, benchmark, information-retrieval, dataset]
processed: true

raw_file: raw/papers/samarinas-2024-procis/paper.pdf
raw_md: raw/papers/samarinas-2024-procis/paper.md
bibtex_file: raw/papers/samarinas-2024-procis/paper.bib
possibly_outdated: false

authors:
  - Chris Samarinas
  - Hamed Zamani
year: 2024
venue: SIGIR 2024
venue_type: conference
arxiv_id:
doi: 10.1145/3626772.3657869
url: https://dl.acm.org/doi/10.1145/3626772.3657869
citation_key: samarinas2024procis
paper_type: benchmark

read_status: unread

domain: ir
---

## Summary

ProCIS introduces a benchmark for proactive document retrieval in multi-party conversations, targeting settings where a system should decide not only what to retrieve but also when to intervene. The benchmark is built from Reddit threads aligned to a Wikipedia corpus and combines large-scale weak supervision with a carefully judged test set. Beyond releasing the dataset, the paper formalizes reactive and proactive contextual suggestion, proposes the time-aware `npDCG` metric for proactive evaluation, and reports baselines spanning lexical, sparse, dense, multi-vector, and LLM-based retrieval. The main empirical takeaway is asymmetric: LMGR is strongest in reactive retrieval, while ColBERT remains stronger in proactive retrieval when paired with a DeBERTa-based engagement classifier.

## Problem & Motivation

Conversational information seeking benchmarks mostly assume a reactive query-response loop where a user explicitly issues a request and the system responds. That setup misses proactive assistance scenarios in which a system monitors an ongoing discussion and decides whether an external resource would add value before anyone asks. The authors argue that progress on proactive conversational search has been bottlenecked by the absence of large-scale training data, reliable test judgments, and an evaluation metric that rewards timely intervention rather than only final ranking quality. ProCIS is proposed to fill that gap with both dataset infrastructure and a benchmark protocol.

## Method

- **Corpus and conversation mining**: the retrieval corpus is a preprocessed Wikipedia dump with `5,315,384` English articles. Candidate conversations come from Reddit dumps covering `2005-2022`; the pipeline filters NSFW content, external links, embedded media, non-English content, HTML markup, and links missing from the Wikipedia corpus.
- **Dataset construction**: after filtering, the collection contains over `2.8M` conversations from `1,893,201` unique posts. Table 1 reports splits of `2,830,107` train, `4,165` dev, `3,385` future-dev, and `100` test conversations; `future-dev` is chronological to stress emerging topics.
- **Weak supervision assumption**: for train/dev/future-dev, a Wikipedia article is treated as relevant when its URL appears in the thread. The authors note that only `63%` of originally mentioned Wikipedia pages were judged truly useful, so large-scale training labels are intentionally sparse and noisy.
- **Crowdsourced test annotation**: each test conversation is pooled with `5` candidate sets of up to `10` documents from BM25, SPLADE, ANCE, ColBERT, and LMGR. Annotation uses `1,000` HITs in `8` batches, `3` workers per HIT, and yields `4,207` document assessments plus highlighted conversational evidence.
- **Quality controls**: workers had to summarize the conversation, assign graded relevance labels `0/1/2`, and highlight supporting utterances for partially relevant or relevant documents. Final labels use majority voting; reported inter-annotator agreement is `Fleiss kappa = 0.6482` at a total cost of `$3,500` (`$1.16` per HIT).
- **Reactive task**: given a conversation `U = {u_1, ..., u_m}`, the model retrieves documents once at the end of the observed conversation. Evaluation uses standard ranking metrics such as `nDCG@k`, `MRR`, `MAP`, and `Recall@k`.
- **Proactive task**: at each turn `i`, a system either waits or emits a ranked list `D_i`. The paper defines `pDCG = (1 / Z) sum_i 1{|D_i| > 0} * DCG(D_i \\ union_{i' < i} D_i')`, where `Z = sum_i 1{|D_i| > 0}` and repeated documents shown in earlier turns are removed from later credit.
- **Time-aware gain**: for a document with ideal first useful turn `l`, the gain is `rel(r_ij) = 0` if `i < l`, else `r_ij / log(1 + i - (l - 1))`. This delays credit for late interventions, and `npDCG = pDCG / ipDCG` normalizes by an oracle proactive policy.
- **Benchmarked retrievers**: the benchmark covers [[bm25]], [[splade]], [[ance]], and [[colbert]], plus [[language-model-grounded-retrieval]] (LMGR). LMGR uses LLM candidate generation, dense retrieval over Wikipedia title-description pairs, and grounding; it generates up to `20` candidates and evaluates grounding with `k in {1, 3, 5}`.
- **Proactive classifier**: proactive engagement is modeled with a [[deberta]]-base binary classifier trained on balanced positive/negative utterance pairs, then composed with the retrieval back end to decide whether retrieval should happen on each turn.

## Key Results

- The benchmark covers broad open-domain conversations: the training split spans `34,785` subreddits, with average conversation length `406.01 +/- 774.67` words and `5.41 +/- 7.81` turns.
- The annotated test set is much denser than raw Reddit hyperlinks: it contains `8.02` relevant documents per conversation on average versus only `1.15` originally mentioned Wikipedia links.
- Human annotation quality is usable for benchmarking at scale: `4,207` assessments from `1,000` HITs with `3` annotators each and `Fleiss kappa = 0.6482`.
- In reactive retrieval, the best non-LLM baseline is [[colbert]] with `nDCG@5 = 0.2091`, `nDCG@20 = 0.2094`, `MRR = 0.5679`, and `R@20 = 0.1778`.
- LMGR is the strongest reactive method: with `k = 5`, it reaches `nDCG@5 = 0.3408`, `nDCG@20 = 0.4524`, `MRR = 0.6300`, `MAP = 0.2663`, `R@5 = 0.2853`, and `R@20 = 0.5306`.
- In proactive retrieval, traditional neural retrievers remain stronger: [[colbert]] + the proactive classifier achieves the best reported `npDCG@5/20/100 = 0.1719 / 0.1944 / 0.2172`, while LMGR with `k = 5` reaches only `0.0781 / 0.1840 / -`.

## Limitations

- The benchmark is restricted to English Reddit threads grounded to Wikipedia, so it may not transfer cleanly to private chats, spoken meetings, enterprise knowledge bases, or multilingual settings.
- Large parts of the training signal are noisy by construction: URL mention is only a proxy for usefulness, and the paper reports that just `63%` of linked Wikipedia pages actually provide useful context.
- The carefully judged test set is small at `100` conversations, which is reasonable for dense annotation but still limits statistical power for fine-grained comparisons.
- Privacy and deployment concerns for systems that monitor live conversations are explicitly acknowledged but left outside the scope of the benchmark.
- LMGR is strong for reactive contextual suggestion but unstable for proactive timing decisions without task-specific adaptation, leaving the hardest part of proactive retrieval only partially solved.

## Concepts Extracted

- [[proactive-retrieval]]
- [[reactive-retrieval]]
- [[conversational-information-seeking]]
- [[mixed-initiative-interaction]]
- [[contextual-suggestion]]
- [[depth-k-pooling]]
- [[relevance-judgment]]
- [[normalized-proactive-discounted-cumulative-gain]]
- [[dense-retrieval]]
- [[multi-vector-retrieval]]
- [[language-model-grounded-retrieval]]

## Entities Extracted

- [[chris-samarinas]]
- [[hamed-zamani]]
- [[procis]]
- [[university-of-massachusetts-amherst]]
- [[wikipedia]]
- [[reddit]]
- [[amazon-mechanical-turk]]
- [[bm25]]
- [[splade]]
- [[ance]]
- [[colbert]]
- [[deberta]]
- [[openchat-3-5]]
- [[mistral-7b]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
