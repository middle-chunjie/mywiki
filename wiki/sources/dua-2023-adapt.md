---
type: source
subtype: paper
title: "To Adapt or to Annotate: Challenges and Interventions for Domain Adaptation in Open-Domain Question Answering"
slug: dua-2023-adapt
date: 2026-04-20
language: en
tags: [odqa, domain-adaptation, retrieval, question-answering, dataset-shift]
processed: true

raw_file: raw/papers/dua-2023-adapt/paper.pdf
raw_md: raw/papers/dua-2023-adapt/paper.md
bibtex_file: raw/papers/dua-2023-adapt/paper.bib
possibly_outdated: true

authors:
  - Dheeru Dua
  - Emma Strubell
  - Sameer Singh
  - Pat Verga
year: 2023
venue: "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)"
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2023.acl-long.807
url: "https://aclanthology.org/2023.acl-long.807"
citation_key: dua2023adapt
paper_type: method

read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper studies whether state-of-the-art open-domain question answering systems trained on Wikipedia actually transfer to distant target domains such as PubMed, StackOverflow, Reddit, news, and legal text. Using seven datasets across five corpora, the authors evaluate end-to-end retrieval-plus-reading performance rather than retrieval alone, derive a shift taxonomy over question/context and answer distributions, and test both zero-shot and few-shot interventions. The core result is that out-of-domain failure is severe even when retrieval metrics appear strong: answer-containing passages frequently do not justify the answer, and adaptation success depends heavily on the shift type. Few-shot large-language-model sentence generation followed by cloze conversion yields the strongest gains, improving end-to-end answer F1 by up to roughly 24 points.

## Problem & Motivation

The paper asks whether ODQA systems trained on Wikipedia and standard QA supervision remain reliable when the corpus, question style, and answer distribution shift substantially. Prior work typically studied only conservative domain shifts or evaluated retrieval in isolation, which hides failures where the retriever surfaces a passage containing the gold string but the reader still cannot justify or extract the right answer. The authors therefore target end-to-end domain generalization, seek a lightweight way to predict when adaptation is likely to work, and compare when it is better to rely on zero-shot interventions, few-shot target examples, or direct annotation.

## Method

- **Task setup**: model ODQA with question `q`, answer `a`, and corpus `C`; the retriever returns `c_q = R(q, C)` and the reader predicts `\hat{a} <- M(a | q, c_q)`.
- **Source domain**: train on English Wikipedia QA using [[natural-questions]] and BoolQ, plus extra cloze questions built by retrieving a sentence with BM25 and replacing the answer span with sentinel markers.
- **Target evaluation**: test on seven datasets across five domains, including Quasar-S, Quasar-T, SearchQA, [[bioasq]], NewsQA, CliCR, and COLIEE; retrieval is scored with `Acc@100`, and reading is scored with token-level `F1`.
- **Retriever/reader stack**: compare `BM25`, Contriever, [[dpr]], and Spider as retrievers; use FiD as the reader, encoding the top `100` retrieved documents in parallel before decoding the answer.
- **Shift taxonomy**: decompose distribution change into four cases over input and output distributions: no shift, [[label-shift]], [[covariate-shift]], and full shift.
- **Input compatibility test**: estimate question-context compatibility over sampled contexts with `p(q, c_g) = R(q, c_g) / sum_{c_k in C} R(q, c_k)`, then compare target and source distances from a uniform prior and the gold distribution.
- **Output compatibility test**: estimate answer compatibility with globally normalized answer likelihoods over sampled answer spans, `p(a_g | q, c_q) = prod_t M(a_g^t | a_g^{<t}, q, c_q) / sum_{a_k in A} prod_t M(a_k^t | a_k^{<t}, q, c_q)`, and compare target distances against a source reference distribution.
- **Zero-shot interventions**: vary one factor at a time using a combined index for context shift, `50k` sampled answer spans for answer-shift experiments, and two target-side question augmentation schemes: standard question generation and [[cloze-question]] creation.
- **Answer sampling**: test random, uniform-over-entity-types, most-frequent, and oracle-target-answer-distribution sampling; coarse entity types are extracted with spaCy.
- **Few-shot data generation**: prompt a [[large-language-model]] with `8` seed examples from the target domain to generate a sentence from a passage, filter outputs that contain numbers, fail to repeat passage text, or have `< 75%` word-set overlap after stopword removal, then convert accepted generations into cloze-style QA pairs.
- **Adaptation training**: retrain retrieval/reading components by mixing source supervision with target-side synthetic data; for the few-shot retriever, train DPR on Natural Questions plus roughly `8k-10k` generated passage-sentence pairs.

## Key Results

- The shift analysis on `100` labeled examples classifies BioASQ and Quasar-T as [[label-shift]], Quasar-S as [[covariate-shift]], SearchQA as no shift, and CliCR / NewsQA as full shift.
- End-to-end zero-shot transfer is brittle: on Quasar-S, `~83%` `Acc@100` with BM25 still corresponds to only `~11%` answer `F1`, and manual analysis finds that about `65%` of answer-containing retrievals are false positives that do not justify the answer.
- Spider, the strongest in-domain dense retriever, drops by about `40%` on NewsQA and about `28%` on Quasar-T and Quasar-S; exposing Spider to a mixed-domain index reduces performance by another `~15%`, while enforcing a minimum context length of `50` words partly recovers it.
- Across datasets, zero-shot interventions help more when shift is milder: label-shift and covariate-shift datasets gain about `8.5%` average `F1`, versus about `3.5%` on full-shift datasets.
- For BioASQ answer-distribution control, uniform entity-type sampling gives the best retriever score among unsupervised options, improving `Acc@100` from `45.35` (random) to `50.02`; for the reader, oracle target-answer sampling reaches `41.33` `F1` with C4 pretraining.
- Cloze QA and standard question generation are both useful, but cloze construction is cheaper: retriever `Acc@100` on Quasar-S rises from `10.24` to `21.79` with cloze QA versus `17.47` with standard QGen, while reader `F1` rises from `50.37` to `66.87` / `68.21`.
- Few-shot DataGen is the strongest reader adaptation: Quasar-S improves from `50.37` to `71.93` `F1`, NewsQA from `12.54` to `22.69`, and COLIEE from `73.39` to `82.23`; on the retriever side, Quasar-S improves from `10.24` to `34.19` `Acc@100`.
- Manual inspection of generated data finds that more than `70%` of sampled few-shot generations are correct, supporting the usefulness of the prompting-and-filtering pipeline.

## Limitations

- The proposed shift diagnosis is not annotation-free: it still assumes access to a small labeled target set, and the paper reports distances using `100` labeled target examples.
- The study is centered on one source regime (Wikipedia with Natural Questions / BoolQ supervision) and one main reader architecture (FiD), so the conclusions may not transfer unchanged to newer retriever-reader stacks.
- Few-shot data generation relies on prompt templates and handwritten heuristics, including role words such as doctor / engineer / journalist / poster and a `75%` overlap filter, which may be domain-specific.
- The evaluation covers only seven datasets and a mostly English setup, with limited evidence for multilingual or broader real-world deployment.
- Some analyses depend on manual audits of `50` failure cases per dataset or `20` generations per dataset, which is useful but still a small sample for characterizing all failure modes.

## Concepts Extracted

- [[open-domain-question-answering]]
- [[domain-adaptation]]
- [[domain-generalization]]
- [[dataset-shift]]
- [[label-shift]]
- [[covariate-shift]]
- [[dense-retrieval]]
- [[question-generation]]
- [[few-shot-learning]]
- [[zero-shot-adaptation]]
- [[cloze-question]]
- [[large-language-model]]

## Entities Extracted

- [[dheeru-dua]]
- [[emma-strubell]]
- [[sameer-singh]]
- [[pat-verga]]
- [[university-of-california-irvine]]
- [[google-research]]
- [[carnegie-mellon-university]]
- [[natural-questions]]
- [[bm25]]
- [[dpr]]
- [[bioasq]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
