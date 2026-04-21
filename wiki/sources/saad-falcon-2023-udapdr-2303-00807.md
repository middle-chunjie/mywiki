---
type: source
subtype: paper
title: "UDAPDR: Unsupervised Domain Adaptation via LLM Prompting and Distillation of Rerankers"
slug: saad-falcon-2023-udapdr-2303-00807
date: 2026-04-20
language: en
tags: [information-retrieval, domain-adaptation, synthetic-query-generation, reranking, distillation]
processed: true

raw_file: raw/papers/saad-falcon-2023-udapdr-2303-00807/paper.pdf
raw_md: raw/papers/saad-falcon-2023-udapdr-2303-00807/paper.md
bibtex_file: raw/papers/saad-falcon-2023-udapdr-2303-00807/paper.bib
possibly_outdated: true

authors:
  - Jon Saad-Falcon
  - Omar Khattab
  - Keshav Santhanam
  - Radu Florian
  - Martin Franz
  - Salim Roukos
  - Avirup Sil
  - Md Arafat Sultan
  - Christopher Potts
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2303.00807
doi:
url: http://arxiv.org/abs/2303.00807
citation_key: saadfalcon2023udapdr
paper_type: method

read_status: unread

domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

UDAPDR addresses zero-shot retrieval degradation under domain shift by using a small amount of expensive LLM prompting to bootstrap corpus-adapted prompts, then scaling synthetic query generation with a cheaper model. The resulting synthetic queries train multiple DeBERTaV3-Large passage rerankers, whose judgments are distilled into a single ColBERTv2 retriever for low-latency deployment. Across LoTTE, BEIR, Natural Questions, and SQuAD, the method consistently improves zero-shot retrieval without requiring in-domain labels, and on LoTTE Lifestyle it keeps ColBERTv2's `35 ms` latency while outperforming explicit reranking pipelines that are `11.8x` to `589x` slower. The paper is best read as an early recipe for LLM-assisted retrieval adaptation rather than as a general replacement for supervised retriever training.

## Problem & Motivation

Neural IR systems often rely on large labeled query-document datasets such as MS MARCO, SQuAD, or NQ, but these labels are unavailable in many target domains and quickly become stale when the document distribution shifts. Prior unsupervised domain adaptation methods use synthetic queries effectively, yet often require millions of generated examples or expensive rerankers at inference time. UDAPDR targets this gap: it assumes only access to unlabeled in-domain passages, uses LLM prompting to cheaply synthesize target-style queries, and tries to retain most reranker gains after distilling them into a single retriever that preserves low latency for deployment.

## Method

- **Stage 1 seed generation**: sample `X ∈ {5, 10, 50, 100}` in-domain passages and use `text-davinci-002` with `5` prompting strategies to generate `5X` high-quality seed queries.
- **Prompt construction**: convert Stage 1 outputs into `Y ∈ {1, 5, 10}` corpus-adapted prompts containing positive and negative query examples for new passages; this is inspired by the Demonstrate stage of DSP.
- **Cheap large-scale query generation**: use Flan-T5 XXL with each corpus-adapted prompt to generate `Z ∈ {1K, 10K, 100K, 1M}` synthetic queries; the paper mainly studies `Z = 10K` and `100K`.
- **Query filtering**: keep only synthetic queries whose gold passage is retrieved in the top `20` by a zero-shot ColBERTv2 retriever, following prior evidence that this improves adaptation quality.
- **Teacher rerankers**: train one DeBERTaV3-Large cross-encoder reranker per prompt-generated query set, so the system uses multiple teachers rather than relying on a single reranker.
- **Reranker optimization**: fine-tune with cross-entropy loss, dropout `0.1`, `1` epoch, learning rate `5e-6`, batch size `32`, linear warmup/decay, and a linear classifier over the final `[CLS]` hidden state.
- **Retriever distillation**: use the trained rerankers to label additional synthetic questions as triples, then distill their scores into a single ColBERTv2 student retriever.
- **Retriever hyperparameters**: distill ColBERTv2 with learning rate `1e-5`, batch size `32`, maximum document length `300`, and a `BERT-Base` encoder.
- **Deployment objective**: evaluate only the distilled ColBERTv2 at inference, so the final system avoids the reranker's large serving-time cost.

## Key Results

- On LoTTE Pooled dev, zero-shot ColBERTv2 improves from `63.7` to `72.1` with `Y = 5, Z = 20K` and to `72.2` with `Y = 10, Z = 10K`; Natural Questions rises from `68.9` to `74.0`, and SQuAD from `65.0` to `73.8`.
- On LoTTE Lifestyle latency tests, UDAPDR (`Y = 5, Z = 20K`) keeps the same `35 ms` query latency as zero-shot ColBERTv2 while raising Success@5 from `64.5` to `74.8`; explicit reranking reaches only `73.3-73.5` while costing `412 ms`, `2060 ms`, or `20600 ms`.
- On LoTTE test sets, UDAPDR raises Forum pooled Success@5 from `62.3` to `70.8` and Search pooled from `71.5` to `76.6`; the paper summarizes this as average gains of `+7.1` and `+3.9`, respectively.
- On BEIR test sets, the authors report an average `+5.2` nDCG@10 improvement over zero-shot ColBERTv2, with examples including Covid `84.7 -> 88.0`, FEVER `78.0 -> 83.2`, and FiQA `45.8 -> 53.5`.
- Component ablations show the strongest LoTTE Pooled dev result (`71.1`) comes from `GPT-3 + Flan-T5 XXL + DeBERTaV3-Large`; replacing GPT-3 with only Flan-T5 XXL drops to `68.0`, and shrinking the reranker to DeBERTaV3-Base drops to `67.0`.

## Limitations

- The method is not annotation-free in a practical sense: it still requires a sizable collection of in-domain passages before adaptation can begin.
- Synthetic queries may inherit biases and hidden contamination from GPT-3 and Flan-T5 pretraining; the paper explicitly notes possible overlap with NQ, SQuAD, and StackExchange-derived LoTTE data.
- Training still depends on substantial GPU compute and storage, especially when generating large synthetic datasets and training multiple rerankers.
- The evaluation is entirely English, so transfer to multilingual or low-resource settings is unverified.
- Scaling synthetic data is not monotonic: the appendix reports that much larger `Z` can fail to help and may even hurt performance.
- The empirical study centers on ColBERTv2-style late-interaction retrieval, so generalization to other retriever families remains an open question.

## Concepts Extracted

- [[information-retrieval]]
- [[domain-adaptation]]
- [[domain-shift]]
- [[synthetic-query-generation]]
- [[query-generation]]
- [[prompt-engineering]]
- [[reranking]]
- [[cross-encoder]]
- [[knowledge-distillation]]
- [[dense-retrieval]]
- [[zero-shot-retrieval]]
- [[late-interaction]]

## Entities Extracted

- [[jon-saad-falcon]]
- [[omar-khattab]]
- [[keshav-santhanam]]
- [[radu-florian]]
- [[martin-franz]]
- [[salim-roukos]]
- [[avirup-sil]]
- [[md-arafat-sultan]]
- [[christopher-potts]]
- [[stanford-university]]
- [[ibm-research-ai]]
- [[gpt-3]]
- [[colbertv2]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
