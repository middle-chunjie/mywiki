---
type: source
subtype: paper
title: Benchmarking and Building Long-Context Retrieval Models with LoCo and M2-BERT
slug: saad-falcon-2024-benchmarking-2402-07440
date: 2026-04-20
language: en
tags: [retrieval, long-context, benchmark, embeddings, state-space-models]
processed: true
raw_file: raw/papers/saad-falcon-2024-benchmarking-2402-07440/paper.pdf
raw_md: raw/papers/saad-falcon-2024-benchmarking-2402-07440/paper.md
bibtex_file: raw/papers/saad-falcon-2024-benchmarking-2402-07440/paper.bib
possibly_outdated: false
authors:
  - Jon Saad-Falcon
  - Daniel Y. Fu
  - Simran Arora
  - Neel Guha
  - Christopher Ré
year: 2024
venue: arXiv
venue_type: preprint
arxiv_id: 2402.07440
doi:
url: http://arxiv.org/abs/2402.07440
citation_key: saadfalcon2024benchmarking
paper_type: benchmark
read_status: unread
domain: ir
---

## Summary

The paper targets long-context retrieval settings where documents routinely exceed `10k` tokens and simple truncation or chunk averaging loses the evidence needed to identify the correct document. It contributes both LoCoV1, a 12-task benchmark spanning law, medicine, science, finance, government, and programming, and M2-BERT, an `80M`-parameter state-space retrieval encoder based on Monarch Mixer that scales to `32k` tokens. The training recipe mixes short and long masked-language-modeling examples from C4, Wikipedia, and BookCorpus, then replaces batch-hungry contrastive retrieval training with orthogonal projection loss so fine-tuning can proceed with effectively single-sample batches. On LoCoV1, M2-BERT-32k beats the strongest truncation baseline, E5-Mistral, by `23.3` points on average while also running substantially faster.

## Problem & Motivation

Existing retrieval benchmarks and embedding models largely reward short-context behavior: many relevant signals appear near the beginning of documents, and the longest documents are only a few thousand tokens, so stronger long-context encoders often yield only marginal gains. The authors argue this is misaligned with practical domains such as legal opinions, government reports, scientific papers, financial filings, and meeting transcripts, where identifying the right document requires integrating evidence distributed across tens of thousands of tokens. This creates three coupled problems: current benchmarks do not measure long-context retrieval reliably, standard Transformer retrievers scale poorly to these lengths, and common contrastive fine-tuning losses degrade when GPU memory limits force tiny batches.

## Method

- **LoCoV1 benchmark**: construct a `12`-task long-context retrieval benchmark from Tau Scrolls, LongBench, QASPER, CourtListener, Australian Legal Case Reports, and StackOverflow. Example average document lengths range from `4,544` tokens (StackOverflow) to `58,129` tokens (QMSUM), and evaluation uses `nDCG@10`.
- **Retriever architecture**: build M2-BERT as an `80M`-parameter encoder on top of Monarch Mixer, a state-space model with subquadratic sequence scaling. Released variants use maximum sequence lengths `S ∈ {128, 2048, 8192, 32768}`.
- **Pretraining objective**: use masked language modeling with training mask probability `` `p = 0.3` `` and validation mask probability `` `p = 0.15` `` so the encoder learns both short query-like inputs and long document-like inputs.
- **Pretraining data mixture**: sample C4, Wikipedia, and BookCorpus equally (`~33%` each). The sequence-length mixture is `30%` variable-length examples and `70%` maximum-length concatenated examples; the paper's concrete proportions are C4/Wikipedia/BookCorpus variable `10/10/10%` and maximum `24/23/23%`.
- **Pretraining optimization**: use linear decay with warmup ratio `` `0.06` ``, learning rate `` `5e-4` ``, epsilon `` `1e-6` ``, betas `` `(0.9, 0.98)` ``, and weight decay `` `1e-5` ``.
- **Warm start for 32k**: random initialization works for `128`, `2k`, and `8k`, but `32k` does not converge reliably. The `32k` model is warm-started from the `8k` checkpoint, and its positional embeddings are extended by replicating the `8k` positional weights.
- **Standard retrieval loss baseline**: multiple negatives ranking loss is written as `` `MNRL({q_k, d_i}) = CE([PCS(q_k, d_i)]_{i=1}^n, [1, ..., n])` ``. At long context, memory forces a sharp reduction in effective negatives, e.g. from about `` `k = 128` `` at `128` tokens to `` `k = 2` `` at `32k`.
- **Prototype-loss attempt**: the authors also test `` `PL(q_k, p_k) = PCS(TM(q_k), SM(q_k)) + PCS(TM(p_k), SM(p_k))` ``, distilling from a short-context teacher, but find the resulting `128`-token geometry transfers poorly to `32k`.
- **Final fine-tuning loss**: adopt orthogonal projection loss, `` `OPL(q_k, p_k) = MSE(PCS(q_k, p_k), y)` `` with `` `y = 1.0` `` for positives and `` `y = 0.0` `` for negatives. OPL is batch-size independent and works with effectively single-sample batches.
- **Fine-tuning hyperparameters**: use Sentence Transformers with learning rate `` `5e-6` ``, true batch size `` `32` ``, `1` epoch, maximum gradient norm `` `1.0` ``, and `32` negative passages per query-positive pair; OPL uses cosine similarity distance.

## Key Results

- On LoCoV1, `M2-BERT-32k` scores `94.7`, versus `79.9` for BM25, `71.4` for E5-Mistral, `64.8` for fine-tuned BGE-Large, `63.2` for OpenAI Ada, and `53.6` for ColBERTv2.
- Relative to the best truncation baseline, `M2-BERT-32k` gains `+23.3` points over E5-Mistral; relative to the best chunked baseline, it gains `+24.4`; relative to BM25, it gains `+14.8`.
- Scaling context length steadily improves M2-BERT: `70.3` at `128`, `82.3` at `2k`, `86.9` at `8k`, and `94.7` at `32k`.
- On a per-task basis, `M2-BERT-32k` beats all baselines on `7/12` LoCoV1 tasks and all Transformer-based models on `10/12` tasks.
- The efficiency table reports `0.0071s` to encode `32,768` tokens for `M2-BERT-32k`, compared with `4.8s` for E5-Mistral, yielding speedups from `3.13x` at `128` tokens to `676x` at `32k`.
- The pretraining ablation shows mixed short/long examples score `55.4` on LoCoV1 at `2k`, compared with `44.9` for long-only and `37.2` for short-only pretraining.
- Warm-starting the `32k` checkpoint reaches MLM train accuracy `33.9` after `6,000` steps, versus `4.8` for cold start.
- For `M2-BERT-32k`, OPL reaches `94.7`, while MNRL reaches `70.4` and prototype loss `63.2`; on BEIR, `M2-BERT-128` scores `38.7` versus `40.0` for SentenceBERT while using `27%` fewer parameters.

## Limitations

- The paper focuses on a single long-context benchmark suite and one primary architecture family, so it does not establish that Monarch Mixer or OPL dominate all long-context retrieval designs.
- Although LoCoV1 covers `12` tasks, it is still an English-centric research benchmark and does not directly test multilingual retrieval, production ANN indexing, or end-to-end retrieval-augmented generation quality.
- The strongest M2-BERT model is specialized for long documents; the short-context `M2-BERT-128` still trails SentenceBERT by `1.3` points on BEIR, so gains are not universal across standard retrieval settings.
- Prototype loss transfers poorly from short to long context, which suggests the training recipe remains somewhat brittle and may depend on careful loss selection and checkpoint initialization.

## Concepts Extracted

- [[long-context-retrieval]]
- [[benchmark]]
- [[information-retrieval]]
- [[dense-retrieval]]
- [[state-space-model]]
- [[masked-language-modeling]]
- [[multiple-negatives-ranking-loss]]
- [[orthogonal-projection-loss]]
- [[chunking]]

## Entities Extracted

- [[jon-saad-falcon]]
- [[daniel-y-fu]]
- [[simran-arora]]
- [[neel-guha]]
- [[christopher-re]]
- [[stanford-university]]
- [[m2-bert]]
- [[loco-v1]]
- [[beir]]
- [[c4]]
- [[bookcorpus]]
- [[colbertv2]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
