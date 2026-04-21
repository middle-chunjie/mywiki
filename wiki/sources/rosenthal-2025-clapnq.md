---
type: source
subtype: paper
title: "CLAPnq: Cohesive Long-form Answers from Passages in Natural Questions for RAG systems"
slug: rosenthal-2025-clapnq
date: 2026-04-20
language: en
tags: [rag, benchmark, lfqa, retrieval, grounding]
processed: true
raw_file: raw/papers/rosenthal-2025-clapnq/paper.pdf
raw_md: raw/papers/rosenthal-2025-clapnq/paper.md
bibtex_file: raw/papers/rosenthal-2025-clapnq/paper.bib
possibly_outdated: false
authors:
  - Sara Rosenthal
  - Avirup Sil
  - Radu Florian
  - Salim Roukos
year: 2025
venue: Transactions of the Association for Computational Linguistics
venue_type: journal
arxiv_id:
doi: 10.1162/tacl_a_00729
url: https://aclanthology.org/2025.tacl-1.3/
citation_key: rosenthal2025clapnq
paper_type: benchmark
read_status: unread
domain: retrieval
---

## Summary

This paper introduces CLAPnq, a benchmark for grounded long-form question answering in retrieval-augmented generation systems built from Natural Questions passages. Unlike prior LFQA datasets that either lack gold evidence or only test generation, CLAPnq pairs concise multi-sentence answers with gold supporting passages and a retrieval corpus, enabling separate evaluation of retrieval, generation with oracle evidence, and full end-to-end RAG. The benchmark explicitly targets five properties of good answers: faithfulness, conciseness, completeness, cohesion, and correct abstention on unanswerable questions. Baseline experiments show that strong decoder LLMs often answer too verbosely and fail to abstain, while a fine-tuned FLAN-T5 model better matches desired answer length and grounding. Overall, the dataset exposes a substantial gap between gold-passage generation and realistic noisy RAG.

## Problem & Motivation

Existing long-form QA benchmarks are poorly matched to grounded RAG evaluation. Some focus on short extractive answers, some provide long answers without gold evidence, and some fix retrieved passages instead of exposing a full corpus. That makes it hard to separately measure retrieval quality, gold-passage generation quality, and the degradation that appears in a full RAG pipeline.

The authors target a more realistic benchmark for grounded long-form QA: answers should be supported by a passage, remain concise rather than copying the whole passage, integrate non-contiguous evidence into a cohesive response, and detect when a question is unanswerable from the provided evidence. CLAPnq is meant to stress exactly those properties and reveal where current RAG systems still fail.

## Method

- **Data source**: start from Natural Questions examples with a long answer but no short answer, excluding tables/lists and favoring passages with more than `5` sentences. This yields `12,657` train and `384` dev candidates for annotation.
- **Two-round annotation**: `7` trained in-house annotators work in Appen. Round 1 selects relevant sentences and writes a concise answer; Round 2 re-annotates cases where the selected evidence is non-consecutive, i.e. there exists at least one skipped sentence between selected spans.
- **Answer design target**: each answer should be faithful, concise, complete, and cohesive. The final dataset keeps answers annotated by more than one person; inter-annotator agreement between round-1 and round-2 answers is measured by `RougeL = 0.67`.
- **Dataset construction**: CLAPnq contains both answerable and unanswerable examples. Answerable items include question, title, gold passage `P`, relevant sentences `RS`, and answer `A`; unanswerables are sampled from NQ examples judged unanswerable and paired with a random passage from the same document as pseudo-context.
- **Retrieval corpus**: build a passage corpus from the original NQ Wikipedia documents, drop passages longer than about `3000` words or shorter than `15` words, remove near-duplicate document variants when `RougeL > 0.90`, and keep multiple gold passages when necessary.
- **Retrieval baselines**: compare BM25, `all-MiniLM-L6-v2`, `BGE-base`, and `E5-base-v2`. BM25 uses ElasticSearch with maximum passage length `512` tokens and overlap stride `256` when splitting long passages.
- **Generation setups**: evaluate zero-shot, `1/0` one-shot, and `1/1` two-shot prompting on gold passages for FLAN-T5-Large, FLAN-T5-XXL, Llama-13B-chat, GPT-3.5, GPT-4, Mistral-7B-Instruct, and fine-tuned models.
- **Fine-tuned model**: `CLAPnq-T5-lg` uses FLAN-T5-Large with `epochs = 6`, `lr = 1e-4`, `batch_size = 32`, `max_input_len = 412`, and `max_output_len = 100`; about `368` training passages (`10%`) are truncated at the tail to fit context.
- **RAG-trained variants**: for retrieval-conditioned training, increase context to `1024`, use `batch_size = 8`, and train for `10` epochs; one variant preserves the gold passage in the top `3` retrieved passages at a random position.
- **Evaluation**: retrieval uses `nDCG@{1,3,5,10}` and `Recall@10`; generation uses RougeL, recall, RougeLp, answer length, and unanswerable accuracy. Human evaluation uses `3` annotators scoring faithfulness and appropriateness on a `1-4` scale plus pairwise preference.

## Key Results

- CLAPnq contains `4,946` questions: `2,555` answerable and `2,391` unanswerable. Answers average `56.8` words over `2.3` sentences, versus average passage length `156` words, and round-1/round-2 agreement reaches `RougeL = 0.67`.
- Retrieval is non-trivial even with modern dense retrievers: on dev, `E5-base-v2` achieves `nDCG@10 = 64` and `Recall@10 = 87`; on test, `E5-base-v2` and `BGE-base` tie at `nDCG@10 = 65`, both with `Recall@10 = 88`.
- In gold-passage generation, `CLAPnq-T5-lg` is the strongest overall model on test with `RougeL = 57.7`, `Recall = 69.5`, `RougeLp = 51.7`, `Len = 351`, and `Unanswerable = 86.8%`.
- Strong decoder LLMs underperform on the benchmark's desired answer style: GPT-3.5 reaches test `RougeL = 40.3`, `Recall = 56.3`, `RougeLp = 29.9`, `Len = 375`, `Unanswerable = 31.3%`, while GPT-4 is even longer and worse on abstention (`Len = 797`, `Unanswerable = 22.2%`).
- End-to-end RAG remains much harder than oracle generation: `E5-base-v2 + CLAPnq-T5-lg` drops to test `RougeL = 41.6`, `Recall = 51.3`, `RougeLp = 55.7`, `Len = 321`, and `Unanswerable = 45.9%`.
- Human evaluation confirms the automatic pattern. In the gold setup, `CLAPnq-T5-lg` scores `Faithful = 3.7`, `Appropriate = 3.7`, `Win-rate = 66%`; in RAG it remains highly faithful (`3.8`) but becomes less appropriate (`3.2`) because retrieval noise changes what can be answered well.

## Limitations

- The benchmark is derived only from Natural Questions and Wikipedia, so it inherits their topical, linguistic, and annotation biases rather than covering broader domains.
- Because the official NQ test set is unavailable, CLAPnq dev/test construction partially reuses NQ train material; this makes contamination from model pretraining on NQ difficult to rule out.
- Manual annotation remains noisy: the authors explicitly note that some answers are likely incorrect or unclear despite the two-round process.
- Automatic metrics and single-gold evaluation can under-credit alternative but grounded RAG answers when retrieved evidence differs from the gold passage.
- Unanswerable examples use a random passage from the source document as pseudo-context, which is useful for benchmarking abstention but is not identical to naturally retrieved failure cases in deployed RAG.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[long-form-question-answering]]
- [[benchmark-dataset]]
- [[passage-retrieval]]
- [[dense-retrieval]]
- [[faithfulness]]
- [[unanswerable-questions]]
- [[human-evaluation]]

## Entities Extracted

- [[sara-rosenthal]]
- [[avirup-sil]]
- [[radu-florian]]
- [[salim-roukos]]
- [[ibm-research-ai]]
- [[natural-questions]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
