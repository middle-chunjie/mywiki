---
type: source
subtype: paper
title: BOOOOKSCORE：A SYSTEMATIC EXPLORATION OF BOOK-LENGTH SUMMARIZATION IN THE ERA OF LLMS.pdf
slug: unknown-nd-booookscorea-2310-00785
date: 2026-04-20
language: en
tags: [summarization, llm, evaluation, long-context, coherence]
processed: true
raw_file: raw/papers/unknown-nd-booookscorea-2310-00785/paper.pdf
raw_md: raw/papers/unknown-nd-booookscorea-2310-00785/paper.md
bibtex_file: raw/papers/unknown-nd-booookscorea-2310-00785/paper.bib
possibly_outdated: true
authors:
  - Yapei Chang
  - Kyle Lo
  - Tanya Goyal
  - Mohit Iyyer
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.00785
doi:
url: https://arxiv.org/abs/2310.00785
citation_key: unknownndbooookscorea
paper_type: benchmark
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature. This paper studies book-length summarization for documents above `100K` tokens that exceed contemporary LLM context windows. It compares two prompting workflows, hierarchical merging and incremental updating, on a newly collected set of 100 recently published books to avoid BookSum-style contamination. The authors annotate `1,193` span-level coherence errors in GPT-4 summaries, derive an eight-type taxonomy, and convert it into BOOOOKSCORE, a sentence-level, reference-free metric that estimates the fraction of error-free sentences in a summary. BOOOOKSCORE tracks human judgments closely and supports broad ablations over chunk size and model choice, showing that GPT-4 and Claude 2 are strongest overall, incremental updating is usually more detailed but less coherent, and Mixtral approaches GPT-3.5-Turbo while LLaMA 2 degrades badly.

## Problem & Motivation

Book-length documents are typically far longer than the context windows of 2023-era LLMs, so practical systems must split the source into chunks and then merge or update partial summaries. Existing datasets such as BookSum are vulnerable to pretraining contamination, and conventional summary metrics do not capture the coherence failures that emerge when chunk-level outputs are stitched together. The paper therefore targets a missing evaluation regime: recently published books, fine-grained human annotation of confusion, and an automatic metric for coherence rather than overlap with unavailable gold summaries.

## Method

- Formalize book summarization under context limit `W` and source length `L >> W`; split the book into non-overlapping chunks `c_1, c_2, ..., c_{ceil(L / C)}` with `C < W`.
- **Hierarchical merging**: summarize each chunk, then recursively merge adjacent partial summaries until one global summary remains; each merge prompt plus its inputs must stay below `W - G_l`, where `G_l` controls summary length at level `l`.
- **Incremental updating**: maintain a running global summary `g_i`; after reading chunk `c_i`, prompt the model to update `g_{i-1}` into `g_i`, and trigger a separate compression prompt whenever the running summary exceeds the target limit `G_n`.
- Build a contamination-resistant evaluation corpus of `100` recently published books with average length `190K` tokens and no publicly available plot summaries found online.
- Run fine-grained human evaluation on GPT-4 summaries generated with `chunk size = 4096` and `G_n = 1200`; hire `4` annotators who each label `25` disjoint summaries by highlighting confusing spans and asking clarification questions.
- Derive an `8`-type coherence taxonomy: entity omission, event omission, causal omission, discontinuity, salience, language, inconsistency, and duplication.
- Define BOOOOKSCORE at sentence level as `` `BOOOOKSCORE(S) = (1 / n) * Σ_i [LLM(E, S, s_i) == No confusion]` ``; the evaluation prompt uses the full summary `S`, target sentence `s_i`, and `2` full worked examples containing `42` sentence-level annotations.
- Systematically evaluate `5` instruction-tuned LLMs with default `chunk size = 2048`, `G_n = 900`, `temperature = 0.5`, and `p = 1`; Claude 2 is additionally tested with `88K` context.

## Key Results

- Human annotators marked `840` coherence errors for GPT-4 incremental summaries versus `353` for hierarchical summaries; omission errors dominate, with entity omission at `7.3% / 3.71%` and event omission at `4.25% / 2.27%` errors per sentence for incremental / hierarchical outputs.
- BOOOOKSCORE annotation precision is `78.2%`, close to human precision `79.7%`; substituting human labels into the score changes GPT-4 system scores only slightly (`82.1` vs `82.4` incremental, `89.4` vs `90.8` hierarchical).
- Under hierarchical merging, Claude 2 (`2048`) achieves the highest BOOOOKSCORE at `91.1` with only `1.3%` repeated trigrams; GPT-4 reaches `89.1`, GPT-3.5-Turbo `84.2`, Mixtral `81.5`, and LLaMA 2 `72.4`.
- Incremental updating is usually less coherent but more detailed: GPT-4 falls from `89.1` to `82.5`, while coarse human preference favors incremental summaries for detail (`83` vs `11`) but hierarchical summaries for overall quality (`54` vs `44`), structure (`59` vs `35`), and logic (`53` vs `38`).
- Large context particularly helps incremental updating for Claude 2: `90.9` BOOOOKSCORE at `88K` context versus `78.6` at `2048`; LLaMA 2 could not perform incremental updating reliably and instead copied source text until the summary limit.
- BOOOOKSCORE makes wider ablations practical by saving roughly `$15K` in additional human evaluation cost and about `500` annotator hours.

## Limitations

- The coherence taxonomy is induced from GPT-4 summaries, so some failure modes of weaker or newer book summarizers may be underrepresented.
- BOOOOKSCORE requires sentence-by-sentence prompting with a long GPT-4 prompt, making it relatively expensive and slow despite still being cheaper than full human evaluation.
- The metric treats all error types uniformly rather than assigning severity weights, which may blur differences between minor and major coherence failures.
- Human validation measures annotation precision but not recall, because the study does not collect overlapping exhaustive annotations for each summary.
- The evaluation protocol centers on coherence and does not directly solve faithfulness assessment for full book-length inputs.

## Concepts Extracted

- [[book-length-summarization]]
- [[large-language-model]]
- [[chunking]]
- [[hierarchical-summarization]]
- [[incremental-updating]]
- [[context-compression]]
- [[reference-free-evaluation]]
- [[coherence-evaluation]]
- [[data-contamination]]
- [[few-shot-prompting]]

## Entities Extracted

- [[yapei-chang]]
- [[kyle-lo]]
- [[tanya-goyal]]
- [[mohit-iyyer]]
- [[university-of-massachusetts-amherst]]
- [[allen-institute-for-ai]]
- [[princeton-university]]
- [[gpt-4]]
- [[claude-2]]
- [[gpt-3-5-turbo]]
- [[mixtral-8x7b]]
- [[llama-2]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
