---
type: source
subtype: paper
title: Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions
slug: trivedi-2023-interleaving-2212-10509
date: 2026-04-20
language: en
tags: [retrieval, reasoning, llm, qa, prompting]
processed: true

raw_file: raw/papers/trivedi-2023-interleaving-2212-10509/paper.pdf
raw_md: raw/papers/trivedi-2023-interleaving-2212-10509/paper.md
bibtex_file: raw/papers/trivedi-2023-interleaving-2212-10509/paper.bib
possibly_outdated: true

authors:
  - Harsh Trivedi
  - Niranjan Balasubramanian
  - Tushar Khot
  - Ashish Sabharwal
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2212.10509
doi:
url: http://arxiv.org/abs/2212.10509
citation_key: trivedi2023interleaving
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper introduces IRCoT, a retrieval strategy for knowledge-intensive multi-step question answering that alternates between generating one chain-of-thought sentence and retrieving more evidence with that sentence as the next query. The core claim is that one-shot question-only retrieval misses later-hop evidence because the right query often depends on partial reasoning. IRCoT uses BM25 retrieval plus a prompted language model reasoner, then feeds the accumulated paragraphs into a separate QA reader. Across HotpotQA, 2WikiMultihopQA, MuSiQue, and IIRC, the method improves retrieval recall and downstream F1 in both in-domain and out-of-distribution settings, reduces factual errors in generated reasoning, and remains effective with smaller Flan-T5 models as well as GPT3.

## Problem & Motivation

One-step retrieve-and-read pipelines are often inadequate for multi-hop open-domain QA because later evidence may have little lexical overlap with the original question. The paper argues that reasoning and retrieval should be mutually conditioning: retrieval helps avoid hallucinated intermediate steps, while partial reasoning exposes better follow-up queries for the next hop. IRCoT is proposed to operationalize this loop in a few-shot setting without supervised retriever training.

## Method

- **Ingredients**: IRCoT combines a base retriever, a zero/few-shot CoT-capable LM, and a small set of annotated demonstrations containing both reasoning chains and supporting paragraphs.
- **Initial retrieval**: retrieve `K` paragraphs with the original question `Q`; for IRCoT the tuned step budget is `K ∈ {2, 4, 6, 8}` and total retrieved paragraphs are capped at `15`.
- **Reason step**: prompt the LM with the question, all collected paragraphs, and previously generated CoT sentences in the format `Q: <question> / A: <CoT-sent-1> ... <CoT-sent-n>`; if the LM emits multiple sentences, only the first new sentence is kept.
- **Retrieve step**: use the last generated CoT sentence as the next query, retrieve `K` more paragraphs, and merge them into the evidence pool. This interleaving continues until the CoT contains `"answer is:"` or a maximum-step threshold is hit.
- **Demonstrations**: each demonstration uses gold supporting paragraphs plus `M ∈ {1, 2, 3}` randomly sampled distractor paragraphs; at test time the model sees all accumulated paragraphs from prior retrieval rounds.
- **Retriever implementation**: the underlying retriever is BM25 in Elasticsearch. The one-step baseline uses question-only retrieval with `K ∈ {5, 7, 9, 11, 13, 15}`.
- **Reader**: a separate QA reader consumes the retrieved paragraphs. GPT3 uses CoT prompting; Flan-T5 readers use direct prompting. The reader shares the same LM family as the IRCoT reasoner.
- **Context budgets**: GPT3 `code-davinci-002` is limited to `8K` word pieces; Flan-T5 inputs are capped at `6K` word pieces due to `80G` A100 memory.
- **Evaluation setup**: experiments cover HotpotQA, 2WikiMultihopQA, answerable MuSiQue, and answerable IIRC, with `100` dev questions for tuning and `500` test questions. The authors manually wrote CoTs for `20` questions per dataset, formed `3` demonstration sets of `15` examples each, and report mean/std over the three sets.

## Key Results

- **Retrieval recall gains over one-step retrieval**: with Flan-T5-XXL, IRCoT improves recall by `+7.9` on HotpotQA, `+14.3` on 2WikiMultihopQA, `+3.5` on MuSiQue, and `+10.2` on IIRC; with GPT3 the gains are `+11.3`, `+22.6`, `+12.5`, and `+21.2`, respectively.
- **Downstream QA gains over OneR QA**: with Flan-T5-XXL, IRCoT improves F1 by `+9.4` (HotpotQA), `+15.3` (2WikiMultihopQA), `+5.0` (MuSiQue), and `+2.5` (IIRC); with GPT3 the gains are `+7.1`, `+13.2`, and `+7.1` on the first three datasets, with no improvement on IIRC.
- **Best reported GPT3 IRCoT QA scores**: `60.7` F1 on HotpotQA, `68.0` on 2WikiMultihopQA, `36.5` on MuSiQue, and `49.9` on IIRC.
- **Factuality**: relative to OneR, IRCoT reduces annotated CoT factual errors by `50%` on HotpotQA and `40%` on 2WikiMultihopQA.
- **Scaling**: IRCoT improves retrieval for all tested Flan-T5 sizes down to `0.2B`; IRCoT with Flan-T5-XL (`3B`) outperforms one-step retrieval + GPT3 (`175B`) QA in the reported datasets.
- **Leaderboard-style comparison**: GPT3-based IRCoT QA reports `58.5` F1 on HotpotQA bridge questions, `60.7` on HotpotQA, `68.0` on 2WikiMultihopQA, and `43.8` on MuSiQue 2-hop, exceeding the non-head-to-head baselines listed in Table 1.

## Limitations

- IRCoT assumes the base LM can generate useful zero/few-shot CoT; the paper notes this is weaker for smaller models below roughly `20B`.
- The method requires long input windows because retrieved paragraphs and demonstrations must fit together in context.
- Computational cost rises with reasoning depth because IRCoT makes an LM call for each CoT sentence rather than one retrieval pass.
- Part of the evaluation depends on OpenAI `code-davinci-002`, which was later deprecated, making exact reproduction harder.
- The paper's SOTA comparison is not head-to-head because competing systems use different corpora, APIs, and model backbones.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[chain-of-thought-prompting]]
- [[open-domain-question-answering]]
- [[multihop-question-answering]]
- [[bm25]]
- [[few-shot-learning]]
- [[hallucination]]
- [[out-of-distribution-generalization]]

## Entities Extracted

- [[harsh-trivedi]]
- [[niranjan-balasubramanian]]
- [[tushar-khot]]
- [[ashish-sabharwal]]
- [[stony-brook-university]]
- [[allen-institute-for-ai]]
- [[code-davinci-002]]
- [[flan-t5-xxl]]
- [[hotpotqa]]
- [[2wikimultihopqa]]
- [[musique]]
- [[iirc]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
