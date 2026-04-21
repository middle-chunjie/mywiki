---
type: source
subtype: paper
title: Retrieval meets Long Context Large Language Models.pdf
slug: unknown-nd-retrieval-2310-03025
date: 2026-04-20
language: en
tags: [retrieval, long-context, llm, question-answering, summarization]
processed: true

raw_file: raw/papers/unknown-nd-retrieval-2310-03025/paper.pdf
raw_md: raw/papers/unknown-nd-retrieval-2310-03025/paper.md
bibtex_file: raw/papers/unknown-nd-retrieval-2310-03025/paper.bib
possibly_outdated: true

authors:
  - Peng Xu
  - Wei Ping
  - Xianchao Wu
  - Lawrence McAfee
  - Chen Zhu
  - Zihan Liu
  - Sandeep Subramanian
  - Evelina Bakhturina
  - Mohammad Shoeybi
  - Bryan Catanzaro
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.03025
doi:
url: https://arxiv.org/abs/2310.03025
citation_key: unknownndretrieval
paper_type: benchmark

read_status: unread

domain: retrieval
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature. This paper compares two routes for handling long inputs in decoder-only LLMs: extending the context window versus adding retrieval at inference time, and then studies whether the two are complementary. Using instruction-tuned GPT-43B and Llama2-70B on seven long-context QA and query-focused summarization benchmarks, the authors show that simple top-`k` retrieval can make a `4K` model competitive with a `16K` long-context variant while using less generation compute. They further find that retrieval still helps `16K`/`32K` models, with the best `Llama2-70B-32k-ret` system outperforming GPT-3.5-turbo-16k on the reported average across seven tasks. The paper's practical message is that sparse external selection and long-context modeling should be combined rather than treated as substitutes.

## Problem & Motivation

Long-context LLMs have become increasingly attractive, but exact self-attention over long sequences remains expensive in both memory and latency. Retrieval-augmented generation offers an alternative by selecting only relevant chunks from a long document or set of documents, potentially approximating sparse attention over the full context at far lower cost. The paper asks three practitioner-facing questions: whether retrieval can match explicit context-window extension, whether long-context models still benefit from retrieval, and which combination gives the best trade-off on downstream long-context QA and summarization tasks with informative queries.

## Method

- **Models studied**: proprietary GPT-43B and public Llama2-70B, both starting from native context length `4,096`; GPT-43B has `48` layers, hidden size `8,192`, and pretraining on `1.1T` tokens, while Llama2-70B has `80` layers, hidden size `8,192`, and pretraining on `2T` tokens.
- **Context-window extension**: extend RoPE-based models with positional interpolation, adapting GPT-43B from `4K -> 16K` and Llama2 from `4K -> 16K/32K`; adaptation uses the Pile with batch size `128` and constant learning rate `5e-6`.
- **Retrieval pipeline**: chunk each document into `300`-word segments, encode the query and chunks independently, score with dense similarity `score(q, c) = q · c`, and concatenate the top `N` chunks in relevance order, where `N in {5, 10, 20}`.
- **Retrievers**: Dragon as a supervised dual encoder, Contriever as an unsupervised dense retriever trained with contrastive learning, and OpenAI `text-embedding-ada-002` with maximum input length `8,191` tokens and embedding dimension `1,536`.
- **Instruction tuning**: build a `102K`-example mixture from Soda, ELI5, FLAN, OpenAssistant, Dolly, and proprietary dialogue data; prompt format is `System + Context + User + Assistant`, and training applies loss only on `{Answer}` with batch size `128`, learning rate `5e-6`, and `1,000` steps.
- **Evaluation suite**: seven zero-shot long-context tasks spanning query-based summarization, single-document QA, and multi-hop QA; metrics are ROUGE geometric mean for QMSum, exact match for QuALITY, and F1 for the remaining five tasks.
- **Length regime analysis**: compare native truncation against long-context models and retrieval-augmented variants, then analyze retriever choice, retrieved-chunk count, and a lost-in-the-middle diagnostic for `Llama2-70B-4k` versus `Llama2-70B-32k`.

## Key Results

- On average score across seven tasks, retrieval makes `4K` models much stronger: GPT-43B improves from `26.44` to `29.32`, and Llama2-70B improves from `31.61` to `36.02`.
- Retrieval-augmented `4K` models become close to longer-context baselines with much lower generation cost: GPT-43B `4K + ret = 29.32` versus GPT-43B `16K = 29.45`; Llama2-70B `4K + ret = 36.02` versus Llama2-70B `16K = 36.78`.
- Retrieval still helps long-context models: Llama2-70B rises from `37.36` to `39.60` at `32K`, and from `36.78` to `37.23` at `16K`.
- The strongest reported system, `Llama2-70B-32k-ret`, reaches `43.6` average over the seven-task comparison in Table 3, beating GPT-3.5-turbo-16k at `42.8` and Davinci003 at `39.2`.
- HotpotQA benefits strongly from longer context even before retrieval, with Llama2-70B improving from `34.64` at `4K` to `43.97` at `16K`; with retrieval at `32K`, it reaches `53.89`.
- More retrieved context is not monotonically better: for Llama2-70B at `32K`, top-`5` gives `39.60`, top-`10` gives `38.98`, and top-`20` drops to `38.38`, consistent with distraction or lost-in-the-middle effects.
- Retrieval improves throughput-relevant efficiency as well: the paper reports the retrieval-augmented `Llama2-70B-32k` setup can be about `4x` faster than the non-retrieval `32K` baseline on NarrativeQA generation.

## Limitations

- The study is limited to seven long-context tasks plus two few-shot tasks, so claims may not transfer to other domains such as code, dialogue agents, or multimodal settings.
- Comparisons focus on large decoder-only models (`43B` and `70B`), while smaller models behave differently; the paper's conclusions are explicitly weaker for `6B`/`7B` systems.
- Retrieval is evaluated with relatively simple chunking and rank-then-concatenate pipelines, not end-to-end learned retrieval-reader training.
- The OpenAI comparisons rely on black-box APIs, so architecture, preprocessing, and exact serving-time behavior are not controlled.
- The best retrieved chunk count depends on context length; adding up to `20` chunks can hurt, which means the proposed recipe does not remove long-context robustness issues such as lost-in-the-middle.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[dense-retrieval]]
- [[dual-encoder]]
- [[contrastive-learning]]
- [[instruction-tuning]]
- [[zero-shot-learning]]
- [[rotary-positional-embedding]]
- [[lost-in-the-middle]]
- [[multihop-question-answering]]

## Entities Extracted

- [[peng-xu]]
- [[wei-ping]]
- [[xianchao-wu]]
- [[lawrence-mcafee]]
- [[chen-zhu]]
- [[zihan-liu]]
- [[sandeep-subramanian-nvidia]]
- [[evelina-bakhturina]]
- [[mohammad-shoeybi]]
- [[bryan-catanzaro]]
- [[nvidia]]
- [[llama-2]]
- [[dragon]]
- [[contriever]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
