---
type: source
subtype: paper
title: "RA-DIT: Retrieval-Augmented Dual Instruction Tuning"
slug: unknown-nd-raditretrievalaugmented-2310-01352
date: 2026-04-20
language: en
tags: [retrieval-augmented-generation, instruction-tuning, llm, dense-retrieval, question-answering]
processed: true

raw_file: raw/papers/unknown-nd-raditretrievalaugmented-2310-01352/paper.pdf
raw_md: raw/papers/unknown-nd-raditretrievalaugmented-2310-01352/paper.md
bibtex_file: raw/papers/unknown-nd-raditretrievalaugmented-2310-01352/paper.bib
possibly_outdated: true

authors:
  - Xi Victoria Lin
  - Xilun Chen
  - Mingda Chen
  - Weijia Shi
  - Maria Lomeli
  - Rich James
  - Pedro Rodriguez
  - Jacob Kahn
  - Gergely Szilvasy
  - Mike Lewis
  - Luke Zettlemoyer
  - Scott Yih
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.01352
doi:
url: https://arxiv.org/abs/2310.01352
citation_key: unknownndraditretrievalaugmented
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

RA-DIT proposes a lightweight recipe for retrofitting pretrained large language models with retrieval rather than relying on expensive retrieval-aware pretraining or naive post-hoc retrieval fusion. The method separates optimization into two stages: retrieval-augmented instruction tuning for the language model and LM-supervised fine-tuning for the retriever. Built on LLaMA plus a DRAGON+ dense retriever, it uses parallel top-`k` retrieval augmentation and weighted ensembling over retrieved chunks. Across knowledge-intensive zero-shot and few-shot benchmarks, the combined system consistently beats vanilla LLaMA and REPLUG, and it remains competitive with Atlas despite avoiding continuous pretraining. The paper's main claim is that decoupled LM and retriever adaptation is an efficient and effective path to stronger retrieval-augmented LLMs.

## Problem & Motivation

The paper targets a practical gap in retrieval-augmented language modeling. End-to-end retrieval-aware pretraining methods such as REALM, RETRO, and Atlas are expensive, while plug-in retrieval methods that attach an off-the-shelf retriever to an LLM leave the model poorly trained to use retrieved evidence and vulnerable to noisy context. RA-DIT argues that a middle ground exists: fine-tune a pretrained LLM to exploit retrieved background information and, separately, fine-tune the retriever to return documents preferred by that LLM. The motivation is especially strong for knowledge-intensive tasks where parametric knowledge is incomplete, stale, or long-tail, but where full retraining of the base model is economically unattractive.

## Method

- **Base architecture**: initialize the system with [[llama]] and a [[dragon-plus]]-style dense retriever. The retriever is dual-encoder / [[bi-encoder]] based, with query and document embeddings scored by `s(q, c) = E_q(q) · E_d(c)`.
- **Parallel retrieval augmentation**: for prompt `x`, retrieve top-`k` chunks `C'` and compute `p_LM(y | x, C') = Σ_{c∈C'} p_LM(y | c ∘ x) · p_R(c | x)`, where `p_R(c | x)` is the top-`k`-normalized retriever distribution.
- **LM fine-tuning objective**: construct retrieval-augmented instruction examples `(c_ij ∘ x_i, y_i)` and optimize only output-token likelihood, `L(D_L) = -Σ_i Σ_j log p_LM(y_i | c_ij ∘ x_i)`.
- **Language-model training data**: `D_L` contains `20` datasets across dialogue, open-domain QA, reading comprehension, summarization, and chain-of-thought / reasoning categories; each example uses top-`\tilde{k} = 3` retrieved chunks during LM fine-tuning.
- **Retriever fine-tuning objective**: define LM-supervised retrieval scores `p_LSR(c | x, y) ∝ exp(p_LM(y | c ∘ x) / τ)` and minimize `KL(p_R(c | x) || p_LSR(c | x, y))` with temperature `τ = 0.01`, updating only the query encoder.
- **Retrieval corpus**: combine December 2021 Wikipedia with 2017-2020 CommonCrawl into a `399M`-chunk datastore; chunks are capped at `200` words and indexed with GPU nearest-neighbor search.
- **LM fine-tuning recipe**: use a data mixture with `10%` unsupervised text and `5%` OASST-1, cap the remaining per-dataset sample count at `η = 7500`, pack examples to sequence length `2048`, and use cosine LR decay from `1e-5` to `1e-7` with `200` warmup steps and `500` total steps.
- **Scale settings**: fine-tune LLaMA `7B`, `13B`, and `65B` on `8`, `16`, and `64` A100 GPUs respectively; for `65B`, use batch size `128`, model parallel `8`, and early stopping after `300` steps based on KILT dev performance.
- **Retriever fine-tuning recipe**: train on a `95%` corpus / `5%` multi-task-instruction mixture, using `900k` self-supervised corpus chunks plus `286k` MTI examples, LR `1e-5`, batch size `32` per GPU, and top-`10` retrieved chunks for LSR supervision.
- **Inference**: use top-`k = 10` retrieved chunks and aggregate weighted answer probabilities across prompts; for generation tasks, decode each augmented prompt independently and select the answer with the highest weighted probability mass.

## Key Results

- **Main zero-shot gains**: RA-DIT `65B` reaches `49.1` average on `MMLU/NQ/TQA/ELI5` versus REPLUG `45.1` and vanilla LLaMA `32.9`; on the broader average it scores `50.5` versus `43.1` and `22.8`.
- **Main five-shot gains**: RA-DIT `65B` reaches `51.8` on the four-task average and `55.2` overall, beating REPLUG `51.1 / 52.7` and vanilla LLaMA `47.2 / 45.0`.
- **Dataset-level improvements**: in 0-shot, RA-DIT improves over REPLUG on MMLU `64.6 vs 59.7`, Natural Questions `35.2 vs 28.8`, FEVER `80.7 vs 73.3`, zsRE `73.7 vs 50.8`, and T-REx `53.1 vs 36.3`.
- **Few-shot benchmark numbers**: in 5-shot, RA-DIT reaches NQ `43.9`, FEVER `90.7`, AIDA `55.8`, and zsRE `72.4`, generally matching or exceeding REPLUG on most knowledge-intensive tasks.
- **Comparison to Atlas**: in the `64`-shot setting, a single RA-DIT `65B` model scores `60.9` average versus Atlas `56.8`, with especially large gains on AIDA `80.5 vs 66.5` and T-REx `72.8 vs 58.9`.
- **Parametric-knowledge retention**: without retrieval, RA-DIT `65B` still improves commonsense reasoning average from `72.1` to `74.5`, suggesting the method does not simply trade away internal reasoning ability.
- **LM-ft ablation**: retrieval-augmented instruction tuning is better than standard instruction tuning under retrieval; with top-`10` chunks in 0-shot dev, RA-IT `65B` scores `51.0` average versus IT `47.7`.
- **Retriever-ft ablation**: query-encoder-only fine-tuning with `95%` corpus + `5%` MTI yields the best retriever result at `58.0` average, above base DRAGON+ at `57.4`; fine-tuning both encoders hurts to `55.8`.
- **Dual-tuning ablation**: combining LM-ft and R-ft gives the strongest `5`-shot dev average, `54.6`, compared with `53.9` for LM-ft only and `54.4` for retriever-ft only.

## Limitations

- Inference cost scales with the number of retrieved chunks because the model runs one LM pass per retrieved passage; the default `k = 10` setting is expensive.
- The method still depends heavily on retriever quality, and the paper shows large variation across corpora, chunk counts, and retriever backbones.
- Retriever fine-tuning updates only the query encoder and keeps top-`10` retrieved chunks / LSR scores fixed during training, which simplifies optimization but may leave gains on the table.
- Evaluation is concentrated on knowledge-intensive QA, fact checking, entity linking, and dialogue benchmarks; broader domains and harder multi-hop synthesis remain underexplored.
- Despite strong average gains, RA-DIT does not dominate every single benchmark or configuration, and Wizard of Wikipedia improvements remain small.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[instruction-tuning]]
- [[large-language-model]]
- [[dense-retrieval]]
- [[bi-encoder]]
- [[lm-supervised-retrieval]]
- [[in-context-learning]]
- [[parametric-knowledge]]
- [[non-parametric-memory]]
- [[approximate-nearest-neighbor-search]]

## Entities Extracted

- [[xi-victoria-lin]]
- [[xilun-chen]]
- [[mingda-chen]]
- [[weijia-shi]]
- [[maria-lomeli]]
- [[rich-james]]
- [[pedro-rodriguez-meta]]
- [[jacob-kahn]]
- [[gergely-szilvasy]]
- [[mike-lewis]]
- [[luke-zettlemoyer]]
- [[scott-wen-tau-yih]]
- [[meta-ai]]
- [[llama]]
- [[dragon-plus]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
