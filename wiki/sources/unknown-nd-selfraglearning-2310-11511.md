---
type: source
subtype: paper
title: "SELF-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection"
slug: unknown-nd-selfraglearning-2310-11511
date: 2026-04-20
language: en
tags: [llm, retrieval, rag, factuality, citation]
processed: true

raw_file: raw/papers/unknown-nd-selfraglearning-2310-11511/paper.pdf
raw_md: raw/papers/unknown-nd-selfraglearning-2310-11511/paper.md
bibtex_file: raw/papers/unknown-nd-selfraglearning-2310-11511/paper.bib
possibly_outdated: true

authors:
  - Akari Asai
  - Zeqiu Wu
  - Yizhong Wang
  - Avirup Sil
  - Hannaneh Hajishirzi
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.11511
doi:
url: https://arxiv.org/abs/2310.11511
citation_key: unknownndselfraglearning
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

SELF-RAG proposes a retrieval-augmented large language model that decides when to retrieve evidence, generates answers conditioned on candidate passages, and emits self-reflective tokens that score passage relevance, claim support, and overall usefulness. Instead of using a separate verifier during inference, the model learns to predict these control signals directly as next tokens, which makes decoding both more interpretable and more configurable. The method combines an offline-trained critic with a generator fine-tuned on roughly 150k instruction-following and knowledge-intensive examples. Across open-domain QA, fact verification, reasoning, and long-form cited generation, SELF-RAG 7B/13B consistently outperforms conventional RAG baselines and often matches or exceeds stronger proprietary systems on factuality and citation precision.

## Problem & Motivation

Standard [[retrieval-augmented-generation]] systems usually prepend a fixed number of retrieved passages whether or not retrieval is actually helpful. The paper argues that this hurts general instruction-following versatility, injects irrelevant context, and still does not guarantee that model outputs are supported by retrieved evidence. SELF-RAG is motivated by the need for a single model that can selectively use external knowledge, explicitly judge whether evidence is relevant and supportive, and expose those judgments during decoding so practitioners can trade off factual grounding, completeness, and fluency. The target setting spans both knowledge-intensive tasks, where retrieval is essential, and open-ended prompts, where forced retrieval can be counterproductive.

## Method

- **Core formulation**: train a generator `M` to emit output segments `y = [y_1, ..., y_T]` plus [[reflection-token]]s, so retrieval decisions and self-evaluations become ordinary next-token predictions rather than a separate inference-time module.
- **Token types**: the model predicts `Retrieve ∈ {yes, no, continue}`, `ISREL ∈ {relevant, irrelevant}`, `ISSUP ∈ {fully supported, partially supported, no support}`, and `ISUSE ∈ {1, 2, 3, 4, 5}` to operationalize [[self-reflection]] for evidence need, relevance, support, and utility.
- **Inference flow**: before each segment, SELF-RAG decides whether to trigger [[adaptive-retrieval]]. If retrieval fires, the retriever returns top-`K` passages, the model generates `K` passage-conditioned continuations in parallel, critiques them, and keeps the best continuation.
- **Adaptive retrieval rule**: retrieval can be triggered when `p(Retrieve=yes) / (p(Retrieve=yes) + p(Retrieve=no)) > δ`; the default threshold is `δ = 0.2` for most tasks and `δ = 0` for ALCE/ASQA-style citation-heavy generation.
- **Critique-guided decoding**: segment scoring uses `f(y_t, d) = p(y_t | x, d, y_<t) + S(critique)`, where `S(critique) = Σ_G w^G s_t^G` over `G ∈ {ISREL, ISSUP, ISUSE}`. Default weights are `w^ISREL = 1.0`, `w^ISSUP = 1.0`, and `w^ISUSE = 0.5`.
- **Critic supervision**: a critic `C` is trained from GPT-4-labeled feedback, with roughly `4k-20k` supervision instances per reflection type. The paper reports `>90%` agreement with GPT-4 predictions on most categories when `C` is initialized from Llama 2 7B.
- **Generator training**: the final generator is trained with the standard next-token objective on a corpus augmented offline with retrieved passages and critique tokens. Retrieved chunks are masked out of the loss, and the final training set totals about `145,619` examples (described in the main text as about `150k`).
- **Data sources and models**: training mixes Open-Instruct subsets with knowledge-intensive data such as KILT sources, ASQA, ARC-Easy, and OpenBookQA. The generator bases are Llama 2 `7B` and `13B`; the base critic is Llama 2 `7B`.
- **Retriever and decoding hyperparameters**: the default retriever is Contriever-MS MARCO; inference uses the top `5` retrieved documents by default, beam width `2` at the segment level, greedy token decoding, and `vllm` for serving. For biography/OpenQA tasks the system adds another `5` web-retrieved documents.
- **Training details**: models are trained for `3` epochs with batch size `128`, peak learning rate `2e-5`, `3%` warmup, linear decay, max length `2048` for `7B` and `1524` for `13B`, using `4` A100 `80GB` GPUs, BF16, DeepSpeed ZeRO-3, and FlashAttention.

## Key Results

- On PopQA, SELF-RAG reaches `54.9` (`7B`) and `55.8` (`13B`) accuracy, beating vanilla Llama 2 (`14.7`/`14.7`), retrieval-augmented Llama 2 (`38.2`/`45.7`), and even Ret-ChatGPT (`50.8`).
- On TriviaQA, SELF-RAG scores `66.4` (`7B`) and `69.3` (`13B`), surpassing Alpaca `13B` with retrieval (`66.9`) and Ret-ChatGPT (`65.7`).
- On closed-set factual tasks, SELF-RAG gets `72.4`/`74.5` on PubHealth and `67.3`/`73.1` on ARC-Challenge, outperforming non-proprietary baselines across both tasks; the `13B` model approaches ChatGPT on ARC (`75.3`).
- On biography generation, SELF-RAG obtains FactScore `81.2` (`7B`) and `80.2` (`13B`), compared with ChatGPT `71.8` and retrieval-augmented Llama 2 `78.0` (`7B`) / `77.5` (`13B`).
- On ASQA-style long-form answers, SELF-RAG `13B` reaches citation precision/recall `70.3`/`71.3` and MAUVE `71.6`; the `7B` model reaches `66.9`/`67.8` precision/recall with MAUVE `74.3`. Precision exceeds Ret-ChatGPT (`65.1`), though recall remains lower than Ret-ChatGPT (`76.6`).
- Ablations on a `50k`-example setting show that removing the retriever drops PopQA from `45.5` to `43.6`, removing the critic drops ASQA EM from `32.1` to `18.1`, always using top-`1` retrieval drops PopQA to `41.8`, and removing `ISSUP` from decoding lowers ASQA EM to `30.6`.

## Limitations

- The method still produces unsupported or partially supported claims; the paper's qualitative examples and human evaluation explicitly show remaining citation and support errors.
- Critique supervision depends on GPT-4-generated labels, so part of the training pipeline inherits proprietary-model cost and reproducibility constraints even though inference does not require GPT-4.
- Inference is more expensive than a plain LM because it may retrieve repeatedly, score multiple passages in parallel, and run segment-level beam search.
- Retrieval quality remains a bottleneck: the paper notes Wikipedia snapshot freshness issues on PopQA and leaves jointly trained or instruction-tuned retrievers for future work.
- Results are concentrated on six English benchmarks and zero-shot prompting; broader multilingual, robustness, and long-horizon agent settings are not evaluated here.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[large-language-model]]
- [[self-reflection]]
- [[reflection-token]]
- [[adaptive-retrieval]]
- [[constrained-decoding]]
- [[factuality]]
- [[citation-grounding]]
- [[instruction-tuning]]
- [[open-domain-question-answering]]
- [[fact-verification]]
- [[long-form-generation]]

## Entities Extracted

- [[akari-asai]]
- [[zeqiu-wu]]
- [[yizhong-wang]]
- [[avirup-sil]]
- [[hannaneh-hajishirzi]]
- [[university-of-washington]]
- [[allen-institute-for-ai]]
- [[ibm-research]]
- [[llama-2]]
- [[gpt-4]]
- [[popqa]]
- [[triviaqa]]
- [[arc-challenge]]
- [[asqa]]
- [[vllm]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
