---
type: source
subtype: paper
title: Teaching language models to support answers with verified quotes
slug: menick-2022-teaching-2203-11147
date: 2026-04-20
language: en
tags: [llm, question-answering, citation, retrieval, rlhf]
processed: true

raw_file: raw/papers/menick-2022-teaching-2203-11147/paper.pdf
raw_md: raw/papers/menick-2022-teaching-2203-11147/paper.md
bibtex_file: raw/papers/menick-2022-teaching-2203-11147/paper.bib
possibly_outdated: true

authors:
  - Jacob Menick
  - Maja Trebacz
  - Vladimir Mikulik
  - John Aslanides
  - Francis Song
  - Martin Chadwick
  - Mia Glaese
  - Susannah Young
  - Lucy Campbell-Gillingam
  - Geoffrey Irving
  - Nat McAleese
year: 2022
venue: arXiv
venue_type: preprint
arxiv_id: 2203.11147
doi:
url: http://arxiv.org/abs/2203.11147
citation_key: menick2022teaching
paper_type: method

read_status: unread
read_date:
rating:

domain: llm
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature. This paper introduces GopherCite, a `280B` open-book QA system that answers with inline verbatim quotes drawn from retrieved documents, so users can directly inspect supporting evidence instead of trusting unsupported generations. The method combines Google-based retrieval, supervised fine-tuning to teach a strict quote syntax, a reward model trained on `33,242` human preference pairs, and RL from human preferences to improve answer quality and abstention. Human evaluation shows strong supported-and-plausible performance on filtered Natural Questions and ELI5 subsets, especially with reranking. The central finding is also a limitation: quote-supported answers are not necessarily true, since models can select misleading but superficially supportive evidence from imperfect sources.

## Problem & Motivation

Standard large language models can answer many factual questions, but their outputs are hard to trust because they often hallucinate fluent but unsupported claims. The paper targets this trust gap by making the model expose evidence inline, reducing verification effort for both end users and human raters. The goal is not just answer correctness, but self-supported question answering: produce a plausible answer together with specific evidence sufficient for a human to judge whether the answer is supported. This also turns grounding quality into a trainable preference-learning problem.

## Method

- **Task format**: the model generates a single string using Inline Evidence syntax, `"% <Claim> % (Document title) % [Quote from document] %"`, factorized as `P(answer, evidence | question, c) = P(answer | question, c) P(evidence | answer, question, c)`.
- **Base model**: all systems are finetuned from Gopher-family LMs; the main system uses the `280B` model, with `1.4B` and `7B` variants used in scale ablations. Tokenization reuses Gopher's SentencePiece vocabulary of `32,000` subwords.
- **Retrieval and conditioning**: the input question is forwarded directly to Google Search. At inference, the system retrieves top-`K` documents, then performs `N > K` generations in round-robin order over documents, each conditioned on as much content as possible from a single document.
- **Context construction**: SFT conditions on up to `4096` subword tokens. The training pipeline randomly varies document count and per-document token budget, truncating around the returned search snippet while ensuring the gold quote remains visible.
- **Bootstrapping data**: because no suitable inline-evidence dataset existed, the authors created about `5,000` high-quality examples by few-shot prompting Gopher, then filtering samples with human ratings.
- **Supervised fine-tuning**: SFT trains only on samples rated both plausible and supported. The `280B` model is trained for `60` SGD steps with batch size `128`; Appendix D specifies Adafactor with learning rate `3e-6`, `128` TPU v3 cores, frozen embeddings, and aggressive early stopping. Held-out verbatim quoting reaches about `75%` even without constrained sampling.
- **Reward modeling**: the reward model is warm-started from a pretrained `7B` Gopher LM and trained on `33,242` pairwise preference comparisons. It predicts both pairwise preference and an auxiliary Supported&Plausible label, with final loss equal to the average of those two objectives.
- **RL fine-tuning**: the policy is initialized from SFT and optimized with synchronous `A2C` toward expected reward `E[r(x,y)]`, plus a KL penalty to the SFT policy: final loss is `alpha * KL + (1 - alpha) * A2C`. Appendix F reports Adafactor, learning rate `2e-6`, effective batch size `16`, gradient clipping `1.0`, freezing the first `60%` of weights (`48/80` transformer layers), and a two-layer value head of width `2048`.
- **Constrained quoting and abstention**: decoding can be constrained so quoted spans are verbatim substrings of the source document. At test time, the system can abstain by outputting `"I don't know"` whenever reward falls below a global threshold.

## Key Results

- On NaturalQuestionsFiltered, the best Google-backed system is `SFT - top@64` with `80.0 ± 6.1` Supported&Plausible (S&P), compared with `50.4 ± 7.7` for `SFT - first answer`, `60.9 ± 7.5` for `RL - first answer`, and `58.3 ± 7.6` for FiD-DPR.
- On ELI5Filtered, the best model is `RL - top@16` at `66.9 ± 7.0` S&P, versus `57.9 ± 7.4` for `SFT - top@64`, `46.3 ± 7.5` for `RL - first answer`, and `42.1 ± 7.4` for a prompted-Gopher ROUGE-evidence baseline.
- Abstention materially improves quality: by declining roughly the least certain `30%` of questions, performance exceeds `90%` S&P on NaturalQuestionsFiltered and `80%` on ELI5Filtered while still attempting about `70%` of questions.
- Preference comparisons show competitiveness with human references but not clear dominance: `SFT - top@64` reaches `49.5 ± 7.8` preference against NQ gold evidence, while `RL - top@64` reaches `42.9 ± 7.4` preference against top Reddit ELI5 answers with URLs.
- Truthfulness remains weak despite supportedness: on TruthfulQA, `SFT + top@16` scores `59.3` S&P but only `22.2` Truthful&Informative and `22.4` Truthful, showing that supported quotes do not guarantee truth.

## Limitations

- Supportedness is not the same as truthfulness: the system can quote misleading or cherry-picked evidence and still appear well supported.
- The method trusts webpages returned by Google Search and does not model source reliability, allowlisting, or source-level uncertainty.
- The system does not learn to search iteratively; it simply forwards the user question to a search engine and reasons over returned documents.
- Human evaluation is conducted on filtered subsets (`115` overlapping NQ questions and `121` overlapping ELI5 questions), so reported numbers are informative but not broad guarantees.
- The method depends on expensive infrastructure: a `280B` generator, TPU-scale SFT/RL, reward modeling, and human preference annotation.

## Concepts Extracted

- [[self-supported-question-answering]]
- [[inline-evidence]]
- [[question-answering]]
- [[retrieval-augmented-generation]]
- [[reward-model]]
- [[reranking]]
- [[constrained-decoding]]
- [[reinforcement-learning-from-human-feedback]]
- [[selective-prediction]]
- [[citation-accuracy]]
- [[factuality]]

## Entities Extracted

- [[jacob-menick]]
- [[maja-trebacz]]
- [[vladimir-mikulik]]
- [[john-aslanides]]
- [[francis-song]]
- [[martin-chadwick]]
- [[mia-glaese]]
- [[susannah-young]]
- [[lucy-campbell-gillingam]]
- [[geoffrey-irving]]
- [[nat-mcaleese]]
- [[gophercite]]
- [[gopher]]
- [[natural-questions]]
- [[eli5]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
