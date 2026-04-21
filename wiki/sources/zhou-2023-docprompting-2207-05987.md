---
type: source
subtype: paper
title: "DocPrompting: Generating Code by Retrieving the Docs"
slug: zhou-2023-docprompting-2207-05987
date: 2026-04-20
language: en
tags: [code-generation, retrieval-augmented-generation, documentation, nlp, benchmark]
processed: true
raw_file: raw/papers/zhou-2023-docprompting-2207-05987/paper.pdf
raw_md: raw/papers/zhou-2023-docprompting-2207-05987/paper.md
bibtex_file: raw/papers/zhou-2023-docprompting-2207-05987/paper.bib
possibly_outdated: true
authors:
  - Shuyan Zhou
  - Uri Alon
  - Frank F. Xu
  - Zhiruo Wang
  - Zhengbao Jiang
  - Graham Neubig
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: "2207.05987"
doi: ""
url: http://arxiv.org/abs/2207.05987
citation_key: zhou2023docprompting
paper_type: method
read_status: unread
domain: nlp
---

## Summary

> ⚠️ **Possibly outdated** — published 2023 in the volatile NLP / code generation domain; API landscapes and stronger LLMs may have shifted the landscape since.

DocPrompting introduces a retrieve-then-generate framework for natural language to code generation that explicitly leverages code documentation. Given a natural language intent, a retriever selects the top-k most relevant documentation entries from a large pool; a generator then conditions on both the intent and retrieved docs to produce code. The method is model-agnostic and supports sparse (BM25) and dense (SimCSE/CodeT5-based) retrieval, and is demonstrated on Bash (tldr benchmark) and Python (CoNaLa benchmark). DocPrompting consistently improves strong baselines — e.g., CodeT5 gains +2.85% pass@1 (52% relative) on CoNaLa and +6.9% exact match on tldr — and critically enables generalization to unseen functions and libraries at test time.

## Problem & Motivation

Existing NL-to-code models assume all required libraries and functions were seen during training. New libraries are introduced continuously, making it impossible for a fixed model to generalize to unseen APIs. Human programmers solve this by reading documentation; DocPrompting brings this same strategy to models. The core insight is that documentation serves as an external, updatable knowledge store that can grow without re-training any model component, enabling generalization to previously unseen functions.

## Method

**Overall framework:** `P(c | D, n) ≈ P(c | D_hat_n, n) · P(D_hat_n | D, n)`, where `n` is the NL intent, `D` is the full documentation pool, and `D_hat_n` is the top-k retrieved subset.

**Two components:**

- **Retriever R:** Scores `s(d_i, n)` between every document `d_i ∈ D` and intent `n`; selects top-`k` docs.
  - *Sparse:* BM25 via Elasticsearch; word-frequency-based.
  - *Dense:* Contrastive training with in-batch negatives. Loss: `L^r = -log[ exp(sim(h_n, h_d+)) / (exp(sim(h_n, h_d+)) + Σ_{d- ∈ B\D*} exp(sim(h_n, h_d-))) ]`. Initializes from SimCSE (RoBERTa) or CodeT5-base encoder. Also mixes weak supervision: same sentence with different dropout masks forms a positive pair.
  - Dense retriever training: `10 epochs`, `batch_size=512`, `lr=1e-5`; takes ~15 hours on a single A6000 GPU.

- **Generator G:** Generates `c` conditioned on intent and retrieved docs.
  - *Joint encoding* (GPT-Neo, Codex): retrieved docs and NL intent concatenated as a single prompt.
  - *Parallel encoding / FiD:* For T5/CodeT5 (max 512-token input), each `(n, d_i)` pair is encoded independently; decoder attends over all encoded pairs. Doc length truncated to `200` tokens.
  - Fine-tuning: `20 epochs`, `lr=4e-5` for single-source generators; `10000 steps`, `lr=5e-5`, `warmup=2000` steps, linear decay, `batch_size=8` for FiD.
  - Codex: 3-shot in-context learning with `code-davinci-001`; nucleus sampling `p=0.95`.

**Documentation pools:**
- *tldr (Bash):* `400k` paragraphs from `1,879` Bash manuals; each paragraph describes one flag/concept.
- *CoNaLa (Python):* `35,763` documents from all Python libraries on DevDocs; one document = one API signature + paragraph.

**Oracle supervision:** Oracle docs `D*_n` annotated via string/function-name matching; used to supervise the dense retriever.

## Key Results

- **tldr (Bash, BM25 retriever, top-10 docs):**
  - CodeT5+DocPrompting: CMD Acc 30.72% vs 14.60% baseline (+110%); EM 9.15% vs 2.18% baseline
  - T5+DocPrompting: EM 9.16% vs 0.76%; charBLEU 31.97 vs 25.48
  - GPT-Neo-1.3B+DocPrompting: EM 9.05% vs 3.12%; charBLEU 30.57 vs 24.70
  - Codex 3-shot+DocPrompting: charBLEU 23.72 vs 16.94 (+6.78)

- **tldr with oracle command name:**
  - T5+DocPrompting: EM 22.55% vs 12.96%; Codex+DocPrompting: EM 32.43% vs 22.44%

- **CoNaLa (Python, CodeT5 dense retriever, top-10 docs):**
  - CodeT5+DocPrompting: BLEU 36.22 vs 34.57 (+1.65); unseen function recall 18.30 vs 9.03 (+2×)
  - Codex+DocPrompting: BLEU 43.47 vs 43.16 (minor; possible data leakage)
  - CodeT5+DocPrompting oracle docs: BLEU 49.04, unseen recall 63.91 (upper bound)

- **Execution-based (pass@k on 100 CoNaLa examples):**
  - pass@1: +2.85% (52% relative); pass@5: +4.45%; pass@10 (reported in abstract): +4.39%

- **DocPrompting vs ExPrompting (retrieving NL-code examples):** DocPrompting far superior — e.g., GPT-Neo-125M: CMD Acc 25.32% vs 6.68%

- **Retrieval recall@10 (dev):** tldr: BM25 59.86%, RoBERTa finetuned 60.33%; CoNaLa: CodeT5 finetuned 55.81%, BM25 only 9.73%

- **n-gram bridging effect:** (NL+retrieved docs)↔code unigram overlap increases from 12% to 24% in tldr, and 30% to 91% in CoNaLa

## Limitations

- Dense retriever requires oracle documentation annotation per training example (string/function-name matching heuristic — brittle for complex functions).
- Generator can misuse retrieved docs: e.g., inheriting arguments from `df.read_csv` docs when calling `df.to_csv`.
- Documentation pools are curated (Bash manuals from manned.org; Python from DevDocs) — not comprehensive, and quality depends on documentation quality.
- Weak supervision via dropout positives may not generalize to all programming languages equally.
- Codex leakage concern: code-davinci-002 may have memorized test sets (CoNaLa from StackOverflow in CommonCrawl); paper uses code-davinci-001 to mitigate.
- Joint retriever+generator training not explored; cascading errors from retrieval mistakes are not addressed.
- Increasing number of retrieved docs `k` beyond 5–10 hurts performance because the generator cannot effectively filter irrelevant docs.

## Concepts Extracted

- [[docprompting]]
- [[natural-language-to-code]]
- [[retrieval-augmented-generation]]
- [[documentation-retrieval]]
- [[retrieval-augmented-code-generation]]
- [[dense-retrieval]]
- [[bm25]]
- [[fusion-in-decoder]]
- [[contrastive-learning]]
- [[in-batch-negatives]]
- [[pass-at-k]]
- [[execution-based-evaluation]]
- [[code-generation]]

## Entities Extracted

- [[shuyan-zhou-cmu]]
- [[uri-alon]]
- [[frank-xu]]
- [[zhiruo-wang]]
- [[zhengbao-jiang]]
- [[graham-neubig]]
- [[carnegie-mellon-university]]
- [[inspired-cognition]]
- [[codet5]]
- [[gpt-neo-1-3b]]
- [[codex]]
- [[roberta]]
- [[conala]]
- [[tldr-benchmark]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
