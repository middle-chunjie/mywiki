---
type: source
subtype: paper
title: "LeanDojo: Theorem Proving with Retrieval-Augmented Language Models"
slug: yang-nd-leandojo
date: 2026-04-20
language: en
tags: [theorem-proving, lean, retrieval, llm, benchmark]
processed: true

raw_file: raw/papers/yang-nd-leandojo/paper.pdf
raw_md: raw/papers/yang-nd-leandojo/paper.md
bibtex_file: raw/papers/yang-nd-leandojo/paper.bib
possibly_outdated: true

authors:
  - Kaiyu Yang
  - Aidan M. Swope
  - Alex Gu
  - Rahul Chalamala
  - Peiyang Song
  - Shixing Yu
  - Saad Godil
  - Ryan Prenger
  - Anima Anandkumar
year: 2023
venue: NeurIPS 2023 Datasets and Benchmarks Track
venue_type: conference
arxiv_id: 2306.15626
doi:
url: https://arxiv.org/abs/2306.15626
citation_key: yangndleandojo
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

LeanDojo introduces an open-source research stack for learning-based theorem proving in Lean and uses it to build ReProver, a retrieval-augmented tactic generator. The toolkit extracts proof trees, premise provenance, and interaction traces from Lean while exposing a reliable programmatic interface to the proof environment. On top of this infrastructure, the paper constructs a `98,734`-theorem benchmark with a harder `novel_premises` split and trains ByT5-based retriever/generator models that run on modest compute. ReProver improves theorem proving over both a no-retrieval baseline and zero-shot GPT-4, showing that explicit premise retrieval materially improves generalization when proofs depend on previously unseen library facts.

## Problem & Motivation

Existing LLM-based theorem provers were largely closed: they relied on private code, private or unreleased datasets, and compute budgets that were hard for academic groups to match. At the same time, theorem proving in Lean depends on finding useful premises from a very large math library, which is difficult to handle with a fixed context window and difficult to study without reliable tool support. The paper therefore targets three bottlenecks at once: reproducible tooling for extracting data and interacting with Lean, an open benchmark that measures harder generalization beyond random train/test overlap, and a prover architecture that augments tactic generation with explicit premise retrieval instead of pure memorization.

## Method

- **LeanDojo toolkit**: instruments Lean's elaborator and proof environment to export proof trees, proof states, tactic traces, and premise provenance. For interaction, it inserts control code as a Lean tactic, reducing proof-checking errors from `21.1%` in `lean-gym` to `1.4%`.
- **Benchmark construction**: extracts `98,734` theorems/proofs and `130,262` premises from mathlib, with splits `train = 94,734`, `val = 2,000`, `test = 2,000`. Besides a random split, it defines `novel_premises`, where test proofs must use at least one premise never used in training.
- **Retriever**: builds on [[dense-passage-retrieval]] for [[premise-selection]]. The query is the current proof state, and the candidate set is restricted to premises accessible from the current theorem. Program analysis shrinks the average candidate pool from about `128k` premises to `33,160`.
- **Hard negatives**: retriever training uses `n = 3` negatives per example, including `k = 1` in-file negative. This targets confusable premises from the same Lean source file rather than relying only on easy random negatives.
- **Tactic generator**: concatenates the state with retrieved premises and feeds the sequence to an encoder-decoder ByT5 model. The generator is trained with cross-entropy on human-written tactics, and the concatenated input is truncated to `2,300` tokens.
- **Optimization**: both retriever and generator use AdamW with `batch_size = 8`; learning rates warm up for `2,000` steps and then follow cosine decay, with `lr_retriever = 1e-4` and `lr_generator = 5e-4`. Training runs with bfloat16 mixed precision and DeepSpeed ZeRO Stage 2 on `1 x NVIDIA A100 80GB`.
- **Proof search**: at inference time the retriever returns `100` premises, the generator emits `64` tactic candidates via beam search, and a [[best-first-search]] procedure prioritizes states by cumulative tactic log-likelihood under a `10`-minute wall-clock limit.

## Key Results

- **Tool reliability**: LeanDojo lowers proof-checking errors on correct human proofs from `21.1%` (`lean-gym`) to `1.4%`.
- **Premise selection, random split**: BM25 reaches `R@1 = 6.7`, `R@10 = 17.2`, `MRR = 0.15`; ReProver's retriever reaches `13.5`, `38.4`, and `0.31`.
- **Premise selection, novel_premises split**: BM25 reaches `5.9 / 15.5 / 0.14`; ReProver reaches `9.1 / 27.6 / 0.24`.
- **Ablations**: retrieving from all premises hurts performance (`R@1 = 11.7`, `R@10 = 36.2` on random), and removing in-file negatives also hurts (`10.8 / 33.1 / 0.25`), validating both retrieval design choices.
- **LeanDojo Benchmark theorem proving**: Pass@1 on the random split is `51.2%` for ReProver, versus `47.6%` without retrieval, `29.0%` for GPT-4, and `23.8%` for `tidy`.
- **Hard split theorem proving**: on `novel_premises`, ReProver reaches `26.3%`, versus `23.2%` without retrieval, `7.4%` for GPT-4, and `5.3%` for `tidy`.
- **Out-of-distribution evaluation**: ReProver reaches `26.5%` Pass@1 on [[minif2f]] and `13.8%` on [[proofnet]]; the appendix reports `33` newly found Lean proofs on MiniF2F and `39` on ProofNet.

## Limitations

- The paper stays with a relatively small `299M`-parameter ByT5 backbone; the authors explicitly note that stronger open LLMs may substantially change the frontier.
- Byte-level modeling lengthens Lean sequences, and the generator can only fit roughly `10-15` retrieved premises into the `2,300`-token input budget.
- Performance drops sharply on the harder `novel_premises` split and on out-of-distribution datasets, so retrieval helps but does not solve generalization.
- The benchmark is centered on Lean/mathlib and depends on the extraction pipeline supporting sufficiently recent Lean repositories; it is not a universal theorem-proving benchmark across assistants.
- Direct comparison with prior strong LLM-based Lean provers remains incomplete because those systems relied on private data, private infrastructure, or RL pipelines the authors could not reproduce.

## Concepts Extracted

- [[retrieval-augmented-generation]]
- [[premise-selection]]
- [[interactive-theorem-proving]]
- [[automated-theorem-proving]]
- [[best-first-search]]
- [[dense-passage-retrieval]]
- [[hard-negative-mining]]
- [[formal-verification]]
- [[data-splitting]]

## Entities Extracted

- [[kaiyu-yang]]
- [[aidan-m-swope]]
- [[alex-gu]]
- [[rahul-chalamala]]
- [[peiyang-song]]
- [[shixing-yu]]
- [[saad-godil]]
- [[ryan-prenger]]
- [[anima-anandkumar]]
- [[leandojo]]
- [[reprover]]
- [[lean]]
- [[mathlib]]
- [[minif2f]]
- [[proofnet]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
