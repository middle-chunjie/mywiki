---
type: source
subtype: paper
title: Repository-Level Prompt Generation for Large Language Models of Code
slug: shrivastava-2023-repositorylevel-2206-12839
date: 2026-04-20
language: en
tags: [llm, code-completion, prompt-generation, repository-context, code-intelligence]
processed: true

raw_file: raw/papers/shrivastava-2023-repositorylevel-2206-12839/paper.pdf
raw_md: raw/papers/shrivastava-2023-repositorylevel-2206-12839/paper.md
bibtex_file: raw/papers/shrivastava-2023-repositorylevel-2206-12839/paper.bib
possibly_outdated: true

authors:
  - Disha Shrivastava
  - Hugo Larochelle
  - Daniel Tarlow
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2206.12839
doi:
url: http://arxiv.org/abs/2206.12839
citation_key: shrivastava2023repositorylevel
paper_type: method

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

The paper proposes Repo-Level Prompt Generator (RLPG), a black-box prompting framework for code LLMs that selects repository-aware prompt context instead of relying only on the default left context in the current file. RLPG defines `63` discrete prompt proposals from `10` prompt sources and `7` context types, then learns a Prompt Proposal Classifier (PPC) to pick the best proposal per completion instance. On single-line Java autocompletion from `47` Google Code repositories, the oracle over proposals reaches `79.63%` success rate versus `58.73%` for Codex, while learned variants RLPG-H and RLPG-R reach `68.51%` and `67.80%`. The main claim is that repository structure and cross-file evidence can materially improve black-box code completion without fine-tuning the underlying model.

## Problem & Motivation

The paper studies single-line code autocompletion in an IDE-like maintenance setting where useful evidence may lie outside the prefix before the cursor. Default Codex prompting only exposes the current file's preceding context, but real repositories contain imports, parent classes, sibling files, similarly named files, and later lines in the current file that may strongly constrain the missing suffix. The authors therefore ask whether prompt construction can explicitly inject repository-level structure into a black-box code model, and whether a learned selector can choose the right prompt pattern on a per-example basis.

## Method

- **Task setup**: for each non-blank, non-comment line, place the hole at the middle character and predict the suffix from the cursor to end-of-line under the [[line-level-maintenance]] setting.
- **Prompt proposal inventory**: a [[prompt-proposal]] is defined by prompt source plus prompt context type. The paper instantiates `10` sources (current file, parent class, import, sibling, similar-name file, child class, and four import-derived variants) and `7` context types (post lines, identifiers, type identifiers, field declarations, string literals, method names, method names plus bodies), giving `63` proposals.
- **Training labels for PPC**: for hole `h`, build a multi-hot target `Y^h = [y_p^h]` where `y_p^h = 1` if proposal `p` is applicable and Codex exactly matches the target hole under that prompt. Applicability is tracked by mask `T^h`, and the loss is `L = (1/N) sum_h (1/M^h) sum_p BCE(yhat_p^h, y_p^h) * T_p^h`.
- **RLPG-H**: encode a hole window with frozen CodeBERT `[CLS]` representation of size `768`, then score proposals with an MLP `768 -> 512 -> 63`. Training uses Adam with learning rate `3e-4` and batch size `64`.
- **RLPG-R**: score proposals using attention between hole-window representation `Q^h` and proposal-context representations `K_p^h, V_p^h`. The model uses `d_k = d_q = d_v = 32`, `tau = 4` heads, `d_model = 768`, dropout `0.25`, and a transformer-like feed-forward block with residual connection plus layer normalization.
- **Prompt composition**: the selected proposal context is prepended to the default Codex context. Standard allocation is `50%` proposal context and `50%` default context, with additional `25/75` and `75/25` variants for post-lines prompts. Contexts are truncated if they exceed budget.
- **Experimental setup**: dataset contains `47` Java repositories from [[google-code-archive]], split into `19/14/14` train/val/test repositories, `2655/1060/1308` files, and `92721/48548/48288` holes. Main completion engine is [[code-davinci-001]] with `temperature = 0.0`, newline stop criterion, completion length `24`, and maximum prompt length `4072`.
- **Baselines**: compare against raw [[codex]], oracle proposal selection, fixed prompt proposal, [[bm25]] over proposal contexts, file-level BM25, identifier-usage baselines, and random context selection.

## Key Results

- Oracle prompt proposals substantially outperform Codex on all splits: train `59.78% -> 80.29%` (`+34.31%` relative), val `62.10% -> 79.05%` (`+27.28%`), test `58.73% -> 79.63%` (`+35.58%`).
- On test success rate, RLPG-H reaches `68.51%` (`+16.65%` relative over Codex), RLPG-R reaches `67.80%` (`+15.44%`), and RLPG-BM25 reaches `66.41%` (`+13.07%`).
- The strongest fixed non-learned strategy is post-lines from the current file, reaching `65.78%` success rate (`+12.00%` relative).
- File-level BM25 reaches `63.14%`, weaker than proposal-aware retrieval; identifier-usage baselines reach `64.93%` and `64.91%`, still below RLPG-H/R.
- Under normalized character edit distance (lower is better), Codex scores `30.73`, RLPG-H `22.55` (`+26.62%` relative improvement), and RLPG-R `23.00` (`+25.14%`).
- Transfer to [[code-cushman-001]] still helps: baseline `58.40%`, RLPG-H `64.74%` (`+10.87%`), RLPG-R `64.79%` (`+10.95%`).
- RLPG-R is computationally lightweight relative to model fine-tuning: `3.6M` parameters and about `9.19` minutes per epoch on one Tesla V100, though label collection still required about `230k` Codex API queries for train plus validation.

## Limitations

- The paper only studies Java single-line autocompletion, so generalization to other languages and longer code edits is unproven.
- Prompt proposals are manually designed and discrete; performance depends on proposal coverage and may miss useful repository relations not encoded in the inventory.
- PPC training is label-expensive because ground truth requires querying Codex for each applicable proposal, totaling about `230k` API calls for train plus validation data.
- RLPG relies on CodeBERT encoders with shorter input limits than Codex, so proposal contexts must be truncated before scoring, potentially discarding the most useful evidence.
- Potential train-test contamination with Codex cannot be fully ruled out because the model's original training data are unknown, even though the dataset avoids GitHub repositories when possible.

## Concepts Extracted

- [[large-language-model]]
- [[code-language-model]]
- [[code-completion]]
- [[prompt-engineering]]
- [[black-box-language-model]]
- [[cross-file-context]]
- [[repository-level-context]]
- [[prompt-proposal]]
- [[prompt-proposal-classifier]]
- [[prompt-composition]]
- [[bm25]]
- [[exact-string-match]]
- [[line-level-maintenance]]

## Entities Extracted

- [[disha-shrivastava]]
- [[hugo-larochelle]]
- [[daniel-tarlow]]
- [[codex]]
- [[code-davinci-001]]
- [[code-cushman-001]]
- [[codebert]]
- [[graphcodebert]]
- [[google-code-archive]]
- [[openai]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
