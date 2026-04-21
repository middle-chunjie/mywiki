---
type: source
subtype: paper
title: "KITAB: Evaluating LLMs on Constraint Satisfaction for Information Retrieval"
slug: unknown-nd-kitabevaluating-2310-15511
date: 2026-04-20
language: en
tags: [ir, llm, benchmark, constraint-satisfaction, hallucination]
processed: true
raw_file: raw/papers/unknown-nd-kitabevaluating-2310-15511/paper.pdf
raw_md: raw/papers/unknown-nd-kitabevaluating-2310-15511/paper.md
bibtex_file: raw/papers/unknown-nd-kitabevaluating-2310-15511/paper.bib
possibly_outdated: true
authors:
  - Marah I. Abdin
  - Suriya Gunasekar
  - Varun Chandrasekaran
  - Jerry Li
  - Mert Yuksekgonul
  - Rahee Ghosh Peshawaria
  - Ranjita Naik
  - Besmira Nushi
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2310.15511
doi:
url: https://arxiv.org/abs/2310.15511
citation_key: unknownndkitabevaluating
paper_type: benchmark
read_status: unread
domain: ir
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

KITAB introduces a benchmark and data-construction pipeline for evaluating how well large language models answer constrained information-retrieval queries such as listing books by an author under lexical, temporal, or entity-based conditions. The dataset covers 611 authors and 12,989 queries over book metadata, and the paper evaluates GPT-4 and GPT-3.5 under no-context, self-context, with-context, all-books, and single-item settings. Even when given complete context, the models frequently fail to satisfy constraints or return complete answer sets, while self-generated context often increases irrelevant or fabricated titles. The paper's main contribution is therefore not a new model, but a controlled benchmark showing that conversational LLMs remain unreliable at filtering information under explicit constraints.

## Problem & Motivation

Existing factuality benchmarks for LLMs are often saturated, contaminated, or focused on single short answers rather than longer outputs that must satisfy explicit constraints. The paper studies constraint satisfaction as an information-retrieval problem: users often ask for lists that must jointly obey author, year, lexical, or named-entity conditions, and failure to respect those conditions leads to omission, irrelevance, or hallucination. The authors therefore build a benchmark that separates failures from missing knowledge, failures from poor filtering, and failures from the retrieval setup itself.

## Method

- **Dataset construction**: sample `20,000` Wikidata authors, filter to authors born after `1850` with `10-300` works, reduce to `1,505`, then cross-reference Open Library and retain authors with at least `5` cleaned works; after additional controls, KITAB contains `611` authors.
- **Query generation**: build `8,239` single-book-constraint queries and `4,750` two-book-constraint queries over lexical, temporal, and entity predicates; `7.99%` of single-constraint queries are unsatisfiable, while only `0.76%` of double-constraint queries are jointly unsatisfiable.
- **Constraint families**: constraints include title `starts-with`, `ends-with`, `word-count`, publication-year ranges, and presence or absence of human or city names; the paper defines constrainedness as `kappa = 1 - S / N` where `S` is the number of satisfying books and `N` is the author's total books.
- **Experimental conditions**: evaluate `all-books`, `no-context`, `with-context`, `self-context`, and `single-item` prompting settings to decouple parametric recall, retrieval-augmented filtering, and single-item verification from list-level reasoning.
- **Models and prompting**: GPT-4 and GPT-3.5 are run at temperature `0` with prompt-specific token budgets such as `400` for `no-context`, `1000` for `with-context`, and `3000` for `self-context`; most prompts request a short chain-of-thought-style reason before each returned title.
- **Matching and clustering**: model outputs are normalized into a set `K = {k_1, ..., k_n}` and matched to ground-truth titles by subset checks or Levenshtein similarity at `80%` threshold, then clustered so near-duplicate mentions of the same book do not over-penalize the model.
- **Metrics**: each answer is scored by irrelevant, satisfying, and unsatisfying cluster fractions `p_irr`, `p_sat`, and `p_unsat` with `p_irr + p_sat + p_unsat = 1`; completeness is `p_comp`, and `all-correct` requires both `p_sat = 1` and `p_comp = 1`.
- **Data verification**: books are cleaned by deduplication, Wikidata/Open Library authorship checks, language filtering, and manual audits showing that missing titles may affect fewer than `5%` of sampled GPT-4 queries and fewer than `6%` of sampled GPT-3.5 queries.

## Key Results

- KITAB contains `611` authors, `8,239` one-constraint queries, and `4,750` two-constraint queries, for `12,989` total benchmark queries.
- On one-constraint queries, GPT-4 scores `0.26 | 0.33 | 0.00` irrelevance, `0.51 | 0.49 | 0.78` satisfaction, `0.24 | 0.26 | 0.70` completeness, and `0.08 | 0.08 | 0.31` all-correct for `no-context | self-context | with-context`.
- On the same setting, GPT-3.5 scores `0.20 | 0.44 | 0.00` irrelevance, `0.44 | 0.26 | 0.68` satisfaction, `0.16 | 0.16 | 0.47` completeness, and `0.07 | 0.02 | 0.15` all-correct.
- For two-constraint queries, GPT-4 all-correct rises only from `0.06` in `no-context` to `0.19` in `with-context`, while GPT-3.5 moves from `0.00` to `0.07`.
- Single-item constraint verification is much easier than list filtering: summary accuracy is `0.80` for GPT-4 and `0.69` for GPT-3.5.
- Context sharply reduces irrelevant titles, but does not solve filtering failures; even with complete context, GPT-4's overall all-correct on one-constraint queries is only `0.31`, and the paper reports `<35%` correctness across popularity bins.

## Limitations

- The benchmark is restricted to English-language book retrieval, so its conclusions may not transfer directly to domains with different structure, metadata quality, or multilingual ambiguity.
- Evaluation depends on Open Library and Wikidata coverage; the paper's manual audit shows some model outputs are real books missing from KITAB, which can slightly overestimate irrelevance.
- Only GPT-4 and GPT-3.5 are evaluated in the main study, so the paper does not compare against classical IR systems, symbolic filtering pipelines, or stronger open-weight LLMs.
- The `with-context` setting assumes an ideal retrieval stage that provides complete author-specific book lists, which is often easier than real deployment.

## Concepts Extracted

- [[constraint-satisfaction]]
- [[information-retrieval]]
- [[large-language-model]]
- [[benchmark-dataset]]
- [[benchmark-evaluation]]
- [[retrieval-augmented-generation]]
- [[chain-of-thought-prompting]]
- [[parametric-knowledge]]
- [[constrainedness]]
- [[information-popularity]]
- [[hallucination]]

## Entities Extracted

- [[marah-i-abdin]]
- [[suriya-gunasekar]]
- [[varun-chandrasekaran]]
- [[jerry-li]]
- [[mert-yuksekgonul]]
- [[rahee-ghosh-peshawaria]]
- [[ranjita-naik]]
- [[besmira-nushi]]
- [[microsoft-research]]
- [[university-of-illinois-urbana-champaign]]
- [[stanford-university]]
- [[gpt-4]]
- [[gpt-3-5]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
