---
type: source
subtype: paper
title: "CROSSCODEEVAL: A Diverse and Multilingual Benchmark for Cross-File Code Completion"
slug: ding-nd-crosscodeeval
date: 2026-04-20
language: en
tags: [benchmark, code-completion, code-llm, retrieval, multilingual]
processed: true
raw_file: raw/papers/ding-nd-crosscodeeval/paper.pdf
raw_md: raw/papers/ding-nd-crosscodeeval/paper.md
bibtex_file: raw/papers/ding-nd-crosscodeeval/paper.bib
possibly_outdated: true
authors:
  - Yangruibo Ding
  - Zijian Wang
  - Wasi Uddin Ahmad
  - Hantian Ding
  - Ming Tan
  - Nihal Jain
  - Murali Krishna Ramanathan
  - Ramesh Nallapati
  - Parminder Bhatia
  - Dan Roth
  - Bing Xiang
year: 2023
venue: NeurIPS 2023
venue_type: conference
arxiv_id: 2310.11248
doi: 10.48550/ARXIV.2310.11248
url: https://arxiv.org/abs/2310.11248
citation_key: dingndcrosscodeeval
paper_type: benchmark
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

CROSSCODEEVAL is a repository-level code completion benchmark designed to force models to use cross-file context rather than only the current file. The dataset contains about `10k` examples from `1,002` permissively licensed GitHub repositories across Python, Java, TypeScript, and C#, and constructs each example by statically identifying API uses that become unresolved when project-local imports are replaced with dummy definitions. The benchmark evaluates both code match and identifier match, and also supports retrieval-based studies by exposing realistic repository context. Results show that strong code LMs remain weak with only in-file context, but improve substantially when given retrieved cross-file evidence, making the benchmark useful for both code completion and code retrieval research.

## Problem & Motivation

Existing code completion benchmarks such as HumanEval and MBPP mostly evaluate single-file prediction, which understates the difficulty of real software repositories where APIs, classes, and helper functions are often defined elsewhere. The paper targets this mismatch by building a benchmark where successful completion must depend on project-local cross-file context, while also trying to minimize data leakage by sampling recent, permissively licensed repositories and filtering against The Stack. The authors also want a benchmark that can diagnose not only generation quality but retrieval quality, because repository-level completion depends on both.

## Method

- **Repository collection**: mine non-fork GitHub repositories created between `2023-03-05` and `2023-06-15`, downloaded on `2023-09-01`, restricted to Python, Java, TypeScript, and C#, with zip size `< 1 MB`, stars `>= 3`, and `10-50` source files.
- **Leakage control**: remove repositories whose code files exactly match files in The Stack, yielding `471` Python, `239` Java, `193` TypeScript, and `99` C# repositories.
- **Cross-file example mining**: replace intra-project imports with empty classes or dummy definitions, then run language-specific static analysis or compilers to surface unresolved members that require cross-file context.
- **Language tooling**: use `Pylint` for Python undefined-member errors, `javac` for Java, `tsc` for TypeScript, and `csc` from Mono for C#; use `tree-sitter` to select a token before the cross-file entity as cursor and to recover statement boundaries.
- **Prompt/reference construction**: split at a randomly selected same-line token before the cross-file entity; if the same undefined name appears multiple times, keep only the first occurrence to reduce in-file leakage.
- **Filtering**: require at least `N = 10/20/30/5` non-import prompt lines for Python/Java/TypeScript/C#, keep references with `3-30` tokens, discard references duplicated elsewhere in the repository, and remove duplicate references.
- **Model-based quality control**: run `starcoderbase-1B` on prompt-only inputs and drop exact matches so examples are less likely to be solvable from in-file clues alone.
- **Evaluation setup**: zero-shot generation with max context lengths `2048` for CodeGen, `4096` for GPT-3.5-Turbo, and `8192` for StarCoder; maximum generation length `50`.
- **Retrieval baseline**: retrieve non-overlapping code chunks of `M = 10` lines using the last `N = 10` in-file lines as query, rank by `BM25`, prepend top-`5` snippets, and cap retrieved context at `512` BPE tokens.
- **Oracle-style upper bound**: a "retrieval w/ ref." condition forms the query from the last `10` lines of `prompt + reference`, estimating the headroom of better retrieval even though it is not usable in deployment.

## Key Results

- Dataset scale: `9,928` examples from `1,002` repositories and `3,534` files across `4` programming languages.
- Per-language examples: Python `2,665`, Java `2,139`, TypeScript `3,356`, C# `1,768`; average prompt lengths range from `71.1` to `116.5` lines.
- With in-file context only, `StarCoder-15.5B` reaches just `8.82%` code EM on Python, `9.96%` on Java, `6.35%` on TypeScript, and `4.47%` on C#.
- Adding retrieved cross-file context raises `StarCoder-15.5B` code EM to `15.72%`, `17.48%`, `8.31%`, and `13.57%` respectively; with retrieval w/ ref., it further rises to `21.01%`, `19.92%`, `11.02%`, and `20.08%`.
- `GPT-3.5-Turbo` also benefits: Java code EM improves from `12.30%` to `19.12%` with retrieval and `22.72%` with retrieval w/ ref.
- Best retrieval methods remain far from solved: for `StarCoder-15.5B`, OpenAI ada retrieval w/ ref. yields identifier EM/F1 of `29.64/58.64` on Python and `27.96/53.53` on C#, still leaving substantial headroom.
- Human validation is strong: annotators judged that cross-file information is necessary in `99%` of sampled Python examples and `100%` of sampled Java examples; only `2%` of Python and `2%` of Java samples were judged directly predictable from current-file context alone.

## Limitations

The benchmark covers only four languages and restricts repositories to relatively small projects (`< 1 MB`, `10-50` files), so it does not fully represent large industrial repositories. The construction pipeline depends on static analysis and compiler diagnostics, which can miss or mislocalize cross-file dependencies. Retrieval is evaluated with fixed `10`-line chunks and mostly zero-shot prompting, so weak results partly reflect prompt formatting and retrieval granularity rather than model capability alone. The paper also cannot fully rule out memorization, and the oracle-like retrieval w/ ref. setting is useful analytically but not deployable.

## Concepts Extracted

- [[cross-file-context]]
- [[static-analysis]]
- [[repository-level-code-completion]]
- [[code-completion-benchmark]]
- [[context-retrieval]]
- [[fill-in-the-middle]]
- [[retrieval-augmented-generation]]
- [[code-language-model]]

## Entities Extracted

- [[yangruibo-ding]]
- [[zijian-wang]]
- [[wasi-uddin-ahmad]]
- [[hantian-ding]]
- [[ming-tan]]
- [[nihal-jain]]
- [[murali-krishna-ramanathan]]
- [[ramesh-nallapati]]
- [[parminder-bhatia]]
- [[dan-roth]]
- [[bing-xiang]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
