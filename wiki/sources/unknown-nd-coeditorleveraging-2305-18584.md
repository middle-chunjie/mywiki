---
type: source
subtype: paper
title: "Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"
slug: unknown-nd-coeditorleveraging-2305-18584
date: 2026-04-20
language: en
tags: [code-editing, code-completion, static-analysis, long-context, software-engineering]
processed: true
raw_file: raw/papers/unknown-nd-coeditorleveraging-2305-18584/paper.pdf
raw_md: raw/papers/unknown-nd-coeditorleveraging-2305-18584/paper.md
bibtex_file: raw/papers/unknown-nd-coeditorleveraging-2305-18584/paper.bib
possibly_outdated: true
authors:
  - Jiayi Wei
  - Greg Durrett
  - Isil Dillig
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2305.18584
doi:
url: https://arxiv.org/abs/2305.18584
citation_key: unknownndcoeditorleveraging
paper_type: method
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

Coeditor studies multi-round code auto-editing: given an original codebase and a sequence of recent edits, the model predicts further edits inside a target region instead of only continuing code left-to-right. The system fine-tunes CodeT5-base with a line-diff encoding, static-analysis-derived signature context, and sparse attention so the query region can interact efficiently with many prior edits and retrieved references. To train it, the authors build PyCommits from 1,650 permissively licensed Python repositories and synthesize repeated-editing examples from commit histories. On single-line completion instances extracted from edits, Coeditor reaches 60.4% exact match with only 220M parameters, substantially ahead of larger infilling baselines; in multi-round evaluation it automates 46.7% of changed lines and saves 28.6% of keystrokes.

## Problem & Motivation

Most code generation work focuses on producing new code, while real software maintenance requires revising existing code after nearby or cross-file changes. Standard completion models do not explicitly observe recent edits, so they cannot reliably infer where similar follow-up changes should happen or how user intent evolves across rounds. This paper formulates auto-editing as predicting a target edit `Delta u` conditioned on prior contextual edits `Delta_1 ... Delta_k` and the original codebase `U`, with the target region allowed to overlap prior edits so users and model can iteratively collaborate on the same function or file region.

## Method

- **Task formulation**: learn `P(Delta u | Delta_k ... Delta_1, U)` for a user-specified edit region, where repeated edits to the same region are allowed across rounds.
- **Line-diff encoding**: input lines are annotated with status tokens `empty`, `<add>`, or `<del>`, and editable lines receive placeholders such as `<1> ... <n>`; outputs are generated as `EncOutput(Delta u) = <1>I_a D_a <2>I_{a+1} D_{a+1} ...`, where insertions are prefixed by `<add>`.
- **Editing constraint**: the decoder is forbidden from emitting `<del>` for a line already marked `<add>`, preventing the model from immediately revising a just-added line and reducing degenerate edit loops.
- **Repository context construction**: lightweight static analysis with [[jedi]] retrieves function signatures or first assignment statements for symbols used in the target region; these retrieved snippets are concatenated into a signature document.
- **Backbone and context scaling**: Coeditor fine-tunes [[codet5]]-base with `220M` parameters. Reference blocks are capped at `512` tokens, the query block at `1024`, and the model uses up to `16.4K` reference tokens at test time while keeping query-to-reference attention global and making reference ordering irrelevant through infinite relative distances.
- **Sparse attention pattern**: each contextual change or signature chunk is placed in a separate reference block; self-attention is dense within a block, but attention across reference blocks is removed to avoid full quadratic cost.
- **Dataset construction**: [[pycommits]] is built from `1,650` open-source Python projects on [[github]], with splits of `1,550/50/50` projects for train/valid/test and at most `1000` commits per project. Training labels are unit modifications, while unit additions/deletions remain visible as context.
- **Synthetic multi-round supervision**: for edits with at least two changed lines, a random subset of changed lines is inlined into the input so the remaining lines become the prediction target, creating repeated-editing examples from single commits.
- **Training setup**: AdamW with learning rate `2e-5`, weight decay `0.01`, batch size `1`, `1.34M` steps, and `1.75` epochs. The maximum reference context grows from `2048` to `4096` and then `8192` tokens during training; training takes about `5` days on a single NVIDIA Quadro RTX 8000 `48GB` GPU.

## Key Results

- **PyCommits-OneLine (5000 instances)**: Coeditor reaches `60.4%` overall EM, with `47.1%` Add EM and `64.9%` Replace EM, versus the best baseline [[incoder]]-6B at `31.3%` overall EM.
- **Model efficiency**: the paper emphasizes that Coeditor uses `220M` parameters, about `30x` smaller than the strongest completion baseline while still nearly doubling its exact-match accuracy.
- **Multi-round editing (5000 test problems)**: average gains rise from `28.5/23.1/19.2` (Lines/Levenshtein/Keystrokes) in single-round mode to `46.7/25.9/28.6` in multi-round mode, with an average of `2.43` rounds.
- **Ablations**: removing explicit diffs drops validation EM from `42.1%` to `26.1%`; removing signature retrieval drops it to `33.3%`; shrinking reference context from `16K` to `2048` tokens yields `39.8%`.
- **Context coverage**: with `16.4K` reference tokens, the model covers `88.8%` of test instances without truncating contextual changes; only `7.8%` of query sequences exceed the `1024`-token query limit.
- **Dataset scale**: PyCommits contains `217K/5006/5854` used commits and `958K/20.1K/22.5K` modified functions across train/valid/test splits, giving the method substantially more realistic edit supervision than prior contextual editing work.

## Limitations

- The system assumes the user already knows which code region should be edited; it does not detect edit locations across the whole repository.
- The decoding scheme cannot revise a line that has already been modified in the input, so partial human edits within a single line are not naturally completed by the model.
- Evaluation simulates user interaction by accepting only exact-matching predicted lines and otherwise applying the next ground-truth change, which is a useful proxy but not a full human study.
- The dataset covers only permissively licensed Python repositories from [[github]], so transfer to other languages, proprietary codebases, or IDE workflows is not established here.
- As a `2023` preprint built on CodeT5-era models, the paper predates later repository-scale code agents and stronger long-context code LMs.

## Concepts Extracted

- [[code-editing]]
- [[multi-round-editing]]
- [[line-diff-encoding]]
- [[infilling]]
- [[static-analysis]]
- [[sparse-attention]]
- [[repository-level-context]]
- [[code-completion]]
- [[relative-positional-encoding]]
- [[long-context-modeling]]

## Entities Extracted

- [[jiayi-wei]]
- [[greg-durrett]]
- [[isil-dillig]]
- [[university-of-texas-at-austin]]
- [[codet5]]
- [[pycommits]]
- [[jedi]]
- [[github]]
- [[vscode]]
- [[incoder]]
- [[santacoder]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
