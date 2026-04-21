---
type: source
subtype: paper
title: "Multi-View Graph Representation for Programming Language Processing: An Investigation into Algorithm Detection"
slug: long-2022-multiview
date: 2026-04-20
language: en
tags: [program-representation, graph-learning, algorithm-detection, code-semantics, programming-language-processing]
processed: true

raw_file: raw/papers/long-2022-multiview/paper.pdf
raw_md: raw/papers/long-2022-multiview/paper.md
bibtex_file: raw/papers/long-2022-multiview/paper.bib
possibly_outdated: true

authors:
  - Ting Long
  - Yutong Xie
  - Xianyu Chen
  - Weinan Zhang
  - Qinxiang Cao
  - Yong Yu
year: 2022
venue: AAAI 2022
venue_type: conference
arxiv_id:
doi: 10.1609/aaai.v36i5.20522
url: https://ojs.aaai.org/index.php/AAAI/article/view/20522
citation_key: long2022multiview
paper_type: method

read_status: unread

domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

This paper proposes MVG, a program representation method that models source code through four complementary graph views: a data-flow graph, a control-flow graph, a read-write graph, and a combined graph. Instead of centering representation on AST-derived syntax, the method emphasizes code semantics by separating operand dependencies, execution order, and operand-operation interactions, then encoding each view with a gated graph neural network before concatenation. The paper evaluates MVG on algorithm detection, using the public [[poj-104]] benchmark and a newly introduced multi-label dataset [[alg-109]]. MVG reaches the best reported results on both datasets, and ablations show that each view matters, with the data-flow view contributing the largest individual gain.

## Problem & Motivation

Prior program-representation methods either rely on execution data, flatten code into token sequences, or build syntax-heavy structures such as ASTs. The authors argue that these approaches underweight program semantics and often crowd heterogeneous signals into a single representation, making it harder to disentangle data dependencies, execution order, and operand-operation interactions. They therefore target a more semantic and more decomposed representation that can better support [[algorithm-detection]], especially on realistic code where multiple algorithms and data structures may co-occur and where superficially similar implementations can correspond to different underlying algorithms.

## Method

- **Four-view program graph**: MVG constructs `DFG`, `CFG`, `RWG`, and `CG` for each program, with `G_i = {V_i, E_i}` and `E_CG = E_DFG ∪ E_CFG ∪ E_RWG`. `DFG` models operand dependencies, `CFG` models execution order, `RWG` is a bipartite graph between operands and operations, and `CG` integrates the other three views.
- **Semantics-first design**: `DFG` uses operation edges and function edges to represent value propagation; `CFG` uses `PosNext`, `NegNext`, `IterJump`, `CallNext`, `ReturnNext`, and `Next`; `RWG` uses `Read` and `Write` edges to link operands with operations explicitly.
- **GGNN encoder per view**: for each node `u`, hidden states are initialized as `h_u^0 = x_u`, where `x_u` is a one-hot node-type vector. Message passing follows `m_{u,v}^t = f_e(h_v^{t-1})`, `m̄_u^t = Mean({m_{u,v}^t}_{v ∈ N(u)})`, and `h_u^t = GRU(h_u^{t-1}, m̄_u^t)`.
- **Graph pooling and fusion**: after `T` propagation steps, each view is pooled as `z_i = MaxPooling({h_u^T}_{u ∈ V_i})`, and the final program vector is `z = z_DFG ⊕ z_CFG ⊕ z_RWG ⊕ z_CG`.
- **Downstream classifiers**: on [[poj-104]], MVG feeds `z` to a two-layer `MLP + Softmax` for `104`-way single-label prediction; on [[alg-109]], it uses `MLP + Sigmoid` with threshold `0.5` for multi-label prediction over `109` algorithms and data structures.
- **Preprocessing and hyperparameters**: a rule-based parser builds `DFG`, `CFG`, `RWG`, and `CG`. The representation dimension is searched over `100, 120, 140, 160, 180, 200` on POJ-104 and `120, 144, 168, 192, 216` on ALG-109; propagation depth `T ∈ {1, 2, 4, 8}`; learning rate is searched over `1e-3, 6e-4, 3e-4, 1e-4` with Adam.

## Key Results

- On [[poj-104]], MVG reaches `94.96%` accuracy, outperforming NCC (`94.83%`), TBCNN (`94.00%`), LRPG (`90.31%`), and Devign (`92.82%`).
- On [[alg-109]], MVG achieves `65.26 ± 0.85` micro-F1, `36.27 ± 0.67` exact match, and `1.03 ± 0.02` Hamming loss, improving over the strongest baseline LRPG at `60.56 ± 0.87` micro-F1 and `30.14 ± 1.33` exact match.
- On the frequent-label subset ALG-10, MVG reaches `80.15 ± 0.86` micro-F1, `58.36 ± 1.99` exact match, and `6.67 ± 0.29` Hamming loss.
- The new [[alg-109]] benchmark contains `11,913` C/C++ programs, `109` labels, average `94.27` lines of code, and `1.94` labels per sample, making it substantially harder than [[poj-104]].
- Ablations on ALG-109 show performance drops for every removed view: `-DFG` to `62.34` micro-F1, `-CFG` to `64.18`, `-RWG` to `64.01`, `-CG` to `64.38`, and `OnlyCG` to `62.02`; adding AST slightly hurts performance at `65.19` micro-F1 and `1.04` Hamming loss.

## Limitations

- The evaluation is concentrated on [[algorithm-detection]] rather than a broader suite of code tasks such as code search, summarization, repair, or generation, so transfer beyond this setting is not established.
- MVG depends on a rule-based parser to construct `DFG`, `CFG`, and `RWG`, which may be brittle across languages, coding styles, and incomplete or noisy code.
- Gains on [[poj-104]] are numerically small because the benchmark is already near saturation and may leak shortcut cues through problem-specific input/output formats.
- The [[alg-109]] dataset is collected from CSDN and manually annotated, so it may inherit platform bias, label imbalance, and annotation subjectivity.
- The conclusion focuses on semantics over syntax for this task; the finding that adding AST is unnecessary may not generalize to all program-analysis tasks.

## Concepts Extracted

- [[program-representation]]
- [[programming-language-processing]]
- [[algorithm-detection]]
- [[multi-view-learning]]
- [[graph-neural-network]]
- [[gated-graph-neural-network]]
- [[data-flow-graph]]
- [[control-flow-graph]]
- [[read-write-graph]]
- [[code-semantics]]
- [[multi-label-classification]]

## Entities Extracted

- [[ting-long]]
- [[yutong-xie]]
- [[xianyu-chen]]
- [[weinan-zhang]]
- [[qinxiang-cao]]
- [[yong-yu]]
- [[shanghai-jiao-tong-university]]
- [[university-of-michigan]]
- [[poj-104]]
- [[alg-109]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
