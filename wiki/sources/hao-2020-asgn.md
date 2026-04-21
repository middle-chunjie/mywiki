---
type: source
subtype: paper
title: "ASGN: An Active Semi-supervised Graph Neural Network for Molecular Property Prediction"
slug: hao-2020-asgn
date: 2026-04-20
language: en
tags: [chemistry, graph-learning, molecular-property-prediction, semi-supervised-learning, active-learning]
processed: true
raw_file: raw/papers/hao-2020-asgn/paper.pdf
raw_md: raw/papers/hao-2020-asgn/paper.md
bibtex_file: raw/papers/hao-2020-asgn/paper.bib
possibly_outdated: false
authors:
  - Zhongkai Hao
  - Chengqiang Lu
  - Zhenya Huang
  - Hao Wang
  - Zheyuan Hu
  - Qi Liu
  - Enhong Chen
  - Cheekong Lee
year: 2020
venue: "Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining"
venue_type: conference
arxiv_id:
doi: 10.1145/3394486.3403117
url: https://dl.acm.org/doi/10.1145/3394486.3403117
citation_key: hao2020asgn
paper_type: method
read_status: unread
domain: chemistry
---

## Summary

ASGN proposes a molecular property prediction framework that combines semi-supervised learning and active learning on top of a message-passing graph neural network. The method separates representation learning from target prediction through a teacher-student design: the teacher jointly optimizes supervised property loss with node-level reconstruction and graph-level clustering losses over labeled and unlabeled molecules, while the student fine-tunes on labeled data only and feeds pseudo labels back to the teacher. The framework then uses teacher embeddings for diversity-based active selection of new molecules to label. On QM9 and OPV, ASGN consistently beats supervised, Mean-Teacher, and InfoGraph baselines, while also improving label efficiency by roughly `2x-3x` in the active-learning setting.

## Problem & Motivation

Accurate molecular property prediction is valuable for chemistry, biology, and materials discovery, but high-quality labels are usually produced by expensive Density Functional Theory calculations. Prior graph neural network approaches such as MPNN and SchNet are effective but strongly data-hungry, making them brittle when only a small fraction of molecules are labeled. The paper argues that unlabeled molecules still encode useful structural and distributional information, yet naive semi-supervised training creates a conflict between representation learning and downstream property optimization. ASGN is motivated by the need to exploit unlabeled molecular graphs, reduce overfitting under scarce labels, and actively choose the most informative molecules for additional labeling.

## Method

- **Backbone MPGNN**: each molecule is a weighted graph `G = (V, E)` with atom features and pairwise distances; the backbone uses `L = 4` message-passing layers and hidden dimension `d = 96` in experiments.
- **Message passing update**: node states are updated as ``z_i^{l+1} = sigma(W^l · AGG(z_i^l, {e(v_i, v_j): v_j in N(v_i)}))`` with `AGG = sum`, so the encoder explicitly aggregates neighbor messages over molecular edges.
- **Distance-aware edge encoding**: messages use Gaussian radial basis features ``e(v_i, v_j)[k] = z_i^l[k] · exp(-gamma (||r_i - r_j|| - d_k)^2)`` with filters spanning `0` to `3 nm` at interval `0.01 nm`.
- **Teacher objective**: the teacher jointly optimizes supervised property regression ``L_p = sum ||y_i - f_theta(z_G_i)||^2``, node-level reconstruction `L_r` for atom types and discretized edge distances, and graph-level clustering `L_c` over `M = 100` clusters, giving ``L_t = sum L_p + sum L_r + sum L_c`` across labeled and unlabeled molecules.
- **Node-level SSL**: the model reconstructs sampled node types and sampled edge-distance bins from node embeddings; only `alpha |G|` edges are sampled so reconstruction cost stays linear in graph size.
- **Graph-level SSL**: graph embeddings are clustered under a uniform prior and optimized through an entropy-regularized optimal-transport objective solved with the Sinkhorn-Knopp algorithm, encouraging chemically meaningful global partitions.
- **Student model and pseudo labels**: after teacher pretraining, weights are transferred to a student network that optimizes only ``L_s = sum ||y_i - f_theta_s(z_G_i)||^2`` on labeled data; the student then predicts pseudo labels for unlabeled molecules and feeds them back into the next teacher iteration.
- **Active learning**: in each iteration the framework selects a diverse batch by maximizing the minimum embedding distance to the current labeled set, i.e. ``argmax_j min_i d(G_i, G_j)`` with `L2` distance in teacher embedding space.
- **Training setup**: the teacher is trained or fine-tuned for `20` epochs per iteration, the student is trained until loss plateaus for about `20` epochs, and active selection adds `1000` new molecules per iteration in the main setup.

## Key Results

- On QM9 effectiveness benchmarks, ASGN reduces MAE from `0.1410` to `0.0562` on `U0`, from `0.1702` to `0.0594` on `U`, from `0.1605` to `0.1190` on HOMO, and from `0.5444` to `0.2818` on `alpha` relative to InfoGraph.
- On OPV, ASGN reaches MAE `0.059` on HOMO and `0.057` on LUMO, improving over supervised (`0.080`, `0.078`), Mean-Teacher (`0.078`, `0.075`), and InfoGraph (`0.077`, `0.076`).
- In the efficiency experiment, the paper reports that ASGN is about `2x-3x` more label-efficient than active-learning baselines and reaches full supervised accuracy with about `50%` of labels on QM9 and `40%` on OPV.
- Ablation on HOMO shows the full method outperforming teacher-only and student-only variants: on QM9 with `5k` labels ASGN gets `0.1190` vs `0.1668` (ASGN-T) and `0.1632` (ASGN-S); on OPV with `5k` labels it gets `0.060` vs `0.080` and `0.076`.
- The framework runs on `1` Tesla V100 GPU and `16` Intel CPUs in the reported setup, with label budgets of `5000` molecules for the main effectiveness comparison.

## Limitations

- The evidence is limited to two quantum-chemistry datasets, QM9 and OPV, so the paper does not establish whether the framework transfers to broader molecular regimes or non-chemistry graph tasks.
- Active learning still assumes access to an external labeling oracle such as DFT, which remains computationally expensive even if fewer labels are needed.
- The method adds multiple interacting components, including clustering, pseudo labeling, and teacher-student iteration, but does not provide a full wall-clock or memory breakdown against all baselines.
- The paper compares mainly with contemporary semi-supervised and active-learning baselines and does not test against later graph pretraining or molecular foundation-model approaches.

## Concepts Extracted

- [[graph-neural-network]]
- [[molecular-property-prediction]]
- [[semi-supervised-learning]]
- [[active-learning]]
- [[teacher-student-framework]]
- [[message-passing-neural-network]]
- [[graph-representation-learning]]
- [[pseudo-labeling]]
- [[molecular-graph]]

## Entities Extracted

- [[zhongkai-hao]]
- [[chengqiang-lu]]
- [[zhenya-huang]]
- [[hao-wang-ustc]]
- [[zheyuan-hu]]
- [[qi-liu]]
- [[enhong-chen]]
- [[cheekong-lee]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
