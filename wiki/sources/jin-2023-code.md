---
type: source
subtype: paper
title: Code Recommendation for Open Source Software Developers
slug: jin-2023-code
date: 2026-04-20
language: en
tags: [code-recommendation, open-source-software, recommender-systems, graph-learning, software-engineering]
processed: true
raw_file: raw/papers/jin-2023-code/paper.pdf
raw_md: raw/papers/jin-2023-code/paper.md
bibtex_file: raw/papers/jin-2023-code/paper.bib
possibly_outdated: false
authors:
  - Yiqiao Jin
  - Yunsheng Bai
  - Yanqiao Zhu
  - Yizhou Sun
  - Wei Wang
year: 2023
venue: Proceedings of the ACM Web Conference 2023
venue_type: conference
arxiv_id:
doi: 10.1145/3543507.3583503
url: https://dl.acm.org/doi/10.1145/3543507.3583503
citation_key: jin2023code
paper_type: method
read_status: unread
domain: software-engineering
---

## Summary

Jin et al. formulate code recommendation for open-source software developers as a file-level matching problem over GitHub repositories and propose CODER, a graph-based framework that fuses code semantics, file-structure hierarchy, and multi-level user behavior. The model first builds file and repository representations by combining CodeBERT-based code segments with contributor-aware attention and a repository hierarchy graph, then jointly models microscopic user-file interactions and macroscopic user-project behaviors through graph propagation. To support evaluation, the paper constructs three large-scale GitHub datasets spanning machine learning, databases, and full-stack development. Across intra-project, cross-project, and cold-start settings, CODER consistently outperforms matrix factorization, neural collaborative filtering, and graph recommendation baselines, with especially large gains on sparse full-stack data.

## Problem & Motivation

The paper studies how to recommend concrete project files as development tasks to open-source contributors. This setting differs from ordinary item recommendation because OSS activity is sparse, multimodal, and hierarchical: developers interact with both files and repositories, code carries semantic information beyond IDs, and files live in directory structures that encode functionality. Existing recommender models mostly treat interactions as flat user-item signals, so they cannot adequately exploit repository hierarchy, source-code semantics, or project-level actions such as stars and forks. The authors therefore aim to jointly model fine-grained user-file expertise, coarse-grained user-project interests, and structural relationships among files to improve recommendation quality, especially for cross-project transfer and cold-start users.

## Method

- **Task formulation**: define users `\mathcal{U}`, files `\mathcal{V}`, and repositories `\mathcal{R}` with file-level interaction matrix `\mathbf{Y} \in {0,1}^{|\mathcal{U}| \times |\mathcal{V}|}` and project-level behavior matrices for multiple behavior types `t \in \mathcal{T}`.
- **Overall architecture**: CODER has two stages. Node semantics modeling learns structure-aware file and repository embeddings; multi-behavioral modeling propagates user-file and user-project interactions and fuses both granularities for prediction.
- **Code-user modality fusion**: each file is partitioned into `N_C = 8` code segments; each user description/history is partitioned into `N_Q = 4` query segments. With code feature map `\mathbf{C}` and user feature map `\mathbf{Q}`, the model computes an affinity matrix `\mathbf{L} = \tanh(\mathbf{C}\mathbf{W}_O\mathbf{Q}^\top)`, attention map `\mathbf{H} = \tanh(\mathbf{W}_C\mathbf{C}^\top + \mathbf{W}_Q(\mathbf{L}\mathbf{Q})^\top)`, weights `\mathbf{a} = \operatorname{softmax}(\mathbf{w}_H^\top \mathbf{H})`, and file representation `\mathbf{h} = \mathbf{a}^\top \mathbf{C}`.
- **Code encoder**: source files are tokenized, chunked at token boundaries, and encoded with pretrained CodeBERT using `6` layers, `12` attention heads, and hidden size `768`; max pooling produces segment embeddings.
- **Structural-level aggregation**: each repository is converted into a hierarchical heterogeneous graph over repository, directory, and file nodes. Directory names are encoded with TF-IDF; repository nodes use owner, creation time, and top-5 languages. A `3`-layer Graph Attention Network updates node states as `\widetilde{\mathbf{h}} = f_{\text{GNN}}(\mathbf{h}, G_S)`.
- **File-level aggregation**: CODER builds a user-file bipartite graph and uses LightGCN-style propagation: `\mathbf{u}_i^{(l)} = \sum_{v_j \in \mathcal{N}_i} \frac{1}{\sqrt{|\mathcal{N}_i||\mathcal{N}_j|}} \mathbf{v}_j^{(l-1)}` and `\mathbf{v}_j^{(l)} = \sum_{u_i \in \mathcal{N}_j} \frac{1}{\sqrt{|\mathcal{N}_j||\mathcal{N}_i|}} \mathbf{u}_i^{(l-1)}`. Mean pooling over `L = 4` layers yields `\mathbf{u}_i^\star` and `\mathbf{v}_j^\star`.
- **Project-level aggregation**: for each project behavior `t`, user and repository embeddings are propagated on a user-repository graph as `\mathbf{Z}_t^{(l)} = \mathbf{D}_t^{-1/2}\mathbf{\Lambda}_t\mathbf{D}_t^{-1/2}\mathbf{Z}_t^{(l-1)}` and averaged across `L = 4` layers to obtain `\mathbf{z}_{t,i}^\star` and `\mathbf{r}_{t,k}^\star`.
- **Prediction head**: macro-level behavior embeddings are aggregated as `\mathbf{z}_i^\star = \textsc{AGG}(\{z_t^\star\})` and `\mathbf{r}_k^\star = \textsc{AGG}(\{r_t^\star\})`, then fused through MLPs: `\mathbf{u}_i = \textsc{MLP}([\mathbf{u}_i^\star \| \mathbf{z}_i^\star])`, `\mathbf{v}_j = \textsc{MLP}([\mathbf{v}_j^\star \| \mathbf{r}_{\phi(j)}^\star])`. File relevance is scored by inner product `s_F(i,j) = \mathbf{u}_i^\top \mathbf{v}_j`.
- **Optimization**: the main objective uses Bayesian Personalized Ranking on files and projects, plus structure contrastive losses on users and files: `\mathcal{L} = \mathcal{L}_F + \lambda_1 \sum_t \mathcal{L}_P^t + \lambda_2 (\mathcal{L}_C^\mathcal{U} + \mathcal{L}_C^\mathcal{V}) + \lambda_3 ||\Theta||_2`, with `\lambda_1 \in {10^-2, 10^-1, 1}`, `\lambda_2 = 10^-6`, `\lambda_3 \in {10^-4, 10^-3, 10^-2}`, and contrastive hop `\eta = 2`.
- **Implementation details**: all models use embedding size `32`, Xavier initialization, Adam, and batch size `1024`; learning rate is searched in `{10^-4, 3 \times 10^-4, 10^-3, 3 \times 10^-3, 10^-2}`.
- **Complexity and runtime**: the overall time complexity is `O((N_C + N_Q)|\mathcal{V}| + |\mathcal{E}| + |\mathbf{A}^+| + \sum_t |\mathbf{\Lambda}_t^+|)`. Average inference time is `1.138 ms` per example, versus `1.073 ms` for LightGCN and `0.804 ms` for MF.

## Key Results

- The paper releases three GitHub-based datasets: ML with `239,232` files, `21,913` users, and `663,046` interactions; DB with `415,154` files, `30,185` users, and `1,935,155` interactions; FS with `568,972` files, `51,664` users, and `1,512,809` interactions.
- On ML intra-project recommendation, CODER reaches `NDCG@50 = 0.132`, `Hit@50 = 0.351`, and `MRR@50 = 0.211`, improving over the best baseline by `11.2%`, `20.5%`, and `5.0%`; at `K = 200`, gains are `17.9%` on NDCG, `15.8%` on Hit, and `8.2%` on MRR.
- On DB, CODER obtains `NDCG@50 = 0.160`, `Hit@50 = 0.390`, and `MRR@50 = 0.260`; at `K = 200`, improvements over the strongest baseline are `27.3%` on `NDCG`, `29.5%` on `Hit`, and `6.0%` on `MRR`.
- On the sparsest FS dataset, CODER reaches `NDCG@50 = 0.146`, `Hit@50 = 0.374`, and `MRR@50 = 0.226`; the paper reports the largest relative gains here, including `37.1%` on `NDCG@50` and `35.6%` on `NDCG@100`.
- For cold-start users with `<= 2` training interactions, CODER still leads strongly: on ML it achieves `NDCG@3 = 0.126` and `MRR@5 = 0.177`; on DB `NDCG@3 = 0.119` and `Hit@5 = 0.287`; on FS `NDCG@5 = 0.137` and `MRR@5 = 0.187`, with relative gains up to `54.0%` on DB `MRR@3`.
- In ablations, removing project-level aggregation hurts the most, followed by disabling structural-level aggregation, indicating that macro-level user-project signals and file hierarchy are the most critical components beyond pretrained code semantics.

## Limitations

- The method only recommends existing files; it cannot score brand-new files because candidate metadata and code semantics are unavailable at prediction time.
- Project hierarchy is modeled through directory names and parent-child links, which is cheaper than AST or data-flow analysis but may miss finer program semantics.
- The code encoder is cached because it is expensive to compute, so the approach still depends on a relatively heavy pretrained model even though final inference is near-LightGCN speed.
- Evaluation is restricted to three GitHub topic clusters and commit-based positive interactions, so generalization to other OSS ecosystems or task definitions remains uncertain.
- The authors do not model social relations among developers, which they explicitly identify as future work for improving user representations.

## Concepts Extracted

- [[code-recommendation]]
- [[heterogeneous-graph]]
- [[graph-neural-network]]
- [[graph-attention-network]]
- [[lightgcn]]
- [[bayesian-personalized-ranking]]
- [[contrastive-learning]]
- [[cold-start-recommendation]]
- [[multimodal-recommendation]]

## Entities Extracted

- [[yiqiao-jin]]
- [[yunsheng-bai]]
- [[yanqiao-zhu]]
- [[yizhou-sun]]
- [[wei-wang-ucla]]
- [[georgia-institute-of-technology]]
- [[university-of-california-los-angeles]]
- [[github]]
- [[codebert]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
