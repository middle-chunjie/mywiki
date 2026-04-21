---
type: source
subtype: paper
title: "ConvTrans: Transforming Web Search Sessions for Conversational Dense Retrieval"
slug: mao-2022-convtrans
date: 2026-04-20
language: en
tags: [conversational-search, dense-retrieval, data-augmentation, query-rewriting, session-graph]
processed: true

raw_file: raw/papers/mao-2022-convtrans/paper.pdf
raw_md: raw/papers/mao-2022-convtrans/paper.md
bibtex_file: raw/papers/mao-2022-convtrans/paper.bib
possibly_outdated: true

authors:
  - Kelong Mao
  - Zhicheng Dou
  - Hongjin Qian
  - Fengran Mo
  - Xiaohua Cheng
  - Zhao Cao
year: 2022
venue: EMNLP 2022
venue_type: conference
arxiv_id:
doi: 10.18653/v1/2022.emnlp-main.190
url: https://aclanthology.org/2022.emnlp-main.190
citation_key: mao2022convtrans
paper_type: method

read_status: unread

domain: retrieval
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature.

ConvTrans is a data augmentation pipeline for training [[conversational-dense-retrieval]] systems when genuine conversational search logs and relevance labels are scarce. The core idea is to start from abundant web search sessions with click-derived supervision, reorganize them into a heterogeneous [[session-graph]], rewrite keyword-style queries into more conversational utterances with two T5-based stages, and then sample pseudo conversations through a constrained random walk. This narrows the gap between web search behavior and conversational search along both session structure and query form. On CAsT benchmarks, the resulting training data supports retrieval quality comparable to expensive human-authored conversational data while scaling to 75,193 sessions from MS MARCO logs.

## Problem & Motivation

Conversational dense retrieval needs training examples where the current utterance must be interpreted jointly with prior turns and prior responses. The paper argues that this setting is bottlenecked by data scarcity: real conversational search systems are not widely deployed, and existing benchmarks or oracle rewrites are manually produced, expensive, and small. Raw web search sessions are abundant and include click-derived relevance labels, but they differ from conversational search in two main ways: their queries are mostly keyword-style and self-contained, and their session structure is noisier, with topic shifts and segmentation artifacts that do not resemble human conversation. ConvTrans is proposed to bridge both gaps instead of only rewriting queries or only generating pseudo labels.

## Method

- **Task setup**: a conversational session is `S^conv = {(u_k, p_k)}_{k=1}^n`, where the conversational session encoder computes `s_k = CSE(u_k, C_k)` under context `C_k = {(u_i, p_i)}_{i=1}^{k-1}` and is trained with the standard ranking loss `L_rank = -log( exp(s_k · p^+) / (exp(s_k · p^+) + Σ exp(s_k · p^-)) )`.
- **Session graph construction**: each raw web session `S^web = {(q_k, p_k)}_{k=1}^m` is converted into a heterogeneous graph whose nodes are queries and whose edges are three query relations: response-induced, topic-shared, and topic-changed.
- **Relation rules**: `q'` is response-induced from `q` if more than half its terms occur in some sentence `p^s` from `q`'s clicked passage, i.e. `|q' ∩ p^s| > |q'| / 2`, with edge weight `|q' ∩ p^s|`; otherwise it is topic-shared if `|q' ∩ q| > |q| / 2`, with weight `|q'| / |q' ∩ q|`; otherwise it is topic-changed with a constant weight.
- **Graph enrichment and pruning**: for each central query, ConvTrans augments the graph with additional satisfactory queries retrieved from the whole session database, then keeps the Top-5 response-induced and Top-5 topic-shared edges with largest weights while retaining original in-session queries preferentially.
- **Two-stage query transformation**: NL-T5 first converts keyword-style queries into natural-language queries using training pairs `(q^kw, q^nl)` built from Quora Question Pairs via KeyBERT keyword extraction; CNL-T5 then converts natural-language queries into conversational natural-language queries.
- **CNL-T5 inputs**: for topic-shared edges, the model predicts `q^cnl = T5([CLS] ◦ q'^nl ◦ [SEP] ◦ q^nl ◦ [SEP])`; for response-induced edges, it predicts `q'^cnl = T5([CLS] ◦ q'^nl ◦ [SEP] ◦ p^s ◦ [SEP])`. Central nodes remain NL-style and only linked nodes are contextualized into conversational form.
- **Training data for rewriters**: NL-T5 is trained on 404,302 Quora queries after keyword extraction; CNL-T5 is trained on [[canard]], using oracle queries from previous turns plus the current turn as input and the human conversational query as target.
- **Session generation**: the final pseudo conversation is sampled by a tailored [[random-walk-sampling]] procedure that starts from the first central node, samples at most `w = 3` topic-shared children, then `0` or `1` response-induced child, moves along the topic-changed edge, and stops at maximum length `T = 10`.
- **Retriever training details**: raw sessions come from the MS MARCO Conversational Search DEV set (`75,193` sessions, `408,389` queries). The conversational session encoder is initialized from [[ance]], trained for `1` epoch with Adam, batch size `16`, learning rate `5e-7`, maximum input length `512`, and maximum previous-response length `384`, while the passage encoder is frozen.

## Key Results

- ConvTrans generates `75,193` training sessions, versus `19,032` for AutoRewriter and `5,644` for CQE / ConvDR-CANARD.
- On CAsT-19, ConvTrans achieves `MRR = 0.732`, `NDCG@3 = 0.453`, `R@20 = 0.189`, and `R@100 = 0.360`, outperforming all compared baselines on all four metrics.
- On the harder CAsT-20 benchmark, ConvTrans reaches `MRR = 0.459`, `NDCG@3 = 0.312`, `R@20 = 0.211`, and `R@100 = 0.387`; it is second-best on MRR / NDCG@3 but best on Recall, with reported relative gains of `7.1%` on `R@20` and `8.7%` on `R@100` over the second-best results.
- ConvTrans remains competitive when training size is controlled: with only `5,644` sessions, it still clearly beats the two AutoRewriter variants and trails CQE only slightly on CAsT-20 `NDCG@3` (`0.283` vs `0.289`).
- Ablations show both major components matter: Direct use of raw sessions is near zero-shot ANCE, while adding only session graphs or only query transformation improves performance but remains well below the full model.
- Query-relation analysis shows topic-shared sampling is more important than response-induced sampling: ConvTrans-TS scores `0.288` / `0.361` on CAsT-20 (`NDCG@3` / `R@100`), ConvTrans-RI only `0.229` / `0.292`, and full ConvTrans `0.312` / `0.387`.

## Limitations

- The session graph uses only three coarse query-relation types, so it does not capture richer conversational behaviors such as returning to earlier topics or referring to older responses.
- Relevance labels are inherited directly from raw web sessions even after query insertion and reordering, so some pseudo conversational turns may carry context-misaligned supervision.
- Scaling data size helps only up to a point: the paper reports that ConvTrans stops yielding significant gains once the generated training pool reaches roughly `30,000` sessions, suggesting a remaining quality gap versus human-authored data.
- The method depends on several heuristic decisions, including lexical overlap thresholds, Top-5 edge pruning, and sampling order, which may limit robustness across domains beyond conversational search.

## Concepts Extracted

- [[conversational-search]]
- [[conversational-dense-retrieval]]
- [[dense-retrieval]]
- [[data-augmentation]]
- [[session-graph]]
- [[query-rewriting]]
- [[conversational-query-rewriting]]
- [[random-walk-sampling]]
- [[sequence-to-sequence]]

## Entities Extracted

- [[kelong-mao]]
- [[zhicheng-dou]]
- [[hongjin-qian]]
- [[fengran-mo]]
- [[xiaohua-cheng]]
- [[zhao-cao]]
- [[renmin-university-of-china]]
- [[universite-de-montreal]]
- [[huawei-poisson-lab]]
- [[cast-19]]
- [[cast-20]]
- [[canard]]
- [[keybert]]
- [[t5]]
- [[ance]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
