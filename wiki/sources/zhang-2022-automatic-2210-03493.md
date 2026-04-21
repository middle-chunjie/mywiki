---
type: source
subtype: paper
title: Automatic Chain of Thought Prompting in Large Language Models
slug: zhang-2022-automatic-2210-03493
date: 2026-04-20
language: en
tags: [chain-of-thought, prompting, llm, reasoning, in-context-learning]
processed: true
raw_file: raw/papers/zhang-2022-automatic-2210-03493/paper.pdf
raw_md: raw/papers/zhang-2022-automatic-2210-03493/paper.md
bibtex_file: raw/papers/zhang-2022-automatic-2210-03493/paper.bib
possibly_outdated: true
authors:
  - Zhuosheng Zhang
  - Aston Zhang
  - Mu Li
  - Alex Smola
year: 2022
venue: ICLR 2023
venue_type: conference
arxiv_id: 2210.03493
doi:
url: http://arxiv.org/abs/2210.03493
citation_key: zhang2022automatic
paper_type: method
read_status: unread
domain: nlp
---

## Summary

⚠ Possibly outdated: published 2022; re-verify against recent literature in LLM prompting.

Manual-CoT prompting achieves strong multi-step reasoning by providing hand-crafted demonstrations, but hand-crafting is expensive and task-specific. This paper proposes **Auto-CoT**, which eliminates manual effort by (1) clustering test questions into `k` groups via k-means over Sentence-BERT embeddings, and (2) selecting the cluster-centroid question from each group and auto-generating its reasoning chain via Zero-Shot-CoT ("Let's think step by step"). The key insight is that diversity-based sampling avoids the *misleading-by-similarity* failure mode, where similar wrong demonstrations cluster together and propagate errors. On ten public reasoning benchmarks (arithmetic, commonsense, symbolic) with GPT-3 (text-davinci-002), Auto-CoT matches or exceeds Manual-CoT, showing that automation with diversity can substitute for human demonstration design.

## Problem & Motivation

Chain-of-thought prompting splits into two paradigms: Zero-Shot-CoT (append "Let's think step by step") and Manual-CoT (hand-craft `k=8` demonstrations with question + rationale + answer). Manual-CoT consistently outperforms Zero-Shot-CoT, but the performance gain hinges entirely on nontrivial human effort to design task-specific, high-quality demonstrations. The natural automation candidate—retrieving the most similar questions and using Zero-Shot-CoT to generate their reasoning chains (Retrieval-Q-CoT)—actually underperforms random sampling. The paper investigates why and proposes a principled alternative.

## Method

- **Retrieval-Q-CoT failure analysis**: Retrieval selects semantically similar questions to a test instance. When Zero-Shot-CoT generates a wrong chain for a similar question, the LLM replicates the same mistake for the test question (misleading by similarity). Unresolved-rate: Retrieval-Q-CoT `46.9%` vs. Random-Q-CoT `25.8%` on MultiArith questions where Zero-Shot-CoT already fails.
- **Frequent-error cluster observation**: k-means on Sentence-BERT embeddings reveals clusters with systematically high Zero-Shot-CoT error rates (e.g., Cluster 2 at `52.3%` error on MultiArith). Similarity-based retrieval risks drawing multiple wrong demonstrations from the same cluster.
- **Auto-CoT Stage 1 — Question Clustering**: Encode each question with Sentence-BERT; cluster all test questions into `k` groups using k-means. Within each cluster, sort questions by ascending distance to cluster center (`q_1^(i), q_2^(i), ...`).
- **Auto-CoT Stage 2 — Demonstration Sampling**: For each cluster `i`, iterate over sorted questions; invoke Zero-Shot-CoT on question `q_j^(i)` to get rationale `r_j^(i)` and answer `a_j^(i)`; accept as demonstration `d^(i) = [Q: q_j^(i), A: r_j^(i) ∘ a_j^(i)]` if the question has `≤ 60 tokens` and rationale has `≤ 5 reasoning steps` (heuristics to filter verbose/noisy outputs); otherwise try next question in cluster.
- **Inference**: Concatenate `k` constructed demonstrations followed by the test question; feed to LLM. `k = 8` for most tasks; `k ∈ {4, 6, 7}` for AQuA, Letter, CSQA, StrategyQA.
- **Streaming extension (Auto-CoT*)**: Bootstrapping variant for the streaming setting where questions arrive in batches. Batch 1 uses Zero-Shot-CoT; from batch 2 onward, use question-chain pairs accumulated so far as demonstrations (Auto-CoT style).
- **LLM**: GPT-3 `text-davinci-002` (175B), greedy decoding, `max_tokens = 256`, `temperature = 0`. Also tested with Codex `code-davinci-002`.

## Key Results

- **Main results (GPT-3, Table 3)**: Auto-CoT vs. Manual-CoT on 10 datasets:
  - MultiArith: `92.0` vs. `91.7`; GSM8K: `47.9` vs. `46.9`; AddSub: `84.8` vs. `81.3`
  - AQuA: `36.5` vs. `35.8`; SingleEq: `87.0` vs. `86.6`; SVAMP: `69.5` vs. `68.9`
  - CSQA: `74.4` vs. `73.5`; StrategyQA: `65.4` vs. `65.4`; Letter: `59.7` vs. `59.0`; Coin: `99.9` vs. `97.2`
  - Auto-CoT matches or exceeds Manual-CoT on all 10 tasks.
- **Codex (Table 4)**: Auto-CoT `93.2` / `62.8` / `91.9` vs. Manual-CoT `96.8` / `59.4` / `84.6` on MultiArith / GSM8K / AddSub; Auto-CoT surpasses Manual-CoT on GSM8K and AddSub.
- **Robustness to wrong demonstrations (Figure 6)**: Auto-CoT degrades much more slowly than In-Cluster Sampling as wrong demonstration proportion increases; still competitive at `50%` wrong demos.
- **Streaming (Figure 7)**: Auto-CoT* at batch `≥ 2` matches Manual-CoT accuracy on MultiArith (`m = 30` questions per batch).
- **Ablation — demonstration components (Table 5)**: Shuffling answers drops accuracy from `91.7%` to `17.0%`; shuffling rationales to `43.8%`; shuffling questions to `73.8%`, confirming rationale-answer consistency is critical.
- **Cluster center sampling (Table 8)**: Questions closest to cluster center yield `93.7%` vs. `89.2%` (random) vs. `88.7%` (farthest) on MultiArith.

## Limitations

- Requires access to the full test set at demonstration construction time (batch assumption); streaming variant exists but adds complexity.
- Greedy decoding from Zero-Shot-CoT still produces wrong chains; the method relies on heuristics (length filters) rather than answer verification to filter bad demonstrations—verified-correct chains (Retrieval-Q-CoT with annotations) are better when ground-truth labels are available.
- k-means hyperparameter `k` is set equal to the number of demonstrations and is not tuned per task; the method is sensitive to how many demonstrations are feasible within context length.
- Evaluated solely on GPT-3 (`text-davinci-002`) and Codex; generalization to instruction-tuned or RLHF-trained LLMs (e.g., ChatGPT, GPT-4) is unverified in this paper.
- All benchmarks are English reasoning tasks; cross-lingual or multilingual behavior is not studied.

## Concepts Extracted

- [[auto-cot]]
- [[chain-of-thought-prompting]]
- [[zero-shot-cot]]
- [[in-context-learning]]
- [[misleading-by-similarity]]
- [[manual-cot]]
- [[k-means-clustering]]
- [[few-shot-prompting]]
- [[self-consistency]]
- [[large-language-model]]
- [[arithmetic-reasoning]]
- [[commonsense-reasoning]]
- [[symbolic-reasoning]]

## Entities Extracted

- [[zhuosheng-zhang]]
- [[aston-zhang]]
- [[mu-li-amazon]]
- [[alex-smola]]
- [[shanghai-jiao-tong-university]]
- [[amazon-web-services]]
- [[gpt-3]]
- [[sentence-bert]]
- [[gsm8k]]
- [[strategyqa]]
- [[multiarith]]
- [[commonsenseqa]]
- [[jason-wei]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
