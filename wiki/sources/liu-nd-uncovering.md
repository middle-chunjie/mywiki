---
type: source
subtype: paper
title: Uncovering and Quantifying Social Biases in Code Generation
slug: liu-nd-uncovering
date: 2026-04-20
language: en
tags: [code-generation, social-bias, fairness, evaluation, llm]
processed: true

raw_file: raw/papers/liu-nd-uncovering/paper.pdf
raw_md: raw/papers/liu-nd-uncovering/paper.md
bibtex_file: raw/papers/liu-nd-uncovering/paper.bib
possibly_outdated: true

authors:
  - Yan Liu
  - Xiaokang Chen
  - Yan Gao
  - Zhe Su
  - Fengji Zhang
  - Daoguang Zan
  - Jian-Guang Lou
  - Pin-Yu Chen
  - Tsung-Yi Ho
year: 2023
venue: NeurIPS 2023
venue_type: conference
arxiv_id:
doi:
url: https://papers.nips.cc/paper_files/paper/2023/hash/071a637d41ea290ac4360818a8323f33-Abstract-Conference.html
citation_key: liunduncovering
paper_type: benchmark

read_status: unread

domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper studies whether modern code generation models express social bias when prompted in human-relevant settings. The authors design a prompt construction paradigm that places two complete functions before an incomplete signature such as `find_ADJ_people(people, HumanAttribute)` to elicit demographic judgments from Codex, InCoder, and CodeGen. They build a 392-prompt dataset spanning 49 modifiers and 8 demographic dimensions, train a code bias classifier for automatic scoring, and introduce three evaluation metrics: Code Bias Score, UnFairness Score, and the standard deviation of demographic frequencies. Across all tested model families, larger models show both better pass@k and more severe social bias, with Codex producing the strongest bias signals overall.

## Problem & Motivation

Prior fairness work mostly studies natural-language tasks, while standard code-generation benchmarks such as HumanEval are largely human-irrelevant and therefore unlikely to surface demographic harms. The paper asks whether social bias is nevertheless encoded in code models and, if so, how it appears in generated code. The motivation is practical: code assistants can be used in human-facing applications such as hiring, education, and law, so biased code completions can propagate stereotypes into downstream systems. The paper therefore reframes code fairness as an evaluation problem requiring specially constructed prompts, demographic coverage, and metrics that capture both overall bias prevalence and demographic disparities.

## Method

- **Prompt paradigm**: construct prompts with two complete seed functions plus an incomplete human-relevant signature `def find_ADJ_people(people, HumanAttribute):`; the adjective slot is a judgmental modifier and `HumanAttribute` is one of 8 demographic dimensions.
- **Modifier space**: instantiate `49` modifiers across 5 categories: `25` RoBERTa-negative, `10` random negative, `10` random positive, `2` comparative negative, and `2` comparative positive.
- **Dataset construction**: combine `49 modifiers x 8 demographic dimensions = 392` prompts; sample `10` completions per prompt for each model, yielding `3920` generated snippets per model family.
- **Prompt-design findings**: appendix ablations show bias elicitation increases with more prompt functions (`0 -> 0.00`, `1 -> 8.53/14.05`, `2 -> 31.55/39.30`, `3 -> 40.01/52.63` CBS for InCoder/CodeGen), more human-relevant functions, and placing the human-relevant function closer to the incomplete signature.
- **Models evaluated**: Codex (`100B+` inferred by authors), InCoder (`1.3B`, `6.7B`), and CodeGen Mono (`350M`, `2.7B`, `6.1B`).
- **Annotation setup**: annotate generated code as biased vs acceptable; split the code-bias dataset into train/dev/test with ratio `7:2:1`, producing combined totals of `5488 / 1568 / 784`.
- **Automatic scorer**: compare three classifiers, namely LSTM without pretrained embeddings, LSTM with word2vec embeddings, and BERT-Base; BERT-Base is selected as the final code bias classifier. Appendix reproducibility details set BERT learning rate to `1e-5` for `5` epochs, and the LSTM baseline uses `2` layers with `100` hidden units each for `5` epochs.
- **Metrics**: compute `CBS = (sum_i 1[P_cls(code_i) >= 0.5] / N) x 100`, `UFS = (f_d1 - f_d2) / max(f_d1, f_d2)` with `f_di = N_di / N_bias`, and `sigma = sqrt((1/M) sum_k (f_dk - f_bar)^2)`; code quality is tracked with [[pass-at-k]] on HumanEval.
- **Decoding analysis**: vary `temperature t` and `top-p` over `{0.1, 0.2, ..., 0.9}` on CodeGen-6B to study how sampling hyperparameters affect bias severity.

## Key Results

- All evaluated model families exhibit non-trivial bias: total CBS is `23.52` for InCoder-1.3B, `32.55` for InCoder-6.7B, `9.36` for CodeGen-350M, `45.15` for CodeGen-2.7B, `62.65` for CodeGen-6.1B, and `82.64` for Codex.
- Larger models are more capable and more biased: Codex reaches `pass@1 = 47.03`, `pass@10 = 74.91`, `pass@100 = 92.14`, but also has the highest bias under every modifier group.
- Bias is sensitive to demographic dimension: for InCoder-6B, standard deviation across valid demographics is especially high for ethnicity (`23.24` under RoBERTa-Neg) and gender (`54.34` under RoBERTa-Neg).
- Fine-grained unfairness can flip direction by modifier and dimension: InCoder-6B reports ethnicity UFS values from `-0.33` to `0.66` and gender UFS values from `-1.00` to `0.68`, indicating uneven treatment across paired demographics.
- Human evaluation confirms the automatic trend: total human CBS is `28.90` for InCoder-1.3B, `38.73` for InCoder-6.7B, `5.69` for CodeGen-350M, `48.45` for CodeGen-2.7B, `68.25` for CodeGen-6.1B, and `84.03` for Codex.
- Sampling choices matter: CBS stays relatively high for `temperature = 0.3-0.5`, falls when `temperature > 0.6`, and peaks around `top-p = 0.8` in the CodeGen-6B analysis.

## Limitations

- The evaluation depends on a crafted prompt template rather than naturally occurring programming workflows, so measured bias may reflect elicitation strength as well as model behavior.
- The automatic scorer is useful but imperfect; the paper explicitly validates it against human annotation because classifier error can affect CBS and UFS.
- Coverage is limited to `8` demographic dimensions and `3` model families, leaving broader populations, languages, and newer code models untested.
- The work diagnoses bias but does not provide a concrete mitigation method or a deployment-time intervention.

## Concepts Extracted

- [[code-generation]]
- [[social-bias]]
- [[ai-fairness]]
- [[prompt-construction]]
- [[code-bias-classifier]]
- [[bias-metric]]
- [[pass-at-k]]
- [[human-evaluation]]
- [[temperature-sampling]]
- [[top-p-sampling]]

## Entities Extracted

- [[yan-liu]]
- [[xiaokang-chen]]
- [[yan-gao]]
- [[zhe-su]]
- [[fengji-zhang]]
- [[daoguang-zan]]
- [[jian-guang-lou]]
- [[pin-yu-chen]]
- [[tsung-yi-ho]]
- [[codex]]
- [[incoder]]
- [[codegen]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
