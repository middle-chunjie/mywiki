---
type: source
subtype: paper
title: At Which Training Stage Does Code Data Help LLMs Reasoning?
slug: unknown-nd-which-2309-16298
date: 2026-04-20
language: en
tags: [llm, reasoning, code, pretraining, instruction-tuning]
processed: true
raw_file: raw/papers/unknown-nd-which-2309-16298/paper.pdf
raw_md: raw/papers/unknown-nd-which-2309-16298/paper.md
bibtex_file: raw/papers/unknown-nd-which-2309-16298/paper.bib
possibly_outdated: true
authors:
  - Yingwei Ma
  - Yue Liu
  - Yue Yu
  - Yuanliang Zhang
  - Yu Jiang
  - Changjian Wang
  - Shanshan Li
year: 2023
venue: arXiv
venue_type: preprint
arxiv_id: 2309.16298
doi:
url: https://arxiv.org/abs/2309.16298
citation_key: unknownndwhich
paper_type: benchmark
read_status: unread
domain: llm
---

## Summary

⚠ Possibly outdated: published 2023; re-verify against recent literature.

This paper asks whether code data helps reasoning most during pre-training, instruction-tuning, or both. Using PanGu-based `2.6B`/`13B` decoder-only models, the authors compare text-only and code-mixed training on roughly `42B` pre-training tokens and about `500K` instruction examples, then evaluate six reasoning tasks spanning logic, law, science, analogy, code QA, and code generation. The main empirical finding is stage specificity: adding code during pre-training improves general reasoning with little negative transfer, while adding code during instruction tuning mainly improves code-centric reasoning and instruction following. Additional experiments show that Chain-of-Thought prompting amplifies these gains, and that a stepwise code-to-text mixing schedule yields the strongest code-task performance. The contribution is primarily an ablation-driven training study rather than a new architecture.

## Problem & Motivation

Prior work had already suggested that code-heavy models often reason better, but those observations were confounded by differences in model scale, training data, and undocumented pipelines. This paper isolates one narrower question: if code really helps reasoning, does it help because it is seen during large-scale pre-training, because it appears during instruction tuning, or because both stages matter differently? The motivation is practical as well as scientific. If the benefit mainly comes from pre-training, then code should be treated as part of the model's foundational corpus; if it mainly appears during tuning, then practitioners can use smaller, cheaper post-training interventions. The paper therefore frames code as a potential source of logical structure and long-range dependency signals for general-purpose reasoning in LLMs.

## Method

- **Comparison setup**: evaluate three regimes on matched PanGu-family models: text-only pre-training (`NL 2.6B`, `NL 13B`), mixed text+code pre-training (`CODE 2.6B` / CodePanGu2.6B), and instruction-tuned variants with or without code instructions.
- **Training objective**: standard autoregressive next-token prediction with `\mathcal{L} = \sum_{i=1}^{n} \log p(x_i \mid x_{<i}; \Theta)`.
- **Architecture**: retain a `32`-layer [[decoder-only-transformer]] with an added top query layer; Appendix B reports hidden size `2560`, FFN size `10240`, and `32` attention heads.
- **Pre-training corpus**: build a text corpus matched to PanGu2.6B (`100GB` from BaiDuQA, CAIL2018, Sogou-CA, Common Crawl, encyclopedias, news, e-books) and add CodeParrot Python data (`~50GB`, `5,361,373` files), yielding roughly `42B` tokens for the mixed setting.
- **Instruction-tuning corpus**: use about `500K` examples from Alpaca-GPT-4, PromptCLUE, CodeAlpaca, CosQA, and MBPP, all converted into a unified instruction format so text-only and code-mixed tuning can be compared.
- **Tokenization**: text-only models use the PanGu vocabulary of size `40,000`; mixed text+code models use the ChatGLM BPE vocabulary of size `130,044`, including extra tokens for repeated spaces (`<|extratoken_X|>`).
- **Optimization and hardware**: train in MindSpore on Ascend 910 clusters with sequence length `1024`; pre-training uses `16` processors and instruction tuning uses `8`; optimizer is Adam with `\beta_1 = 0.9`, `\beta_2 = 0.95`, cosine schedule, warm-up `500`, initial learning rate `1e-4` for pre-training and `2e-5` for instruction tuning, and final learning rate `1e-6`.
- **Evaluation protocol**: test six reasoning tasks across five domains using prompt-based zero-shot evaluation; understanding tasks are cast as perplexity comparison, while MBPP is generated directly. The paper additionally tests CoT prompts and three code/text mixing schedules: uniform `5:3`, stepwise increase (`7:3 -> 7:3 -> 6:4 -> 5:5`), and stepwise decrease (`5:5 -> 6:4 -> 7:3 -> 7:3`).

## Key Results

- **Pre-training with code improves general reasoning**: `CODE 2.6B` beats `NL 2.6B` on Logic (`40.90` vs `36.36`), JEC-QA (`28.70` vs `27.00`), ScienceQA (`46.06` vs `45.93`), E-KAR (`36.12` vs `32.24`), CosQA (`50.50` vs `47.01`), and MBPP BLEU (`5.06` vs `0.52`).
- **Code pre-training can outperform larger text-only models**: `CODE 2.6B` surpasses `NL 13B` on JEC-QA (`28.70` vs `27.00`), ScienceQA (`46.06` vs `45.18`), E-KAR (`36.12` vs `35.52`), CosQA (`50.50` vs `46.85`), and MBPP (`5.06` vs `1.34`), though not on Logic (`40.90` vs `45.45`).
- **Code instructions mainly help code-centric tasks**: for PanGu `2.6B`, moving from text-only tuning (`NN`) to code-mixed tuning (`NC`) raises CosQA from `45.20` to `48.18` and MBPP from `0.00` to `5.61`, but hurts ScienceQA (`44.45` to `43.44`) and E-KAR (`30.45` to `28.66`).
- **Using code in both stages yields the strongest code ability**: `CC 2.6B` reaches CosQA `52.48` and MBPP `24.88`, far above the text-only tuned baseline.
- **Chain-of-Thought is especially effective for the code-pretrained model**: `CODE 2.6B` improves from `46.06` to `70.30` on ScienceQA and from `36.12` to `72.84` on E-KAR when CoT is provided.
- **Dynamic mixing matters**: the stepwise-decrease schedule gives the best code-task results among tested strategies, with CosQA `52.48` and MBPP `24.88`.
- **Negative transfer is limited but not zero**: code data barely changes `C^3` (`54.14` to `54.30` in pre-training), but DuReader drops notably in both pre-training (`F1 15.29` to `8.73`) and instruction tuning (`F1 34.85` to `25.05`).

## Limitations

- The empirical study is restricted to the PanGu architecture family; Appendix H explicitly says the authors did not verify the conclusions on more LLM backbones because of resource constraints.
- The paper does not introduce a genuinely new model design, so its claims are about training-stage effects rather than a new reasoning mechanism.
- The proposed dynamic mixing strategy is only validated during instruction tuning; the authors leave pre-training-stage validation of mixing schedules to future work because of compute cost.
- Some instruction-tuned results degrade on non-code tasks, especially DuReader, so the claim of "almost no negative transfer" does not hold uniformly across all downstream settings.
- The explanation for why code helps reasoning is partly inferential, based on case studies about code logic and dependency structure rather than direct mechanistic evidence.

## Concepts Extracted

- [[large-language-model]]
- [[code-language-model]]
- [[instruction-tuning]]
- [[chain-of-thought]]
- [[data-mixture]]
- [[autoregressive-language-model]]
- [[decoder-only-transformer]]
- [[perplexity]]

## Entities Extracted

- [[yingwei-ma]]
- [[yue-liu]]
- [[yue-yu]]
- [[yuanliang-zhang]]
- [[yu-jiang]]
- [[changjian-wang]]
- [[shanshan-li]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
