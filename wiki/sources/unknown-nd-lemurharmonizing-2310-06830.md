---
type: source
subtype: paper
title: "Lemur: Harmonizing Natural Language and Code for Language Agents"
slug: unknown-nd-lemurharmonizing-2310-06830
date: 2026-04-20
language: en
tags: [agents, llm, code, pretraining, instruction-tuning]
processed: true

raw_file: raw/papers/unknown-nd-lemurharmonizing-2310-06830/paper.pdf
raw_md: raw/papers/unknown-nd-lemurharmonizing-2310-06830/paper.md
bibtex_file: raw/papers/unknown-nd-lemurharmonizing-2310-06830/paper.bib
possibly_outdated: false

authors:
  - Yiheng Xu
  - Hongjin Su
  - Chen Xing
  - Boyu Mi
  - Qian Liu
  - Weijia Shi
  - Binyuan Hui
  - Fan Zhou
  - Yitao Liu
  - Tianbao Xie
  - Zhoujun Cheng
  - Siheng Zhao
  - Lingpeng Kong
  - Bailin Wang
  - Caiming Xiong
  - Tao Yu
year: 2024
venue: ICLR 2024 Spotlight
venue_type: conference
arxiv_id: 2310.06830
doi: 10.48550/arXiv.2310.06830
url: https://arxiv.org/abs/2310.06830
citation_key: unknownndlemurharmonizing
paper_type: method

read_status: unread

domain: agents
---

## Summary

Lemur studies how to build open models that are simultaneously strong at natural language interaction and executable code generation for language-agent workloads. Starting from Llama-2-70B, the paper performs continual pretraining on a 90B-token corpus with a 10:1 code-to-text ratio, then instruction-finetunes on roughly 300K mixed text-and-code examples to produce Lemur-Chat. The central claim is that agent performance depends on harmonizing these two capability clusters rather than maximizing one alone. Across standard text and coding benchmarks, Lemur improves the average score of the base model, and on agent evaluations covering tool use, environment feedback, natural-language feedback, and partially observable environments, Lemur-Chat is the strongest open model reported in the paper and substantially narrows, though does not close, the gap to GPT-4.

## Problem & Motivation

The paper argues that language agents require more than conversational fluency: they must also generate grounded, executable actions in environments such as Python interpreters, SQL terminals, browsers, and embodied simulators. Existing open models in 2023-2024 tended to specialize either in natural language or in code, which made them brittle in multi-turn agent settings where planning, communication, tool calling, debugging, and environment interaction must work together. Lemur is motivated by the hypothesis that agent backbones should balance natural-language reasoning with programming competence instead of optimizing either dimension in isolation.

## Method

- **Base model**: initialize from `Llama-2-70B` and adapt it into `Lemur-70B`, then instruction-tune it into `Lemur-70B-Chat`.
- **Continual pretraining corpus**: train on `90B` effective tokens with a `10:1` code-to-text ratio, i.e. `90.9%` code and `9.1%` text.
- **Code mixture**: build the code corpus mainly from The Stack, with Python dominant at `72.73%` of code tokens (`65.46B` effective tokens); smaller slices include SQL (`4.63B`), shell (`1.64B`), notebooks (`1.54B`), JavaScript (`1.52B`), C (`1.44B`), PHP (`1.19B`), and C++ (`1.09B`).
- **Text mixture**: use RefinedWeb, RedPajama/CommonCrawl-derived text, Wikipedia, Books, ArXiv, StackExchange, and DM Mathematics; perform extensive deduplication after aggregation.
- **Pretraining objective and optimization**: train on a `TPUv4-512` pod with sequence packing, batch size `4M` tokens, Adam with peak learning rate `4e-5`, `beta1 = 0.9`, `beta2 = 0.95`, gradient clipping at `1.0`, and cosine decay after `2000` warmup steps down to `10%` of the peak learning rate.
- **Instruction fine-tuning data**: build Lemur-Chat from about `300K` mixed examples from OpenAssistant 1 (`34,546`), OpenOrca (`200,000`), ShareGPT and ChatLogs (`81,319`), and Evol-CodeAlpaca (`51,952` before the paper's later cleaning summary).
- **Instruction fine-tuning hyperparameters**: train for `2` epochs with Adam, learning rate `2e-5`, and batch size `128`.
- **Evaluation design**: evaluate both fundamental capabilities and agent capabilities. Text/code evaluation spans `8` benchmarks: MMLU, BBH, GSM8K, HumanEval, MBPP, Spider, MultiPL-E, and DS-1000.
- **Agent evaluation axes**: measure tool augmentation, environment-feedback self-debugging, natural-language feedback following, and exploration in partially observable environments such as InterCode-CTF, WebArena, and ALFWorld.
- **Action representation study**: in WebArena, compare direct symbolic action prediction against a Python-style intermediate representation such as `type(id: int, content: str, press_enter_after: bool)`, then deterministically map the Python form back to executable actions.

## Key Results

- On the 8-way text/code average in Table 3, `Lemur-70B` reaches `47.9` vs `43.6` for `Llama-2-70B` (`+4.3`), and `Lemur-70B-Chat` reaches `55.0` vs `40.2` for `Llama-2-70B-Chat` (`+14.8`).
- On tool-augmented reasoning (Table 4), `Lemur-70B-Chat` achieves `31.65` micro-average, outperforming `Llama-2-70B-Chat` (`20.25`) and `CodeLlama-34B-INST` (`14.87`).
- On environment-feedback benchmarks (Table 5), `Lemur-70B-Chat` scores `46.67` on M-HumanEval, `17.58` on M-MBPP, `73.79` on IC-SQL, and `75.68` on RoboCodeGen, consistently ahead of the two open baselines.
- On natural-language feedback following (Table 6), `Lemur-70B-Chat` improves from `30.31` to `38.50` micro-average with GPT-4 teacher feedback, for `Delta_feedback = 8.19`, larger than `5.31` for `Llama-2-70B-Chat` and `4.20` for `CodeLlama-34B-INST`.
- On partially observable environments (Table 7), `Lemur-70B-Chat` scores `22.00` on InterCode-CTF, `5.30` on WebArena, and `59.70` on ALFWorld; the prose later reports `5.79` for WebArena, so the table value is the more reliable reference.
- The open-source gap remains substantial: GPT-4 still reports `66.77` on tool augmentation micro-average and `84.33` on ALFWorld, indicating that Lemur narrows but does not close the agent-performance gap.

## Limitations

- The method is demonstrated on a single very large backbone (`70B`), so it does not establish whether the same text-code harmonization recipe transfers cleanly to smaller models.
- The code-to-text ratio `10:1` is motivated empirically, but the paper explicitly notes that it does not provide a systematic scaling-law-style study of optimal mixture ratios because of compute limits.
- Absolute agent performance remains modest on several hard settings, especially WebArena (`5.30`) and InterCode-CTF (`22.00`), which suggests that harmonization alone is insufficient for robust real-world agents.
- GPT-4 remains clearly stronger across most agent benchmarks, so the approach improves open models without reaching proprietary frontier performance.
- The evaluation is benchmark-centric and mostly English-centric; the paper does not deeply study safety, long-horizon deployment reliability, or cross-domain generalization beyond the selected tasks.

## Concepts Extracted

- [[large-language-model]]
- [[language-agent]]
- [[continual-pretraining]]
- [[instruction-tuning]]
- [[code-language-model]]
- [[tool-augmentation]]
- [[execution-feedback]]
- [[partial-observability]]
- [[data-deduplication]]
- [[code-execution]]
- [[intermediate-representation]]

## Entities Extracted

- [[yiheng-xu]]
- [[hongjin-su]]
- [[chen-xing]]
- [[boyu-mi]]
- [[qian-liu]]
- [[weijia-shi]]
- [[binyuan-hui]]
- [[fan-zhou]]
- [[yitao-liu]]
- [[tianbao-xie]]
- [[zhoujun-cheng]]
- [[siheng-zhao]]
- [[lingpeng-kong]]
- [[bailin-wang]]
- [[caiming-xiong]]
- [[tao-yu]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
