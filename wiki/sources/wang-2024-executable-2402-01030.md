---
type: source
subtype: paper
title: Executable Code Actions Elicit Better LLM Agents
slug: wang-2024-executable-2402-01030
date: 2026-04-20
language: en
tags: [agents, llm, code, tool-use, instruction-tuning]
processed: true

raw_file: raw/papers/wang-2024-executable-2402-01030/paper.pdf
raw_md: raw/papers/wang-2024-executable-2402-01030/paper.md
bibtex_file: raw/papers/wang-2024-executable-2402-01030/paper.bib
possibly_outdated: false

authors:
  - Xingyao Wang
  - Yangyi Chen
  - Lifan Yuan
  - Yizhe Zhang
  - Yunzhu Li
  - Hao Peng
  - Heng Ji
year: 2024
venue: ICML 2024
venue_type: conference
arxiv_id: 2402.01030
doi:
url: http://arxiv.org/abs/2402.01030
citation_key: wang2024executable
paper_type: method

read_status: unread

domain: agents
---

## Summary

The paper argues that LLM agents should emit executable Python code rather than text or JSON tool calls, turning actions into a unified interface that can reuse variables, compose multiple tools, and react to execution feedback across turns. The authors evaluate `17` models on API-Bank and on their new `82`-task M3ToolEval benchmark, showing that CodeAct is usually the strongest or tied-strongest action format; the best closed model (`gpt-4-1106-preview`) reaches `74.4%` success with `5.5` average turns on M3ToolEval. They then build `CodeActInstruct`, a `7,139`-trajectory instruction-tuning corpus spanning information seeking, code generation, tabular reasoning, external memory, and robot planning, and train `CodeActAgent` on Llama-2 and Mistral backbones. The Mistral-based agent improves substantially on agent tasks while preserving general capabilities.

## Problem & Motivation

Prior LLM agents usually express actions as text strings or JSON objects with a fixed schema. That design works for simple tool invocation, but it narrows the action space to predefined tools and makes control flow, data flow, and multi-tool composition awkward or impossible. The paper's core motivation is that modern LLMs already see large amounts of code during pretraining, so executable code may be a more natural action language than specialized text or JSON formats. Using Python code also exposes mature software libraries and automated feedback such as tracebacks, which could let agents solve harder tasks and improve from interaction rather than only from one-shot action prediction.

## Method

- **General interaction model**: the paper formalizes three roles, user, agent, and environment. For agent-environment interaction, CodeAct emits Python code actions `a_t`, executes them in an interpreter, and feeds execution outputs or errors back as the next observation.
- **Unified action space**: CodeAct replaces task-specific text/JSON schemas with executable Python, so one action can express variable reuse, tool composition, `if` branches, and `for` loops without defining new tool wrappers.
- **Atomic tool-use study**: on API-Bank level-1 tasks, each model must emit exactly one tool call in one of three formats: Python function call, JSON object, or text expression. Correctness is measured by matching execution outputs to ground truth.
- **Complex benchmark**: the authors introduce M3ToolEval with `82` human-curated tasks spanning web browsing, finance, travel planning, science, and information processing. Each interaction is capped at `10` turns, and answers are scored by exact match.
- **CodeActInstruct data construction**: the training mixture covers information seeking (HotpotQA), software-package use (APPS and MATH), external memory (WikiTableQuestion with `sqlite3`/`pandas` variants), and robot planning (ALFWorld). The released mixture contains `7,139` trajectories and `10,581,681` tokens.
- **Trajectory filtering**: the data selection process keeps trajectories that demonstrate improvement from interaction, especially cases where the model first fails, observes execution feedback, and later repairs the solution through self-debugging.
- **Teacher model mix**: trajectory generation uses `gpt-3.5-turbo-0613`, `gpt-3.5-turbo-0613-16k`, `claude-1-instant`, `claude-2`, and `gpt-4-0613`; the final curated set keeps `411` GPT-4 trajectories and `6,728` GPT-3.5/Claude trajectories.
- **CodeActAgent training**: the authors perform full-parameter supervised fine-tuning on Llama-2 `7B` with sequence length `4,096` and Mistral `7B` with sequence length `16,384`, mixing CodeActInstruct with general conversation data.
- **Evaluation suite**: trained agents are tested on MINT with interaction budget `k = 5`, on out-of-domain text-action tasks from MiniWob++ and ScienceWorld, and on generic LLM tasks including MMLU, HumanEval, GSM8K, and MTBench.

## Key Results

- On API-Bank atomic tool calls, CodeAct is the best-performing format for `8/17` models overall. For example, Llama-2-13B reaches `38.1%` with CodeAct versus `37.3%` with text and `8.5%` with JSON; Claude-2 reaches `76.7%` with CodeAct versus `73.7%` text and `59.4%` JSON.
- On M3ToolEval, CodeAct yields the highest success rate for `12/17` models and the fewest average turns for `12/17` models.
- The strongest reported M3ToolEval result is `gpt-4-1106-preview` with `74.4%` success and `5.5` average turns under CodeAct, compared with `53.7%` and `7.7` turns for text, and `52.4%` and `7.6` turns for JSON.
- The paper reports a large open/closed gap on complex agent tasks: the best open-source model reaches only `13.4%` success on M3ToolEval, while the best closed-source model reaches `74.4%`.
- CodeActInstruct contains `1,664` HotpotQA, `1,732` MATH, `647` APPS, `1,065` WikiTableQuestion, and `2,031` ALFWorld trajectories, totaling `7,139` instances and `10.58M` tokens.
- Relative to prior same-backbone agent-tuning work, the paper states that CodeActInstruct yields `24%` and `119%` relative improvement over AgentLM and FireAct.
- CodeActAgent (Mistral `7B`) reaches `57.4` on MINT in-domain, `32.4` on MINT out-of-domain, `12.2` on M3ToolEval, `46.2` on MiniWob++, `59.1` on MMLU, `34.7` on HumanEval, `58.0` on GSM8K, and `8.2` on MTBench.

## Limitations

- Absolute performance on realistic agent tasks is still limited for open models; even the best open-source model reaches only `13.4%` success on M3ToolEval.
- The Llama-2-based CodeActAgent does not improve on M3ToolEval (`0.0%`), indicating strong backbone sensitivity and possible training artifacts.
- CodeActAgent remains an early prototype and can hallucinate execution state, for example by assuming variable contents without explicitly printing or checking them.
- The training domains overlap part of MINT (for example ALFWorld and MATH), so the paper has to separate in-domain and out-of-domain results; broader generalization is still only partially established.
- Letting an agent execute arbitrary code raises safety concerns beyond benchmark accuracy, including possible sandbox-break or cyber-attack scenarios if safeguards are weak.

## Concepts Extracted

- [[large-language-model]]
- [[tool-use]]
- [[tool-augmented-agent]]
- [[code-as-actions]]
- [[action-space]]
- [[multi-turn-interaction]]
- [[agent-environment-interaction]]
- [[instruction-tuning]]
- [[self-debugging]]
- [[external-memory]]

## Entities Extracted

- [[xingyao-wang]]
- [[yangyi-chen]]
- [[lifan-yuan]]
- [[yizhe-zhang]]
- [[yunzhu-li]]
- [[hao-peng]]
- [[heng-ji]]
- [[codeact]]
- [[codeactinstruct]]
- [[codeactagent]]
- [[api-bank]]
- [[m3tooleval]]
- [[mint]]
- [[llama-2]]
- [[mistral-7b]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
