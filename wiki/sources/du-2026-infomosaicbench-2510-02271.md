---
type: source
subtype: paper
title: "InfoMosaic-Bench: Evaluating Multi-Source Information Seeking in Tool-Augmented Agents"
slug: du-2026-infomosaicbench-2510-02271
date: 2026-04-20
language: en
tags: [agents, benchmark, mcp, information-seeking, tool-use]
processed: true

raw_file: raw/papers/du-2026-infomosaicbench-2510-02271/paper.pdf
raw_md: raw/papers/du-2026-infomosaicbench-2510-02271/paper.md
bibtex_file: raw/papers/du-2026-infomosaicbench-2510-02271/paper.bib
possibly_outdated: false

authors:
  - Yaxin Du
  - Yuanshuo Zhang
  - Xiyuan Yang
  - Yifan Zhou
  - Cheng Wang
  - Gongyi Zou
  - Xianghe Pang
  - Wenhao Wang
  - Menglan Chen
  - Shuo Tang
  - Zhiyu Li
  - Feiyu Xiong
  - Siheng Chen
year: 2026
venue: arXiv
venue_type: preprint
arxiv_id: 2510.02271
doi:
url: https://arxiv.org/abs/2510.02271
citation_key: du2026infomosaicbench
paper_type: benchmark

read_status: unread
read_date:
rating:

domain: agents
---

## Summary

InfoMosaic-Bench introduces a benchmark for evaluating whether tool-augmented LLM agents can solve information-seeking tasks that genuinely require combining open-web evidence with domain-specific tools. The dataset contains `621` synthesized problems across six settings (medical/biology, finance, maps, video, web, and multi-domain) and exposes agents to `77` MCP tools spanning `7` servers. Its core contribution is InfoMosaic-Flow, a two-stage pipeline that first grounds candidate tasks in verified tool outputs and then iteratively removes shortcut conditions that make them solvable by trivial web search. Across `14` state-of-the-art agents, the paper shows that web-only agents remain weak on multi-source reasoning, while domain tools provide selective but inconsistent gains because present-day agents still struggle with robust tool selection and usage.

## Problem & Motivation

The paper argues that current information-seeking agents over-rely on open-web search, which is noisy, incomplete, and often insufficient for high-stakes or domain-specific questions. At the same time, the rise of MCP-compatible tools makes it possible to access structured sources such as biomedical databases, financial feeds, maps, and video metadata, but it is unclear whether LLM agents can actually coordinate those sources effectively. Existing benchmarks either test web browsing in isolation or isolated API execution correctness; they do not directly measure whether an agent can seek, combine, and reason over heterogeneous evidence sources in a single task.

## Method

- **Benchmark scope**: InfoMosaic-Bench contains `621` tasks across `6` domains and `77` tools from `7` servers, with condition-level gold labels and tool-call traces for both final-answer and diagnostic evaluation.
- **Task formalization**: each instance is written as `` `τ = (q, T_avail, K, GT)` ``, where `q` is the query, `T_avail` is the available tool set, `K` is the tool-call budget, and `GT` is the ground-truth answer; correctness is evaluated by `` `E(H, τ) -> {0, 1}` `` over the interaction history `H`.
- **Agent framework**: the benchmarked agent uses a ReAct-style setup with OpenAI-style function calling, tool metadata serialized as JSON Schema, and a Python sandbox that executes translated tool calls; the max budget is `K = 20` tool invocations.
- **InfoMosaic-Flow Stage 1**: an organizer-worker pipeline synthesizes tasks. The organizer proposes scenarios from seed sources (e.g. Wikipedia, Baidu Baike, Qunar, NCI IDs), decomposes them into subtasks, and dispatches them as `` `executor(subtask, domain)` `` to a domain worker that gathers verifiable evidence from the domain toolset.
- **Cross-source composition**: the organizer integrates returned evidence into a multi-condition question whose answer depends on combining multiple tool outputs rather than a single lookup.
- **InfoMosaic-Flow Stage 2**: a web-only verifier stress-tests the draft through three substeps: condition decomposing, condition fuzzing, and concluding. The draft is revised until `` `(i) web-search-only fails` `` and `` `(ii) no single condition determines the answer` ``.
- **Quality control**: automatic filters remove under-constrained or incoherent items via minimum tool-call filtering, answer-evidence consistency checks, and coherence filtering; human annotators then revise or discard remaining problematic items.
- **Ablation signals**: removing the executor reduces average tool calls from `59.1` to `7.8` and unique tools from `43.1` to `5.6`, while skipping stage-2 web verification inflates shortcut-prone accuracy to `45.1%`.

## Key Results

- Dataset composition: `621` problems, `77` MCP tools, `7` servers, and `6` domains/settings.
- Web-only performance is weak: GPT-5 is the strongest web-only agent at `38.18%` accuracy and `67.48%` pass rate; the best open model reported is GLM-4.5 at `20.61%` accuracy.
- Domain tools help selectively rather than uniformly: for GPT-5, domain-tool access yields `+7.41` points on Map and `+10.00` on Video, but `-9.73` on Medical/Biology and `-9.00` on Finance.
- Tool handling remains a major bottleneck: `22.4%` of failures are attributed to incorrect tool usage or tool selection.
- More tool calls help only up to a point: performance generally improves before plateauing after roughly `8` tool calls; effective tool-call capacity correlates with overall accuracy at `R^2 = 0.57`.
- Human validation is strong: three annotators reviewed `120` sampled items and reported high agreement with `Cohen's kappa = 0.92`.
- Failure analysis shows GPT-5 web-only errors are dominated by Retrieval Miss (`39.6%`) and Overgeneralization (`28.2%`), indicating retrieval and evidence selection failures more than purely final-step reasoning errors.

## Limitations

- The benchmark is synthesized through an agentic pipeline rather than collected from natural user logs, so ecological validity for organic research workflows remains uncertain.
- Reported agent results are tied to one ReAct-style framework, a JSON-schema tool interface, and a fixed `K = 20` budget, so some failures may reflect framework constraints as much as model limitations.
- Coverage is restricted to `77` tools across six domains; the conclusions may not fully transfer to other MCP ecosystems, modalities, or interactive environments.
- Final-answer evaluation is not exact-match-only and uses an LLM judge for semantic alignment, which introduces some judge-dependent variance.

## Concepts Extracted

- [[tool-augmented-agent]]
- [[large-language-model]]
- [[multi-source-information-seeking]]
- [[model-context-protocol]]
- [[domain-specific-tool]]
- [[web-search]]
- [[organizer-worker-architecture]]
- [[iterative-refinement]]
- [[benchmark-dataset]]
- [[agentic-data-synthesis]]

## Entities Extracted

- [[yaxin-du]]
- [[yuanshuo-zhang]]
- [[xiyuan-yang]]
- [[yifan-zhou]]
- [[cheng-wang]]
- [[gongyi-zou]]
- [[xianghe-pang]]
- [[wenhao-wang]]
- [[menglan-chen]]
- [[shuo-tang]]
- [[zhiyu-li]]
- [[feiyu-xiong]]
- [[siheng-chen]]

## Contradictions

<!-- None yet; first source on these concepts. -->

## My Notes

<!-- LLM never overwrites this section. Only appends timestamped blocks via ADD-NOTE operation. -->
