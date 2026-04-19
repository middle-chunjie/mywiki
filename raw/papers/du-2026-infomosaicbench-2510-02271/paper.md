InfoMosaic-Bench: Evaluating Multi-Source  Information Seeking in Tool-Augmented Agents
==========================================================================================

Yaxin Du1 Yuanshuo Zhang1 Xiyuan Yang1 Yifan Zhou2 Cheng Wang1  
Gongyi Zou4 Xianghe Pang1 Wenhao Wang3 Menglan Chen1 Shuo Tang1  
Zhiyu Li5 Feiyu Xiong5 Siheng Chen1,5  
1Shanghai Jiao Tong University2The Chinese University of Hong Kong3Zhejiang University  
4University of Oxford5MemTensor (Shanghai) Technology Co., Ltd  
Corresponding Author: sihengc@sjtu.edu.cn

###### Abstract

Information seeking is a fundamental requirement for humans. However, existing LLM agents rely heavily on open-web search, which exposes two fundamental weaknesses: online content is noisy and unreliable, and many real-world tasks require precise, domain-specific knowledge unavailable from the web. The emergence of the Model Context Protocol (MCP) now allows agents to interface with thousands of specialized tools, seemingly resolving this limitation. Yet it remains unclear whether agents can effectively leverage such tools—and more importantly, whether they can integrate them with general-purpose search to solve complex tasks.
Therefore, we introduce InfoMosaic-Bench, the first benchmark dedicated to multi-source information seeking in tool-augmented agents. Covering six representative domains (medicine, finance, maps, video, web, and multi-domain integration), InfoMosaic-Bench requires agents to combine general-purpose search with domain-specific tools. Tasks are synthesized with InfoMosaic-Flow, a scalable pipeline that grounds task conditions in verified tool outputs, enforces cross-source dependencies, and filters out shortcut cases solvable by trivial lookup. This design guarantees both reliability and non-triviality.
Experiments with 14 state-of-the-art LLM agents reveal three findings: (i) web information alone is insufficient, with GPT-5 achieving only 38.2% accuracy and 67.5% pass rate; (ii) domain tools provide selective but inconsistent benefits, improving some domains while degrading others; and (iii) 22.4% of failures arise from incorrect tool usage or selection, highlighting that current LLMs still struggle with even basic tool handling.

[https://github.com/DorothyDUUU/Info-Mosaic](https://github.com/DorothyDUUU/Info-Mosaic "")

[<img src='figs/hf_logo.png' alt='[Uncaptioned image]' title='' width='12' height='12' /> https://huggingface.co/datasets/Dorothydu/InfoMosaic_Bench](https://huggingface.co/datasets/Dorothydu/InfoMosaic_Bench "")

1 Introduction
--------------

<img src='x1.png' alt='Refer to caption' title='' width='830' height='467' />

*Figure 1: Overview of InfoMosaic-Bench. The benchmark evaluates multi-source information seeking in tool-augmented agents. (Left) Example query illustrating that single-source web search often fails, while multi-source tool use is required. (Center) Dataset statistics, including 621 samples across six domains, 77 MCP tools, and 14 models (7 closed- and 7 open-sourced). (Right) Radar plot showing domain-wise accuracy across models and the pie chart illustrating sample distribution across domains.*

Access to high-quality information is the fundamental driver of enhanced cognition, optimized decision-making, innovation, and societal progress. Each major advance in intelligent systems has been closely tied to progress in how they acquire and organize information: the advent of PageRank search engines*(Page et al., [1999])* made the web navigable at scale; the breakthrough of large language models (LLMs)*(Brown et al., [2020]; OpenAI, [2023]; Kaplan et al., [2020])* shifted information access from explicit retrieval to leveraging vast pre-trained knowledge; and most recently, web-search-augmented LLM agents*(Nakano et al., [2021])*, such as various deep research product*(OpenAI, [2025]; Perplexity AI, [2025]; Google, [2025])*, have transformed information seeking into an iterative process of querying, browsing, and synthesizing evidence. Already in wide use, these agents are now becoming indispensable, powering high-frequency workflows in science*(Chai et al., [2025])*, business, and everyday decision-making*(Shen, [2024])*.

Despite their growing adoption, today’s agents remain fundamentally limited by their heavy reliance on open-web search. Online content is noisy, inconsistently formatted, and often unreliable*(Wenzek et al., [2020]; Vosoughi et al., [2018])*, making it insufficient for high-stakes applications. More importantly, many real-world tasks require precise, verifiable, and domain-specific knowledge that web search simply cannot provide. For example, a financial analyst risks misleading conclusions without structured access to corporate filings and market data*(Loughran \& McDonald, [2011])*; a medical assistant cannot ensure patient safety without curated drug–side effect databases*(U.S. Food and Drug Administration, [2022])*; and even route planning requires geospatial applications and transport schedules that cannot be recovered from fragmented web pages. These scenarios reveal a deeper challenge: general-purpose web search is not enough—reliable agents must integrate both general web information and specialized, domain-specific sources.

In parallel, the ecosystem of information-seeking tools is expanding rapidly. With the advent of the Model Context Protocol (MCP) tools*(Hou et al., [2025])*, LLM agents can access thousands of heterogeneous data sources, ranging from biomedical databases*(Flotho et al., [2025])* to financial feeds*(Zeng, [2025])* and mapping services. Such advancements substantially enrich agent–environment interaction and broaden their information-seeking potential. While this shift appears to overcome the limitations of relying solely on general-purpose web search, it also raises two critical open questions: (1) How effectively can LLM agents leverage domain-specific tools to access information within each individual field? (2) More importantly, can they seamlessly integrate general-purpose search with multiple specialized tools to tackle complex, multi-source information-seeking tasks?

To answer these questions, we propose InfoMosaic-Bench, the first benchmark dedicated to evaluating the ability of LLM agents to perform multi-source information search using external tools. InfoMosaic-Bench comprises 621 synthesized tasks and 77 tools across six domains—medical/biology, finance, maps, video, web and multi-source seeking. This benchmark directly targets the two open challenges identified above and enables evaluation of both domain-specific tool usage and the harder setting of seeking of multi-source information. Unlike existing benchmarks, which either focus on generic web search in single source with single tool (like BrowseComp*(Wei et al., [2025])* and WebWalkerQA*(Wu et al., [2025])*) or correctness of isolated tool calls ($\tau$-Bench*(Yao et al., [2024])*, MCP-Bench*(Wang et al., [2025])*), InfoMosaic-Bench uniquely evaluates agents’ ability to solve multi-source information-seeking tasks using contemporary and domain-specific MCP tools, with verified outputs ensuring reliability and non-triviality.

A key challenge in constructing such a benchmark is how to design tasks that inherently require multi-source search, rather than being solvable by a single tool or trivial web lookup.
In practice, human curation has two limitations: no single author has broad cross-domain expertise, and crafting coherent multi-source tasks demands dozens of iterative tool calls, which is rarely sustainable by hand.
Therefore, we propose InfoMosaic-Flow, an agentic data synthesis pipeline for multi-source information seeking task. The key idea is to leverage an organizer–workers architecture, where a single organizer acts as the commander, coordinating multiple domain-specific workers to enable scalable, cross-tool data synthesis. The organizer handles high-level planning, while each worker, tied to a particular domain, executes assigned tasks with its tools and returns precise results. This design enables integrative use of domain tools while maintaining robust reasoning, producing coherent and cross-tool grounded outputs. At last, we enforce multi-stage quality control combining automatic and carefully guided manual checks to guarantee reliability and difficulty.

After conducting extensive experiments, the results reveal three key findings: (1) Web information alone is insufficient for precise domain reasoning: even GPT-5 attains only 38.2% accuracy, showing that open-web search cannot meet the information needs of domain-specific tasks. (2) Domain tools offer selective but limited benefits: they improve performance in Map and Video but degrade in Medical, Finance, and Multi-domain, indicating that current agents are still far from being able to effectively exploit domain-specific tools within each field. (3) Many failures come from incorrect domain-tool usage and selection: nearly 22.4% of failures come from wrong tool usage and tool selection, demonstrating that agents lack the competence to reliably in even basic tool handling.

Looking forward, the benchmark exposes a fundamental gap: today’s models excel at web search yet remain unable to reliably exploit domain tools or combine them effectively. Closing this gap is not a minor improvement, but a prerequisite for deploying trustworthy agents in high-stakes domains such as medicine, finance, and scientific discovery.

Our contributions are as follows:

* •

    We identify the challenge that reliance on general-purpose web search is inadequate, and that no benchmark evaluates whether agents can leverage diverse domain-specific tools for reliable information seeking. We propose InfoMosaic-Bench to fill this gap.

* •

    We propose InfoMosaic-Flow, an automated two-stage synthesis pipeline that grounds tasks in domain-wise tool evidence and refines them with web-based verification.

* •

    Experiments show that relying on web search alone is insufficient for precise reasoning, while domain-specific tools can unlock additional capabilities, but current agents fail to robustly use them, leading to selective and inconsistent gains.

2 Related Work
--------------

### 2.1 Tool-Using LLMs

Early work on tool-augmented reasoning explored how to disentangle internal reasoning from external actions. ReAct*(Yao et al., [2023a])* pioneered this idea by interleaving chain-of-thought with explicit tool calls (e.g., search, calculator), enabling models to iteratively refine answers with external evidence. Building on this, Toolformer*(Schick et al., [2023])* showed that LLMs can self-supervise when and how to call APIs, while systems such as ToolLLM*(Qin et al., [2023])* and EasyTool*(Yuan et al., [2024])* scaled the breadth of API coverage and improved robustness of invocation. As LLM capabilities advanced, the focus shifted from invoking single APIs to handling long-horizon search and orchestration.Works such as Search-o1*(Li et al., [2025b])*, WebThinker*(Li et al., [2025c])*, and R1-Searcher*(Song et al., [2025])* focus on persistent retrieval and orchestration in web search, highlighting the strengths and limitations of single-channel search-augmented reasoning. In contrast, the introduction of the Model Context Protocol (MCP)*(Hou et al., [2025])* expands tool use from web-only retrieval to a broad ecosystem of heterogeneous domain-specific tools, raising the new challenge of coordinating and integrating evidence across multiple sources. Our work targets precisely this gap, proposing a benchmark dedicated to evaluating multi-source information seeking in tool-augmented agents.

### 2.2 Benchmarks for Tool-Using Agents

There are three parallel lines of work for benchmark tool-augmented LLMs:
1) API-centric benchmarks. ToolBench*(Qin et al., [2023])* and related datasets*([Patil et al.,] ; Yao et al., [2024]; Chen et al., [2025])* evaluate an agent’s ability to discover, select, and call APIs correctly. These efforts provide valuable coverage of API functionality and invocation robustness, but the evaluation typically centers on single-tool correctness rather than multi-source synthesis.
2) Web/search-oriented benchmarks. Datasets such as BrowseComp*(Wei et al., [2025])*, WebWalkerQA*(Wu et al., [2025])*, and MM-BrowseComp*(Li et al., [2025a])* evaluate agent’s ability to engage with the open web, combining capabilities such as query reformulation, long-horizon reasoning, and information extraction from complex webpages. These benchmarks highlight the reasoning dimension of search-augmented agents and have advanced our understanding of how models operate in noisy and partially observable environments. However, the scope of tool use remains narrow: agents are restricted to web search and browsing, without evaluating whether they can coordinate multiple types of tools or integrate evidence beyond a single retrieval channel.
3) MCP-style tool suites. More recently, benchmarks have emerged around the MCP ecosystem, including MCP-Universe*(Luo et al., [2025])*, MCP-Radar*(Gao et al., [2025])*, MCP-Zero*(Fei et al., [2025])*, and MCP-Bench*(Wang et al., [2025])*. These benchmarks expose agents to large-scale, heterogeneous tool environments and focus on aspects such as tool invocation correctness, execution robustness under complex tool spaces, or zero-shot tool discovery. However, they generally stop short of evaluating information seeking and long-horizon reasoning across tools. In short, none of them systematically evaluate whether LLM agents can reliably seek, combine, and reason over heterogeneous evidence sources. InfoMosaic-Bench fills this gap, providing a benchmark where every task is grounded in tool evidence and demands genuine multi-source reasoning.

3 Methodology
-------------

To construct a benchmark that reliably evaluates the ability of LLM agents to integrate evidence across multiple heterogeneous tools, we propose InfoMosaic-Flow, a scalable synthesis pipeline that generates tasks requiring non-trivial multi-source reasoning. This section introduces the overall design in Sec.[3.1] and describes our quality control procedure in Sec.[3.2].

<img src='x2.png' alt='Refer to caption' title='' width='747' height='324' />

*Figure 2: Overview of InfoMosaic-Flow. The synthesis pipeline is laid on an organizer–workers architecture, where a single organizer acts as the commander, coordinating multiple domain-specific workers. Stage 1: Information Seeking composing interdependent constraints and grounding them with verified multi-tool outputs to form initial QA pairs; Stage 2: Iterative Refinement revising drafts, pruning shortcuts, and enforcing multi-source reasoning.*

### 3.1 Dataset Construction

Fig.[2] shows the overall generation with organizer-worker system. Specifically, an organizer is responsible for reasoning and formulating constraints/verification, while a worker is activated as a tool-calling event and follows instructions from the organizer to perform continuous tool calls and return consolidated evidence. This dual-agent designation has two advantages: (1) it isolates execution, preserving reasoning depth. The organizer handles decomposition or constraint reasoning; the worker handles fine-grained tool calls and evidence consolidation. This functional separation mitigates execution-induced noise in multi-step inference and reduces retrofitting of constraints to available tools.
(2) It also expands exploration, improves multi-source coupling. The organizer remains tool-agnostic and selects only the target domain, while the executor freely chooses among all tools within that domain to satisfy the plan. This decoupling turns each subtask into a combinatorial search over the domain toolset, increasing tool diversity and coverage (see App.[A.6]).

Overall, the pipeline proceeds in two stages: (i) Information Seeking, the synthesizer composes interdependent constraints and the executor grounds them with verified outputs from multiple tools to form the initial QA pair; and (ii) Iterative Refinement, where drafts are repeatedly challenged and revised to prune single-source shortcuts, leaving only tasks that genuinely require multi-source reasoning. Fig.[2] illustrates the multi-domain instantiation: the other domains generation process simply by eliminating executor toolsets with medical/biology, finance, maps, video, and web, respectively.

#### 3.1.1 Stage 1: Information Seeking

This stage takes seed data as input and generates a coherent multi-condition problem grounded in domain-wise evidence. The core idea of this stage is to use seed data to propose various scenarios that could diversify domain-wise tool callings to include different domain-specific information for problem synthesis. In addition, instead of relying on static templates or noisy web content, we actively query specialized tools and compose their retrieved information into tasks. This design guarantees that every problem is grounded in verifiable evidence and requires reasoning across multiple sources.

In this stage, the organizer in the agentic system is the synthesizer, and the worker is the executor with respective domain tools. The process of synthesizer works in the following steps:
(1) Scenario Proposing. Starting from various seeds (e.g., Wikipedia or Baidu Baike, Qunar web, NCI IDs), the synthesizer proposes candidate scenarios that guide the construction of problems. This helps to provide diverse contexts that naturally invoke heterogeneous tools, increasing diversity beyond narrow or contrived tool-calling flows.
(2) Domain Information Gathering. The synthesizer reasons step by step and emits high-level instructions (subtasks) that trigger tool calls, i.e. ‘executor(subtask, domain)‘. The executor, equipped with the domain toolset, selects and composes tools to retrieve verifiable domain-specific facts and returns organized evidence. The synthesizer consumes this evidence, updates the plan, and issues the next instruction.
(3) Integrating. Finally, the synthesizer organizes the validated tool results into a coherent multi-sourced problem which requires multiple tool calls and cross-condition reasoning.

Overall, this design has two benefits. (i) First, by hiding tool internals, the synthesizer focuses on maintaining coherence and naturalness in the problem statement, rather than overfitting to tool quirks. (ii) Second, the information gathering loop enlarges the exploration space and includes diverse tools. As a result, Stage 1 ensures the generated problems are both coherent and inherently multi-source.

#### 3.1.2 Stage 2: Iterative Refinement

While Stage 1 ensures that every synthesized problem is executable, the resulting tasks may still be trivial in practice. In particular, some problems can be answered by satisfying only a single clue, or even by issuing a generic web query without invoking multiple tools. Such cases do not reflect the real challenges faced by agentic systems, where reliable reasoning requires integrating evidence from multiple heterogeneous sources.

To eliminate these trivial cases, we introduce an iterative refinement stage. In this stage, the worker is a Verifier with only web-search tool, attempting to answer the Refiner-assigned problem through web search. The refinement proceeds in three steps:
(i) Condition Decomposing: the Refiner breaks down the synthesized problem into individual conditions and asks the Verifier to solve them independently;
(ii) Condition Fuzzing: if any condition proves too revealing (e.g., directly exposing the answer via a single search), the Refiner rewrites, augments, or combines it with others to reduce shortcut solutions;
(iii) Concluding: once no condition can independently yield the answer and the Verifier fails to solve the task via search alone, the refined set of conditions is recomposed into the final question.

The refinement process is repeated until two criteria are simultaneously met: (i) the problem cannot be solved by web search alone, and (ii) no single condition is sufficient to determine the answer. This ensures that each admitted question is both challenging and robust, demanding genuine multi-source reasoning. This refinement stage guarantees difficulty, since trivial shortcuts are removed and the remaining tasks genuinely require reasoning over multiple sources, which ensures that InfoMosaic-Bench provides a robust testbed for evaluating multi-source information seeking.

### 3.2 Quality Control

To ensure the reliability of InfoMosaic-Bench, we adopt a serious of quality control processes. We first apply two automated checks, including Tool-Call Filtering, Answer–Evidence Consistency, and Coherence Filtering, to remove trivial, noisy, or ill-formed tasks. After automatic filtering, we further conduct manual screening and revision by human annotators, who correct or discard problematic cases to improve factual alignment, coherence, and difficulty.

The automatic quality checks are as follows: (1)Tool-Call Filtering. We first enforce a minimum tool-call threshold in Stage 1, discarding samples below to eliminate under-constrained, low-seeking tasks and keep the benchmark focused on non-trivial multi-source reasoning. (2) Answer–Evidence Consistency. To guarantee traceability, we retain only items whose final answers are exactly derivable from collected tool outputs, ensuring every sample is grounded in verifiable information sources.
(3) Coherence Filtering. We further remove tasks that exhibit incoherent or ill-formed conditions, such as contradictory constraints or unnatural phrasing, to guarantee that each problem maintains semantic coherence across its question, intermediate conditions, and final answer.

Human Selection and Refinement. After automatic filtering, we further review a sample for consistency, coherence, and difficulty (App.[A.10.1]); problematic items are revised or discarded. A dedicated user study (Sec.[5.4]) further confirms reliability for evaluating multi-source seeking.

4 Dataset
---------

*Table 1: Key Statistics of InfoMosaic-Bench.*

| Domain | | Sample | | --- | | Number | | | MCP Tool | | --- | | Number | |
| --- | --- | --- | --- | --- | --- | --- |
| Medical/Biology | 83 | 15 |
| Web | 100 | 2 |
| Video | 100 | 11 |
| Finance | 100 | 29 |
| Map | 135 | 20 |
| Multi-Domain | 103 | - |
| Total | 621 | 77 |

Statistics. Table[1] summarizes the composition of InfoMosaic-Bench. The dataset contains 621 problems across 5 domains (medicine/biology, finance, maps, video, web), plus an additional set of explicitly cross-domain tasks. Each problem is paired with condition-level gold labels and traces of tool calls, enabling both final-answer evaluation and fine-grained diagnostic analysis. In total, InfoMosaic-Bench incorporates 77 distinct tools spanning 7 servers, combined with condition-level supervision, ensuring that the benchmark provides a challenging and reliable testbed for evaluating multi-source information seeking. Detailed tool information are described in App.[A.8.2].

Evaluation Metrics. We report both Accuracy and Pass Rate. Accuracy measuring strict end-to-end task success, reflecting whether the agent can complete information seeking and reasoning holistically. Pass Rate, in contrast, evaluates provides a more fine-grained view of agent performance based on associated test cases (subquestions with gold answers or subgoal checks).

5 Experiments
-------------

### 5.1 Experimental Setup

Models Evaluated. In our experiments, we evaluate 7 closed-source and 7 open-source LLMs. Details of these LLMs can be found in Table [5].

Agent Framework. For the agent framework, we adapt the most popular framework ReAct*(Yao et al., [2023b])* equipped with OpenAI’s tool-calling interface*(OpenAI, [2025])* and a Python Sandbox*(Pang et al., [2025])* from which LLMs receive the tool execution results. More details can be found in Sec.[A.3.2].

Evaluation. As introduced in Sec. [4], we use Accuracy (Acc) and Pass Rate (PR) as our evaluation metrics. Instead of relying solely on exact match, we leverage an LLM to judge whether the predicted answers align with the references, which alleviates cases where semantically correct outputs cannot be captured by string matching. The detailed evaluation prompts are provided in the App.[A.10.2].

### 5.2 Main Results

InfoMosaic-Bench demonstrates that web search alone is insufficient for multi-source reasoning. Table[2] reports results for 14 state-of-the-art LLM agents limited to a web-search tool. We observe that:
(1) Current agent system performs poorly on this task. Even the best closed-source model (GPT-5) attains only 38.2% accuracy and a 67.5% pass rate. (2) Closed-source models exceed open-source ones by 15–20% on accuracy, yet both are constrained by web information.
(3) Pass rates consistently surpass exact accuracy, reflecting that agents often satisfy some conditions but fail to integrate all of them into a correct final answer.

*Table 2: Comparison of 14 LLM agents equipped with a web search tool on InfoMosaic-Bench, evaluated across six domains and the overall average. Metrics include Accuracy (Acc) and Pass Rate. The best overall Accuracy and Pass Rate is highlighted in bold.*

|  |  |  |  |  |  |  | Overall | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Map | | Medical/ | | --- | | Biology | | Video | Web | Finance | | Multi- | | --- | | domain | | Acc | Passrate |
| Close-Sourced Model | | | | | | | | |
| GPT-5 | 32.59 | 53.10 | 36.00 | 29.00 | 41.00 | 41.75 | 38.18 | 67.48 |
| o3 | 40.74 | 44.79 | 23.00 | 28.71 | 45.00 | 35.78 | 36.35 | 64.96 |
| Grok-4 | 9.63 | 39.02 | 33.00 | 10.00 | 43.88 | 19.42 | 25.42 | 39.44 |
| Claude-4.0-Sonnet | 17.04 | 20.48 | 18.00 | 3.00 | 27.00 | 10.68 | 15.94 | 36.47 |
| Qwen2.5-Max | 7.41 | 7.23 | 5.00 | 0.00 | 9.00 | 1.94 | 5.15 | 15.72 |
| Gemini-2.5-Flash | 11.11 | 10.84 | 3.00 | 2.00 | 9.00 | 6.82 | 7.25 | 28.63 |
| o4-mini | 24.44 | 25.30 | 24.00 | 8.00 | 39.00 | 24.27 | 24.15 | 61.67 |
| Open-Sourced Model | | | | | | | | |
| GLM-4.5 | 24.44 | 27.71 | 24.00 | 11.00 | 22.00 | 14.56 | 20.61 | 26.98 |
| Kimi-K2 | 14.81 | 19.28 | 8.00 | 1.00 | 18.00 | 0.00 | 10.14 | 39.72 |
| Qwen3-235B-A22B | 5.19 | 19.28 | 6.00 | 0.00 | 23.00 | 3.88 | 9.02 | 31.40 |
| Qwen3-32B | 8.15 | 8.43 | 4.00 | 1.00 | 23.00 | 1.94 | 7.73 | 33.80 |
| DeepSeek-V3 | 9.63 | 7.23 | 1.00 | 0.00 | 16.00 | 2.91 | 6.28 | 25.40 |
| Qwen3-Coder | 9.63 | 4.82 | 6.00 | 0.00 | 9.00 | 1.71 | 5.44 | 19.25 |
| Llama-4-Scout | 0.74 | 4.82 | 0.00 | 0.00 | 22.00 | 2.91 | 4.83 | 21.03 |

InfoMosaic-Bench reveals stark differences across domains, especially in video, map, and multi-domain. The bottom-right radar in Fig.[1] summarizes domain-wise accuracy for six models on InfoMosaic-Bench. We find that (1) agents with only web search are highly uneven across domains: best scores reach 53% (Medical/Biology) and 45% (Finance) but drop to 36% (Video) and 40.74% (Map). (2) Capabilities vary by domain—for example, Grok-4 does relatively well on Video, whereas GPT-5 struggles there.

### 5.3 Analysis

#### 5.3.1 Domain Tool Analysis

Comparison with only web search tool. To evaluate how LLMs perform with domain-specific tools, we conduct experiments under two settings. For each single-domain evaluation, the agent is provided only with that domain’s specialized tools (for multi-domain, all tools are provided). We report answer accuracy for GLM-4.5 and GPT-5 under only web search tool and domain-tool settings in Table[3]. We find that: (1) On average, domain tools yield only marginal gains, indicating that the bottleneck is not tool availability but tool use—how agents plan, select, parameterize, and time their calls.
(2) Both GPT-5 and GLM-4.5 see clear gains in map and video domains because these tasks depend on structured, exclusive signals (e.g., spatial queries, video metadata) that web search cannot reliably provide.
(3) Accuracy drops on multi-domain tasks with many tools, highlighting cross-source orchestration issues: selecting and chaining tools raises planning complexity and error propagation.

Tool use result analysis. To analyze the reason why the accuracy will drop in certain domains. We collect and categorize the results of tool calls into four types: usage error (wrong function calling), selection error (wrong tool selections), invalid result (successful but irrelevant/unhelpful calling), and valid result (successful and useful tool calling). Fig.[4(a)] shows the distributions of these types in each domain. We have the following findings:
(1) As shown by the line, better tool usage yields more useful information and leads to stronger model performance.
(2) Tool usage error rate correlates with tool complexity. Bio and Multi-domain with larger parameter numbers exhibit higher usage-error rates. Finance and Multi-domain host the largest toolsets and show markedly higher selection error rates, implying larger tool inventories increase selection risk.
(3) Most tool results are unhelpful and contribute little to answering the question.

*Table 3: Comparison between web-only and domain-tool agent accuracy across six domains and the overall average for GLM-4.5 and GPT-5. The colored numbers indicate the difference between domain-tool and web-only settings. Values in green denote accuracy improvements when domain tools are used, while values in red denote decreases.*

|  | Tool | Map | Medical/Bio | Video | Web | Finance | Multi-domain | Overall |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| GLM-4.5 | web | 24.44 | 27.71 | 24.00 | 11.00 | 22.00 | 14.56 | 20.61 |
| | domain | 30.37+5.93 | 34.94+7.23 | 25.00+1.00 | 7.00-4.00 | 20.00-2.00 | 12.62-1.94 | 21.51+0.90 |
| GPT-5 | web | 32.59 | 53.10 | 36.00 | 29.00 | 41.00 | 41.75 | 38.18 |
| | domain | 40.00+7.41 | 43.37-9.73 | 46.00+10.00 | 32.00+3.00 | 30.00-9.00 | 39.81-1.94 | 38.61+0.43 |

#### 5.3.2 Scaling Analysis

We analyze the relationship between the number of tool calls and performance. Fig.[3] and [3] report results of GLM-4.5 across three representative domains (Finance, Bio, and Video), showing how accuracy and pass rate change with the number of tool calls. Fig.[3] compares 4 different models, indicating the growth of input token length as the number of tool calls increases. We find that: (1) Acc and PR generally increase with more tool calls.
(2) Performance plateaus after 8 tool calls, with further tool calls sometimes reducing accuracy and pass rates due to redundant rather than useful information.
(3) In Fig.[3], input token length rises quickly with the number of tool calls until a turning point, after which extra calls add little context. This turning point reflects each model’s effective tool-usage limit. Across models, it correlates with overall accuracy ($R^{2}\=0.57$), indicating a moderate positive link between effective tool-call capacity and task performance.

<img src='x3.png' alt='Refer to caption' title='' width='830' height='402' />

*(a) Num of tool calls v.s. Acc*

<img src='x4.png' alt='Refer to caption' title='' width='830' height='402' />

*(b) Num of tool calls v.s. Avg PR*

<img src='x5.png' alt='Refer to caption' title='' width='830' height='400' />

*(c) Num of too calls v.s. Avg Length*

*Figure 3: Relationship between performance and input length. (a) and (b) show results of GLM-4.5’s Acc and PR across Finance, Bio, and Video domains. (c) compares different models, showing average input token length against the number of tool calls.*

#### 5.3.3 Failure Mode Analysis

Fig.[4(b)] shows GPT-5’s web-only failure distribution on InfoMosaic-Bench using a six-class, primary-cause label (Retrieval Miss, Tool Misuse, Reasoning Gap, Confirmation Bias, Overgeneralization, Context Misread) which is detailed in Appendix[A.4]. Retrieval Miss (39.6%) and Overgeneralization (28.2%) dominate, indicating failures stem mainly from retrieval and evidence selection rather than final-step reasoning—underscoring the need for domain tools and stronger search orchestration.

### 5.4 Human Study

We conduct a human study on 120 randomly sampled problems across domains: three NLP-trained graduate annotators independently rated pre- and post-check versions on factual alignment, coherence, and task difficulty (guidelines in Appendix[A.10.1]). Fig.[4(c)] shows (1) high pre-check scores, indicating strong baseline quality from tool-grounded synthesis; and (2) the largest post-check gain in factual alignment, correcting evidence–answer mismatches. Agreement is high (Cohen’s $\kappa\=0.92$), confirming reliability.

<img src='x6.png' alt='Refer to caption' title='' width='830' height='520' />

*(a) Distribution of tool-call result types in 6 domains.*

<img src='x7.png' alt='Refer to caption' title='' width='830' height='555' />

*(b) Distribution of 6 error types only using the web search.*

<img src='x8.png' alt='Refer to caption' title='' width='830' height='565' />

*(c) Comparison of human evaluation before and after refinement.*

*Figure 4: Distribution analysis. Fig.[4(a)] shows the distributions of 4 tool calling result categories in 6 domains, Fig.[4(b)] shows 6 failure modes of GPT-5 with only web search, and Fig.[4(c)] shows improvements after refinement.*

6 Conclusion
------------

In this work, we propose InfoMosaic-Bench, the first benchmark in evaluating multi-source information seeking in tool-augmented agents, spanning 6 domains with 77 heterogeneous tools. To construct domain-wise and cross-domain information seeking tasks, we also proposed InfoMosaic-Flow, a scalable synthesis methodology. Our extensive experiments demonstrate that (i) web search alone is insufficient for domain-specific information seeking tasks, (ii) current agents remain disproportionately better at web search than at leveraging domain-specific tools. We expect InfoMosaic-Bench to catalyze a shift from web-only search to principled, auditable multi-tool information seeking, accelerating progress on high-stakes domains such as finance and science. Future work may extend the synthesis pipeline to additional modalities, interactive environments, pushing LLM agents closer to real-world deployment.

References
----------

* Brown et al. (2020)Tom B. Brown, Benjamin Mann, Nick Ryder, and et al.Language models are few-shot learners.In *Advances in Neural Information Processing Systems 33 (NeurIPS 2020)*, 2020.
* Chai et al. (2025)Jingyi Chai, Shuo Tang, Rui Ye, Yuwen Du, Xinyu Zhu, Mengcheng Zhou, Yanfeng Wang, Yuzhi Zhang, Linfeng Zhang, Siheng Chen, et al.Scimaster: Towards general-purpose scientific ai agents, part i. x-master as foundation: Can we lead on humanity’s last exam?*arXiv preprint arXiv:2507.05241*, 2025.
* Chen et al. (2025)Chen Chen, Xinlong Hao, Weiwen Liu, Xu Huang, Xingshan Zeng, Shuai Yu, Dexun Li, Shuai Wang, Weinan Gan, Yuefeng Huang, et al.Acebench: Who wins the match point in tool learning?*arXiv e-prints*, pp. arXiv–2501, 2025.
* Fei et al. (2025)Xiang Fei, Xiawu Zheng, and Hao Feng.Mcp-zero: Proactive toolchain construction for llm agents from scratch.*arXiv preprint arXiv:2506.01056*, 2025.
* Flotho et al. (2025)Matthias Flotho, Ian Ferenc Diks, Philipp Flotho, Leidy-Alejandra G Molano, Pascal Hirsch, and Andreas Keller.Mcpmed: A call for mcp-enabled bioinformatics web services for llm-driven discovery.*arXiv preprint arXiv:2507.08055*, 2025.
* Gao et al. (2025)Xuanqi Gao, Siyi Xie, Juan Zhai, Shqing Ma, and Chao Shen.Mcp-radar: A multi-dimensional benchmark for evaluating tool use capabilities in large language models.*arXiv preprint arXiv:2505.16700*, 2025.
* Google (2025)Google.Notebooklm: Ai research tool \& thinking partner, 2025.URL <https://notebooklm.google/>.
* Hou et al. (2025)Xinyi Hou, Yanjie Zhao, Shenao Wang, and Haoyu Wang.Model context protocol (mcp): Landscape, security threats, and future research directions.*arXiv preprint arXiv:2503.23278*, 2025.
* Kaplan et al. (2020)Jared Kaplan, Sam McCandlish, Tom Henighan, and et al.Scaling laws for neural language models.*arXiv preprint*, 2020.
* Li et al. (2025a)Shilong Li, Xingyuan Bu, Wenjie Wang, Jiaheng Liu, Jun Dong, Haoyang He, Hao Lu, Haozhe Zhang, Chenchen Jing, Zhen Li, et al.Mm-browsecomp: A comprehensive benchmark for multimodal browsing agents.*arXiv preprint arXiv:2508.13186*, 2025a.
* Li et al. (2025b)Xiaoxi Li, Guanting Dong, Jiajie Jin, Yuyao Zhang, Yujia Zhou, Yutao Zhu, Peitian Zhang, and Zhicheng Dou.Search-o1: Agentic search-enhanced large reasoning models.*arXiv preprint arXiv:2501.05366*, 2025b.
* Li et al. (2025c)Xiaoxi Li, Jiajie Jin, Guanting Dong, Hongjin Qian, Yutao Zhu, Yongkang Wu, Ji-Rong Wen, and Zhicheng Dou.Webthinker: Empowering large reasoning models with deep research capability.*arXiv preprint arXiv:2504.21776*, 2025c.
* Loughran \& McDonald (2011)Tim Loughran and Bill McDonald.When is a liability not a liability? textual analysis, dictionaries, and 10-ks.*Journal of Finance*, 66(1):35–65, 2011.doi: 10.1111/j.1540-6261.2010.01625.x.Available at SSRN: [https://ssrn.com/abstract\=1331573](https://ssrn.com/abstract=1331573 "").
* Luo et al. (2025)Ziyang Luo, Zhiqi Shen, Wenzhuo Yang, Zirui Zhao, Prathyusha Jwalapuram, Amrita Saha, Doyen Sahoo, Silvio Savarese, Caiming Xiong, and Junnan Li.Mcp-universe: Benchmarking large language models with real-world model context protocol servers.*arXiv preprint arXiv:2508.14704*, 2025.
* Nakano et al. (2021)Reiichiro Nakano, Jacob Hilton, Suchir Balaji, and et al.Webgpt: Browser-assisted question-answering with human feedback.*arXiv preprint*, 2021.
* OpenAI (2023)OpenAI.Gpt-4 technical report.*arXiv preprint*, 2023.URL [https://arxiv.org/abs/2303.08774](https://arxiv.org/abs/2303.08774 "").
* OpenAI (2025)OpenAI.Introducing deep research, February 2025.URL [https://openai.com/index/introducing-deep-research/](https://openai.com/index/introducing-deep-research/ "").
* OpenAI (2025)OpenAI.Function calling — openai api guide.[https://platform.openai.com/docs/guides/function-calling](https://platform.openai.com/docs/guides/function-calling ""), 2025.accessed 2025-09-22.
* Page et al. (1999)Lawrence Page, Sergey Brin, Rajeev Motwani, and Terry Winograd.The pagerank citation ranking: Bringing order to the web.Technical Report 1999-66, Stanford InfoLab, November 1999.URL <http://ilpubs.stanford.edu/422/>.
* Pang et al. (2025)Xianghe Pang, Shuo Tang, Rui Ye, Yuwen Du, Yaxin Du, and Siheng Chen.Browsemaster: Towards scalable web browsing via tool-augmented programmatic agent pair.*arXiv preprint arXiv:2508.09129*, 2025.
* (21)Shishir G Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E Gonzalez.The berkeley function calling leaderboard (bfcl): From tool use to agentic evaluation of large language models.In *Forty-second International Conference on Machine Learning*.
* Perplexity AI (2025)Perplexity AI.Introducing perplexity deep research, 2025.URL [https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research](https://www.perplexity.ai/hub/blog/introducing-perplexity-deep-research "").
* Qin et al. (2023)Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, et al.Toolllm: Facilitating large language models to master 16000+ real-world apis.*arXiv preprint arXiv:2307.16789*, 2023.
* Schick et al. (2023)Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom.Toolformer: Language models can teach themselves to use tools.*Advances in Neural Information Processing Systems*, 36:68539–68551, 2023.
* Shen (2024)Zhuocheng Shen.Llm with tools: A survey.*arXiv preprint arXiv:2409.18807*, 2024.
* Song et al. (2025)Huatong Song, Jinhao Jiang, Yingqian Min, Jie Chen, Zhipeng Chen, Wayne Xin Zhao, Lei Fang, and Ji-Rong Wen.R1-searcher: Incentivizing the search capability in llms via reinforcement learning.*arXiv preprint arXiv:2503.05592*, 2025.
* U.S. Food and Drug Administration (2022)U.S. Food and Drug Administration.Clinical decision support software: Guidance for industry and food and drug administration staff (final guidance, september 2022).<https://www.fda.gov/media/109618/download>, 2022.Docket No. FDA-2017-D-6569.
* Vosoughi et al. (2018)Soroush Vosoughi, Deb Roy, and Sinan Aral.The spread of true and false news online.*science*, 359(6380):1146–1151, 2018.
* Wang et al. (2025)Zhenting Wang, Qi Chang, Hemani Patel, Shashank Biju, Cheng-En Wu, Quan Liu, Aolin Ding, Alireza Rezazadeh, Ankit Shah, Yujia Bao, et al.Mcp-bench: Benchmarking tool-using llm agents with complex real-world tasks via mcp servers.*arXiv preprint arXiv:2508.20453*, 2025.
* Wei et al. (2025)Jason Wei, Zhiqing Sun, Spencer Papay, Scott McKinney, Jeffrey Han, Isa Fulford, Hyung Won Chung, Alex Tachard Passos, William Fedus, and Amelia Glaese.Browsecomp: A simple yet challenging benchmark for browsing agents.*arXiv preprint arXiv:2504.12516*, 2025.
* Wenzek et al. (2020)Guillaume Wenzek, Marie-Anne Lachaux, Alexis Conneau, Vishrav Chaudhary, Francisco Guzmán, Armand Joulin, and Edouard Grave.CCNet: Extracting high quality monolingual datasets from web crawl data.In Nicoletta Calzolari, Frédéric Béchet, Philippe Blache, Khalid Choukri, Christopher Cieri, Thierry Declerck, Sara Goggi, Hitoshi Isahara, Bente Maegaard, Joseph Mariani, Hélène Mazo, Asuncion Moreno, Jan Odijk, and Stelios Piperidis (eds.), *Proceedings of the Twelfth Language Resources and Evaluation Conference*, pp. 4003–4012, Marseille, France, May 2020. European Language Resources Association.ISBN 979-10-95546-34-4.URL [https://aclanthology.org/2020.lrec-1.494/](https://aclanthology.org/2020.lrec-1.494/ "").
* Wu et al. (2025)Jialong Wu, Wenbiao Yin, Yong Jiang, Zhenglin Wang, Zekun Xi, Runnan Fang, Linhai Zhang, Yulan He, Deyu Zhou, Pengjun Xie, et al.Webwalker: Benchmarking llms in web traversal.*arXiv preprint arXiv:2501.07572*, 2025.
* Yao et al. (2023a)Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.React: Synergizing reasoning and acting in language models.In *International Conference on Learning Representations (ICLR)*, 2023a.
* Yao et al. (2023b)Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.ReAct: Synergizing reasoning and acting in language models.In *International Conference on Learning Representations (ICLR)*, 2023b.
* Yao et al. (2024)Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan.tau-bench: A benchmark for tool-agent-user interaction in real-world domains.*arXiv preprint arXiv:2406.12045*, 2024.
* Yuan et al. (2024)Siyu Yuan, Kaitao Song, Jiangjie Chen, Xu Tan, Yongliang Shen, Ren Kan, Dongsheng Li, and Deqing Yang.Easytool: Enhancing llm-based agents with concise tool instruction.*arXiv preprint arXiv:2401.06201*, 2024.
* Zeng (2025)Yifan Zeng.Quantmcp: Grounding large language models in verifiable financial reality.*CoRR*, 2025.

Appendix A Appendix
-------------------

### A.1 Used Data Source

Table[4] lists the data sources used to generate seed data and seeking information. We adhere to all knowledge-use policies of these websites and products.

*Table 4: List of data sources.*

| Name | Website | Type |
| --- | --- | --- |
| Wikipedia | [www.wikipedia.org](https://www.wikipedia.org/ "") | Seed data source |
| Baidu Baike | [baike.baidu.com](https://baike.baidu.com/ "") | Seed data source |
| Qunar | [www.qunar.com](https://www.qunar.com/ "") | Seed data source |
| NORD | [rarediseases.org](https://rarediseases.org "") | Seed data source |
| ClinicalTrials.gov | [clinicaltrials.gov](https://clinicaltrials.gov "") | Seed data source |
| AMap MCP | [github.com/sugarforever/amap-mcp-server](https://github.com/sugarforever/amap-mcp-server "") | MCP server |
| Google Map MCP | [github.com/cablate/mcp-google-map](https://github.com/cablate/mcp-google-map "") | MCP server |
| Bio MCP | [github.com/genomoncology/biomcp](https://github.com/genomoncology/biomcp "") | MCP server |
| YouTube MCP | [github.com/jikime/py-mcp-youtube-toolbox](https://github.com/jikime/py-mcp-youtube-toolbox/tree/main "") | MCP server |
| FMP MCP | [github.com/cdtait/fmp-mcp-server](https://github.com/cdtait/fmp-mcp-server "") | MCP server |

### A.2 Problem Formulation

##### Notation.

Let the domain be $\mathcal{D}$, the queries in $\mathcal{D}$ be $\mathcal{Q}_{\mathcal{D}}$ , and the user query be $q\in\mathcal{Q}_{\mathcal{D}}$.
Let the set of available tools be

|  | $\mathcal{T}_{\text{avail}}\={\mathtt{T}_{1},\ldots,\mathtt{T}_{m}},$ |  |
| --- | --- | --- |

where each tool $\mathtt{T}_{i}$ is specified by interface metadata (name, description, parameter schema, output schema). Let $\mathtt{GT}$ denote the ground-truth answer to $q$ and $K$ be the max tool calling limit. A task instance denoted as

|  | $\tau\=(q,\mathcal{T}_{\text{avail}},K,\mathtt{GT}).$ |  |
| --- | --- | --- |

##### Evaluation.

Let $\mathcal{M}\={M_{1},M_{2},\ldots,M_{n}}$ denote the set of LLMs and
$\mathcal{A}\={A_{1},A_{2},\ldots,A_{p}}$ the set of agent frameworks.
For a given $(M,A)\in\mathcal{M}\times\mathcal{A}$ and task $\tau$, the interaction generates a message history

|  | $H\=(h_{1},h_{2},\ldots,h_{T}),$ |  |
| --- | --- | --- |

where each $h_{t}\=(r_{t}^{(M,A)},r_{t}^{\text{tool}})$ records the responses from agents and any tool invocations. The evaluation function

|  | $E:(H,\tau)\;\to\;{0,1}$ |  |
| --- | --- | --- |

assigns $1$ if the user query is correctly solved under predefined success criteria, and <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="A1.SS2.SSS0.Px2.p1.m7" intent=":literal"><mn>0</mn></math> -->0 otherwise.
Correctness is determined by automated checks (e.g., verifying output content or exactly matching). Details of evaluation could be found in Sec.[5.1].

### A.3 Experiment Setup

#### A.3.1 Models

We evaluate a broad spectrum of current top tier LLMs, including both close-sourced and open-sourced models. Table[5] summarizes their key features such as model size, release date, openness, and access links. Through the evaluation of these latest state-of-the-art LLMs, we ensure the validity and reliability of our main conclusions.

*Table 5: Details about evaluated LLMs.*

| Model | Size | Release Date | Status | Link |
| --- | --- | --- | --- | --- |
| GPT-5 | — | 2025-08-07 | Closed | [https://platform.openai.com/docs/models/gpt-5](https://platform.openai.com/docs/models/gpt-5 "") |
| Grok-4 | — | 2025-07-09 | Closed | [https://x.ai/news/grok-4](https://x.ai/news/grok-4 "") |
| Claude-4.0-Sonnet | — | 2025-05-23 | Closed | <https://www.anthropic.com/claude/sonnet> |
| o3 | — | 2025-04-16 | Closed | <https://platform.openai.com/docs/models/o3> |
| o4-mini | — | 2025-04-16 | Closed | [https://platform.openai.com/docs/models/o4-mini](https://platform.openai.com/docs/models/o4-mini "") |
| Qwen2.5-Max | — | 2025-01-25 | Closed | [https://qwenlm.github.io/zh/blog/qwen2.5-max/](https://qwenlm.github.io/zh/blog/qwen2.5-max/ "") |
| Gemini-2.5-flash | — | 2025-06-17 | Closed | <https://deepmind.google/models/gemini/flash/> |
| GLM-4.5 | 355B | 2025-07-28 | Open | [https://huggingface.co/zai-org/GLM-4.5](https://huggingface.co/zai-org/GLM-4.5 "") |
| Qwen3-235B-A22B | 235B | 2025-04-29 | Open | [https://huggingface.co/Qwen/Qwen3-235B-A22B](https://huggingface.co/Qwen/Qwen3-235B-A22B "") |
| Qwen3-32b | 32.8B | 2025-04-29 | Open | [https://huggingface.co/Qwen/Qwen3-32B](https://huggingface.co/Qwen/Qwen3-32B "") |
| Llama-4-Scout | 109B | 2025-04-05 | Open | [https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E](https://huggingface.co/meta-llama/Llama-4-Scout-17B-16E "") |
| DeepSeek-V3 | 685B | 2025-03-24 | Open | [https://huggingface.co/deepseek-ai/DeepSeek-V3-0324](https://huggingface.co/deepseek-ai/DeepSeek-V3-0324 "") |
| Kimi-K2 | 1T | 2025-07-11 | Open | [https://huggingface.co/moonshotai/Kimi-K2-Instruct](https://huggingface.co/moonshotai/Kimi-K2-Instruct "") |
| Qwen3-Coder | 30.5B | 2025-07-22 | Open | [https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct "") |

#### A.3.2 Agent Framework

We build our agent framework based on ReAct*(Yao et al., [2023a])*, which serves as the foundation of the agent’s reasoning and actions. The framework integrates a Python Sandbox and follows the OpenAI function calling interface*(OpenAI, [2025])*, enabling the LLM to invoke external tools and consume their outputs in a structured manner.

Concretely, we implement multi-turn interactions where tool metadata is serialized into JSON Schema format and provided to the LLM through the function calling interface. The LLM responds with a structured tool call request, which we automatically translate into a Python code snippet. This code is then executed inside the Python Sandbox*(Pang et al., [2025])*, and the execution results (standard outputs or errors) are captured. Finally, the returned results are appended to the dialogue history and passed back to the LLM as additional context for subsequent reasoning steps. We set the tool calling limit to 20, ensuring that the agent terminates tool calls after repeated unsuccessful attempts to solve the problem.

This design allows the framework to (i) support multiple domain-specific tools in a uniform schema, (ii) enforce execution safety through sandbox isolation, and (iii) tightly couple the reasoning trace of the LLM with actual tool outputs.

### A.4 Details of Failure Modes Analysis

We define six categories of failure: (1) Retrieval Miss, where the agent fails to extract key information already present in the tool or knowledge base, often producing “cannot determine” answers or overlooking relevant facts; (2) Tool Misuse, where the agent misinterprets tool outputs or provides incorrect parameters or commands, leading to irrelevant, distorted, or erroneous results (e.g., miscalculations); (3) Reasoning Gap, where correct information fragments are retrieved but not coherently linked, resulting in logical jumps, causal confusion, or inconsistent conclusions; (4) Confirmation Bias, where the agent selectively emphasizes evidence supporting its initial hypothesis while downplaying or ignoring contradictory tool outputs; (5) Overgeneralization and Hallucination, where insufficient evidence leads to unfounded guesses or fabricated details not supported by tool results; and (6) Instruction and Context Misunderstanding, where the agent misinterprets the user’s intent or task context, producing answers that may be partially correct but irrelevant to the actual query.

### A.5 t-SNE Visualization of Query Embeddings

Fig.[5] reveals clear domain-level clustering of query embeddings, indicating strong semantic separability across domains. Limited overlap suggests occasional cross-domain ambiguity, while dispersion reflects intra-domain diversity.

<img src='x9.png' alt='Refer to caption' title='' width='830' height='713' />

*Figure 5: t-SNE visualization of query embeddings across domains. Each point is a query, colored by domain.*

### A.6 Validating the Synthesis Pipeline

We perform ablations to verify the necessity of the two core components of InfoMosaic-Flow: web-based verification and planner–executor interaction.

Effect of Planner–Executor Interaction. The default pipeline employs GPT-5 as the planner and executor pair.
As shown in Table[6], removing the executor collapses the synthesis space. With executor, synthesized tasks involve on average 59.1 tool calls across 43.1 unique tools. Without executor, this drops sharply to 7.8 calls across 5.6 tools, indicating severe loss of heterogeneity and task richness.

Effect of Web-Evolving Verification. Stage-2 verification is implemented with GPT-5 acting as both the main synthesizer and executor. Table[7] show that without this step, many tasks collapse to trivial single-source queries, yielding inflated accuracy (45.1%). To test pruning strategies, we substitute the executor with GPT-5. In Quick Fuzz mode, the GPT-5 executor rewrites conditions without invoking web tools, modestly reducing shortcuts but leaving residual triviality (39.7%). Advanced Fuzz applies iterative probing and constraint rewriting, lowering accuracy to 31.3%, which indicates stronger resistance to shortcuts and closer alignment with true multi-source reasoning.

*Table 6: Effect of planner–executor design. Removing the executor collapses tool usage and reduces task complexity.*

| Executor Setting | Avg. Tool Calls | Avg. Tools Used |
| --- | --- | --- |
| With Executor | 59.1 | 43.1 |
| Without Executor | 7.8 | 5.6 |

*Table 7: Effect of web-evolving verification. Without pruning, many tasks collapse to trivial web lookups, inflating accuracy. Advanced fuzzing most effectively suppresses shortcuts.*

| Verification Setting | Accuracy (%) |
| --- | --- |
| w/o Condition Pruning | 45.1 |
| Quick Fuzz (GPT-5, no web) | 41.7 |
| Advanced Fuzz (GPT-5, web) | 31.3 |

Web-evolving verification prevents trivial shortcuts, and planner–executor interaction ensures sufficient task richness. Both are indispensable for generating a benchmark that reflects realistic multi-source reasoning challenges.

### A.7 Agent Problem-Solving Analysis

#### A.7.1 Patterns Analysis in Searching Problems

After analyzing the trajectories of Agents solving dataset problems, we can find that advanced Agents with higher evaluation scores can already exhibit clear, structured thought processes and steps, which are not prompted by humans. Based on our analysis, GPT-5 demonstrates a specific, sequential long-range “Searching-Reasoning-Evaluating” trajectory when solving problems in the map domain, which is: “Broad Search $\to$ Targeted Information Retrieval $\to$ Solution Evaluation $\to$ Response Calibration”.

Broad Search: In the initial stage, the agent autonomously chooses and prefers to use generalized keyword searches, such as maps_text_search and maps_around_search, which retrieve keywords and return a larger number of candidate answers and their accurate place IDs. The main purpose of this step is to include the correct answer through a generalized search, thereby constraining and narrowing down the broad search space into a list of choices.

Targeted Information Retrieval: After searching for candidate answers, the agent proactively invokes more fine-grained search tools to perform a deep, targeted search for specific POI. For example, agent may call functions maps_search_detail with an accurate ID provided by broad-search tools or functions named maps_distance for targeted information seeking. The use of these tools enables the model to conduct precise searches on the candidate answers, allowing it to further filter out unreasonable options and constrain the search space to a smaller set of candidates.

Solution Evaluation: After filtering, the model is left with 3 to 4 highly similar candidate answers. At this point, in addition to using deep search tools, the model integrates all the information to perform its final reasoning and selection.

Response Calibration: Eventually, agent call tools to validate the candidate answer.

<img src='x10.png' alt='Refer to caption' title='' width='789' height='525' />

*Figure 6: Heatmaps for sequence of tool calls made by the GPT-5 when solving map domain problems.*

Fig. [6] shows the frequency of making diverse tool calls when solving map problems. We calculate the frequency of invoked tools in tool-call sequence of GPTT-5 and. We segment the action trajectory into 8 relative positions, representing different stages of the entire thought process. It is obvious that the agent prefers to call broad search tools (maps_text_search and maps_around_search) mostly at the beginning of the action trajectory. Then, deep and targeted search tools including maps_search_detail and google_maps_search_places begin to emerge. At last, in the validation, more tool calls of maps_distance are used for checking the candidate answer.

#### A.7.2 Error Steps Analysis

Based on our segmentation of agent’s tool-calling trajectory, we further investigated the specific step where the agent’s error occurred. The results are shown in Fig. [7]. We find that agents are most prone to errors at Broad Search, namely when the agent fails to retrieve the correct answer and add it to the candidate list, especially when agent is performing keyword searching. Improper keywords composition may lead to an incorrect or empty search result hit.

<img src='x11.png' alt='Refer to caption' title='' width='581' height='349' />

*Figure 7: Distribution of agent errors across the segmented steps of the tool-calling trajectory*

The following example111Model response in Chinese, translated in English demonstrates how the agent’s use of incorrect keyword combinations (either too complex or too generalized) resulted in the tool’s search failing to return a hit. In this example, too restritive searching keywords lead to empty response.

[⬇](data:text/plain;base64,ewogICAgICAiZnVuY3Rpb24iOiB7CiAgICAgICAgImFyZ3VtZW50cyI6ICJ7InF1ZXJ5IjogIll1ZWNoZW5nIERpc3RyaWN0LCBTaGFveGluZywgU2NlbmljIHNwb3RzIGFuZCBzdG9uZSBicmlkZ2VzIG5lYXIgQ2hhbmdxaWFvIFpoaWppZSJ9IiwKICAgICAgICAibmFtZSI6ICJnb29nbGVtYXBfc2VhcmNoX3BsYWNlcyIKICAgICAgfSwKICAgICAgInR5cGUiOiAiZnVuY3Rpb24iCn0KewogICAgICAicm9sZSI6ICJ0b29sIiwKICAgICAgImNvbnRlbnQiOiAieyd0b29sX3Jlc3VsdCc6ICdUaGUgQmFzaWMgaW5mb3JtYXRpb24gZm9yIHRoZSBxdWVyeTogWXVlY2hlbmcgRGlzdHJpY3QsIFNoYW94aW5nLCBTY2VuaWMgc3BvdHMgYW5kIHN0b25lIGJyaWRnZXMgbmVhciBDaGFuZ3FpYW8gWmhpamllIHdpdGggMCByZXN1bHRzJ30KfQ==)

{

"function":{

"arguments":"{"query":"YuechengDistrict,Shaoxing,ScenicspotsandstonebridgesnearChangqiaoZhijie"}",

"name":"googlemap_search_places"

},

"type":"function"

}

{

"role":"tool",

"content":"{’tool_result’:’TheBasicinformationforthequery:YuechengDistrict,Shaoxing,ScenicspotsandstonebridgesnearChangqiaoZhijiewith0results’}

}

### A.8 Representative Benchmark Instances

#### A.8.1 Examples

The following examples illustrate instances from various domains in our dataset, showcasing how queries are structured with associated testcases and ground truths (GT). In cases where a testcase condition can be decomposed into a subquestion:subanswer format, the corresponding GT is provided and is not null, ensuring verifiable details. However, if a condition lacks an explicit subanswer (e.g., due to its nature as a direct factual check without granular breakdown), it is split into multiple independent testcases for thorough inspection and validation.

| Bio Domain | |
| --- | --- |
| Query | Identify a European technology company headquartered in Espoo, Finland, that reported a net loss in Q1 2025 (quarter ending March 31, 2025) and has a stock beta below 1. What is the ticker of this company? |
| Testcase (Condition) | Headquartered in Espoo, Finland |
| Testcase (Ground_truth) | Espoo, Finland |
| Testcase (Condition) | Reported a net loss in Q1 2025 (quarter ending March 31, 2025) |
| Testcase (Ground_truth) | Net income: -59,000,000 EUR (Q1 2025, period ending 2025-03-31) |
| Testcase (Condition) | Has a stock beta below 1 |
| Testcase (Ground_truth) | Beta: 0.626 |
| GT (Company Ticker) | NOK |

*Table 8: Example instance from the Bio domain.*

| Map Domain | |
| --- | --- |
| Query | 在大连市中山区，寻找一个符合以下特征的地标： 1. 在行政区划上属于中山区（区划代码210202） 2. 从青泥洼桥地铁站步行至此约需31分59秒，步行距离约2.40公里 3. 从大连站驾车到此全程约4.22公里，预计用时约12分59秒；而从大连站乘坐地铁5号线至劳动公园站并步行到达的公共交通方案总用时约42分22秒，因此驾车更快超过5分钟 该地标的名称是？ |
| Testcase (Condition) | 在行政区划上属于中山区（区划代码210202） |
| Testcase (Ground_truth) | null |
| Testcase (Condition) | 从青泥洼桥地铁站步行至此约需31分59秒，步行距离约2.40公里 |
| Testcase (Ground_truth) | null |
| Testcase (Condition) | 从大连站驾车到此全程约4.22公里，预计用时约12分59秒；而从大连站乘坐地铁5号线至劳动公园站并步行到达的公共交通方案总用时约42分22秒，因此驾车更快超过5分钟 |
| Testcase (Ground_truth) | null |
| GT (Toponym) | 大连观光塔 |

*Table 9: Example instance from the Map domain.*

| Finance Domain | |
| --- | --- |
| Query | Identify a publicly traded technology company satisfying the following criteria: (1) Headquarters located in Austin, Texas; (2) Positive net income reported in Q1 2025 (quarter ending March 31, 2025), but negative net income in Q2 2025 (quarter ending June 30, 2025); (3) Year-to-date stock price change as of September 19, 2025, between 10% and 20%, with beta coefficient exceeding 5.0; (4) Q2 2025 revenue (quarter ending June 30, 2025) between $70 million and $80 million USD. |
| Testcase (Condition) | Its headquarters is located in Austin, Texas. |
| Testcase (Ground_truth) | Austin, Texas |
| Testcase (Condition) | It reported positive net income in Q1 2025 (quarter ending March 31, 2025) but negative net income in Q2 2025 (quarter ending June 30, 2025). |
| Testcase (Ground_truth) | Q1 2025 net income: $580,693,000 (positive); Q2 2025 net income: -$936,799,000 (negative) |
| Testcase (Condition) | Its year-to-date stock price change as of September 19, 2025, was between 10% and 20%, and its beta coefficient exceeds 5.0. |
| Testcase (Ground_truth) | YTD change as of 2025-09-19: 15.68%; Beta: 6.6067 |
| Testcase (Condition) | Its Q2 2025 revenue (quarter ending June 30, 2025) was between $70 million and $80 million USD. |
| Testcase (Ground_truth) | $78,628,000 |
| GT (Stock Symbol) | CORZ |

*Table 10: Example instance from the Finance domain.*

| Video Domain | |
| --- | --- |
| Query | On YouTube, there is a video that meets all of the following conditions: 1) Uploaded in September 2025; 2) Duration is approximately 2 minutes and 36 seconds; 3) From a channel with over 1.7 million subscribers; 4) A top comment with over 1,000 likes mentions both ”Earthquake” and ”Baby scene”. Which video is this? Provide its URL.. |
| Testcase (Condition) | Uploaded in September 2025 |
| Testcase (Ground_truth) | null |
| Testcase (Condition) | Duration is approximately 2 minutes and 36 seconds; |
| Testcase (Ground_truth) | null |
| Testcase (Condition) | From a channel with over 1.7 million subscriber |
| Testcase (Ground_truth) | null |
| Testcase (Condition) | A top comment with over 1,000 likes mentions both ”Earthquake” and ”Baby scene” |
| Testcase (Ground_truth) | null |
| GT (URL) | [https://www.youtube.com/watch?v\=Rev9xjajSlM](https://www.youtube.com/watch?v=Rev9xjajSlM "") |

*Table 11: Example instance from the Video domain.*

| Web Domain | |
| --- | --- |
| Query | In a transoceanic venture undertaken by two vessels, the smaller craft had a commander with a rank-style title, while another man actually held the practical authority for navigation. The sailing schedule slipped because that man went inland to collect debts, and much later a regional museum hosted a public lecture centered on him. Who was he? |
| Testcase (Condition) | In a transoceanic venture undertaken by two vessels |
| Testcase (Ground_truth) | Ark and Dove |
| Testcase (Condition) | the smaller craft had a commander with a rank-style title |
| Testcase (Ground_truth) | Captain Wintour |
| Testcase (Condition) | Historic St. Mary’s City (HSMC) |
| Testcase (Ground_truth) | Richard Orchard |
| GT (Person Name) | Richard Orchard |

*Table 12: Example instance from the Web domain.*

| Multi-Domain | |
| --- | --- |
| Query | Identify the stock symbol of a U.S.-listed company fulfilling all criteria: (1) Headquarters in a U.S. city renowned for its high density of universities; (2) Early-year quarterly report indicates minimal revenue and a per-share loss slightly less than one dollar; (3) Share price more than doubled over a mid-2025 season, with trailing six-month total return around 80–100% by early August; (4) Develops therapeutics via a modality that directly edits DNA sequences for monogenic disorders. |
| Testcase (Condition) | Headquarters in a U.S. city renowned for its high density of universities; |
| Testcase (Ground_truth) | Cambridge, Massachusetts, USA |
| Testcase (Condition) | Early-year quarterly report indicates minimal revenue and a per-share loss slightly less than one dollar |
| Testcase (Ground_truth) | Q1 2025 revenue $\approx$ $4.658 million; diluted EPS$\approx$ -$0.92 |
| Testcase (Condition) | Share price more than doubled over a mid-2025 season, with trailing six-month total return around 80–100% by early August) |
| Testcase (Ground_truth) | Share price rose from $1.13 (2025-04-01) to $2.51 (2025-07-31), $\approx$ +122%; 6-month total return as of 2025-08-01 $\approx$ +92% |
| Testcase (Condition) | Develops therapeutics via a modality that directly edits DNA sequences for monogenic disorders. |
| Testcase (Ground_truth) | CRISPR gene editing therapeutics (direct DNA sequence editing for monogenic diseases) |
| GT (Stock Symbol) | EDIT |

*Table 13: Example instance from the Multi-Domain.*

#### A.8.2 List of Tools Used in the Benchmark

The following presents a comprehensive list of all tools used in our benchmark, categorized by their functional domain. Each tool is listed with its name and a brief description.

##### Domain: Map (AMap Services and Google Maps Services)

Tools for geographic information and routing services within China, powered by the AMap server.

| Tool Name | Description |
| --- | --- |
| maps_regeocode | Converts a longitude/latitude coordinate into an administrative region address. |
| maps_geo | Converts a structured address into longitude/latitude coordinates. |
| maps_ip_location | Determines the geographic location based on an IP address. |
| maps_weather | Retrieves real-time weather information for a specified city. |
| maps_bicycling_by_address | Plans a bicycle route between two locations using addresses. Unless you have a specific reason to use coordinates, it’s recommended to use this tool. |
| maps_bicycling_by_coordinates | Plans a bicycle route between two coordinates. |
| maps_direction_walking_by_address | Plans a walking route between two locations using addresses. Unless you have a specific reason to use coordinates, it’s recommended to use this tool. |
| maps_direction_walking_by_coordinates | Plans a walking route based on start and end longitude/latitude coordinates. |
| maps_direction_driving_by_address | Plans a driving route between two locations using addresses. Unless you have a specific reason to use coordinates, it’s recommended to use this tool. |
| maps_direction_driving_by_coordinates | Plans a driving route based on start and end longitude/latitude coordinates. |
| maps_direction_transit_integrated_by_address | Plans a public transit route between two locations using addresses. Requires origin and destination city names for cross-city transit. |
| maps_direction_transit_integrated_by_coordinates | Plans a public transit route based on start and end coordinates. Requires origin and destination city names. |
| maps_distance | Measures the distance (driving, walking, or straight-line) between two coordinates. |
| maps_text_search | Searches for Points of Interest (POI) by keyword within a specified city. |
| maps_around_search | Searches for POIs near a specified coordinate and radius. |
| maps_search_detail | Retrieves detailed information for a POI by its ID. |

Tools for interacting with Google Maps for global geographic searches and directions.

| Tool Name | Description |
| --- | --- |
| googlemap_search_places | Performs a fuzzy search for places on Google Maps. |
| google_map_get_place_details | Retrieves detailed information (reviews, hours, etc.) for a specific place. |
| google_map_get_place_id | Returns the place ID(s) for places matching a search query. |
| google_map_get_map_direction | Fetches step-by-step travel directions between two locations. |

##### Domain: finance (Financial Data Services)

Tools for accessing stock, market, commodity, and cryptocurrency data, primarily powered by the FMP server.

| Tool Name | Description |
| --- | --- |
| get_company_notes | Gets detailed information about company-issued notes and debt instruments. |
| get_income_statement | Retrieves the income statement for a company. |
| get_quote | Gets the current stock quote information. |
| get_quote_change | Gets stock price change over different time periods. |
| get_aftermarket_quote | Gets aftermarket trading quote information. |
| get_price_change | Gets price changes for a stock based on historical data. |
| search_by_symbol | Searches for stocks by ticker symbol. |
| search_by_name | Searches for stocks by company name (English only). |
| get_ratings_snapshot | Gets analyst ratings snapshot for a company. |
| get_financial_estimates | Gets analyst financial estimates for a company. |
| get_price_target_news | Gets the latest analyst price target updates. |
| get_price_target_latest_news | Gets the latest price target announcements with pagination. |
| get_company_dividends | Gets dividend history for a specific company. |
| get_dividends_calendar | Gets a calendar of upcoming dividend events for all stocks. |
| get_index_list | Gets a list of available market indices. |
| get_index_quote | Gets the current quote for a market index. |
| get_biggest_gainers | Gets a list of stocks with the biggest percentage gains. |
| get_biggest_losers | Gets a list of stocks with the biggest percentage losses. |
| get_most_active | Gets a list of most actively traded stocks by volume. |
| get_market_hours | Gets the current market hours status for a specific stock exchange. |
| get_commodities_list | Gets a list of available commodities. |
| get_commodities_prices | Gets current prices for commodities. |
| get_historical_price_eod_light | Gets historical price data for a commodity. |
| get_crypto_list | Gets a list of available cryptocurrencies. |
| get_crypto_quote | Gets current quotes for cryptocurrencies. |
| get_forex_list | Gets a list of available forex pairs. |
| get_forex_quotes | Gets the current quote for a forex pair. |
| get_ema | Gets Exponential Moving Average (EMA) values for a stock. |

##### Domain: Medical/Biology (Biomedical Research Services)

Tools for searching and retrieving data from biomedical databases such as PubMed, ClinicalTrials.gov, and MyVariant.info.

| Tool Name | Description |
| --- | --- |
| search | Universal search across all biomedical domains with unified query language. |
| fetch | Retrieve detailed information for any biomedical record; auto‑detects domain if not provided. |
| article_searcher | Searches PubMed/PubTator3 for research articles and preprints about genes, variants, diseases, or chemicals. |
| article_getter | Fetches detailed information (abstract, full text) for a specific article by its identifier. |
| trial_searcher | Searches ClinicalTrials.gov for clinical studies based on conditions, interventions, location, etc. |
| trial_getter | Fetches comprehensive details for a specific clinical trial by its NCT ID. |
| trial_protocol_getter | Fetches core protocol information (title, summary, design, eligibility) for a clinical trial. |
| trial_references_getter | Fetches publications and references linked to a clinical trial. |
| trial_outcomes_getter | Fetches outcome measures and results data for a clinical trial. |
| trial_locations_getter | Fetches contact and location details for sites participating in a clinical trial. |
| variant_searcher | Searches MyVariant.info for genetic variant database records (frequencies, significance, predictions). |
| variant_getter | Fetches comprehensive details for a specific genetic variant by its ID. |
| gene_getter | Get gene information from MyGene.info, including official name, aliases, genomic location and database links. |
| disease_getter | Get disease information from MyDisease.info, including definition, synonyms, ontology IDs and phenotypes. |
| drug_getter | Get drug or chemical information from MyChem.info, including structure, mechanism, indications, trade names and identifiers. |

##### Domain: web (General Web Services)

General-purpose tools for web searching and content parsing.

| Tool Name | Description |
| --- | --- |
| web_search | Performs a general web search for information. |
| web_parse | Parses a specific webpage or image URL to extract information based on a user query. |

##### Domain: Video (Media Services)

Tools for searching and retrieving information from YouTube.

| Tool Name | Description |
| --- | --- |
| google_search_images | Searches Google Images for pictures. |
| google_search_videos | Searches Google Videos for video content. |
| search_videos | Searches for YouTube videos with advanced filtering options. |
| get_video_details | Gets detailed information about a specific YouTube video. |
| get_channel_details | Gets detailed information about a specific YouTube channel. |
| get_video_comments | Retrieves comments for a specific YouTube video. |
| get_video_transcript | Retrieves the transcript for a specific YouTube video. |
| get_related_videos | Gets a list of videos related to a specific YouTube video. |
| get_trending_videos | Gets a list of trending videos on YouTube for a specific region. |
| get_video_enhanced_transcript | Advanced tool for extracting, filtering, and searching within YouTube video transcripts. |

### A.9 Distributions of Question and Answer Length

Fig. [9] and Fig. [9] illustrate the distribution of question and ground truth answer (GT) lengths, measured in tokens, for Chinese and English questions.

Both question and answer lengths exhibit skewed distributions, with most samples concentrated in shorter ranges and a long tail extending to larger lengths. This indicates that the benchmark primarily consists of concise inputs and outputs while still including a subset of more complex, lengthier cases.

<img src='x12.png' alt='Refer to caption' title='' width='365' height='362' />

*Figure 8: Question Length Distribution (Chinese vs English).*

<img src='x13.png' alt='Refer to caption' title='' width='365' height='365' />

*Figure 9: Ground Truth (GT) Length Distribution (Chinese vs English).*

### A.10 Comprehensive Evaluation and Analytical Protocols

#### A.10.1 Human Study

The Human Evaluation Prompt provides detailed guidelines for assessing the Factual Alignment, Coherence, and Difficult of InfoMosaic-Bench items across three evaluation dimensions.

[⬇](data:text/plain;base64,V2hhdCB5b3Ugd2lsbCBzZWUgcGVyIGl0ZW0KLSBUYXNrIHN0YXRlbWVudCAodGhlIHF1ZXN0aW9uIHVzZXJzIG11c3QgYW5zd2VyKQotIEdyb3VuZCBUcnV0aAotIENvbmRpdGlvbnMgKGludGVybWVkaWF0ZSByZXF1aXJlbWVudHMpCgpJbXBvcnRhbnQgcnVsZXMKMS4gVXNlIG9ubHkgdGhlIGdpdmVuIG1hdGVyaWFscyAodGFzay9jb25kaXRpb25zL0dUKS4gWW91IGNhbiB1c2Ugb3V0c2lkZSBzZWFyY2ggb3IgcHJpb3Iga25vd2xlZGdlLgoyLiBKdWRnZSBxdWFsaXR5IG9mIHRoZSBkYXRhc2V0IGl0ZW0sIG5vdCB0aGUgZ2VuZXJhbCB0cnV0aCBvZiB0aGUgd29ybGQuCgpEaW1lbnNpb24gQSDigJQgRmFjdHVhbCBBbGlnbm1lbnQgKEFuc3dlcuKAk0NvbmRpdGlvbiBDb25zaXN0ZW5jeSkKUXVlc3Rpb246IERvZXMgdGhlIEdyb3VuZCBUcnV0aCBmYWl0aGZ1bGx5IGNvcnJlc3BvbmQgdG8gdGhlIHN0YXRlZCBjb25kaXRpb25zPyBTcGVjaWZpY2FsbHksIGNhbiBlYWNoIGNvbmRpdGlvbiBiZSB0cmFjZWQgdG8gcGFydCBvZiB0aGUgYW5zd2VyLCBhbmQgZG9lcyB0aGUgYW5zd2VyIHJlbHkgb25seSBvbiBpbmZvcm1hdGlvbiB0aGF0IGlzIGNvdmVyZWQgYnkgdGhlIGNvbmRpdGlvbnM/CkhvdyB0byBqdWRnZToKIENoZWNrIHdoZXRoZXIgdGhlIGNvbmRpdGlvbnMg4oaUIEdyb3VuZCBUcnV0aCBtYXBwaW5nIGlzIGNvbnNpc3RlbnQuIFRoZSBhbnN3ZXIgc2hvdWxkOgoxLiBCZSBkZXJpdmFibGUgYnkgc2F0aXNmeWluZyBhbGwgbGlzdGVkIGNvbmRpdGlvbnMuCjIuIE5vdCBpbmNsdWRlIGV4dHJhIGluZm9ybWF0aW9uIGJleW9uZCB0aGUgY29uZGl0aW9ucy4KMy4gTm90IGJlIHNvbHZhYmxlIGJ5IGlnbm9yaW5nIHNvbWUgY29uZGl0aW9ucy4KTGFiZWxzIChjaG9vc2Ugb25lKToKLSBBNSDigJMgRnVsbHkgYWxpZ25lZDogRXZlcnkgY29uZGl0aW9uIGlzIHJlZmxlY3RlZCBpbiB0aGUgZmluYWwgYW5zd2VyOyBubyBpbmNvbnNpc3RlbnQgY29uZGl0aW9ucyBmb3IgR3JvdW5kIFRydXRoLgotIEEzIOKAkyBQYXJ0aWFsbHkgYWxpZ25lZDogQW5zd2VyIGNvdmVycyB0aGUgbWFpbiBpbnRlbnQsIGJ1dCBzb21lIGNvbmRpdGlvbnMgYXJlIG5vdCBkZXNjcmliaW5nIHRoZSBncm91bmQgdHJ1dGgKLSBBMSDigJMgTWlzYWxpZ25lZDogRmluYWwgYW5zd2VyIGRvZXMgbm90IHJlc3BlY3QgY29uZGl0aW9ucyAoY29udHJhZGljdGluZywgb3IgaXJyZWxldmFudCBjb25kaXRpb25zKS4KCkRpbWVuc2lvbiBCIOKAlCBDb2hlcmVuY2UgKFNlbWFudGljICYgTG9naWNhbCBDb2hlcmVuY2UpClF1ZXN0aW9uOiBJcyB0aGUgaXRlbSB3ZWxsLWZvcm1lZCBhcyBhIGRhdGFzZXQgZXhhbXBsZT8gQXJlIHRoZSB0YXNrLCBjb25kaXRpb25zLCBhbmQgYW5zd2VyIG11dHVhbGx5IGNvbnNpc3RlbnQsIHVuYW1iaWd1b3VzLCBhbmQgZXhlY3V0YWJsZT8KSG93IHRvIGp1ZGdlOiBDaGVjayB0aGUgY2xhcml0eSwgbm9uLWNvbnRyYWRpY3Rpb24sIGFuZCByZWZlcmVudGlhbCBjb2hlcmVuY2UgYW1vbmcgdGFzayDihpQgY29uZGl0aW9ucyDihpQgZ3JvdW5kIHRydXRoLgpMYWJlbHMgKGNob29zZSBvbmUpOgotIEI1IOKAkyBDb2hlcmVudDogVGFzayBpcyBjbGVhcjsgY29uZGl0aW9ucyBhcmUgaW50ZXJwcmV0YWJsZSBhbmQgbm9uLWNvbmZsaWN0aW5nOyBubyB1bm5hdHVyYWwgb3Igc2VsZi1jb250cmFkaWN0b3J5IHBocmFzaW5nLgotIEIzIOKAkyBNaW5vciBpc3N1ZXM6IFNtYWxsIGFtYmlndWl0eSwgbWlsZCByZWR1bmRhbmN5LCBvciBzbGlnaHRseSBhd2t3YXJkIHBocmFzaW5nIHRoYXQgZG9lcyBub3QgYmxvY2sgZXhlY3V0aW9uIG9yIGludGVycHJldGF0aW9uLgotIEIxIOKAkyBJbmNvaGVyZW50OiBDb250cmFkaWN0b3J5IGNvbmRpdGlvbnMsIGlsbC1mb3JtZWQgcmVmZXJlbmNlcyAoZS5nLiwgdW5kZWZpbmVkIGVudGl0eSksIG9yIGNvbmRpdGlvbnMgdGhhdCBjYW5ub3QgYmUgZXhlY3V0ZWQgd2l0aCB0aGUgZ2l2ZW4gdG9vbHMuCk5vdGVzOgotIElmIGEgc2luZ2xlIGNvbmRpdGlvbiBkaXJlY3RseSBnaXZlcyBhd2F5IHRoZSBhbnN3ZXIsIG1hcmsgQjEvQjAgZGVwZW5kaW5nIG9uIHNldmVyaXR5IGFuZCBleHBsYWluICh0aGlzIHdpbGwgYWxzbyBhZmZlY3QgRGlmZmljdWx0eSBiZWxvdykuCgpEaW1lbnNpb24gQyDigJQgVGFzayBEaWZmaWN1bHR5IChOZWVkIGZvciBNdWx0aS1Ub29sIC8gTXVsdGktU3RlcCBSZWFzb25pbmcpClF1ZXN0aW9uOiBXb3VsZCBhIGNvbXBldGVudCBhZ2VudCBuZWVkIG11bHRpcGxlIHRvb2xzIGFuZC9vciBtdWx0aXBsZSBzdGVwcyB0byBzb2x2ZSB0aGlzIHRhc2sgdXNpbmcgdGhlIHByb3ZpZGVkIGRvbWFpbiB0b29scz8KSG93IHRvIGp1ZGdlOiBGb2N1cyBvbiBuZWNlc3NpdHkgKG5vdCBqdXN0IHRoZSBjb3VudCBvZiBjYWxscykuIElmIGEgc2luZ2xlIGNvbmRpdGlvbiBvciBhIHNpbmdsZSB0b29sIHN1ZmZpY2VzLCB0aGUgdGFzayBpcyB0cml2aWFsLgpMaWtlcnQgc2NvcmUgKDHigJM1KToKLSBDMSDigJMgVHJpdmlhbDogU29sdmFibGUgdmlhIG9uZSB0b29sIG9yIG9uZSBjb25kaXRpb247IG5vIGludGVncmF0aW9uIG5lZWRlZC4KLSBDMiDigJMgRWFzeTogTW9zdGx5IG9uZSBzb3VyY2UgcGx1cyBtaW5pbWFsIGxvb2t1cC4KLSBDMyDigJMgTW9kZXJhdGU6IFJlcXVpcmVzIGNvbWJpbmluZyDiiaUyIHBpZWNlcyBvZiBldmlkZW5jZSBvciBzZXF1ZW50aWFsIHN0ZXBzLCBidXQgc3RyYWlnaHRmb3J3YXJkLgotIEM0IOKAkyBIYXJkOiBDbGVhciBuZWVkIHRvIGFnZ3JlZ2F0ZSBtdWx0aXBsZSB0b29scy9jb25kaXRpb25zOyBub24tb2J2aW91cyBjb21wb3NpdGlvbi4KLSBDNSDigJMgVmVyeSBoYXJkOiBMb25nLWhvcml6b24gaW50ZWdyYXRpb24gYWNyb3NzIHNldmVyYWwgdG9vbHMvY29uZGl0aW9uczsgY2FyZWZ1bCBhbGlnbm1lbnQgbmVlZGVkLgpIZXVyaXN0aWNzOgotIElmIHdlYi1saWtlIHNpbmdsZSByZXRyaWV2YWwgY291bGQgYW5zd2VyIGl0IOKGkiBDMeKAk0MyLgotIElmIGF0IGxlYXN0IHR3byBpbmRlcGVuZGVudCBjb25kaXRpb25zIG11c3QgYmUgc2F0aXNmaWVkL2NvbWJpbmVkIOKGkiBDM+KAk0M1IChwaWNrIGJhc2VkIG9uIGNvbXBsZXhpdHkpLg==)

Whatyouwillseeperitem

-Taskstatement(thequestionusersmustanswer)

-GroundTruth

-Conditions(intermediaterequirements)

Importantrules

1.Useonlythegivenmaterials(task/conditions/GT).Youcanuseoutsidesearchorpriorknowledge.

2.Judgequalityofthedatasetitem,notthegeneraltruthoftheworld.

DimensionA—FactualAlignment(Answer–ConditionConsistency)

Question:DoestheGroundTruthfaithfullycorrespondtothestatedconditions?Specifically,caneachconditionbetracedtopartoftheanswer,anddoestheanswerrelyonlyoninformationthatiscoveredbytheconditions?

Howtojudge:

Checkwhethertheconditions↔GroundTruthmappingisconsistent.Theanswershould:

1.Bederivablebysatisfyingalllistedconditions.

2.Notincludeextrainformationbeyondtheconditions.

3.Notbesolvablebyignoringsomeconditions.

Labels(chooseone):

-A5–Fullyaligned:Everyconditionisreflectedinthefinalanswer;noinconsistentconditionsforGroundTruth.

-A3–Partiallyaligned:Answercoversthemainintent,butsomeconditionsarenotdescribingthegroundtruth

-A1–Misaligned:Finalanswerdoesnotrespectconditions(contradicting,orirrelevantconditions).

DimensionB—Coherence(Semantic\&LogicalCoherence)

Question:Istheitemwell-formedasadatasetexample?Arethetask,conditions,andanswermutuallyconsistent,unambiguous,andexecutable?

Howtojudge:Checktheclarity,non-contradiction,andreferentialcoherenceamongtask↔conditions↔groundtruth.

Labels(chooseone):

-B5–Coherent:Taskisclear;conditionsareinterpretableandnon-conflicting;nounnaturalorself-contradictoryphrasing.

-B3–Minorissues:Smallambiguity,mildredundancy,orslightlyawkwardphrasingthatdoesnotblockexecutionorinterpretation.

-B1–Incoherent:Contradictoryconditions,ill-formedreferences(e.g.,undefinedentity),orconditionsthatcannotbeexecutedwiththegiventools.

Notes:

-Ifasingleconditiondirectlygivesawaytheanswer,markB1/B0dependingonseverityandexplain(thiswillalsoaffectDifficultybelow).

DimensionC—TaskDifficulty(NeedforMulti-Tool/Multi-StepReasoning)

Question:Wouldacompetentagentneedmultipletoolsand/ormultiplestepstosolvethistaskusingtheprovideddomaintools?

Howtojudge:Focusonnecessity(notjustthecountofcalls).Ifasingleconditionorasingletoolsuffices,thetaskistrivial.

Likertscore(1–5):

-C1–Trivial:Solvableviaonetooloronecondition;nointegrationneeded.

-C2–Easy:Mostlyonesourceplusminimallookup.

-C3–Moderate:Requirescombining≥2piecesofevidenceorsequentialsteps,butstraightforward.

-C4–Hard:Clearneedtoaggregatemultipletools/conditions;non-obviouscomposition.

-C5–Veryhard:Long-horizonintegrationacrossseveraltools/conditions;carefulalignmentneeded.

Heuristics:

-Ifweb-likesingleretrievalcouldanswerit→C1–C2.

-Ifatleasttwoindependentconditionsmustbesatisfied/combined→C3–C5(pickbasedoncomplexity).

#### A.10.2 Benchmark Evaluation Prompt

Below are the exact evaluation prompts used to extract answers, verify correctness, and assess sub-answers for our benchmark.

##### extract_answer_template

[⬇](data:text/plain;base64,WW91IGFyZSBhIGhlbHBmdWwgQUkgYXNzaXN0YW50IHRhc2tlZCB3aXRoIGV4dHJhY3RpbmcgdGhlIGZpbmFsIGFuc3dlciBmcm9tIGEgcHJvdmlkZWQgc29sdXRpb24uCgoqKklucHV0OioqCjEuIEEgcHJvYmxlbSBzdGF0ZW1lbnQsIHByZWZpeGVkIHdpdGggIj09PVByb2JsZW06IDxwcm9ibGVtPiIuCjIuIEEgc29sdXRpb24gdG8gdGhlIHByb2JsZW0sIHByZWZpeGVkIHdpdGggIj09PVNvbHV0aW9uOiIuCgoqKlByb2JsZW0gYW5kIFNvbHV0aW9uOioqCj09PVByb2JsZW06IHt0YXNrfQoKPT09U29sdXRpb246IHtvcGVyYXRlZF90ZXh0fQoKKipJbnN0cnVjdGlvbnM6KioKLSBDYXJlZnVsbHkgYW5hbHl6ZSB0aGUgc29sdXRpb24gYW5kIGV4dHJhY3QgdGhlIGZpbmFsIGFuc3dlciBpbiByZXBseTogIlRoZSBhbnN3ZXIgaXMgPGFuc3dlciBleHRyYWN0ZWQ+IGluIHJlcGx5Ii4KLSBJZiB0aGUgc29sdXRpb24gZG9lcyBub3QgY29udGFpbiBhIGZpbmFsIGFuc3dlciAoZS5nLiwgb25seSByZWFzb25pbmcsIGNvZGUgd2l0aG91dCBleGVjdXRpb24sIG9yIGluY29tcGxldGUgaW5mb3JtYXRpb24pLCByZXNwb25kIHdpdGg6ICJUaGUgcmVwbHkgZG9lc24ndCBjb250YWluIGFuIGFuc3dlci4iCi0gRW5zdXJlIHRoYXQgdGhlIGV4dHJhY3RlZCBhbnN3ZXIgaXMgZXhhY3RseSBhcyBwcmVzZW50ZWQgaW4gdGhlIHNvbHV0aW9uLiBEbyBub3QgaW5mZXIgb3IgdXNlIGV4dGVybmFsIGtub3dsZWRnZS4gRG8gbm90IGV4ZWN1dGUgdGhlIGNvZGUgeW91cnNlbGYuCi0gUmVtZW1iZXIsIE5ldmVyIGV4ZWN1dGUgdGhlIGNvZGUgeW91cnNlbGYhIE5ldmVyIGRvaW5nIGFueSBjb21wdXRhdGlvbiB5b3Vyc2VsZiEgSnVzdCBleHRyYWN0IGFuZCBvdXRwdXQgdGhlIGV4aXN0aW5nIGFuc3dlciEK)

YouareahelpfulAIassistanttaskedwithextractingthefinalanswerfromaprovidedsolution.

**Input:**

1.Aproblemstatement,prefixedwith"\=\=\=Problem:<problem>".

2.Asolutiontotheproblem,prefixedwith"\=\=\=Solution:".

**ProblemandSolution:**

\=\=\=Problem:{task}

\=\=\=Solution:{operated_text}

**Instructions:**

-Carefullyanalyzethesolutionandextractthefinalanswerinreply:"Theansweris<answerextracted>inreply".

-Ifthesolutiondoesnotcontainafinalanswer(e.g.,onlyreasoning,codewithoutexecution,orincompleteinformation),respondwith:"Thereplydoesn’tcontainananswer."

-Ensurethattheextractedanswerisexactlyaspresentedinthesolution.Donotinferoruseexternalknowledge.Donotexecutethecodeyourself.

-Remember,Neverexecutethecodeyourself!Neverdoinganycomputationyourself!Justextractandoutputtheexistinganswer!

##### eval_prompt_template

[⬇](data:text/plain;base64,WW91IGFyZSBhIGhlbHBmdWwgQUkgYXNzaXN0YW50LiBZb3Ugd2lsbCB1c2UgeW91ciBjb2RpbmcgYW5kIGxhbmd1YWdlIHNraWxscyB0byB2ZXJpZnkgdGhlIGFuc3dlci4KWW91IGFyZSBnaXZlbjoKICAxLiBBIHByb2JsZW0sIHdoaWNoIGlzIGdvaW5nIHRvIHN0YXJ0IGxpa2UgIj09PVByb2JsZW06IDxwcm9ibGVtPiIuCiAgMi4gQSBncm91bmQgdHJ1dGggYW5zd2VyLCB3aGljaCBpcyBnb2luZyB0byBzdGFydCBsaWtlICI9PT1Hcm91bmQgdHJ1dGggYW5zd2VyOiIuCiAgMy4gQSByZXBseSB3aXRoIHRoZSBhbnN3ZXIgdG8gdGhlIHByb2JsZW0sIHdoaWNoIGFyZSBnb2luZyB0byBzdGFydCBsaWtlICI9PT1SZXBseToiLgpQbGVhc2UgZG8gdGhlIGZvbGxvd2luZzoKMS4gRXh0cmFjdCB0aGUgYW5zd2VyIGluIHJlcGx5OiAiVGhlIGFuc3dlciBpcyA8YW5zd2VyIGV4dHJhY3RlZD4gaW4gcmVwbHkiLgoyLiBDaGVjayB3aGV0aGVyIHRoZSBhbnN3ZXIgaW4gcmVwbHkgbWF0Y2hlcyB0aGUgZ3JvdW5kIHRydXRoIGFuc3dlci4gV2hlbiBjb21wYXJpc29uIGlzIG5vdCBvYnZpb3VzIChmb3IgZXhhbXBsZSwgMypcXHNxcnQoNikgYW5kIDcuMzQ4KSwgeW91IG1heSBjb21wYXJlIGJ5IGNhbGN1bGF0aW9uLCBhbGxvd2luZyBhIHNtYWxsIG1hcmdpbiBvZiBlcnJvci4KMy4gQWZ0ZXIgZXZlcnl0aGluZyBpcyBkb25lLCBwbGVhc2UgZ2l2ZSBlYWNoIHJlcGx5IGEgY29tbWVudCBsaWtlIHRoZSBmb2xsb3dpbmcgb3B0aW9uczoKICAtICJUaGUgYW5zd2VyIGlzIGNvcnJlY3QuIgogIC0gIlRoZSBhbnN3ZXIgaXMgYXBwcm94aW1hdGVkIGJ1dCBzaG91bGQgYmUgY29ycmVjdC4gQ29ycmVjdCBBbnN3ZXI6IDxncm91bmQgdHJ1dGggYW5zd2VyPiB8IEFuc3dlciBleHRyYWN0ZWQ6IDxhbnN3ZXIgZXh0cmFjdGVkPi4iCiAgLSAiVGhlIGFuc3dlciBpcyBpbmNvcnJlY3QuIENvcnJlY3QgQW5zd2VyOiA8Z3JvdW5kIHRydXRoIGFuc3dlcj4gfCBBbnN3ZXIgZXh0cmFjdGVkOiA8YW5zd2VyIGV4dHJhY3RlZD4uIgogIC0gIlRoZSByZXBseSBkb2Vzbid0IGNvbnRhaW4gYW4gYW5zd2VyLiIKSGVyZSBhcmUgdGhlIHByb2JsZW0sIHRoZSBncm91bmQgdHJ1dGggYW5zd2VyIGFuZCB0aGUgcmVwbHk6Cj09PVByb2JsZW06IHt0YXNrfQoKPT09R3JvdW5kIHRydXRoIGFuc3dlcjoge2dyb3VuZF90cnV0aH0KCj09PVJlcGx5OiB7b3BlcmF0ZWRfdGV4dH0=)

YouareahelpfulAIassistant.Youwilluseyourcodingandlanguageskillstoverifytheanswer.

Youaregiven:

1.Aproblem,whichisgoingtostartlike"\=\=\=Problem:<problem>".

2.Agroundtruthanswer,whichisgoingtostartlike"\=\=\=Groundtruthanswer:".

3.Areplywiththeanswertotheproblem,whicharegoingtostartlike"\=\=\=Reply:".

Pleasedothefollowing:

1.Extracttheanswerinreply:"Theansweris<answerextracted>inreply".

2.Checkwhethertheanswerinreplymatchesthegroundtruthanswer.Whencomparisonisnotobvious(forexample,3*\\sqrt(6)and7.348),youmaycomparebycalculation,allowingasmallmarginoferror.

3.Aftereverythingisdone,pleasegiveeachreplyacommentlikethefollowingoptions:

-"Theansweriscorrect."

-"Theanswerisapproximatedbutshouldbecorrect.CorrectAnswer:<groundtruthanswer>|Answerextracted:<answerextracted>."

-"Theanswerisincorrect.CorrectAnswer:<groundtruthanswer>|Answerextracted:<answerextracted>."

-"Thereplydoesn’tcontainananswer."

Herearetheproblem,thegroundtruthanswerandthereply:

\=\=\=Problem:{task}

\=\=\=Groundtruthanswer:{ground_truth}

\=\=\=Reply:{operated_text}

##### eval_testcase_prompt_template

[⬇](data:text/plain;base64,WW91IGFyZSBhIGhlbHBmdWwgQUkgYXNzaXN0YW50LiBZb3Ugd2lsbCB1c2UgeW91ciBjb2RpbmcgYW5kIGxhbmd1YWdlIHNraWxscyB0byB2ZXJpZnkgdGhlIHN1YmFuc3dlci4KWW91IGFyZSBnaXZlbjoKICAxLiBBIHNldCBvZiBzdWJwcm9ibGVtcyBhbmQgaXRzIGNvcnJlc3BvbmRpbmcgc3ViYW5zd2Vycywgd2hpY2ggYXJlIGdvaW5nIHRvIHN0YXJ0IGxpa2UgIj09PVN1YnByb2JsZW1zIGFuZCBzdWJhbnN3ZXJzIi4KICAyLiBBIHJlcGx5IHdpdGggdGhlIHN1YmFuc3dlciB0byB0aGUgc3VicHJvYmxlbSwgd2hpY2ggYXJlIGdvaW5nIHRvIHN0YXJ0IGxpa2UgIj09PVJlcGx5OiIuClBsZWFzZSBkbyB0aGUgZm9sbG93aW5nOgoxLiBNYXRjaGluZyB0aGUgc3VicXVlc3Rpb25zIGFuZCBjb21wYXJlIHRoZSBzdWJhbnN3ZXIgaW4gcmVwbHkgd2l0aCB0aGUgZ3JvdW5kIHRydXRoIHN1YmFuc3dlci4KMi4gQ2hlY2sgd2hldGhlciB0aGUgc3ViYW5zd2VyIGluIHJlcGx5IG1hdGNoZXMgdGhlIGdyb3VuZCB0cnV0aCBzdWJhbnN3ZXIuIFdoZW4gY29tcGFyaXNvbiBpcyBub3Qgb2J2aW91cyAoZm9yIGV4YW1wbGUsIDMqXFxzcXJ0KDYpIGFuZCA3LjM0OCksIHlvdSBtYXkgY29tcGFyZSBieSBjYWxjdWxhdGlvbiwgYWxsb3dpbmcgYSBzbWFsbCBtYXJnaW4gb2YgZXJyb3IuIEhlcmUgYXJlIHNvbWUgcHJpbmNpcGxlczoKLSBJZiB0aGUgZ3JvdW5kIHRydXRoIHN1YmFuc3dlciBpcyBhIG51bWVyaWNhbCB2YWx1ZSwgeW91IG1heSBjb21wYXJlIHRoZSBudW1lcmljYWwgdmFsdWUgaW4gdGhlIHN1YmFuc3dlciB3aXRoIHRoZSBncm91bmQgdHJ1dGggc3ViYW5zd2VyLCBhbGxvd2luZyBhIHNtYWxsIG1hcmdpbiBvZiBlcnJvci4KLSBJZiB0aGUgZ3JvdW5kIHRydXRoIHN1YmFuc3dlciBpcyBhIHN0cmluZywgeW91IG1heSBjb21wYXJlIHRoZSBzdHJpbmcgaW4gdGhlIHN1YmFuc3dlciB3aXRoIHRoZSBncm91bmQgdHJ1dGggc3ViYW5zd2VyLCBjYXNlLWluc2Vuc2l0aXZlLCBhbmQganVzdGlmeSBpZiBpdCBpcyB0aGUgc2FtZSBhcyB0aGUgZ3JvdW5kIHRydXRoIHN1YmFuc3dlci4KMy4gQWZ0ZXIgZXZlcnl0aGluZyBpcyBkb25lLCBwbGVhc2UgZ2l2ZSBlYWNoIHJlcGx5IGEgY29tbWVudCBsaWtlIHRoZSBmb2xsb3dpbmcgZm9ybWF0IHRvIGp1c3RpZnkgaWYgdGhlIHN1YmFuc3dlciBpcyBjb3JyZWN0IG9yIGluY29ycmVjdDoKYGBganNvbgp7ewoic3VicXVlc3Rpb24iOiAiQ09SUkVDVHxJTkNPUlJFQ1QiLAoic3VicXVlc3Rpb24iOiAiQ09SUkVDVHxJTkNPUlJFQ1QiCn19CmBgYApIZXJlIGFyZSB0aGUgcHJvYmxlbSwgdGhlIGdyb3VuZCB0cnV0aCBhbnN3ZXIgYW5kIHRoZSByZXBseToKPT09U3VicHJvYmxlbXMgYW5kIHN1YmFuc3dlcnM6Cnt0YXNrfQoKPT09UmVwbHk6CntzdWJhbnN3ZXJ9CgojIyBPdXRwdXQgZm9ybWF0OiBZb3UgbXVzdCBmb2xsb3cgdGhlIGZvbGxvd2luZyBmb3JtYXQKYGBganNvbgp7ewoic3VicXVlc3Rpb24iOiAiQ09SUkVDVHxJTkNPUlJFQ1QiLAoic3VicXVlc3Rpb24iOiAiQ09SUkVDVHxJTkNPUlJFQ1QiCn19CmBgYA==)

YouareahelpfulAIassistant.Youwilluseyourcodingandlanguageskillstoverifythesubanswer.

Youaregiven:

1.Asetofsubproblemsanditscorrespondingsubanswers,whicharegoingtostartlike"\=\=\=Subproblemsandsubanswers".

2.Areplywiththesubanswertothesubproblem,whicharegoingtostartlike"\=\=\=Reply:".

Pleasedothefollowing:

1.Matchingthesubquestionsandcomparethesubanswerinreplywiththegroundtruthsubanswer.

2.Checkwhetherthesubanswerinreplymatchesthegroundtruthsubanswer.Whencomparisonisnotobvious(forexample,3*\\sqrt(6)and7.348),youmaycomparebycalculation,allowingasmallmarginoferror.Herearesomeprinciples:

-Ifthegroundtruthsubanswerisanumericalvalue,youmaycomparethenumericalvalueinthesubanswerwiththegroundtruthsubanswer,allowingasmallmarginoferror.

-Ifthegroundtruthsubanswerisastring,youmaycomparethestringinthesubanswerwiththegroundtruthsubanswer,case-insensitive,andjustifyifitisthesameasthegroundtruthsubanswer.

3.Aftereverythingisdone,pleasegiveeachreplyacommentlikethefollowingformattojustifyifthesubansweriscorrectorincorrect:

‘‘‘json

{{

"subquestion":"CORRECT|INCORRECT",

"subquestion":"CORRECT|INCORRECT"

}}

‘‘‘

Herearetheproblem,thegroundtruthanswerandthereply:

\=\=\=Subproblemsandsubanswers:

{task}

\=\=\=Reply:

{subanswer}

##Outputformat:Youmustfollowthefollowingformat

‘‘‘json

{{

"subquestion":"CORRECT|INCORRECT",

"subquestion":"CORRECT|INCORRECT"

}}

‘‘‘
