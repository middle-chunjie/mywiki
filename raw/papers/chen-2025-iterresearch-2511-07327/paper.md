IterResearch: Rethinking Long-Horizon Agents via Markovian State Reconstruction
================================================================================

Guoxin Chen1,2, Zile Qiao2,∗, Xuanzhong Chen2, Donglei Yu2, Haotian Xu3, Wayne Xin Zhao1,∗  
Ruihua Song1,∗, Wenbiao Yin2, Huifeng Yin2, Liwen Zhang2, Kuan Li2, Minpeng Liao2  
Yong Jiang2, Pengjun Xie2, Fei Huang2, Jingren Zhou2  
1Gaoling School of Artificial Intelligence, Renmin University of China  
2Tongyi Lab, Alibaba Group, 3OpenRLHF  
{gx.chen.chn, batmanfly}@gmail.com, songruihua_bloon@outlook.com  
{qiaozile.qzl, yongjiang.jy}@alibaba-inc.com

###### Abstract

Recent advances in deep-research agents have shown promise for autonomous knowledge construction through dynamic reasoning over external sources.
However, existing approaches rely on a mono-contextual paradigm that accumulates all information in a single, expanding context window, leading to context suffocation and noise contamination that limit their effectiveness on long-horizon tasks. We introduce IterResearch, a novel iterative deep-research paradigm that reformulates long-horizon research as a Markov Decision Process with strategic workspace reconstruction. By maintaining an evolving report as memory and periodically synthesizing insights, our approach preserves consistent reasoning capacity across arbitrary exploration depths. We further develop Efficiency-Aware Policy Optimization (EAPO), a reinforcement learning framework that incentivizes efficient exploration through geometric reward discounting and enables stable distributed training via adaptive downsampling. Extensive experiments demonstrate that IterResearch achieves substantial improvements over existing open-source agents with average +14.5pp across six benchmarks and narrows the gap with frontier proprietary systems. Remarkably, our paradigm exhibits unprecedented interaction scaling, extending to 2048 interactions with dramatic performance gains (from 3.5% to 42.5%), and serves as an effective prompting strategy, improving frontier models by up to 19.2pp over ReAct on long-horizon tasks.
These findings position IterResearch as a versatile solution for long-horizon reasoning, effective both as a trained agent and as a prompting paradigm for frontier models.

††footnotetext: ∗Corresponding Authors.<img src='x1.png' alt='[Uncaptioned image]' title='' width='20' height='20' /> [Code](https://github.com/Alibaba-NLP/DeepResearch "")

<img src='x2.png' alt='Refer to caption' title='' width='830' height='415' />

*Figure 1: Performance of IterResearch against state-of-the-art open-source long-horizon agents.*

1 Introduction
--------------

Recent advances in deep-research agents represent a transformative shift for Large Language Models (LLMs), moving beyond passive knowledge acquisition from the model itself towards autonomous agents that construct knowledge through dynamic reasoning over external sources*[dr, google_dr, grok3, Perplexity, claude_deep_research, kimi-researcher]*.
These frontier proprietary systems have demonstrated remarkable performance on long-horizon tasks that require sustained reasoning and information-seeking capabilities over extended interactions.

When tackling long-horizon tasks, recent works*[chen2025cpo, song2025r1, zheng2025deepresearcher, jin2025search, Li2025webthinker, li2025websailor, tao2025webshaper]* typically append all retrieved information and intermediate reasoning steps to a single, continuously expanding context window, which we term the mono-contextual paradigm.
While straightforward to implement, this paradigm fundamentally undermines the sustained reasoning capabilities required for long-horizon tasks:
(1) context suffocation: as the context window fills with all prior interactions, the available space for model reasoning progressively shrinks, forcing increasingly constrained responses that ultimately degrade into premature or superficial conclusions.
(2) noise contamination: irrelevant information from web searches and early exploration errors become permanently embedded in the context, creating cascading interference that dilutes signal quality throughout the entire reasoning process.

To address these limitations, we introduce IterResearch, a novel Iterative Deep-Research Paradigm that fundamentally reimagines how autonomous agents maintain sustained reasoning capacity in long-horizon scenarios.
Our key insight is that effective long-horizon research requires periodic synthesis and strategic forgetting—capabilities absent in current mono-contextual approaches.
Specifically, we extend the Markov Decision Process (MDP) framework for deep research with a distinctive state design: rather than maintaining an ever-expanding history, each state is a strategically reconstructed workspace containing only essential elements: the question, an evolving report serving as the agent’s memory, and the immediate context needed for current reasoning.
This Markovian structure, where future exploration depends only on the current reconstructed state rather than the entire history, enables the agent to maintain consistent reasoning capacity across arbitrary exploration depths while naturally circumventing the degradation that plagues mono-contextual approaches.

To fully realize this paradigm’s potential, we develop Efficiency-Aware Policy Optimization (EAPO), a reinforcement learning framework specifically designed for training IterResearch.
EAPO addresses two critical challenges unique to our iterative paradigm:
First, recognizing that not all successful trajectories are equally valuable, we introduce efficiency-aware rewards that geometrically discount based on trajectory length—agents reaching correct conclusions through concise, focused exploration receive higher rewards than those requiring extensive iterations.
Second, since our paradigm naturally decomposes trajectories into independent training samples at each round, we employ adaptive downsampling to handle the variable sample counts based on data-parallel size, ensuring stable distributed training while preserving over 99% of training data.

Extensive experiments demonstrate that IterResearch significantly outperforms existing open-source agents, achieving an average improvement of 14.5 percentage points (pp) across six challenging benchmarks.
More remarkably, IterResearch narrows the performance gap with frontier proprietary systems, even surpassing some on these benchmarks.
Furthermore, our work reveals three fundamental insights about deep-research agents.
First, our iterative paradigm unlocks extreme interaction scaling—a capability theoretically extensible to infinite depths yet structurally infeasible for current mono-contextual approaches.
To our knowledge, we are the first to successfully extend agents to 2048 interactions with only 40K context length, exhibiting dramatic performance improvements (3.5% → 42.5%) as maximum interactions increase from 2 to 2048, suggesting that the perceived difficulty of long-horizon tasks may stem from insufficient exploration capacity.
Second, we observe cross-paradigm knowledge transfer: trajectories generated by IterResearch significantly enhance mono-contextual agents, demonstrating that our paradigm induces superior exploration behaviors that create high-quality training signals transferable even across paradigmatically different approaches.
Third, our iterative paradigm serves as an effective prompting strategy: without any training, simply applying it to frontier models yields substantial improvements over the standard mono-contextual approach, ReAct*[yao2023react]*, particularly on long-horizon tasks (+12.7-19.2pp on BrowseComp), revealing that IterResearch offers a model-agnostic solution to long-horizon reasoning.
These results confirm the effectiveness of our iterative paradigm in enabling both deeper exploration and higher-quality reasoning in long-horizon scenarios.

In summary, our main contributions can be summarized as follows:

* •

    We propose IterResearch, a novel iterative deep-research paradigm that reformulates long-horizon research as an MDP with strategic workspace reconstruction, maintaining sustained reasoning capacity through periodic synthesis and an evolving report memory—eliminating the context suffocation and noise contamination that plague mono-contextual approaches.

* •

    We develop Efficiency-Aware Policy Optimization (EAPO) with geometric discounted rewards that incentivize efficient exploration and adaptive downsampling for stable distributed training, enabling effective learning from our paradigm’s unique trajectory structure.

* •

    We demonstrate IterResearch’s exceptional capabilities and broader impact: (1) achieving an average 14.5 pp improvement across six challenging benchmarks; (2) exhibiting interaction scaling to 2048 interactions with dramatic performance gains; (3) enabling cross-paradigm knowledge transfer to enhance mono-contextual agents; (4) providing a model-agnostic prompting strategy that significantly improves frontier models on long-horizon tasks without training.

2 Related Work
--------------

Retrieval-Augmented Generation (RAG). RAG is a crucial approach to overcome knowledge limitations of large language models (LLMs) by integrating external information sources*[nakano2021webgpt, yu2024rankrag, AsaiWWSH24, wei2025instructrag, chen2025cpo, jin2025search, song2025r1, zheng2025deepresearcher]*.
However, traditional RAG methods are typically confined to static retrieval environments, such as Wikipedia, with limited exploration spaces, making them inadequate for complex, long-horizon reasoning tasks that require dynamic information gathering.

Deep Research. Recent advances in deep research*[dr, google_dr, grok3, kimi-researcher]* have transcended RAG’s limitations by deploying autonomous agents in real-world environments, demonstrating remarkable capabilities in navigating complex web environments and synthesizing information from diverse sources.
However, existing open-source methods*[li2025search, Li2025webthinker, tao2025webshaper, li2025websailor]* predominantly adopt a mono-contextual paradigm, continuously appending all retrieved information and reasoning steps to a single expanding context. This linear accumulation leads to progressive workspace suffocation and irreversible noise contamination, limiting their effectiveness in long-horizon tasks.
In contrast, our IterResearch reimagines deep research by formalizing it as a Markov Decision Process with workspace reconstruction mechanism, eliminating accumulation-induced degradation and enabling sustained reasoning capacity at arbitrary research depths—a critical advantage absent in existing approaches.

3 Methodology
-------------

In this section, we detail IterResearch, which extends the Markov Decision Process framework to deep research through strategic workspace reconstruction (§[3.1]), as illustrated in Figure[2].
Then, we further introduce Efficiency-Aware Policy Optimization for training (§[3.2]).

<img src='x3.png' alt='Refer to caption' title='' width='830' height='416' />

*Figure 2: (Top) The mono-contextual approach linearly accumulates all information into a single, ever-expanding context, leading to context suffocation and noise contamination. (Bottom) IterResearch models deep research as an extended MDP with workspace reconstruction.
Each round begins with a reconstructed workspace $s_{t}$ containing the question, an evolving report $\mathcal{M}_{t}$, and immediate context.
The agent generates structured decisions $d_{t}\=$ (Think, Report, Action) and interacts with environment $\mathcal{E}$.
The transition function $\mathcal{T}$ reconstructs the workspace, maintaining the Markov property while preventing context bloat and enabling sustained reasoning and information-seeking.*

### 3.1 Iterative Deep-Research Paradigm

#### 3.1.1 Markov Decision Process Formulation

We model IterResearch as an extended Markov Decision Process defined by the tuple $\langle\mathcal{S},\mathcal{D},\mathcal{E},\mathcal{T}\rangle$, where the agent conducts research through iterative rounds of exploration and synthesis to enable unbounded exploration.

* •

    State Space $\mathcal{S}$: Each state $s_{t}\=(q,\mathcal{M}_{t},{a_{t-1},\text{TR}_{t-1}})$ represents the agent’s workspace, comprising question $q$, an evolving report $\mathcal{M}_{t}$ that compresses all critical findings from previous rounds, and the immediate context (action $a_{t-1}$ and tool response $\text{TR}_{t-1}$) from last interaction.

* •

    Decision Space $\mathcal{D}$: At each state $s_{t}$, the agent generates a structured decision $d_{t}\=(\text{Think}_{t},\mathcal{M}_{t+1},a_{t})$ where:
    (1) Think: Reasoning about current progress and identifying information gaps.
    (2) Report ($\mathcal{M}_{t+1}$): Updated report serving as the agent’s compressed memory, incorporating new findings from $\text{TR}_{t-1}$ while preserving essential insights from $\mathcal{M}_{t}$ and filtering noise.
    (3) Action: The agent’s next operation, which can be either a tool call to gather information or a final answer when the agent determines it can adequately address the question.

* •

    Environment $\mathcal{E}$: External tools (Google Search, Google Scholar, Web Browser, Python) that return responses $\text{TR}_{t}\=\mathcal{E}(a_{t})$ containing requested information or computation results.

* •

    Transition Function $\mathcal{T}$: Deterministically maps ($s_{t}\xrightarrow{d_{t},\text{TR}_{t}}s_{t+1}$) current state, decision, and tool response to the next state.
    Unlike mono-contextual approaches that accumulate context, we *reconstruct* the workspace, maintaining only the question $q$, agent-updated report $\mathcal{M}_{t+1}$, and latest interaction ${a_{t},\text{TR}_{t}}$, preventing context blowup.

The complete research process of IterResearch can be formalized as a sequence of state transitions driven by the agent policy $\pi$:

|  | $\begin{cases}\text{Decision:}\&d_{t}\=\pi(s_{t})\=(\text{Think}_{t},\mathcal{M}_{t+1},a_{t})\\[3.0pt] \text{Transition:}\&s_{t+1}\=\mathcal{T}(s_{t},d_{t},\mathcal{E}(a_{t}))\=(q,\mathcal{M}_{t+1},{a_{t},\text{TR}_{t}})\end{cases}$ |  | (1) |
| --- | --- | --- | --- |

where $\text{TR}_{t}\=\mathcal{E}(a_{t})$, initial state $s_{0}\=(q,\mathcal{M}_{0},\emptyset)$ with empty report $\mathcal{M}_{0}$.
The iterative process generates a trajectory $\tau\={(s_{0},d_{0},\text{TR}_{0}),(s_{1},d_{1},\text{TR}_{1}),\ldots,(s_{T},d_{T})}$ terminating when $a_{T}\=\texttt{answer}$.
Unlike mono-contextual approaches where context grows linearly with trajectory length, our workspace reconstruction maintains bounded memory footprint—the report $\mathcal{M}_{t}$ synthesizes findings rather than accumulating raw observations, enabling sustained reasoning quality over extended research trajectories.

#### 3.1.2 Markovian Workspace Reconstruction

The cornerstone of our paradigm is workspace reconstruction, which fundamentally departs from traditional linear accumulation approaches*[Li2025webthinker, li2025websailor, tao2025webshaper]*.
While existing methods suffer from $O(t)$ context growth leading to inevitable performance degradation, we introduce a principled reconstruction mechanism that maintains bounded workspace complexity while preserving complete task-relevant information through selective compression.

At round $t$, the workspace $s_{t}$ contains only three essential components: (1) the question $q$, providing the constant objective; (2) the evolving report $\mathcal{M}_{t}$, serving as compressed memory of all critical findings; and (3) the immediate context ${a_{t-1},\text{TR}_{t-1}}$ from the last interaction.
The key insight is that the report $\mathcal{M}_{t+1}$ is naturally generated by the LLM as part of its structured decision output $d_{t}\=(\text{Think}_{t},\mathcal{M}_{t+1},a_{t})$.
This natural flow leverages the LLM’s inherent capabilities for information compression and relevance filtering, without requiring explicit algorithmic intervention.

As shown in Eq.[1], the transition function $\mathcal{T}$ implements strategic forgetting by reconstructing the workspace at each round.
The historical trajectory $(s_{0},d_{0},\text{TR}_{0},...,s_{t-1},d_{t-1},\text{TR}_{t-1})$ is deliberately discarded, with only the synthesized knowledge preserved in $\mathcal{M}_{t+1}$.
This design ensures a constant workspace regardless of trajectory length, in stark contrast to mono-contextual approaches:

|  | $\underbrace{s_{t}^{\text{mono}}\=[q,a_{0},\text{TR}_{0},...,a_{t-1},\text{TR}_{t-1}]}_{\text{Mono-contextual (ReAct): }{\color[rgb]{0,0,1}\definecolor[named]{pgfstrokecolor}{rgb}{0,0,1}\mathcal{O}(t)\text{ growth}}}\quad\text{vs.}\quad\underbrace{s_{t}^{\text{iter}}\=(q,\mathcal{M}_{t},{a_{t-1},\text{TR}_{t-1}})}_{\text{IterResearch (Ours): }{\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\mathcal{O}(1)\text{ constant}}}$ |  | (2) |
| --- | --- | --- | --- |

Through the markovian workspace, the agent maintains consistent reasoning capacity throughout the research process, avoiding the performance degradation that inevitably occurs when context windows approach their limits.
Furthermore, through end-to-end training (§[3.2]), the agent progressively learns to synthesize reports that effectively filter noise and preserve essential information.
Thus, irrelevant information or errors from early rounds cannot directly propagate to future decisions—they must first pass through the agent’s synthesis to be incorporated into the report.
This selective retention ensures that the Markov property holds: the current state $s_{t+1}$ contains all decision-relevant information, making the full history unnecessary for optimal decision-making.
The transformative impact of this design manifests in interaction scaling.
While mono-contextual approaches typically fail or degrade severely beyond dozens of interactions due to context limitations, our IterResearch enables theoretically unbounded exploration, sustaining consistent reasoning quality at arbitrary depths.
This scaling capability, empirically validated through experiments with up to 2048 interactions (§[4.4]), fundamentally expands the scope of problems that deep-research agents can tackle.

### 3.2 Efficiency-Aware Policy Optimization

#### 3.2.1 Discounted Reward Shaping for Efficiency

While the Markovian workspace reconstruction ensures scalable exploration, a critical question remains: how can we train agents to not just explore deeply, but to do so efficiently? We now address this challenge by introducing an efficiency-aware policy optimization framework.

In deep research tasks, the agent receives a binary reward signal $R_{T}\in{0,1}$ only upon termination, where $R_{T}\=1$ if the final answer is correct and <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S3.SS2.SSS1.p2.m3" intent=":literal"><mn>0</mn></math> -->0 otherwise.
This terminal-only reward stems from the inherent difficulty of evaluating intermediate research steps—it is challenging to determine the value of any particular search query or exploratory action*[chen2025cpo]*.

However, this sparse signal alone is insufficient for guiding efficient learning, as it treats all successful trajectories equally regardless of their computational cost.
An agent that arrives at the correct answer in 5 well-chosen steps should be preferred over one that requires 20 steps of meandering exploration, even if both ultimately succeed.
This efficiency consideration is not merely about computational resources: in real-world deployment, each interaction incurs API costs, and unnecessary exploration can lead to increased latency.
To address these issues, we introduce a reward shaping mechanism based on geometric discounting from MDP theory*[bellman1957markovian]*:

|  | $r_{t}\=\gamma^{T-t}\cdot R_{T},\quad\gamma\in(0,1)$ |  | (3) |
| --- | --- | --- | --- |

where $T$ is the terminal step, $t$ is the current step, and $\gamma$ is the discount factor.
This exponential decay creates an implicit efficiency pressure: actions contributing to earlier task completion receive proportionally higher rewards, naturally incentivizing more direct exploration strategies while maintaining the simplicity of terminal-only evaluation.

#### 3.2.2 Policy Optimization with Multi-Round Trajectories

A distinctive feature of our iterative paradigm is that each trajectory naturally decomposes into multiple independent training samples (one per round), whereas one trajectory typically yields a single training sample in mono-contextual approaches.
Specifically, for each question $q$, we perform $G$ rollouts generating $G$ independent trajectories.
Each trajectory $\tau_{i}$ unfolds over $T_{i}$ rounds, where round $t$ produces a state-decision pair $(s_{i,t},d_{i,t})$ following our MDP formulation (Eq.[1]).

This yields a rich training corpus $\mathcal{C}\={(s_{i,t},d_{i,t},r_{i,t}):i\in[1,G],t\in[1,T_{i}]}$ with $\sum_{i\=1}^{G}T_{i}$ samples, far exceeding the $G$ trajectory-level samples from traditional approaches.
While this paradigm significantly enriches training data, the variable sample count across questions requires careful handling for distributed training.
We address this through adaptive downsampling that reduces the training corpus to the largest multiple of data parallel (DP) size:

|  | $|\mathcal{C}_{\text{train}}|\=\left\lfloor\frac{|\mathcal{C}|}{\text{DP}_{\text{size}}}\right\rfloor\times\text{DP}_{\text{size}}$ |  | (4) |
| --- | --- | --- | --- |

This approach ensures minimal data loss (typically $<1\%$ of samples) while maintaining uniform sampling across trajectories.
To optimize IterResearch, we integrate our geometric discounted rewards and adaptive downsampling with the Group Sequence Policy Optimization (GSPO) algorithm*[zheng2025group]*, enabling stable training on variable-length trajectories:

|  | $\mathcal{J}(\theta)\=\mathbb{E}_{q\sim\mathcal{Q},\mathcal{C}_{\text{train}}\sim\pi_{\theta_{\text{old}}}(\cdot|q)}\left[\frac{1}{|\mathcal{C}_{\text{train}}|}\sum_{i\=1}^{G}\sum_{t\=1}^{T_{i}}\min(\rho_{i,t}(\theta)\hat{A}_{i,t},\text{clip}(\rho_{i,t}(\theta),1-\varepsilon,1+\varepsilon)\hat{A}_{i,t})\right]$ |  | (5) |
| --- | --- | --- | --- |

where all $\sum_{i\=1}^{G}T_{i}$ rounds from the $G$ trajectories for question $q$ form one group, with normalized advantages computed across all samples within this group $\hat{A}_{i,t}\=\frac{r_{i,t}-\mu_{r}}{\sigma_{r}}$, $\mathcal{Q}$ is the training set, and $\rho_{i,t}(\theta)$ is the importance ratio based on sequence likelihood*[zheng2023click]*.

4 Experiments
-------------

### 4.1 Experimental Setup

Datasets. To rigorously assess the effectiveness of our IterResearch, we evaluate on six challenging benchmarks including Humanity’s Last Exam (HLE) *[hle]*, BrowseComp *[bc_en]*, BrowseComp-zh *[bc_zh]*, GAIA *[mialon2023gaia]*, Xbench-DeepSearch *[xbench]*, SEAL-0 *[pham2025sealqa]*.
These benchmarks comprehensively assess the essential capabilities for effective deep research in multi-step tool use, web navigation, complex reasoning, long-horizon information-seeking, and cross-lingual synthesis.

Baselines. We comprehensively compare our IterResearch against state-of-the-art methods including:
(1) Direct Inference: We evaluate frontier LLMs including GPT-4o and GPT-4.1*[gpt_4o]*, o4-mini*[openai_o3_o4_mini]*, and DeepSeek-R1-0528*[guo2025deepseek]*.
(2) Proprietary Deep-Research System: We compare with commercial deep-research systems including OpenAI’s Deep Research*[dr]*, Perplexity Research*[Perplexity]*, Gemini Deep Research*[google_dr]*, Grok3-ResearchSearch*[grok3]*, and Kimi-Researcher*[kimi-researcher]*.
(3) Open-source Agents: Recent open-source deep-research agents including Search-o1*[li2025search]*, WebThinker*[Li2025webthinker]*, WebDancer*[wu2025webdancer]*, WebSailor*[li2025websailor]*, Asearcher*[gao2025beyond]*, and MiroThinker*[MiroThinker]*.

Implementation Details. We implement our IterResearch using Qwen3-30B-A3B*[yang2025qwen3]* as the backbone model, considering both model performance and computational efficiency.
Our training follows a two-stage process: we first employ rejection sampling fine-tuning (RFT)*[yuan2023scaling]* to equip the model with our iterative deep-research paradigm capabilities, then apply reinforcement learning to further enhance its search strategy and reasoning abilities.
For brevity, we provide comprehensive training details and hyperparameters in Appendix[C.3].

### 4.2 Main Results

*Table 1: Main results across six deep-research benchmarks. We report accuracy (%) for all metrics. The best results are in bold, and the second best among open-source agents are underlined.*

| Model | Tools | HLE | BC | BC-zh | GAIA | Xbench-DS | SEAL-0 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| \cellcolor[HTML]E5E5FCDirect Inference | | | | | | | |
| GPT-4o | ✗ | 2.3 | 0.6 | 6.2 | 17.5 | - | - |
| GPT-4.1 | ✗ | 4.9 | 1.5 | 14.4 | 22.3 | - | - |
| o4-mini | ✗ | 18.9 | 6.1 | 15.2 | 33.3 | 60.0 | 4.5 |
| DeepSeek-R1-0528 | ✗ | 17.7 | 2.0 | 26.3 | 16.5 | - | 5.4 |
| \cellcolor[HTML]E5E5FCProprietary Deep-Research System | | | | | | | |
| OpenAI DeepResearch | ✓ | 26.6 | 51.5 | 42.9 | 67.4 | - | - |
| Perplexity Research | ✓ | 21.1 | - | 22.6 | - | - |  |
| Gemini DeepResearch | ✓ | 26.9 | - | - | - | 50.0 | - |
| Grok3-ResearchSearch | ✓ | - | - | 12.9 | - | 50.0 | - |
| Kimi-Researcher | ✓ | 26.9 | - | - | - | 69.0 | 36.0 |
| \cellcolor[HTML]E5E5FCOpen-source Agents | | | | | | | |
| Search-o1-QwQ | ✓ | 5.4 | 2.8 | 17.9 | 39.8 | 40.3 | - |
| WebThinker-QwQ | ✓ | 6.8 | 2.8 | 7.3 | 48.5 | 32.8 | - |
| WebDancer-QwQ | ✓ | 7.6 | 3.8 | 18.0 | 51.5 | 40.0 | 20.7 |
| Asearcher-Web-QwQ | ✓ | 12.5 | 5.2 | 15.6 | 52.8 | 42.1 | - |
| WebSailor-32B | ✓ | 9.6 | 10.5 | 25.5 | 53.2 | 53.3 | 16.2 |
| WebSailor-72B | ✓ | 9.8 | 12.0 | 30.1 | 55.4 | 55.0 | 19.8 |
| MiroThinker-14B${}_{\text{v0.2}}$ | ✓ | 20.0 | 14.1 | 26.6 | 62.1 | 47.0 | - |
| MiroThinker-32B${}_{\text{v0.2}}$ | ✓ | 19.1 | 17.2 | 29.4 | 64.1 | 56.0 | - |
| IterResearch-30B-A3B | ✓ | 28.8 | 37.3 | 45.2 | 72.8 | 71.0 | 39.6 |
| + Improvement |  | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 8.8}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 20.1}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 15.8}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 8.7}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 15.0}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 18.9}$ |

Table[1] presents the comprehensive evaluation results across six challenging benchmarks. First, IterResearch outperforms all existing open-source agents, with an average margin of 14.5 percentage points across the six benchmarks.
More remarkably, it demonstrates competitive or superior performance compared to proprietary deep-research systems—surpassing OpenAI’s DeepResearch on HLE and BrowseComp-zh, while achieving comparable results on BrowseComp and GAIA.
These results confirm that our iterative paradigm successfully bridges the gap between open-source and commercial systems. Second, the consistent improvements across benchmarks with distinct characteristics validate our core design principles.
On information-seeking benchmarks requiring extensive web navigation (BrowseComp, BrowseComp-zh, SEAL-0), our method demonstrates substantial advantages over mono-contextual baselines.
These tasks particularly suffer from context suffocation in traditional approaches, as agents must navigate through numerous web pages while synthesizing vast amounts of information.
Our workspace reconstruction mechanism maintains consistent reasoning capacity by strategically compressing findings into the evolving report, preventing the inevitable degradation that plagues mono-contextual methods.
On complex reasoning benchmarks demanding deep analytical capabilities (HLE, GAIA, Xbench-DS), the advantage stems from our ability to mitigate noise contamination.
While mono-contextual approaches irreversibly accumulate errors and irrelevant information throughout their trajectories, our iterative paradigm provides natural breakpoints for filtering noise through periodic synthesis.
The evolving report preserves only validated findings while discarding exploratory dead-ends, enabling more focused reasoning in subsequent rounds.
These consistent improvements across diverse task types demonstrate that the iterative deep-research paradigm provides a principled solution to the fundamental limitations of linear information accumulation

### 4.3 Ablation Study

To thoroughly understand the contributions of our approach, we conduct comprehensive ablation studies examining both the effectiveness of our Efficiency-Aware Policy Optimization (EAPO) and the fundamental advantages of our iterative paradigm over traditional mono-contextual approaches.

(1) Effectiveness of Efficiency-Aware Policy Optimization. The upper section of Table[2] demonstrates the impact of our EAPO compared to standard GSPO and SFT.
Analysis of average interactions reveals that EAPO requires 18.04 turns, compared to GSPO’s 19.13 turns and SFT’s 16.45 turns.
While EAPO and GSPO achieve comparable accuracy across benchmarks, the critical distinction emerges in interaction efficiency: EAPO reduces average interactions by 5.7% while maintaining or improving accuracy.
This validates our core hypothesis that geometric discounted rewards successfully incentivize the discovery of more efficient research strategies—agents learn to reach correct conclusions through more focused, deliberate exploration rather than exhaustive searching.

(2) Superiority of the Iterative Paradigm. To rigorously validate our paradigm’s advantages, we conduct a controlled comparison using identical training data across different paradigms.
The middle section of Table[2] reveals striking performance gaps: our iterative paradigm outperforms the mono-contextual baseline (Mono-Agent) by an average of 12.6 percentage points across all benchmarks, with particularly dramatic improvements on long-horizon information-seeking tasks (BC: +11.8%, BC-zh: +10.6%).
Notably, to ensure the mono-contextual agent operates at its optimal capacity and mitigate the inevitable context accumulation issues inherent to its design, we deliberately equipped it with a substantially larger context window (64K vs. our 40K tokens).
This substantial performance gap persists despite providing the mono-contextual approach with more context length, which confirms our theoretical analysis: workspace suffocation fundamentally limits mono-contextual approaches—simply expanding the context window cannot resolve this limitation.
In contrast, our workspace reconstruction mechanism maintains consistent reasoning quality at arbitrary depths through strategic information compression and filtering, enabling effective handling of long-horizon tasks that overwhelm traditional approaches regardless of their context size.

*Table 2: Ablation studies on training methodology and paradigm design. The paradigm ablation uses identical training data and external environment to ensure fair comparison.*

|  | HLE | BC | BC-zh | GAIA | Xbench-DS | SEAL-0 | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| \cellcolor[HTML]E5E5FCAblation on Methodology | | | | | | | |
| IterResearch-EAPO | 28.8 | 37.3 | 45.2 | 72.8 | 71.0 | 39.6 | 49.1 |
| IterResearch-GSPO | 28.2 | 38.3 | 45.6 | 70.9 | 67.0 | 39.6 | 48.3 |
| IterResearch-SFT | 25.3 | 34.9 | 40.8 | 68.9 | 65.0 | 37.8 | 45.5 |
| \cellcolor[HTML]E5E5FCAblation on Paradigm (Cross-Paradigm Knowledge Transfer) | | | | | | | |
| Mono-Agent | 18.7 | 25.4 | 34.6 | 62.1 | 55.0 | 23.4 | 36.5 |
| Mono-Agent + Iter | 25.4 | 30.1 | 40.4 | 63.1 | 62.0 | 30.6 | 41.9 |
| + Improvement | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 6.7}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 4.7}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 5.8}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 1.0}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 7.0}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 7.2}$ | ${\color[rgb]{1,0,0}\definecolor[named]{pgfstrokecolor}{rgb}{1,0,0}\uparrow 5.4}$ |

(3) Cross-Paradigm Knowledge Transfer. An unexpected yet significant finding emerges: trajectories generated by our iterative paradigm can enhance mono-contextual agents when incorporated into their training data.
As shown in the bottom rows of Table[2], augmenting Mono-Agent with iterative-paradigm data while maintaining total data volume (Mono-Agent + Iter) yields consistent improvements across most benchmarks, with an average gain of 5.4 percentage points.
The fact that trajectories generated through our iterative paradigm can enhance mono-contextual agents indicates that our paradigm induces superior research behaviors that create higher-quality training signals, partially transferable even across paradigmatically different approaches.

### 4.4 Scaling on Interaction

<img src='x4.png' alt='Refer to caption' title='' width='830' height='526' />

*Figure 3: Interaction Scaling.*

A fundamental advantage of our iterative paradigm is its ability to maintain consistent performance at arbitrary interaction depths—a property critical for tackling genuinely complex long-horizon tasks that may require extensive exploration.
To empirically validate this capability, we conduct scaling experiments on BrowseComp (200 subset), the most interaction-intensive benchmark in our evaluation suites.
Figure[3] presents our scaling analysis as we exponentially increase the maximum allowed turns from 2 to 2048, a range that would be computationally prohibitive for mono-contextual approaches due to context window limitations.
Two key insights emerge from these results:

First, performance scales gracefully with interaction budget. Accuracy improves from 5.5% with only 2 turns to 50.1% at 2048 turns, with the steepest gains occurring between $2^{4}$ and $2^{7}$ turns.
This demonstrates that complex information-seeking tasks genuinely benefit from extended exploration—a capability that mono-contextual approaches cannot provide due to inevitable context overflow.
Notably, 2048 turns represents an extreme challenge that is currently infeasible for mono-contextual agents due to catastrophic context accumulation, yet our approach operates smoothly within its constant 40K token workspace through Markovian state reconstruction. Second, the agent learns intelligent resource allocation. Despite having access to 2048 turns, the agent uses only 80.1 turns on average, indicating adaptive termination once sufficient information is gathered rather than exhaustively consuming the budget.
Notably, the growth pattern of average turns mirrors the accuracy curve—both increase rapidly in the $2^{4}$-$2^{7}$ range before plateauing—suggesting that exploration depth naturally aligns with task complexity.
This sublinear growth in average turns (compared to exponentially increasing budget) demonstrates that the agent develops increasingly efficient search strategies as more interactions become available, rather than simply extending existing patterns.

### 4.5 IterResearch as a Effective Prompting Strategy in Long-Horizon Tasks

<img src='x5.png' alt='Refer to caption' title='' width='747' height='284' />

*Figure 4: Performance comparison between IterResearch and ReAct as Prompting Strategies.*

Having demonstrated IterResearch’s effectiveness as a trained agent, we investigate whether our iterative paradigm can serve as an effective prompting strategy for long-horizon tasks without any training.
We compare with ReAct*[yao2023react]*, the prevailing mono-contextual prompting paradigm, using frontier models o3*[openai_o3_o4_mini]* and DeepSeek-V3.1*[dpsk_v31]*.

Figure[4] reveals that IterResearch consistently outperforms ReAct across all benchmarks, with particularly dramatic improvements on the most challenging long-horizon task BrowseComp (o3: +12.7pp, DeepSeek: +19.2pp).
These gains validate two key insights:
(1) The iterative paradigm with workspace reconstruction provides a more effective cognitive structure for long-horizon reasoning, enabling models to maintain focus through periodic synthesis rather than drowning in accumulated context.
(2) The paradigm’s benefits are model-agnostic—both o3 and DeepSeek model architectures exhibit substantial improvements, suggesting that our approach addresses fundamental limitations in how current models handle extended reasoning chains rather than model-specific weaknesses.
The improvements peak on BrowseComp—the most exploration-intensive benchmark—confirming that our paradigm’s advantages scale with task horizon length, making it particularly valuable for complex real-world problems.

5 Conclusion
------------

In this work, we presented IterResearch, a novel iterative deep-research paradigm that addresses the context suffocation and noise contamination plaguing mono-contextual approaches in long-horizon tasks.
By extending the Markov Decision Process to deep research with strategic workspace reconstruction and developing Efficiency-Aware Policy Optimization for effective training, we achieved substantial improvements over existing agents (average +14.5pp across six benchmarks).
Furthermore, our experiments reveal three transformative insights: this iterative paradigm enables unprecedented interaction scaling to 2048 interactions with dramatic performance gains (3.5% to 42.5%), serves as an effective prompting strategy that improves frontier models by up to 19.2pp, and induces superior exploration behaviors transferable across different paradigms.
These findings establish that iteration with strategic synthesis, rather than accumulation, is fundamental to conquering long-horizon reasoning challenges, providing both a powerful agent architecture and a versatile framework applicable across different models and paradigms.

Appendix A Addtional Related Work
---------------------------------

Memory Mechanisms in LLMs. Memory mechanisms have emerged as a critical component for extending LLM capabilities beyond single-turn interactions*[du2025rethinking, zhang2025learn]*.
While early works explored explicit memory architectures with separate storage and retrieval modules*[yang2024text]*, recent approaches have focused on memory management for LLM agents.
MemoryLLM*[hu2025evaluating]* and MEM1*[zhou2025mem1]* investigate how agents can learn to synthesize and utilize memory across multi-turn interactions, while Memory-R1*[yan2025memory]* employs reinforcement learning to train agents for adaptive memory management.
MemAgent*[yu2025memagent]* and MemOS*[li2025memos]* further advance this direction by introducing memory operating systems that unify representation, scheduling, and evolution of memories as manageable system resources.
However, these memory-centric approaches primarily focus on explicit memory module design or retrieval optimization within fixed context windows, fundamentally differing from our approach.
IterResearch naturally integrates memory through the evolving report $\mathcal{M}_{t}$ within our Markovian workspace reconstruction—rather than maintaining separate memory modules or databases, our report serves as a compressed, task-focused memory that is seamlessly updated through the agent’s structured decisions.
This design eliminates the overhead of explicit memory management while ensuring that memory evolution is intrinsically aligned with the research trajectory, enabling more efficient and coherent long-horizon exploration.

Appendix B More Analysis
------------------------

### B.1 Theoretical Motivation: Efficiency through Discounting

The discounted reward formulation in Eq.[3] elegantly encodes a preference for efficiency that emerges naturally from the MDP framework. To illustrate this, consider two successful research trajectories for the same question: trajectory $\tau_{A}$ reaching the correct answer in $T_{A}\=5$ steps, and trajectory $\tau_{B}$ requiring $T_{B}\=20$ steps.

Under our discounting scheme with $\gamma\=0.995$, each step in the trajectories receives different rewards based on its temporal distance from the terminal state. For any intermediate step $t$, the rewards are:

|  | $\displaystyle r_{t}^{A}$ | $\displaystyle\=\gamma^{T_{A}-t}\cdot R_{T}\=\gamma^{5-t}$ |  | (6) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle r_{t}^{B}$ | $\displaystyle\=\gamma^{T_{B}-t}\cdot R_{T}\=\gamma^{20-t}$ |  | (7) |
| --- | --- | --- | --- | --- |

This creates a fundamental learning signal: earlier steps in shorter trajectories receive substantially higher rewards than corresponding steps in longer trajectories. To illustrate the magnitude of this difference, consider the reward at step $t\=3$:

|  | $\displaystyle r_{3}^{A}$ | $\displaystyle\=\gamma^{5-3}\=\gamma^{2}\approx 0.99$ |  | (8) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle r_{3}^{B}$ | $\displaystyle\=\gamma^{20-3}\=\gamma^{17}\approx 0.918$ |  | (9) |
| --- | --- | --- | --- | --- |

The 7.8% reward difference for the same step position creates a strong gradient that guides the policy toward more efficient research strategies. This consistent multiplicative advantage across all shared steps systematically guides the policy toward discovering more efficient research strategies.

* •

    Redundant exploration: Searching for similar information multiple times delays progress, with each redundant step reducing future rewards by factor $\gamma$

* •

    Circular reasoning: Revisiting previously explored hypotheses without new insights wastes steps, exponentially diminishing the trajectory’s total return

* •

    Unfocused browsing: Following tangential information that doesn’t contribute to the final answer accumulates geometric penalties

Importantly, this efficiency incentive emerges without any explicit length penalty or auxiliary objectives—it is an inherent property of geometric discounting applied to our MDP formulation. The discount factor $\gamma$ serves as a single hyperparameter that controls the trade-off between exploration thoroughness and efficiency: values closer to 1 allow more exploratory behavior, while smaller values create stronger pressure for direct problem-solving. Our empirical choice of $\gamma\=0.995$ strikes a balance that permits necessary exploration while maintaining sufficient efficiency pressure, as validated by the 5.7% reduction in average trajectory length observed in our ablation studies (Table[2]).

### B.2 Computational Complexity Analysis.

Unlike mono-contextual approaches where context size grows as $O(t\cdot|\text{TR}|)$ with $t$ rounds and average response size $|\text{TR}|$, our algorithm maintains a constant workspace size of $O(|\mathcal{M}|+|\text{TR}|)$, where $|\mathcal{M}|$ is the report size bounded by design through the agent’s learned synthesis behavior. This ensures consistent computational efficiency regardless of the research depth. Table[3] provides a detailed complexity comparison.
In the Table, $t$ is the number of rounds, $|\text{TR}|$ is the average tool response size, $|\mathcal{M}|$ is the bounded report size, and $L$ is the model’s context limit.
The key distinctions are:

*Table 3: Computational complexity comparison between paradigms.*

| Metric | Mono-contextual | IterResearch (Ours) |
| --- | --- | --- |
| Used Context Size | $O(t\cdot|\text{TR}|)$ | $O(|\mathcal{M}|+|\text{TR}|)$ |
| Attention Computation | $O((t\cdot|\text{TR}|)^{2})$ | $O((|\mathcal{M}|+|\text{TR}|)^{2})$ |
| Effective Reasoning Window | $O(\max(0,L-t\cdot|\text{TR}|))$ | $O(L-|\mathcal{M}|-|\text{TR}|)$ |
| Maximum Rounds | $O(L/|\text{TR}|)$ | $O(\infty)$ (theoretically unbounded) |

* •

    Used Context Size: Mono-contextual approaches accumulate all past responses, growing linearly with rounds until reaching the context limit. Our approach maintains constant size through workspace reconstruction, with the report $\mathcal{M}$ serving as a compressed memory that synthesizes all essential findings.

* •

    Attention Computation: The quadratic attention cost becomes prohibitive for mono-contextual approaches as $t$ increases, with complexity scaling as $O((t\cdot|\text{TR}|)^{2})$. Our bounded workspace ensures consistent computational cost of $O((|\mathcal{M}|+|\text{TR}|)^{2})$ per round, independent of trajectory length.

* •

    Effective Reasoning Window: In mono-contextual approaches, the available context for new reasoning diminishes as $\max(0,L-t\cdot|\text{TR}|)$, eventually reaching zero when accumulated history exhausts the context limit. Our approach maintains a consistent reasoning window of $L-|\mathcal{M}|-|\text{TR}|$ across all rounds, ensuring sustainable reasoning capacity throughout the research process.

* •

    Maximum Rounds: Mono-contextual approaches face a hard limit of approximately $L/|\text{TR}|$ rounds before context overflow. In contrast, our iterative paradigm is theoretically unbounded—as long as $|\mathcal{M}|+|\text{TR}|<L$ (which is maintained through report synthesis), the agent can continue exploration indefinitely.

These complexity advantages become critical in long-horizon tasks: while mono-contextual approaches face inevitable failure when $t\cdot|\text{TR}|>L$ (context overflow), our approach can theoretically extend to arbitrary depths. This theoretical advantage translates to practical benefits, as empirically demonstrated in our scaling experiments (Figure[3]), where we successfully extend agents to 2048 interactions using only 40K context length—a feat structurally impossible for mono-contextual approaches.

The constant complexity also ensures predictable resource consumption: each round requires approximately the same computational resources regardless of position in the trajectory, enabling better resource planning and allocation in deployment scenarios. This predictability, combined with the unbounded exploration capability, makes our iterative paradigm particularly suitable for genuinely complex research tasks that may require extensive investigation.

### B.3 Extrapolation Beyond Training Horizon

A remarkable property of our iterative paradigm is its ability to extrapolate far beyond the training horizon. While we train with $T_{\max}\=32$ to promote efficient research strategies, the learned agent can seamlessly operate with $T_{\max}\=2048$ or even higher during inference—a 64× extrapolation factor that would be structurally impossible for mono-contextual approaches.

This extrapolation capability is enabled by two fundamental design choices:

* •

    Markovian Workspace: Each round’s decision depends only on the current reconstructed state $(q,\mathcal{M}_{t},{a_{t-1},\text{TR}_{t-1}})$, not on absolute position $t$ or the full trajectory history. This position-agnostic design ensures that the agent’s decision-making process remains consistent whether at round 10 or round 1000.

* •

    Report-based Memory: The evolving report $\mathcal{M}_{t}$ provides a scale-invariant representation of research progress. Unlike raw trajectory accumulation, the report’s bounded complexity ensures the state distribution remains stable regardless of trajectory length, allowing coherent reasoning at any depth.

We deliberately constrain training to $T_{\max}\=32$ for strategic reasons: (1) it provides sufficient signal for learning effective research strategies while keeping computational costs manageable, and (2) it creates pressure for the agent to develop concise exploration patterns rather than relying on exhaustive search. This constrained training paradoxically enhances extrapolation—by learning to maximize information gain within limited rounds, the agent develops robust strategies that scale gracefully when given additional capacity.

Our experiments (Figure[3]) empirically validate this extrapolation capability: agents trained with $T_{\max}\=32$, achieve 42.5% accuracy on BrowseComp when extended to $T_{\max}\=2048$ during inference, compared to only 15.2% with $T_{\max}\=32$. This dramatic improvement demonstrates that the agent effectively utilizes the additional exploration capacity without any degradation in decision quality or coherence.

##### Contrast with Mono-contextual Limitations.

Mono-contextual approaches face fundamental barriers to such extreme extrapolation:

* •

    Position Embedding Overflow: Absolute position encodings trained on sequences of length 32 often produce undefined or degraded representations beyond the training range

* •

    Attention Pattern Collapse: Attention distributions learned on short sequences fail to generalize to dramatically longer contexts, leading to degenerate focus patterns

* •

    Context Saturation: The accumulated context from 2048 rounds would exceed most models’ context limits, causing hard failures rather than graceful degradation

##### Theoretical Foundation.

The extrapolation capability stems directly from our MDP formulation where the optimal policy is defined over states, not trajectory positions. Since our state space $\mathcal{S}$ and decision space $\mathcal{D}$ remain constant regardless of horizon length, a policy learned on shorter trajectories naturally generalizes to longer ones, provided the state distribution remains similar. The report synthesis mechanism ensures this distributional stability by maintaining bounded complexity $O(|\mathcal{M}|)$ regardless of trajectory length, preventing the distribution shift that would otherwise occur with unbounded context accumulation.

This extrapolation capability fundamentally expands the applicability of our approach: agents can be efficiently trained on moderate-length trajectories yet deployed on arbitrarily complex tasks requiring extensive exploration, providing a practical path to handling real-world research challenges of unknown complexity.

### B.4 Training Dynamics of Efficiency-Aware Policy Optimization

<img src='x6.png' alt='Refer to caption' title='' width='324' height='233' />

<img src='x7.png' alt='Refer to caption' title='' width='310' height='219' />

*Figure 5: Training dynamics of our RL. (Left) Training Rewards Curve. (Right) Accuracy Curve.*

Figure[5] illustrates the training dynamics of our EAPO framework across 150 optimization steps.

##### Reward Convergence.

The left panel demonstrates stable convergence with training rewards increasing from 0.55 to approximately 0.72, representing a 30.9% improvement. The smooth EMA curve exhibits only minor oscillations, confirming that our adaptive downsampling successfully handles variable sample counts from our iterative paradigm while maintaining stable gradient signals. The consistent upward trend without plateauing suggests the geometric discounting continues to provide meaningful learning signals throughout training.

##### Performance Evolution.

The right panel reveals distinct learning patterns that reflect fundamental differences in task characteristics:

* •

    BrowseComp (English): The sharp performance jump from 32% to 39% at step 50 followed by stabilization suggests the agent discovers critical search strategies—likely effective query reformulation or result filtering patterns specific to English web content. The subsequent plateau indicates these strategies generalize robustly.

* •

    BrowseComp-zh (Chinese): The monotonic improvement from 40% to 45% reflects a smoother optimization landscape, possibly due to more structured Chinese web content or different information organization patterns that allow incremental strategy refinement.

The correlation between reward growth and performance improvement validates our core hypothesis: geometric discounted rewards successfully guide the agent toward more efficient exploration. Notably, the reward improvement (30.9%) exceeds the accuracy gains (BC: 18.8%, BC-zh: 12.5%), indicating the agent learns not just to solve tasks but to solve them efficiently. This is empirically confirmed in our ablation studies (Table[2]), where EAPO achieves 5.7% shorter trajectories than standard GSPO while maintaining comparable accuracy, demonstrating that our reward design successfully shapes more focused exploration behaviors without compromising task performance.

Appendix C More Implementation Details
--------------------------------------

In this section, we provide a comprehensive implementation details of our proposed method. For additional insights and more intricate details, we refer the reader to our [Github Repo](https://github.com/Alibaba-NLP/DeepResearch "").

### C.1 Algorithmic Framework

Algorithm[1] presents the complete procedure of our iterative deep-research paradigm.

*Algorithm 1  Iterative Deep-Research (IterResearch)*

1:Question $q$, Agent model $\pi$, Environment $\mathcal{E}$, Max rounds $T_{\max}$

2:Final answer to $q$

3:Initialize: $\mathcal{M}_{0}\leftarrow\emptyset$, $s_{0}\leftarrow(q,\mathcal{M}_{0},\emptyset)$, $t\leftarrow 0$ $\triangleright$Empty report and context









12:return final answer

The algorithm proceeds through discrete research rounds, where each round $t$ follows a structured sequence:

1. 1.

    Decision Generation (Lines 3-4): The agent $\pi$ processes the current state $s_{t}$ to produce a structured decision $d_{t}$, which is then parsed into three components: reasoning ($\text{Think}_{t}$), updated report ($\mathcal{M}_{t+1}$), and next action ($a_{t}$).

2. 2.

    Termination Check (Lines 5-7): If the agent outputs $a_{t}\=\texttt{answer}$, the algorithm terminates with the agent’s final answer. This allows autonomous determination of information sufficiency.

3. 3.

    Tool Execution (Line 8): For non-terminal actions, the environment $\mathcal{E}$ executes the requested tool (search, browse, compute) and returns the response $\text{TR}_{t}$.

4. 4.

    Workspace Reconstruction (Line 9): The crucial step distinguishing our paradigm—instead of appending to an ever-growing context, we reconstruct a bounded workspace containing only the question $q$, updated report $\mathcal{M}_{t+1}$, and latest interaction ${a_{t},\text{TR}_{t}}$.

##### Report Evolution Mechanism.

The report $\mathcal{M}_{t+1}$ serves as the agent’s evolving memory, dynamically synthesizing information across rounds. At each step, the agent updates the report by incorporating new findings from $\text{TR}_{t}$ while preserving essential insights from $\mathcal{M}_{t}$. This selective retention ensures critical findings persist while redundant information is filtered, maintaining bounded complexity regardless of trajectory length.

##### Termination Conditions.

The algorithm terminates under two conditions:
(1) Natural Termination: The agent determines sufficient information has been gathered and outputs an answer (Line 6).
(2) Forced Termination: The round counter reaches $T_{\max}$ (Line 11), preventing infinite loops.

### C.2 Tool Environment

Our environment $\mathcal{E}$ provides four complementary tools that enable comprehensive research capabilities. Each tool is designed to handle specific aspects of the research process, from information gathering to computational analysis.
We provide the detailed tool schema in Appendix[E.1].
We implement the tool environment using production-grade APIs and services:

* •

    Google Search: Returns top-10 search results with snippets for general web queries. The tool accepts multiple queries in a single call, enabling efficient batch searching. Each result includes title, URL, and a brief snippet, providing the agent with sufficient context to determine relevance before deeper exploration.

* •

    Google Scholar: Returns top-10 search results with snippets for academic papers, citations, and scholarly metadata. Similar to web search, it supports batch queries and returns structured bibliographic information including authors, publication venues, citation counts, and abstract snippets. The tool also includes fallback to general web search for comprehensive coverage.
    Both Google Search and Google Scholar are accessed via SerpAPI111<https://serpapi.com/>, providing reliable and rate-limited access to search results.

* •

    Visit (Web Browser): Enables detailed content extraction from specific URLs with goal-oriented summarization. The agent specifies both the target URLs and a specific goal (e.g., "find the methodology section" or "extract statistical results"), allowing focused information extraction. The tool handles both HTML webpages and PDF documents, automatically detecting and parsing the appropriate format. Our summarization model (Qwen3-30B-A3B) processes the raw content with the agent’s goal to produce concise, relevant summaries.
    We employ Jina Reader222<https://jina.ai/> for robust web content extraction.

* •

    Python Interpreter: Executes arbitrary Python code in a secure, sandboxed environment for computational tasks and data analysis. The interpreter comes with standard libraries (NumPy, Pandas, Matplotlib, etc.) pre-installed and can handle complex calculations, data manipulations, and logical operations. All outputs must be explicitly printed, ensuring clear communication of results back to the agent.
    We use Code Sandbox333<https://github.com/bytedance/SandboxFusion>, ensuring secure and isolated computation.

### C.3 Implementation Details

This section provides comprehensive implementation details of our IterResearch.
We also provide all of the code and training data for easy reproduction.

*Table 4: Key hyperparameters in the supervised warm-up phase.*

| Hyperparameter | Value |
| --- | --- |
| Learning Rate | 1e-5 |
| Batch size | 512 |
| #Epochs | 3 |
| Chat template | Qwen [yang2025qwen3] |
| Maximum Context Length (Prompt + Response) | 40960 |
| Warmup ratio | 0.03 |
| LR scheduler type | Cosine |

##### Supervised Fine-tuning Phase.

Since existing LLMs lack inherent capabilities for our iterative deep-research paradigm, we conduct a two-stage data preparation process: Stage 1: High-quality QA Collection. We curate 30K high-quality question-answer pairs from recent web research datasets*[li2025websailor, tao2025webshaper, chen2025expanding, webresearcher]*. These pairs are filtered based on answer quality, factual accuracy, and research complexity to ensure they require genuine multi-step investigation. Stage 2: Trajectory Synthesis. To bridge the gap between standard QA pairs and our iterative paradigm, we employ Qwen3-235B-A22B*[yang2025qwen3]* to synthesize research trajectories following our framework. This process yields 110K training trajectories with an average of 3.7 rounds per trajectory, providing rich supervision for learning the iterative research pattern.
We utilize Slime444<https://github.com/THUDM/slime> as our training framework for the initial supervised fine-tuning phase. The detailed hyper-parameters for this phase are presented in Table[4].

*Table 5: Key hyperparameters in the RL phase.*

| Hyperparameter | Value |
| --- | --- |
| Learning Rate | 1e-6 |
| Base model | Qwen3-30B-A3B[yang2025qwen3] |
| Batch size | 16 |
| Group size per Question ($G$) | 16 |
| temperature | 1.0 |
| top p | 0.95 |
| KL loss coefficient ($\lambda$) | 0. |
| entropy coefficient | 0. |
| Maximum Context Length (Prompt + Response) | 40960 |
| Maximum interaction rounds ($T_{\max}$) | 32 |

##### Reinforcement Learning Phase.

We employ a strategic data selection process to identify questions with optimal learning potential:
(1) Difficulty Calibration: Using the best checkpoint from SFT, we evaluate each of the 30K questions with 5 independent trials, recording success rates.
(2) Learning Zone Selection: We retain questions with success rates between 20%-60% (1-3 correct out of 5 attempts), identifying 4,096 questions that fall within the model’s "zone of proximal development"—challenging enough to provide learning signal but achievable enough to generate successful trajectories. Questions that are too easy ($>60\%$ success) provide weak learning signals, while overly difficult questions ($<20\%$ success) lead to sparse rewards and unstable training.
Table[5] summarizes the key hyperparameters used during the reinforcement learning phase.
We also use Slime as our RL frameowrk due to its efficient and easy to use.

*Table 6: Maximum round settings across different stages and benchmarks.*

| Stage/Benchmark | $T_{\max}$ |
| --- | --- |
| Training | 32 |
| Inference Phase: |  |
| GAIA[mialon2023gaia] | 32 |
| HLE[hle] | 64 |
| BrowseComp-zh[bc_zh] | 64 |
| BrowseComp[bc_en] | 256 |

##### Maximum Round Settings.

We adopt task-adaptive $T_{\max}$ values to balance training efficiency with inference flexibility. Table[6] summarizes our configuration across different stages and benchmarks.
We constrain $T_{\max}\=32$ during both SFT and RL phases to instill efficiency-oriented behaviors. This limit, combined with our geometric reward discounting (Equation[3]), creates strong incentives for the agent to develop concise research strategies rather than exhaustive exploration patterns.
During inference, we adjust $T_{\max}$ based on benchmark characteristics.
This adaptive configuration ensures that simple tasks remain efficient while complex questions have sufficient exploration budget, all while maintaining the efficiency patterns learned during training.

##### Reward Design.

We employ LLM-as-judge evaluation following established practices*[chen2025cpo]*. Specifically, we use Qwen3-235B-A22B to assess answer correctness:

|  | $\displaystyle R_{T}\=\begin{cases}1.0\&\text{if answer is correct}\\ 0.0\&\text{otherwise}\end{cases}$ |  | (10) |
| --- | --- | --- | --- |

Appendix D Case Study of IterResearch
-------------------------------------

We present a representative example demonstrating how IterResearch solves a complex biology question through iterative research. This case highlights three key capabilities: (1) evolving report synthesis, (2) efficient information gathering, and (3) autonomous termination decision.


Appendix E Instruction Templates
--------------------------------

### E.1 Tool Schema Specification

The agent interacts with tools through a structured schema that defines available functions and their parameters. Below we present the complete tool specifications used in our system.

*Listing 1: Google Search Tool Schema*

[⬇](data:text/plain;base64,ewogICJ0eXBlIjogImZ1bmN0aW9uIiwKICAiZnVuY3Rpb24iOiB7CiAgICAibmFtZSI6ICJnb29nbGVfc2VhcmNoIiwKICAgICJkZXNjcmlwdGlvbiI6ICJQZXJmb3JtIEdvb2dsZSB3ZWIgc2VhcmNoZXMgdGhlbiByZXR1cm5zIGEKICAgICAgICAgICAgICAgICAgICBzdHJpbmcgb2YgdGhlIHRvcCBzZWFyY2ggcmVzdWx0cy4gQWNjZXB0cwogICAgICAgICAgICAgICAgICAgIG11bHRpcGxlIHF1ZXJpZXMuIiwKICAgICJwYXJhbWV0ZXJzIjogewogICAgICAidHlwZSI6ICJvYmplY3QiLAogICAgICAicHJvcGVydGllcyI6IHsKICAgICAgICAicXVlcnkiOiB7CiAgICAgICAgICAidHlwZSI6ICJhcnJheSIsCiAgICAgICAgICAiaXRlbXMiOiB7InR5cGUiOiAic3RyaW5nIn0sCiAgICAgICAgICAibWluSXRlbXMiOiAxLAogICAgICAgICAgImRlc2NyaXB0aW9uIjogIlRoZSBsaXN0IG9mIHNlYXJjaCBxdWVyaWVzLiIKICAgICAgICB9CiAgICAgIH0sCiAgICAgICJyZXF1aXJlZCI6IFsicXVlcnkiXQogICAgfQogIH0KfQ==)

{

"type":"function",

"function":{

"name":"google_search",

"description":"PerformGooglewebsearchesthenreturnsa

stringofthetopsearchresults.Accepts

multiplequeries.",

"parameters":{

"type":"object",

"properties":{

"query":{

"type":"array",

"items":{"type":"string"},

"minItems":1,

"description":"Thelistofsearchqueries."

}

},

"required":["query"]

}

}

}

*Listing 2: Google Scholar Tool Schema*

[⬇](data:text/plain;base64,ewogICJ0eXBlIjogImZ1bmN0aW9uIiwKICAiZnVuY3Rpb24iOiB7CiAgICAibmFtZSI6ICJnb29nbGVfc2Nob2xhciIsCiAgICAiZGVzY3JpcHRpb24iOiAiTGV2ZXJhZ2UgR29vZ2xlIFNjaG9sYXIgdG8gcmV0cmlldmUgcmVsZXZhbnQKICAgICAgICAgICAgICAgICAgICBpbmZvcm1hdGlvbiBmcm9tIGFjYWRlbWljIHB1YmxpY2F0aW9ucy4gVGhpcwogICAgICAgICAgICAgICAgICAgIHRvb2wgYWxzbyByZXR1cm5zIHJlc3VsdHMgZnJvbSBHb29nbGUgc2VhcmNoLiIsCiAgICAicGFyYW1ldGVycyI6IHsKICAgICAgInR5cGUiOiAib2JqZWN0IiwKICAgICAgInByb3BlcnRpZXMiOiB7CiAgICAgICAgInF1ZXJ5IjogewogICAgICAgICAgInR5cGUiOiAiYXJyYXkiLAogICAgICAgICAgIml0ZW1zIjogeyJ0eXBlIjogInN0cmluZyJ9LAogICAgICAgICAgIm1pbkl0ZW1zIjogMSwKICAgICAgICAgICJkZXNjcmlwdGlvbiI6ICJUaGUgbGlzdCBvZiBzZWFyY2ggcXVlcmllcy4iCiAgICAgICAgfQogICAgICB9LAogICAgICAicmVxdWlyZWQiOiBbInF1ZXJ5Il0KICAgIH0KICB9Cn0=)

{

"type":"function",

"function":{

"name":"google_scholar",

"description":"LeverageGoogleScholartoretrieverelevant

informationfromacademicpublications.This

toolalsoreturnsresultsfromGooglesearch.",

"parameters":{

"type":"object",

"properties":{

"query":{

"type":"array",

"items":{"type":"string"},

"minItems":1,

"description":"Thelistofsearchqueries."

}

},

"required":["query"]

}

}

}

*Listing 3: Visit (Web Browser) Tool Schema*

[⬇](data:text/plain;base64,ewogICJ0eXBlIjogImZ1bmN0aW9uIiwKICAiZnVuY3Rpb24iOiB7CiAgICAibmFtZSI6ICJWaXNpdCIsCiAgICAiZGVzY3JpcHRpb24iOiAiVmlzaXQgd2VicGFnZShzKSBvciBwYXBlcihzKSBhbmQgcmV0dXJuCiAgICAgICAgICAgICAgICAgICAgdGhlIHN1bW1hcnkgb2YgdGhlIGNvbnRlbnQuIiwKICAgICJwYXJhbWV0ZXJzIjogewogICAgICAidHlwZSI6ICJvYmplY3QiLAogICAgICAicHJvcGVydGllcyI6IHsKICAgICAgICAidXJsIjogewogICAgICAgICAgInR5cGUiOiAiYXJyYXkiLAogICAgICAgICAgIml0ZW1zIjogeyJ0eXBlIjogInN0cmluZyJ9LAogICAgICAgICAgIm1pbkl0ZW1zIjogMSwKICAgICAgICAgICJkZXNjcmlwdGlvbiI6ICJUaGUgVVJMKHMpIHRvIHZpc2l0LiIKICAgICAgICB9LAogICAgICAgICJnb2FsIjogewogICAgICAgICAgInR5cGUiOiAic3RyaW5nIiwKICAgICAgICAgICJkZXNjcmlwdGlvbiI6ICJUaGUgZ29hbCBvZiB0aGUgdmlzaXQuIgogICAgICAgIH0sCiAgICAgICAgInBhcnNlX3R5cGUiOiB7CiAgICAgICAgICAidHlwZSI6ICJzdHJpbmciLAogICAgICAgICAgImVudW0iOiBbImh0bWwiLCAicGRmIl0sCiAgICAgICAgICAiZGVmYXVsdCI6ICJodG1sIiwKICAgICAgICAgICJkZXNjcmlwdGlvbiI6ICJTcGVjaWZ5ICdodG1sJyBvciAncGRmJyBmb3JtYXQuIgogICAgICAgIH0KICAgICAgfSwKICAgICAgInJlcXVpcmVkIjogWyJ1cmwiLCAiZ29hbCJdCiAgICB9CiAgfQp9)

{

"type":"function",

"function":{

"name":"Visit",

"description":"Visitwebpage(s)orpaper(s)andreturn

thesummaryofthecontent.",

"parameters":{

"type":"object",

"properties":{

"url":{

"type":"array",

"items":{"type":"string"},

"minItems":1,

"description":"TheURL(s)tovisit."

},

"goal":{

"type":"string",

"description":"Thegoalofthevisit."

},

"parse_type":{

"type":"string",

"enum":["html","pdf"],

"default":"html",

"description":"Specify’html’or’pdf’format."

}

},

"required":["url","goal"]

}

}

}

*Listing 4: Python Interpreter Tool Schema*

[⬇](data:text/plain;base64,ewogICJ0eXBlIjogImZ1bmN0aW9uIiwKICAiZnVuY3Rpb24iOiB7CiAgICAibmFtZSI6ICJQeXRob25JbnRlcnByZXRlciIsCiAgICAiZGVzY3JpcHRpb24iOiAiRXhlY3V0ZXMgUHl0aG9uIGNvZGUgaW4gYSBzZWN1cmUgc2FuZGJveC4KICAgICAgICAgICAgICAgICAgICBEZXNpZ25lZCBmb3IgY2FsY3VsYXRpb25zLCBkYXRhIG1hbmlwdWxhdGlvbnMsCiAgICAgICAgICAgICAgICAgICAgYW5kIGdlbmVyYWwgcHJvZ3JhbW1pbmcgdGFza3MuIiwKICAgICJwYXJhbWV0ZXJzIjogewogICAgICAidHlwZSI6ICJvYmplY3QiLAogICAgICAicHJvcGVydGllcyI6IHsKICAgICAgICAiY29kZSI6IHsKICAgICAgICAgICJ0eXBlIjogInN0cmluZyIsCiAgICAgICAgICAiZGVzY3JpcHRpb24iOiAiVGhlIFB5dGhvbiBjb2RlIHRvIGV4ZWN1dGUuIE91dHB1dAogICAgICAgICAgICAgICAgICAgICAgICAgIG11c3QgdXNlIHByaW50KCkgZnVuY3Rpb25zLiIKICAgICAgICB9CiAgICAgIH0sCiAgICAgICJyZXF1aXJlZCI6IFsiY29kZSJdCiAgICB9CiAgfQp9)

{

"type":"function",

"function":{

"name":"PythonInterpreter",

"description":"ExecutesPythoncodeinasecuresandbox.

Designedforcalculations,datamanipulations,

andgeneralprogrammingtasks.",

"parameters":{

"type":"object",

"properties":{

"code":{

"type":"string",

"description":"ThePythoncodetoexecute.Output

mustuseprint()functions."

}

},

"required":["code"]

}

}

}

### E.2 Instruction of our IterResearch
