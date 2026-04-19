Q-RAG: Long Context Multi‑Step Retrieval  via Value‑Based Embedder Training
============================================================================

Artyom Sorokin1,2,
Nazar Buzun3,
Alexander Anokhin 1,
Oleg Inozemcev1,
Egor Vedernikov1,  
Petr Anokhin2,
Mikhail Burtsev4,
Trushkov Alexey6,
Yin Wenshuai5,
Evgeny Burnaev1,2  
1Applied AI, Moscow, Russia  
2Learnable Intelligence Lab, Moscow, Russia  
3CILAB.AI, Moscow, Russia  
4London Institute for Mathematical Sciences, London, UK  
5Higher School of Economics, Moscow, Russia  
6Independent Researcher  
griver29@gmail.com, n.buzun@cilab.ai

###### Abstract

Retrieval-Augmented Generation (RAG) methods enhance LLM performance by efficiently filtering relevant context for LLMs, reducing hallucinations and inference cost.
However, most existing RAG methods focus on single-step retrieval, which is often insufficient for answering complex questions that require multi-step search.
Recently, multi-step retrieval approaches have emerged, typically involving the fine-tuning of small LLMs to perform multi-step retrieval.
This type of fine-tuning is highly resource-intensive and does not enable the use of larger LLMs.
In this work, we propose Q-RAG, a novel approach that fine-tunes the Embedder model for multi-step retrieval using reinforcement learning (RL).
Q-RAG offers a competitive, resource-efficient alternative to existing multi-step retrieval methods for open-domain question answering and achieves state-of-the-art results on the popular long-context benchmarks Babilong and RULER for contexts up to 10M tokens.

1 Introduction
--------------

Large language models (LLMs) have achieved impressive results across a wide range of tasks*(alphaevolve_novikov2025; deepseek_guo2025; llm_survey_yang2025)*.
However, they still face some several fundamental limitations such as static knowledge, computational inefficiency on long contexts, degraded performance caused by attention dilution, and hallucinations*(ruler_hsieh2024; babilong2024; long_context_survey_liu2025)*. Retrieval-Augmented Generation (RAG) is one of the most widely used techniques to address these issues*(rag_defence_yu2024)*.

RAG works by extracting only the most relevant parts from a large external corpus or context, such as newly added knowledge or lengthy texts. This allows LLMs to operate on shorter and more focused inputs, improving efficiency and output quality. Most current RAG methods rely on single-step retrieval. This setup performs well in relatively simple tasks like Needle-in-a-Haystack*(ruler_hsieh2024)*. Still, more complex problems require multi-step interaction with the context. Multi-step retrieval can be viewed as a form of search-based reasoning. There are several existing approaches to multi-step retrieval reasoning. One direction involves constructing a knowledge graph from the retrieved information *(ma2025largelanguagemodelsmeet; graph_reader_li2024)*. These methods are often slow at inference time, since the LLM must process the entire context to build the graph for each new input. Another line of work uses LLM agents, which interleave RAG queries with LLM-generated instructions *(singh2025agenticretrievalaugmentedgenerationsurvey; arigraph_anokhin2024)*. These systems are sensitive to noisy or inaccurate retrieved passages, which may disrupt the generation of future queries. This shows the need for joint optimization of the retrieval and generation components. Recently, methods have emerged that fine-tune LLMs to interact more effectively with retrieval tools*(r1_searcher_song2025; search_r1_2025; research_chen2025)*. These methods tend to perform better, but they require expensive fine-tuning of the LLM itself. This makes them impractical for large models and limits accessibility for most researchers and practitioners.

In this work, we focus on developing a resource-efficient multi-step RAG approach using reinforcement learning. Instead of fine-tuning an LLM, we train an agent that performs retrieval directly in the latent space of text chunk embeddings. This allows us to learn a compact and efficient model using value-based RL methods.

Our approach achieves state-of-the-art results on long-context commonsence reasoning, multi-hop QA, and NIAH tasks with contexts up to 10 million tokens. It also performs competitively on open-domain QA benchmarks such as Musique and HotpotQA*(hotpotqa_yang2018; hotpotqa_yang2018)*, while being significantly faster and cheaper to train and run compared to existing multi-step RAG methods.
Our contributions are the following:

* •

    We propose a new method for training a multi-step retrieval agent using temporal difference reinforcement learning.

* •

    We achieve state-of-the-art results on benchmarks that require commonsense reasoning and NIAH tasks over ultra long contexts (up to 10M tokens).

* •

    We introduce a new way to incorporate temporal information into the multi-step embedder, enabling temporal reasoning during retrieval. Our temporal reasoning mechanism generalizes well to long contexts at inference time.

2 Related Works
---------------

There are several main directions for tackling complex retrieval scenarios on long context tasks.

A highly popular approach involves building fine-tuning free LLM Agents that combine off-the-shelf retrievers with LLMs, such as Search-o1*(search_o1_li2025)*. Many of these works further enhance retrieval quality by constructing large knowledge graphs over the context, which, while requiring little additional training, are extremely slow at inference due to the need for LLMs to process the entire context, e.g. GraphReader*(graph_reader_li2024)*, HippoRAG*(hipporag_jimenez2024)*, AriGraph*(arigraph_anokhin2024)*.

Another line of work fine-tunes LRMs to perform multi-step retrieval, allowing the model to generate intermediate search queries inside the reasoning for long contexts. The first work to apply this idea was IM-RAG*(im_rag_yang2024)*, which fine-tuned the LLM with a frozen embedder using PPO*(ppo_schulman2017)*. More recent papers, such as R1-Searcher*(r1_searcher_song2025)*, Search-R1*(search_r1_2025)*, RAG-RL*(rag_rl_huang2025)*, and ReSearcher*(research_chen2025)*, extended this direction by employing GRPO*(grpo_deepseek2024)* for the task. Unlike these methods, which freeze the embedder and fine-tune the LLM, our approach fine-tunes only the embedder, allowing it to pair with LLMs of any size, including proprietary ones, while keeping fine-tuning efficient and inexpensive.

A different approach is to fine-tune the retriever itself using feedback from the LLM, as in RePlug*(replug_shi2024)*. This direction is most similar to ours, but RePlug did not address multi-step reasoning or use reinforcement learning in this setting. BeamRetriever*(beam_retriever_zhang2024)* achieves state-of-the-art results on short-context QA by training a reranker for BeamSearch-style planning. In contrast, Q-RAG trains the embedder with reinforcement learning, enabling faster inference and better scalability to long contexts through efficient vector similarity instead of transformer-based trajectory scoring.

Extremely long-sequence processing is demonstrated by models that combine recurrence with Transformer architecture. The Mamba family of state space models*(gu2024mambalineartimesequencemodeling)* replaces attention with structured recurrent dynamics, offering linear-time scalability and strong performance on long sequences, though often at the cost of weaker in-context learning and less expressive token-to-token interaction compared to Transformer-based architectures.
The Recurrent Memory Transformer (RMT)*(rmt2022)* introduces segment-level recurrence by passing memory tokens between fixed-size segments, enabling Q\&A on sequences up to 10M tokens. Titans*(titans2024)* frames recurrent memory training as a meta-learning problem, showing scaling beyond 2M tokens. Building on this idea, ATLAS*(atlas2025)* increases memory capacity, achieving better long-context performance than both RMT and Titans. The Associative Recurrent Memory Transformer (ARMT)*(armt2024)* employs quasi-linear, associative attention in each layer and attains the best long-context scores among recurrent models. Our approach outperforms all of these models on contexts beyond 1M tokens while belonging to a different class of methods.

LongRoPE2*(longrope2_shang2025)* tackles the positional encoding bottleneck, extending the effective context window of pre-trained LLMs to 128K tokens while retaining short-context performance through RoPE rescaling and mixed-window training.

3 Methods
---------

<img src='x1.png' alt='Refer to caption' title='' width='789' height='395' />

*Figure 1: Q-RAG agent interacts with multi-step retrieval environment. The starting state $s_{0}$ contains the initial query $q$. At the start of the episode, the agent embeds all chunks of the long context ${\mathbb{C}}$. At each step $t$, the agent computes a vector embedding of the current state $s_{t}$, which includes $q$ and all previously selected chunks. For every chunk $c^{i}\in{\mathbb{A}}_{t}$, the utility of retrieving it is evaluated by the $Q$-function $Q_{\theta}(s_{t},a\=c^{i})$. The policy $\pi_{\theta}$ selects the next chunk from ${\mathbb{A}}_{t}$ with probability proportional to its $Q_{\theta}(s_{t},c^{i})$ value.*

### 3.1 Preliminaries

Let $\mathcal{D}$ be a dataset of triples $({\mathbb{C}},q,y)$, where ${\mathbb{C}}$ is a long context, $q$ is an initial query, and $y$ is the gold answer. The query $q$ can be either a user question about ${\mathbb{C}}$ or a generated claim whose factuality or consistency with earlier parts of ${\mathbb{C}}$ must be verified. We assume ${\mathbb{C}}$ is pre-segmented into non-overlapping111Chunk overlapping may complicate the explanation but does not affect our proposed solution. text chunks ${\mathbb{C}}\={c^{(i)}}_{i\=1}^{m}$ in document order. The agent’s goal is to identify the information in ${\mathbb{C}}$ that is missing from $q$ but necessary to produce the correct answer $y$. We model multi-step retrieval as a finite-horizon Markov Decision Process, or MDP $({\mathbb{S}},{\mathbb{A}},p,r,\gamma)$, where ${\mathbb{A}}$ is the action space, ${\mathbb{S}}$ is the state space, $r$ is the reward function, $p$ is the (deterministic) transition function, and $\gamma\in[0,1]$ is the discount factor. At step $t\=0$, the action set is ${\mathbb{A}}_{0}\={\mathbb{C}}$, where an action $a_{t}\in{\mathbb{A}}_{t}$ selects one chunk. At later steps, previously selected chunks are removed so ${\mathbb{A}}_{t}\={\mathbb{C}}\setminus{a_{0},\dots,a_{t-1}}$. Superscripts indicate document positions and subscripts indicate episode timesteps. The notation $a^{i}$ (equivalently $c^{(i)}$) denotes the chunk/action at position $i$ in the document; selecting the chunk with index $i$ at step $t$ is written $a^{i}_{t}$. Symbols $c$ and $a$ are used interchangeably, depending on context.

States are ordered lists that always begin with the query, $s_{t}\=\mathrm{ord}([q,a_{0},\dots,a_{t-1}])$, where $\mathrm{ord}(\cdot)$ sorts by the original document order to avoid permutation ambiguity; the initial state contains only the query, $s_{0}\=[q]$. Transitions are deterministic, $p(s_{t},a_{t})\=\mathrm{ord}([q,a_{0},\dots,a_{t-1},a_{t}])$. An episode terminates either when a step budget $T$ is reached or when a special Stop action is taken.

When supervision provides a set of support facts $F^{\star}\subseteq C$, we use a sparse terminal reward: the reward is <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S3.SS1.p3.m2" intent=":literal"><mn>0</mn></math> -->0 at all intermediate steps, and at the end of the episode it is $1$ if all support facts are included in the final state (otherwise <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S3.SS1.p3.m4" intent=":literal"><mn>0</mn></math> -->0). When only answer supervision is available, one could instead use an LLM to generate $\hat{y}$ from the final state and define a terminal reward via an answer-quality metric (e.g., exact match or F1). In this work we do not pursue LLM-based rewards; all reported experiments rely on the support-fact signal, and exploring LLM-based reward design is left for future work.

### 3.2 Value-based RL for Embedder Fine-Tuning

Action selection in multi-step retrieval is performed by a value-based agent. Specifically, maximum-entropy reinforcement learning *(max_entr_rl_ziebart2010; sac_haarnoja2018)* is adopted together with the corresponding definitions of the soft $Q^{\pi}$ and $V^{\pi}$ value functions for a policy $\pi$:

|  | $\displaystyle Q^{\pi}(s,a)$ | $\displaystyle\=r(s,a)+\gamma V^{\pi}(s^{\prime}\=p(s,a))$ |  | (1) |
| --- | --- | --- | --- | --- |
|  | $\displaystyle V^{\pi}(s)$ | $\displaystyle\=\mathbb{E}_{a\sim\pi(\cdot|s)}\left[Q^{\pi}(s,a)-\alpha\log\pi(a|s)\right]$ |  | (2) |
| --- | --- | --- | --- | --- |

Here, $\alpha>0$ is a temperature that controls the strength of exploration. This choice is primarily motivated by the need for effective exploration in the long-context multi-step retrieval environment. In Q-RAG, Q function is approximated using two embedders for states and actions. The state embedder $E_{s}(s_{t};\theta_{1})\in\mathbb{R}^{d}$ produces vector embedding for the current state $s_{t}$, while the action embedder $E_{a}(a^{i},i;\theta_{2})\in\mathbb{R}^{d}$ employ rotary position embeddings to encode both the candidate chunk content and its document-position index $i$. Q values are then estimated by an inner product between two embeddings: $Q_{\theta}(s,a^{i})\=\langle E_{s}(s;\theta_{1}),E_{a}(a^{i},i;\theta_{2})\rangle$. This factorization is theoretically grounded; we derive its convergence guarantees with explicit rates in Appendix [A].
Given $Q_{\theta}$, the chunk selection probability is computed using a Boltzmann policy:

|  | $\pi(a_{t}|s_{t})\=\frac{\exp\frac{1}{\alpha}\left(Q_{\theta}(s_{t},a_{t})-q\right)}{\sum_{a\in\mathcal{A}_{t}}\exp\frac{1}{\alpha}(Q_{\theta}(s_{t},a)-q)}$ |  | (3) |
| --- | --- | --- | --- |

with $q\=\max_{a\in\mathcal{A}_{t}}Q_{\theta}(s_{t},a)$ and temperature $\alpha$ annealed from an initial value to zero during training (proportionally to the learning rate).

As the backbone Temporal Difference learning algorithm, we adopt the recent PQN method by *pqn_gallici2024*. Compared to DQN *(dqn_mnih2015)*, PQN removes the need for a replay buffer. In our setting with a large number of chunks, a replay buffer would require re-embedding all document chunks for each sample drawn from the replay buffer to estimate $V/Q$ values for subsequent states $s_{t+1}$. Which significantly slows the training process and increases memory space requirements. Using PQN enables an on-policy value-based training that avoids these costs. The key departures in Q-RAG, relative to the original PQN backbone, are the use of soft value functions and target networks. Ablation results demonstrating the benefit of these choices are reported in Section[4.5].

As the training target, rather than the one-step return (see r.h.s. in Eq.[1]), a $\lambda$-return is used to improve stability and learning speed:

|  | $G_{t}^{\lambda}\=(1-\lambda)\sum_{n\=1}^{T-t-1}\lambda^{\,n-1}\,G_{t:t+n}\;+\;\lambda^{\,T-t-1}G_{t},$ |  |
| --- | --- | --- |

where $G_{t:t+n}\=\sum^{n}_{k\=1}\gamma^{k-1}r_{t+k}+V_{\theta^{\prime}}(s_{t+n})$. The approximation of the state value function can be computed from Q values in the case of discrete actions:

|  | $V_{\theta^{\prime}}(s_{t})\=\alpha\log\sum_{a\in\mathcal{A}_{t}}\exp\left(\frac{Q_{\theta^{\prime}}(s_{t},a)}{\alpha}\right)$ |  | (4) |
| --- | --- | --- | --- |

Here $\theta^{\prime}$ denotes slowly updated target network parameters. The model parameters $\theta$ are finetuned to minimize the mean squared error to the $\lambda$-returns:

|  | $\mathcal{L}_{Q}\=\mathbb{E}[(Q_{\theta}(s_{t},a_{t})-G^{\lambda}_{t})^{2}]$ |  | (5) |
| --- | --- | --- | --- |

The Q-RAG pseudocode is presented in Algorithm[1].

*Algorithm 1  Q-RAG*

1:Hyperparameters:

2:Environments count $K$, retrieval steps $T$, temperature $\alpha$, TD parameter $\lambda$, EMA $\tau$.

3:Initialize:

4:State embedder $E_{s}(s;\theta_{1})$

5:Action embedder $E_{a}(a^{i},i;\theta_{2})$ with position $i$

6:Critic $Q_{\theta}(s,a^{i})\=E_{s}(s;\theta_{1})^{T}E_{a}(a^{i},i;\theta_{2})$

7:Critic target $Q_{\theta^{\prime}}(s,a^{i})$

8:procedure ComputeTargets(${s_{t},a_{t},r_{t},v_{t}}_{t\=1}^{T+1}$)

9:Initialize $\lambda$-returns $G_{T}\=r_{T}+\gamma v_{T+1}$

10: for $t\=T-1$ downto $1$ do

11:$G_{t}\=r_{t}+\gamma\big[(1-\lambda)v_{t+1}+\lambda G_{t+1}\big]$

12: end for

13: return ${G_{t}}_{t\=1}^{T}$

14:end procedure

15:Training (one update step)

16:for env $k\in 1,\ldots,K$ in parallel do

17:$s_{1},\mathcal{A}_{1}\=\text{ResetQueryAndContext()}$

18:Compute $E_{a}\=E_{a}(\mathcal{A};\theta)$ and $E^{\prime}_{a}\=E_{a}(\mathcal{A};\theta^{\prime})$

19: for step $t\in 1,\ldots,T+1$ do

20:$a_{t}\sim\text{softmax}_{a\in\mathcal{A}_{t}}\frac{1}{\alpha}E_{s}(s;\theta)^{T}E_{a}$

21:$v_{t}\=\alpha\log{\sum_{a\in\mathcal{A}}}\exp\frac{1}{\alpha}E_{s}(s;\theta^{\prime})^{T}E^{\prime}_{a}$

22:$r_{t}\=\text{ComputeReward}(s_{t},a_{t})$

23:$s_{t+1}\=\text{concatenate}(s_{t},a_{t})$

24:$\mathcal{A}_{t+1}\=\mathcal{A}_{t}\setminus{a_{t}}$

25: end for

26:$\mathcal{B}\={s_{t},a_{t},r_{t},v_{t}}_{t\=1}^{T+1}$

27:${G_{t}^{k}}_{t\=1}^{T}\=\text{ComputeTargets}(\mathcal{B})$

28:end for

29:$\nabla\mathcal{L}_{Q}\=\frac{1}{TK}\sum_{k\=1}^{K}\sum_{t\=1}^{T}\nabla_{\theta}(Q_{\theta}(s^{k}_{t},a^{k}_{t})-G^{k}_{t})^{2}$

30:Update $\theta$ using $\nabla\mathcal{L}_{Q}$

31:Update target parameters: $\theta^{\prime}\leftarrow\tau\theta+(1-\tau)\theta^{\prime}$

### 3.3 Temporal reasoning for long-context search

When dealing with narrative text, the information contained in a text chunk $c$ may be insufficient to determine whether $c$ helps us answer the question $q$. For example, we may need to know what happened before some specific event. A standard retriever can find several relevant text chunks that specify the character’s location, but choosing the correct one can be impossible without taking into account temporal information.
To address this, we propose a *relative postional encoding* of chunks that explicitly encodes their position with respect to the facts already extracted into the state.
At step $t$, let $S_{t}\={i_{1}<\dots<i_{k}}$ be the (sorted) document indices of selected chunks and ${\mathbb{A}}_{t}$ the set of available actions. The indices in $S_{t}$ partition the document into $k{+}1$ disjoint intervals: “before the earliest selected fact”, “between consecutive selected facts”, and “after the latest selected fact.” The relative positional mapping $\rho_{t}:\mathbb{N}\to\mathbb{R}^{+}$ assigns to every original chunk index a real-valued index that (i) identifies the interval it belongs to and (ii) preserves the relative order between chunks. This mapping makes explicit *between which extracted facts* a chunk lies, while remaining invariant to global shifts of absolute positions.

Formally, the interval boundaries are defined as $b_{0}{\=}1$, $b_{j}{\=}i_{j}$ for $j{\=}1{:}k$, and $b_{k+1}{\=}m{+}1$ for ${\mathbb{C}}\={c^{(i)}}_{i\=1}^{m}$. To compute relative index $\rho_{t}(i)$ for a chunk $c^{i}$, find the unique $j$ such that $b_{j}\leq i<b_{j+1}$ and set

|  | $\rho_{t}(i)\;\=\;j\,\delta\;+\;\ell\,\frac{i-b_{j}}{\,b_{j+1}-b_{j}\,},$ |  | (6) |
| --- | --- | --- | --- |

where $\delta>0$ is the inter-interval step and $\ell\in(0,\delta)$ controls the within-interval resolution (e.g., $\delta{\=}10$, $\ell{\=}9$ in our experiments). In the action embedder, the absolute position is replaced by the relative one,

|  | $E_{a}\big(a^{i},\,i;\theta_{2}\big)\;\Rightarrow\;E_{a}\big(a^{i},\,\rho_{t}(i);\theta_{2}\big),$ |  | (7) |
| --- | --- | --- | --- |

which allows the Q-function to exploit the spatial relation of candidates to already retrieved evidence while retaining local order within each interval.
This design allows the retrieval agent to perform strongly not only on fact-finding over disjoint document collections, but also on long-form narrative tasks, enabling Q-RAG to compete with recurrent transformers*(rmt2022; armt2024; atlas2025; titans2024)* and other long context approaches.

4 Experiments
-------------

### 4.1 Experimental Setup

We evaluate our approach, Q-RAG, on tasks that cover commonsence reasoning, temporal reasoning, a bunch of needle in a haystack tasks and open-domain multi-hop question answering tasks on context lengths that range from 4k tokens to 10M tokens per sample.
For commonsence and temporal reasoning we use Babilong benchmark*(babilong2024)*, for Needle-in-a-Haystack we use RULER benchmark*ruler_hsieh2024*. For open-domain multi-hop QA we use HotpotQA *hotpotqa_yang2018*, Musique *musique_trivedi2022* and RULER benchmarks. Babilong and RULER require long contexts. Musique and HotpotQA use short contexts.

Baselines differ by task.
Computing a uniform set of baselines across all datasets is difficult and time-consuming.
Many methods do not release code.
Some methods were evaluated only on some of these datasets.
Even when the tasks match, the experimental settings often differ for the same benchmarks.
Some baselines provide code but require heavy resources (e.g., at least 8$\times$A100 GPUs *search_r1_2025; r1_searcher_song2025; rag_rl_huang2025*) to fine-tune, which are unavailable for us.
Therefore, we report three types of baselines, and we mark each baseline in tables accordingly:

* •

    $\times$ Ablation: baselines that test the effectiveness of our proposed modifications.

* •

    $\checkmark$ Reproduced: baselines that we finetuned and/or evaluated on our datasets using released code or publicly available checkpoints.

* •

    $\circ$ Reported: baselines whose scores we take directly from the original papers.

### 4.2 Commonsense reasoning on ultra-long contexts

<img src='figures/babilong_avg_ans_v2.png' alt='Refer to caption' title='' width='598' height='345' />

*(a)*

<img src='figures/babilong_qa3_ans.png' alt='Refer to caption' title='' width='598' height='364' />

*(b)*

*Figure 2: Comparison of answer accuracy on the long-context benchmark Babilong. Solid lines denote methods fine-tuned on the Babilong, while dashed lines denote zero-shot methods. a) Average performance across tasks Q1–QA5. b) Performance on the hardest task, QA3, which requires the longest reasoning chain and temporal awareness.*

On the BabiLong*babilong2024* benchmark, we compared our method with the state-of-the-art long-context processing approaches, including Titans*titans2024*, Atlas*atlas2025*, ARMT*armt2024*, RMT*rmt2022*, as well as proprietary LLMs and LLM-based agents. The results for most of these baselines were taken directly from the respective original papers. As shown in Figure [3(c)], our approach achieves the highest average performance on BabiLong in ultra-long contexts ranging from 1 to 10 million tokens, demonstrating superior generalization to long contexts compared to other specialized long-context methods.

In Figure [3(a)], we present separate results for the QA3 subtask, which is the hardest subtask in the Babilong benchmark, which specifically requires the multistep search of at least 3 different facts and temporal reasoning. Experimental results show that the majority of models perform worst on the QA3 subtask. As the results indicate, alternative long-context approaches show even greater performance degradation on this task with increasing context length. In contrast, Q-RAG shows virtually no degradation, with the largest performance gap over all baselines observed on this most challenging subtask. We additionally fine-tuned the Beam-Retriever baseline specifically on the QA3 subtasks, given its strong performance on open-domain QA datasets. However, this method failed to solve the task. Note that some methods, such as Titans*titans2024* and Atlas*atlas2025*, are absent from the Figure as they did not report detailed breakdowns by a subtask.

### 4.3 Needle in a Haystack and Long Context QA

*Table 1: Results on the RULER benchmark, evaluating long-context retrieval performance across various context lengths. S (Single-needle): Find one value for one key. MK (Multi-keys): Find one value for one key among many. MV (Multi-values): Find all values for one key. MQ (Multi-query): Answer multiple questions over the context. MH QA: open domain multi-hop question answering.*

| Length | Methods | S | | | MK | | | MV | MQ | NIAH Avg. | MH QA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | | 1-st | 2-nd | 3-rd | 1-st | 2-nd | 3-rd | | | | |
| 4K | $\circ$Titans | 98.4 | 99.8 | 89.4 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| | $\circ$Atlas | 99.2 | 100 | 90.6 | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| $\circ$Mamba2-Hybrid | 100 | 100 | 95.7 | 89.5 | 95.5 | 96 | 97.9 | 97.6 | 96.5 | 48.8 |
| $\circ$LongRoPe2-8B | 100 | 100 | 99 | 100 | 100 | 100 | 99 | 99.7 | 99.7 | 60 |
| $\checkmark$Beam-Retriever | 100 | 100 | 98 | 98 | 98 | 97 | 98 | 99 | 98.5 | 28.3 |
| Q-RAG | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 67 |
| 16K | $\circ$Titans | 96.2 | 80.2 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| | $\circ$Atlas | 97 | 84 | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| $\circ$Mamba2-Hybrid | 100 | 100 | 81.5 | 92 | 92.2 | 83 | 89.8 | 90.2 | 91.1 | 44 |
| $\circ$LongRoPe2-8B | 100 | 100 | 100 | 99 | 100 | 98 | 95 | 98.2 | 98.8 | 58 |
| $\checkmark$Beam-Retriever | 100 | 100 | 97 | 96.5 | 96 | 95 | 80 | 98 | 95.3 | 28.3 |
| Q-RAG | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 67 |
| 32K | $\circ$Mamba2-Hybrid | 100 | 100 | 96.7 | 84 | 76.5 | 81.5 | 84.3 | 80.9 | 88.0 | 38.5 |
| | $\circ$LongRoPe2-8B | 100 | 100 | 100 | 99 | 98 | 100 | 98 | 96.2 | 98.9 | 55 |
| Q-RAG | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 67 |
| 128K | $\circ$LongRoPe2-8B | 100 | 100 | 99 | 96 | 91 | 94 | 96.5 | 97 | 96.7 | 50 |
| | Q-RAG | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 62 |
| 1M | Q-RAG | 100 | 100 | 100 | 100 | 98.5 | 99.0 | 100 | 100 | 99.7 | 57 |

While reasoning tasks are crucial for evaluating advanced retrieval systems, a substantial portion of real-world applications reduces to Needle-in-a-Haystack (NIAH) problems, making it equally important that models deliver consistently strong performance on these tasks.

RULER is a dataset that includes many long-context tasks. Most of these tasks follow the NIAH formulation. The NIAH setup evaluates the ability to retrieve a specific “needle” from a long distracting “haystack”.

For RULER benchmark we use Titans*titans2024*, Atlas*atlas2025*, Mamba2*mamba2_waleffe2024*, and LongRope2*longrope2_shang2025* as baselines. Titans, Atlas are recurrent transformers. Mamba2 is a state space model (SSM) that combines transformer components with SSM. LongRope2 is a method for extending the effective context window of LLMs. All methods were fine-tuned either directly on RULER (Titans, Atlas, Mamba2) or on related synthetic NIAH-style datasets (LongRope2). Q-RAG was also fine-tuned on the NIAH subtasks. For the Multi-hop QA RULER subtask, Q-RAG was fine-tuned on HotpotQA and evaluated on the Multi-hop QA subtask out-of-distribution.

The results are shown in Table[1]. Q-RAG achieves near-perfect performance on all NIAH subtasks. Q-RAG embedder was trained on 4K-length documents and generalizes to context lengths up to 1M tokens without loss of accuracy. On the Multi-hop QA subtask, Q-RAG shows significantly better results than all our baselines at all context lengths we consider. Some degradation with increasing context length starts only from 128K.

### 4.4 Open-domain Question Answering

For our experiments on the HotPotQA and Musique datasets, we compared our method against several strong baselines. The first baseline is Beam Retriever, which enables multi-step retrieval by training a model to score sequences of retrieved chunks. During evaluation, Beam-Retriever is given the oracle number of supporting facts (i.e., the gold hop count) and always retrieves exactly that many facts. Although this approach is slower than traditional retrieval methods and does not scale well to longer contexts, it achieves state-of-the-art results on HotPotQA. Another baseline we considered is SearchR1, a recent method from a family of approaches that train the LLM itself to compose text queries for multi-step retrieval. Additionally, we evaluated the performance of LLM-agent-based methods, including GraphReader and HippoRAG.
Q-RAG and Beam-Retriever were fine-tuned on HotPotQA and evaluated on Musique for out-of-distribution testing. Baseline numbers were taken directly from the corresponding papers. Missing entries indicate metrics not reported by the original authors.

The comparison results are presented in Table[2]. Our method achieves fact retrieval accuracy on par with Beam Retriever, surpasses all other baselines on HotPotQA, and matches the performance of full-LLM-tuning Search-R1 while outperforming all alternatives on the out-of-distribution Musique dataset, resulting in the best overall performance across benchmarks. For both methods involving retrieval mechanism fine-tuning (Q-RAG and Beam Retriever), we used the QwQ-32B model to produce the final answer.

*Table 2: Comparison of methods on HotPotQA and Musique benchmarks. Bold text and underline denote the best and second best scores respectively.*

|  | HotPotQA | | | | Musique (OOD) | | | | Avg | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Methods | Fact F1 | Fact EM | Ans F1 | Ans EM | Fact F1 | Fact EM | Ans F1 | Ans EM | Ans F1 | Ans EM |
|  | Finetuned on HotPotQA | | | | | | |  |  |  |
| Plan Q-RAG | 0.95 | 0.91 | 0.76 | 0.60 | 0.69 | 0.53 | 0.51 | 0.36 | 0.64 | 0.48 |
| Q-RAG | 0.93 | 0.89 | 0.76 | 0.59 | 0.71 | 0.55 | 0.52 | 0.37 | 0.64 | 0.48 |
| $\checkmark$Beam-Retriever | 0.97 | 0.94 | 0.77 | 0.61 | 0.61 | 0.36 | 0.40 | 0.27 | 0.59 | 0.44 |
| $\checkmark$Search-r1 | 0.81 | 0.66 | 0.65 | 0.52 | 0.71 | 0.55 | 0.51 | 0.39 | 0.58 | 0.46 |
| $\circ$RAG-RL | 0.82 | – | 0.69 | – | 0.65 | – | 0.47 | – | 0.58 | – |
| $\times$Multi-step RAG w.o. FT | 0.73 | 0.54 | 0.65 | 0.50 | 0.51 | 0.30 | 0.40 | 0.27 | 0.53 | 0.39 |
|  | Zero Shot methods | | | | | | |  |  |  |
| $\checkmark$GraphReader | – | – | 0.46 | 0.24 | – | – | 0.40 | 0.20 | 0.43 | 0.22 |
| $\checkmark$Single step RAG | – | – | 0.53 | 0.39 | – | – | 0.28 | 0.17 | 0.41 | 0.28 |

### 4.5 Ablation Study

<img src='x2.png' alt='Refer to caption' title='' width='829' height='703' />

*(a)*

<img src='x3.png' alt='Refer to caption' title='' width='829' height='705' />

*(b)*

<img src='figures/wall_clock_inf1.png' alt='Refer to caption' title='' width='598' height='353' />

*(c)*

*Figure 3: Ablation for (a) policy entropy coefficient ($\alpha$) in soft Q function and (b) for $\lambda$-return parameter. Inference runtime comparison (c), context length, tokens on x-axes.*

To assess the impact of the architectural choices in Q-RAG, an ablation study was conducted on the Babylon-QA3 task. This benchmark was selected because it is among the most challenging long-context tasks used in the experiments and it supports evaluation at arbitrary context lengths. The following baselines were compared against Q-RAG:

* •

    Multi-step RAG w.o. FT. This baseline reproduces the full Q-RAG retrieval pipeline and uses the same state and action embedders, but relies on their original pretrained weights without any reinforcement learning fine-tuning. This setting tests whether RL fine-tuning of the embedders is beneficial for multi-step retrieval quality.

* •

    Multi-step RAG w. SFT. This baseline applies supervised fine-tuning using ground-truth support facts as supervision. The loss follows the objective used in BeamRetriever for trajectory supervision, adapted to the multi-step retrieval setting. This setting isolates the effect of RL by comparing it to supervised learning on the same supervision signal.

* •

    Q-RAG w.o. target. This variant removes target networks from the PQN-based value learning, following the original PQN recipe without target parameters. It measures the contribution of target networks to stability and performance in the Q-RAG training loop.

* •

    Q-RAG w.o. Soft-Q. This variant replaces the maximum-entropy (soft) value functions with standard (non-entropy-regularized) Q-learning objectives. It evaluates the effect of entropy regularization and the soft value formulation on retrieval performance.

All baselines were evaluated with three random seeds. Table[3] reports results at a 32k-token context length on QA3. Figure[3] shows the sensitivity of Q-RAG to the $\lambda$-return parameter and the temperature $\alpha$ (the strength of entropy regularization) on QA2 and QA3.

*Table 3: Ablation results on Babilong QA3. Table shows F1 score for support facts retrieval. All values are averaged over 3 runs with different seeds.*

| Method | 1K | 4K | 32K | 128K | 1M |
| --- | --- | --- | --- | --- | --- |
| Q-RAG | $97.8\pm 0.17$ | $97.4\pm 0.14$ | $97.1\pm 0.08$ | $96.8\pm 0.08$ | $96.5\pm 0.16$ |
| $\times$Q-RAG w.o. Soft-Q | $95.9\pm 0.70$ | $95.5\pm 0.80$ | $94.5\pm 0.50$ | $94.0\pm 0.30$ | $93.3\pm 0.45$ |
| $\times$Q-RAG w.o. Target | $79.2\pm 26.0$ | $78.1\pm 26.6$ | $77.6\pm 27.2$ | $77.4\pm 27.3$ | $75.9\pm 28.2$ |
| $\times$Multi-Step RAG w. SFT | $20.33\pm 0.32$ | $20.87\pm 0.35$ | $20.10\pm 0.20$ | $18.30\pm 0.36$ | — |
| $\times$Multi-Step RAG w.o. FT | $15.52\pm 0.11$ | $16.38\pm 0.10$ | $15.51\pm 0.16$ | $15.34\pm 0.12$ | — |

5 Conclusion
------------

This work introduced Q-RAG, a resource-efficient method for multi-step retrieval trained with reinforcement learning directly in the latent space of text-chunk embeddings. Across long-context benchmarks (e.g., Babilong, RULER) and open-domain QA datasets (e.g., Musique, HotpotQA), Q-RAG attains state-of-the-art or highly competitive results. Its advantage over baselines widens as context length grows, and performance shows minimal degradation even at ultra-long scales.

A key practical benefit is compute efficiency: all training was performed on a single A100 GPU with 80 GB memory, whereas recent RL-based multi-step retrievers such as Search-R1/R1-Searcher typically report training on clusters of about eight A100 GPUs. By fine-tuning only the embedder while keeping the LLM frozen, Q-RAG remains easy to pair with powerful pre-trained or proprietary LLMs, enabling efficient training, flexible deployment, and strong retrieval over very long contexts.

Looking ahead, promising directions include using structured LLM feedback as a reward signal, strengthening compositional and temporal reasoning directly in the embedding space, and exploring tighter integration with generation while preserving the method’s efficiency and scalability.

Appendix A Inner product approximation for Q-function
------------------------------------------------------

The Universal Approximation Theorem (UAT) states that neural networks with a single hidden layer can approximate any continuous function arbitrarily well under mild conditions. In this section, we prove a variant of the UAT for functions decomposed as an inner product involving Rotary Position Embedding (RoPE). Specifically, we show that any continuous q-function $Q(s,a^{i})$ defined on a compact domain can be approximated by functions of the form:

|  | $F(s,a^{i})\=\langle E_{s}(s),E_{a}(a^{i},i)\rangle,\quad E_{a}(a^{i},i)\=R_{\text{pos}(i)}E_{a}(a^{i}),$ |  | (8) |
| --- | --- | --- | --- |

where $E_{s}$ and $E_{a}$ are continuous vector functions (e.g., neural networks) and $R_{t}$ is the RoPE matrix of dimension $r$ (even) parameterized by $t\=\text{pos(i)}$:

|  | $R_{t}\=\bigoplus_{j\=1}^{r/2}\begin{bmatrix}\cos(\theta_{j}t)\&-\sin(\theta_{j}t)\\ \sin(\theta_{j}t)\&\cos(\theta_{j}t)\end{bmatrix},$ |  | (9) |
| --- | --- | --- | --- |

where $\theta_{j}$ are fixed frequencies. For notational simplicity in the following derivations, we introduce the following conventions:

|  | $(x,y):\=(s,a),\quad t:\=\text{pos}(i),\quad h(x):\=E_{s}(s),\quad g(y):\=E_{a}(a^{i}).$ |  |
| --- | --- | --- |

For simplicity, we assume the domains of $x$, $y$ and $t$ are continuous, corresponding to the embeddings of text tokens.

###### Theorem 1.

Let $X\subset\mathbb{R}^{d_{x}}$, $Y\subset\mathbb{R}^{d_{y}}$, and $T\subset\mathbb{R}$ be compact sets, and define the compact domain $K\=X\times Y\times T$. Let $C(K,\mathbb{R})$ be the space of continuous real-valued functions on $K$ equipped with the uniform norm. Let $R_{t}$ be the RoPE matrix of dimension $r$, defined as a block-diagonal rotation matrix ([9]). Define the function class:

|  | $\mathcal{A}\=\left{F(x,y,t)\=\langle h(x),R_{t}g(y)\rangle\mid h\in C(X,\mathbb{R}^{r}),\;g\in C(Y,\mathbb{R}^{r})\right}.$ |  | (10) |
| --- | --- | --- | --- |

Then $\mathcal{A}$ is dense in $C(K,\mathbb{R})$. That is, for any $f\in C(K,\mathbb{R})$ and $\epsilon>0$, there exist continuous functions $h:X\to\mathbb{R}^{d}$ and $g:Y\to\mathbb{R}^{d}$ such that:

|  | $\sup_{(x,y,t)\in K}\left|f(x,y,t)-\langle h(x),R_{t}g(y)\rangle\right|<\epsilon.$ |  | (11) |
| --- | --- | --- | --- |

###### Proof.

We prove the result via the Stone-Weierstrass theorem, which states that if a subalgebra $\mathcal{A}\subset C(K,\mathbb{R})$ contains the constant functions and separates points, then $\mathcal{A}$ is dense in $C(K,\mathbb{R})$. Thus, we show that $\mathcal{A}$ satisfies these requirements.

#### $\mathcal{A}$ is a subalgebra.

We prove closure under addition, scalar multiplication, and multiplication of two arbitrary elements.

Scalar multiplication: Let $F(x,y,t)\=\langle h(x),R_{t}g(y)\rangle\in\mathcal{A}$ and $c\in\mathbb{R}$. Define $h^{\prime}(x)\=ch(x)$. Then $cF(x,y,t)\=\langle h^{\prime}(x),R_{t}g(y)\rangle\in\mathcal{A}$.

Addition: Let $F_{1}(x,y,t)\=\langle h_{1}(x),R_{t}g_{1}(y)\rangle$ and $F_{2}(x,y,t)\=\langle h_{2}(x),R_{t}g_{2}(y)\rangle$. Define $h(x)\=[h_{1}(x);h_{2}(x)]\in\mathbb{R}^{2d}$ and $g(y)\=[g_{1}(y);g_{2}(y)]\in\mathbb{R}^{2d}$, and let $\widetilde{R}_{t}$ be a block-diagonal extension of $R_{t}$. Then

|  | $\langle h(x),\widetilde{R}_{t}g(y)\rangle\=\langle h_{1}(x),R_{t}g_{1}(y)\rangle+\langle h_{2}(x),R_{t}g_{2}(y)\rangle\=F_{1}(x,y,t)+F_{2}(x,y,t)\in\mathcal{A}.$ |  |
| --- | --- | --- |

Multiplication: Let $F_{1}$ and $F_{2}$ as above. Note that:

|  | $F_{1}(x,y,t)F_{2}(x,y,t)\=\langle h_{1}(x)\otimes h_{2}(x),(R_{t}g_{1}(y))\otimes(R_{t}g_{2}(y))\rangle.$ |  |
| --- | --- | --- |

Since $(R_{t}g_{1}(y))\otimes(R_{t}g_{2}(y))\=(R_{t}\otimes R_{t})(g_{1}(y)\otimes g_{2}(y))$, and $R_{t}\otimes R_{t}$ is a block-diagonal rotation matrix with angles $\theta_{j}+\theta_{k}$ (a RoPE matrix of dimension $d^{2}$), define $h(x)\=h_{1}(x)\otimes h_{2}(x)\in\mathbb{R}^{d^{2}}$, $g(y)\=g_{1}(y)\otimes g_{2}(y)\in\mathbb{R}^{d^{2}}$, and let $\widetilde{R}_{t}$ be the RoPE matrix with frequencies ${\theta_{j}+\theta_{k}}$. Then:

|  | $F_{1}(x,y,t)F_{2}(x,y,t)\=\langle h(x),\widetilde{R}_{t}g(y)\rangle\in\mathcal{A}.$ |  |
| --- | --- | --- |

Thus, $\mathcal{A}$ is a subalgebra.

#### $\mathcal{A}$ contains the constant functions.

Show the constant function $1$ is in $\mathcal{A}$. Augment the dimension: let $d^{\prime}\=d+1$, and define $h(x)\=(1,0,\dots,0)^{T}\in\mathbb{R}^{d^{\prime}}$, $g(y)\=(1,0,\dots,0)^{T}\in\mathbb{R}^{d^{\prime}}$. Define a modified RoPE matrix $R_{t}^{\prime}$ that acts as the identity on the first coordinate and as $R_{t}$ on the remaining $d$ coordinates. Then

|  | $\langle h(x),R_{t}^{\prime}g(y)\rangle\=1.$ |  |
| --- | --- | --- |

#### $\mathcal{A}$ separates points.

Let $(x_{1},y_{1},t_{1})\neq(x_{2},y_{2},t_{2})\in K$. Construct $F\in\mathcal{A}$ such that $F(x_{1},y_{1},t_{1})\neq F(x_{2},y_{2},t_{2})$.

Case 1: $x_{1}\neq x_{2}$ or $y_{1}\neq y_{2}$. Choose $g(y)\=v$ (a constant non-zero vector) and let $h$ be continuous with $h(x_{1})\neq h(x_{2})$. Then $F(x,y,t)\=\langle h(x),R_{t}v\rangle$. Since $R_{t}v$ traces a circle (for $v$ with at least two non-zero components), for generic $v$, $R_{t_{1}}v$ and $R_{t_{2}}v$ are not orthogonal to $h(x_{1})-h(x_{2})$, so $F(x_{1},y_{1},t_{1})\neq F(x_{2},y_{2},t_{2})$.
The case when $y_{1}\neq y_{2}$ is identical to the 1st case.

Case 2: $t_{1}\neq t_{2}$. Choose $h(x)\=w$ and $g(y)\=v$. Then $F(x,y,t)\=\langle w,R_{t}v\rangle$. Since $t\mapsto R_{t}v$ is injective (for $v\neq 0$ and non-zero frequencies), $R_{t_{1}}v\neq R_{t_{2}}v$. Choose $w$ not orthogonal to $R_{t_{1}}v-R_{t_{2}}v$, so $F(x_{1},y_{1},t_{1})\neq F(x_{2},y_{2},t_{2})$.

Thus, by the Stone-Weierstrass theorem, $\mathcal{A}$ is dense in $C(K,\mathbb{R})$.

∎

Theorem 1 establishes that our architecture is capable of approximating any continuous function arbitrarily well. However, it does not specify how complex the network needs to be to achieve a given accuracy. The following quantitative result addresses this by providing an explicit convergence rate dependent on the smoothness of the target function.

###### Theorem 2(Convergence Rate).

Let a target function $f\in C(K,\mathbb{R})$ belong to the Sobolev space $H^{s,\infty}(K)$, i.e., $f$ has bounded derivatives up to order $s$.
Then, for any integer $r>0$, there exist feature maps $h:X\to\mathbb{R}^{r}$ and $g:Y\to\mathbb{R}^{r}$ such that

|  | $\sup_{(x,y,t)\in K}|f(x,y,t)-\langle h(x),R_{t}g(y)\rangle|\leq C\cdot r^{-s/(2d_{x}+2d_{y}+2)},$ |  |
| --- | --- | --- |

where $C$ depends on $\|f\|_{H^{s,\infty}}$, the diameters of $X,Y,T$, and the frequencies $\theta_{j}$.

###### Proof.

Since $R_{t}$ incorporates trigonometric functions, we first expand $f$ in a Fourier series in $t$:

|  | $f(x,y,t)\=\sum_{k\in\mathbb{Z}}a_{k}(x,y)e^{ikt},$ |  |
| --- | --- | --- |

where the Fourier coefficients satisfy:

|  | $|a_{k}(x,y)|\leq\frac{\|f\|_{H^{s,\infty}}}{(1+|k|)^{s}}.$ |  |
| --- | --- | --- |

Truncate to $|k|\leq N$, with error:

|  | $\Big|f(x,y,t)-\sum_{|k|\leq N}a_{k}(x,y)e^{ikt}\Big|\leq C_{1}N^{-s}.$ |  |
| --- | --- | --- |

Make an approximation of $a_{k}(x,y)$ by inner products:

|  | $a_{k}(x,y)\approx\langle h_{k}(x),g_{k}(y)\rangle.$ |  |
| --- | --- | --- |

Using the result on inner product approximation by Lemma [1], for each $k$ there exist functions $h_{k},g_{k}$ such that:

|  | $|a_{k}(x,y)-\langle h_{k}(x),g_{k}(y)\rangle|\leq C_{2}r_{k}^{-s/(d_{x}+d_{y})}.$ |  |
| --- | --- | --- |

Choose $r_{k}\sim r^{1/(2N+1)}$ so that total dimension $\sum_{|k|\leq N}r_{k}\sim(2N+1)r^{1/(2N+1)}$.
Define functions $h,g$
and the RoPE matrix to have frequencies $\theta_{j}\=j$ for $j\=1,\dots,N$ such that

|  | $\langle h(x),R_{t}g(y)\rangle\=\sum_{|k|\leq N}\langle h_{k}(x),R_{t}^{(k)}g_{k}(y)\rangle,$ |  |
| --- | --- | --- |

where $R_{t}^{(k)}$ is the block corresponding to frequency $k$. By design, $R_{t}^{(k)}g_{k}(y)$ produces terms like $e^{ikt}g_{k}(y)$, so:

|  | $\langle h(x),R_{t}g(y)\rangle\=\sum_{|k|\leq N}\langle h_{k}(x),g_{k}(y)\rangle e^{ikt}.$ |  |
| --- | --- | --- |

The error is bounded by:

|  | $\epsilon\=|f(x,y,t)-\langle h(x),R_{t}g(y)\rangle|\leq C_{1}N^{-s}+C_{2}\sum_{|k|\leq N}r_{k}^{-s/(d_{x}+d_{y})}.$ |  |
| --- | --- | --- |

With $r_{k}\sim r^{1/(2N+1)}$ and $N\sim\log r$, we obtain:

|  | $\epsilon\leq C\cdot r^{-s/(2d_{x}+2d_{y}+2)}.$ |  |
| --- | --- | --- |

∎

###### Lemma 1.

Let $\Omega_{x}\subset\mathbb{R}^{d_{x}}$ and $\Omega_{y}\subset\mathbb{R}^{d_{y}}$ be compact domains with Lipschitz boundaries. Consider a symmetric, positive-definite kernel $a:\Omega_{x}\times\Omega_{y}\to\mathbb{R}$ such that $a\in H^{s}(\Omega_{x}\times\Omega_{y})$ for some smoothness parameter $s>(d_{x}+d_{y})/2$. Let $d\=d_{x}+d_{y}$ be the total dimension.

Then, for any integer $r>0$, there exist feature maps $h:\Omega_{x}\to\mathbb{R}^{r}$ and $g:\Omega_{y}\to\mathbb{R}^{r}$ such that the following uniform approximation bound holds:

|  | $\sup_{x\in\Omega_{x},\,y\in\Omega_{y}}\left|a(x,y)-\langle h(x),g(y)\rangle_{\mathbb{R}^{r}}\right|\leq C\cdot r^{-s/d}\cdot\|a\|_{H^{s}}.$ |  |
| --- | --- | --- |

Here, $C>0$ is a constant depending on $s$, $d_{x}$, $d_{y}$, and the domains $\Omega_{x},\Omega_{y}$, but independent of $r$ and $a$.

###### Proof.

Consider a countable orthonormal basis of eigenfunctions ${\phi_{i}}\subset L^{2}(\Omega_{x})$ and ${\psi_{i}}\subset L^{2}(\Omega_{y})$ with corresponding non-negative eigenvalues ${\lambda_{i}}$ (ordered non-increasingly, $\lambda_{1}\geq\lambda_{2}\geq...>0$) such that:

|  | $a(x,y)\=\sum_{i\=1}^{\infty}\lambda_{i}\phi_{i}(x)\psi_{i}(y),$ |  |
| --- | --- | --- |

where the convergence is absolute and uniform on $\Omega_{x}\times\Omega_{y}$.
The optimal rank-$r$ approximation is given by truncating to the first $r$ terms:

|  | $a_{r}(x,y):\=\sum_{i\=1}^{r}\lambda_{i}\phi_{i}(x)\psi_{i}(y).$ |  |
| --- | --- | --- |

Denote the approximation error as the tail of the infinite series:

|  | $e_{r}(x,y)\=a(x,y)-a_{r}(x,y)\=\sum_{i\=r+1}^{\infty}\lambda_{i}\phi_{i}(x)\psi_{i}(y),$ |  |
| --- | --- | --- |

such that

|  | $|e_{r}(x,y)|\leq\sum_{i\=r+1}^{\infty}\lambda_{i}|\phi_{i}(x)||\psi_{i}(y)|.$ |  |
| --- | --- | --- |

The key is that the smoothness of $a$ governs the decay rate of the eigenvalues $\lambda_{i}$. From Theorem 3.1 *birman1972spectral* follows that for an operator with kernel in $H^{s}(\Omega_{x}\times\Omega_{y})$, the eigenvalues satisfy:

|  | $\lambda_{i}\leq C_{1}\cdot i^{-(1+2s/d)}\cdot\|a\|_{H^{s}},$ |  |
| --- | --- | --- |

where $C_{1}$ depends on the domain and dimension.

The condition $s>d/2$ implies that $H^{s}$ is continuously embedded in the space of continuous functions. Furthermore, one can show the eigenfunctions are uniformly bounded:

|  | $\|\phi_{i}\|_{L^{\infty}(\Omega_{x})}\leq C_{2},\quad\|\psi_{i}\|_{L^{\infty}(\Omega_{y})}\leq C_{2},$ |  |
| --- | --- | --- |

where $C_{2}$ is a constant independent of $i$.

Combining the eigenvalue decay and eigenfunction bounds gives a pointwise error estimate:

|  | $|e_{r}(x,y)|\leq C_{2}^{2}\sum_{i\=r+1}^{\infty}\lambda_{i}\leq C_{1}C_{2}^{2}\|a\|_{H^{s}}\sum_{i\=r+1}^{\infty}i^{-(1+2s/d)}.$ |  |
| --- | --- | --- |

The tail of the series can be bounded by an integral:

|  | $\sum_{i\=r+1}^{\infty}i^{-(1+\alpha)}\leq\int_{r}^{\infty}t^{-(1+\alpha)}dt\=\frac{1}{\alpha}r^{-\alpha},\quad\text{where }\alpha\=\frac{2s}{d}.$ |  |
| --- | --- | --- |

Substituting $\alpha\=2s/d$ yields:

|  | $|e_{r}(x,y)|\leq C_{1}C_{2}^{2}\|a\|_{H^{s}}\cdot\frac{d}{2s}\cdot r^{-2s/d}.$ |  |
| --- | --- | --- |

A more refined analysis (*NovakWozniakowski2008*) shows the supremum norm error decays as $O(r^{-s/d})$. Thus, we obtain:

|  | $\sup_{x,y}|e_{r}(x,y)|\leq C\cdot r^{-s/d}\cdot\|a\|_{H^{s}}.$ |  |
| --- | --- | --- |

∎

Appendix B Planning for Multi-Step Retrieval
---------------------------------------------

We *can* apply planning at the multi-step retrieval stage, formulating source selection as a search over the space of action trajectories; see § 4.4 for an application. In the spirit of *Beam-Retriever*, we can run beam search where candidates are ranked by the learned action-value $Q_{\theta}(s,a)$. However, our planning is computationally cheaper because $Q_{\theta}$ is computed as a *dot product* of state and action embeddings,
$Q_{\theta}(s,a)\=\langle E_{s}(s),\,E_{a}(a)\rangle,$
so no new transformer forward passes are required for each candidate chunk, whereas *Beam-Retriever* relies on a transformer reranker over trajectories, incurring fresh forward passes at every expansion. Details of the embedding-based scoring are provided in § 3.2. At inference, we perform *beam search over $Q$* and *deterministically* expand the top-$k$ actions by $Q_{\theta}$.

Appendix C Training details
---------------------------

We trained the model with AdamW (learning rate $1.5\times 10^{-5}$, $\beta_{1}{\=}0.9$, $\beta_{2}{\=}0.98$, $\epsilon{\=}10^{-6}$, weight decay $5\times 10^{-4}$). The learning rate followed a linear schedule: we used a warm-up of $1{,}000$ steps, then linearly decayed the rate to $10\%$ of its initial value over the remaining training steps. We applied gradient clipping with a maximum $\ell_{2}$ norm of $2.0$ and used gradient accumulation for $8$ steps. The base mini-batch size was $12$; with accumulation this yields an effective batch size of $12\times 8\=96$ per update (scaled by the number of devices if using distributed training).

In the objective and algorithmic components we set $\gamma{\=}0.99$, $\alpha{\=}0.05$, $\lambda{\=}0.5$, and $\tau{\=}0.02$. Action representations were capped at a maximum length of $220$ tokens.

#### Models per benchmark.

For open-domain QA benchmarks (HotPotQA, Musique), we trained an multilingual-e5-large encoder. For Ruler and Babilong, we trained facebook/contriever. The end-to-end training of a single model did not exceed 12 hours on a single A100-80GB GPU.
