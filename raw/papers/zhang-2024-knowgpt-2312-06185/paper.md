KnowGPT: Black-Box Knowledge Injection for Large Language Models
=================================================================

Qinggang Zhang1*, Junnan Dong1*, Hao Chen1, Xiao Huang1, Daochen Zha2, Zailiang Yu3  
1The Hong Kong Polytechnic University, 2Rice University, 3Zhejiang Lab  
{qinggangg.zhang,hanson.dong}@connect.polyu.hk, sundaychenhao@gmail.com  
xiaohuang@comp.polyu.edu.hk, daochen.zha@rice.edu, yuzl@zhejianglab.com

###### Abstract

Generative Large Language Models (LLMs), such as ChatGPT, offer interactive APIs that can answer common questions at a human-expert level. However, these models often give inaccurate or incorrect responses when faced with questions requiring domain-specific or professional-specific knowledge not covered in their training corpus. Furthermore, many state-of-the-art LLMs are not open-source, making it challenging to inject knowledge with model APIs only. In this work, we introduce KnowGPT, a black-box knowledge
injection framework for LLMs in question answering. KnowGPT leverages deep reinforcement learning (RL) to extract relevant knowledge from Knowledge Graphs (KGs) and use Multi-Armed Bandit (MAB) to construct the most suitable prompt for each question.
Our extensive experiments on three benchmark datasets showcase that KnowGPT significantly enhances the existing methods. Notably, KnowGPT achieves an average improvement of 23.7% over ChatGPT and an average improvement of 2.9% over GPT-4. Additionally, KnowGPT attains a 91.6% accuracy on the OpenbookQA official leaderboard, which is comparable to human-level performance.

11footnotetext: Equal contribution.

1 Introduction
--------------

Generative large language models (LLMs) have surprised the world with their superior performance*Kung et al. ([2023](#bib.bib17 "")); Zha et al. ([2023](#bib.bib48 ""))*, especially with the emergence of ChatGPT and GPT-4*OpenAI ([2023](#bib.bib27 ""))*. Nonetheless, LLMs are often criticized for their limited factual knowledge and propensity to produce hallucinations, wherein the model fabricates incorrect statements on tasks beyond their knowledge and perception*Amaro et al. ([2023](#bib.bib1 "")); Shen et al. ([2023](#bib.bib31 "")); Gravel et al. ([2023](#bib.bib11 ""))*. Consider an ecological domain-specific question from OpenbookQA*Mihaylov et al. ([2018](#bib.bib26 ""))* in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ KnowGPT: Black-Box Knowledge Injection for Large Language Models"). ChatGPT erroneously provides the response “energy” when asked about the portion of nutrients. This inaccuracy could stem from its potential lack of knowledge of carbs and their relationship to nutrients.

A promising avenue for addressing the above issue entails the integration of Knowledge Graphs (KGs) into LLMs. KGs, such as Yago*Suchanek et al. ([2007](#bib.bib34 ""))*, Freebase*Bollacker et al. ([2008](#bib.bib3 ""))*, and ConceptNet*Speer et al. ([2017](#bib.bib32 ""))* represent relationships among real-world entities in a structured form as triples *(head, relation, tail)*. The enormous factual knowledge stored in KGs holds the potential to significantly enhance the accuracy of LLMs’ responses. For instance, in Figure[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ KnowGPT: Black-Box Knowledge Injection for Large Language Models"), ChatGPT could correct itself by leveraging the related background knowledge in ConceptNet*Speer et al. ([2017](#bib.bib32 ""))*.

Recently, various *white-box* knowledge injection methods have been proposed to incorporate KGs into LLMs. These methods assume that we know everything about LLMs, such as model architectures and parameters.
Many algorithms have been proposed to integrate KGs into LLMs*Pan et al. ([2023](#bib.bib28 ""))*. Early studies directly concatenate the entities from KGs with textual sentences as the input to train LLMs based on cross-modal representation alignment*Sun et al. ([2020](#bib.bib35 "")); Liu et al. ([2020b](#bib.bib23 ""))*. Later research incorporates KGs at an implicit level by either combining text representation and KG embedding with the attention mechanism*Feng et al. ([2020](#bib.bib10 "")); Yasunaga et al. ([2021](#bib.bib46 "")); Lin et al. ([2019](#bib.bib20 "")); Dong et al. ([2023a](#bib.bib8 ""))* or designing various fusion methods to integrate KGs and texts through a tailored encoder*Zhang et al. ([2019](#bib.bib50 "")); Su et al. ([2021](#bib.bib33 ""))*.

<img src='x1.png' alt='Refer to caption' title='' width='207' height='121' />

*Figure 1: A real-world question from OpenbookQA. ChatGPT could effectively correct the answer given the scientific reasoning background from ConceptNet (blue: question concepts, red: answers, grey: entities not present in questions).*

Unfortunately, many state-of-the-art LLMs are confined to a *black-box* role in practical applications. For instance, ChatGPT and GPT-4 exclusively grant access through their APIs, which means we can only retrieve model responses by submitting textual inputs, with model specifics inaccessible. This lack of access prevents us from employing the aforementioned *white-box* knowledge injection techniques. Even though white-box approaches could be applied to open-source LLMs, such as BLOOM*Scao et al. ([2022](#bib.bib30 ""))* and LLaMA*Touvron et al. ([2023](#bib.bib41 ""))*, they often incur significant computation costs due to updating model weights*Liu et al. ([2022](#bib.bib24 ""))*. Thus, we ask: *Can we develop a black-box knowledge injection framework that can efficiently and effectively integrate KGs into LLMs with APIs only?*

Achieving this goal is nontrivial because of two challenges in constructing model inputs, or prompts. ❶ Identifying the most relevant knowledge is difficult. Real-world KGs often consist of millions of triples, whereas LLMs are typically restricted by limited input lengths (e.g., 2048 tokens for ChatGPT and 4096 tokens for GPT-4). Hence, careful selection of the most informative triples from KGs becomes essential. ❷ Effectively encoding KG knowledge is hard. It is observed that even minor variations in prompts conveying the same semantic meaning can yield drastically different responses from LLMs*OpenAI ([2023](#bib.bib27 ""))*. As a result, a customized approach to encoding factual knowledge from extracted KGs for each question is often required to achieve the best performance.

In this work, we propose KnowGPT, a black-box knowledge injection framework for LLMs in question answering. To address challenge ❶, we leverage deep reinforcement learning (RL) to extract paths from source entities mentioned in the question to the target entities within the potential answers. To encourage the agent to discover more informative paths, we devise a tailored reward scheme that promotes the reachability, context-relatedness, and conciseness of the extracted paths. Then, a policy network is trained to maximize the reward using training questions and applied to unseen questions. To tackle challenge ❷, we introduce a prompt construction strategy based on Multi-Armed Bandit (MAB). Given several path extraction strategies and prompt templates, a MAB is learned to select the most effective combination for each question by balancing exploration and exploitation. The learned MAB is then applied to new questions to select path extraction strategies and prompt templates automatically. Our main contributions are summarized as follows:

* •

    Formally define the problem of *black-box knowledge injection for LLMs*, which integrates KGs into LLMs with model APIs only.

* •

    Propose KnowGPT, a black-box knowledge injection framework for LLMs, which leverages deep RL for path extraction in KGs and MAB for prompt construction.

* •

    Implement KnowGPT upon two real-world KGs, ConceptNet*Speer et al. ([2017](#bib.bib32 ""))* and USMLE*Yasunaga et al. ([2021](#bib.bib46 ""))*, with ChatGPT APIs. Experiments on three QA benchmarks show that KnowGPT outperforms the state-of-the-art baselines by a large margin. Notably, KnowGPT attains a 91.6% accuracy on the OpenbookQA leaderboard, which is comparable to human performance.

<img src='x2.png' alt='Refer to caption' title='' width='438' height='97' />

*Figure 2: The overall architecture of our proposed black-box knowledge injection framework, i.e., KnowGPT. Given the question context and answer choices, we retrieve a question-specific subgraph from the real-world KG. Path Extraction is first dedicated to searching for the most informative and concise reasoning background subject to the context. Then the prompt translation module is optimized to prioritize the combination of knowledge and formats subject to the given question.*

2 Problem Statement
-------------------

We formally define the problem of *black-box knowledge injection for LLMs* in complex question answering. We represent each question as a question context $\mathcal{Q}\={\mathcal{Q}_{s},\mathcal{Q}_{t}}$, where $\mathcal{Q}_{s}\={e_{1},e_{2},...,e_{m}}$ is a set of $m$ *source entities*, and $\mathcal{Q}_{t}\={e_{1},e_{2},...,e_{n}}$ is a set of $n$ *target entities*. Following prior work*Feng et al. ([2020](#bib.bib10 "")); Yasunaga et al. ([2022](#bib.bib47 ""))*, $\mathcal{Q}_{s}$ is extracted by concept recognition, and we assume it is given in our problem. Similarly, each target entity in $\mathcal{Q}_{t}$ is extracted from a corresponding candidate answer. We denote an LLM as $f$, a real-world KG as $\mathcal{G}$, which consists of triples (head entity, relation, tail entity), denoted as $(h,r,t)$. In our setting, we only have access to the APIs of $f$. However, we can employ open-source lightweight language models (not $f$), like Bert-Base*Devlin et al. ([2018](#bib.bib7 ""))*, to obtain text embeddings. Using the above notations, we describe our problem below. 
Given a question context $\mathcal{Q}$, an LLM $f$, and a KG $\mathcal{G}$, we aim to learn a prompting function $f_{\text{prompt}}(\mathcal{Q},\mathcal{G})$, which generates a prompt x that incorporates the context of $\mathcal{Q}$ and the factual knowledge in $\mathcal{G}$, such that the prediction of the LLM $f(\textbf{x})$ can output the correct answers for $\mathcal{Q}$.

3 KnowGPT Framework
-------------------

Learning the prompting function $f_{\text{prompt}}(\mathcal{Q},\mathcal{G})$ involves two challenges, i.e., what knowledge should be used in $\mathcal{G}$, and how to construct the prompt. To address these challenges, we present KnowGPT, which extracts subgraphs (paths) with deep RL and then constructs the prompt with MAB. An overview of our framework is shown in Figure[2](#S1.F2 "Figure 2 ‣ 1 Introduction ‣ KnowGPT: Black-Box Knowledge Injection for Large Language Models").

### 3.1 Path Extraction with Deep Reinforcement Learning

Intuitively, the relevant reasoning background lies in a question-specific subgraph $\mathcal{G}_{\text{sub}}$ that contains all the source entities $\mathcal{Q}_{s}$, target entities $\mathcal{Q}_{t}$, and their neighbors. An ideal subgraph $\mathcal{G}_{\text{sub}}$ is expected to have the following properties: $(i)$ $\mathcal{G}_{\text{sub}}$ encompasses as many source and target entities as possible, $(ii)$ the entities and relations within $\mathcal{G}_{\text{sub}}$ exhibit a strong relevance to the question context, and $(iii)$ $\mathcal{G}_{\text{sub}}$ is concise with little redundant information such that it can be fed into LLMs with limited input lengths.

However, it is challenging to find such a $\mathcal{G}_{\text{sub}}$ since extracting a subgraph is NP-hard. To effectively and efficiently find a satisfactory $\mathcal{G}_{\text{sub}}$, we develop a tailored path extraction method, named $\mathcal{P}_{\text{RL}}$, that employs deep RL to sample reasoning paths in a trial-and-error fashion. Specifically, we assume $\mathcal{G}_{\text{sub}}$ is constructed based on a set of reasoning paths $\mathcal{P}\={\mathcal{P}_{1},\mathcal{P}_{2},...,\mathcal{P}_{m}}$, where each path $\mathcal{P}_{i}\={(e_{i},r_{1},t_{1}),(t_{1},r_{2},t_{2}),...,(t_{|\mathcal{P}_{i}|-1},r_{|\mathcal{P}_{i}|},t_{|\mathcal{P}_{i}|})}$ is a path in $\mathcal{G}$ starting from the $i$-$th$ source entity in $\mathcal{Q}_{s}$, and $|\mathcal{P}_{i}|$ is the path length. $\mathcal{G}_{\text{sub}}$ encompasses all the entities and relations appeared in $\mathcal{P}$. We model the sampling of each reasoning path as a Markov Decision Process (MDP) with state, action, transition, and reward, defined as follows.

* •

    State: A state indicates the current location in KG, i.e., one of the entities in KG. Specifically, it represents the spatial change from entity $h$ to $t$. Inspired by the prior study*Xiong et al. ([2017](#bib.bib45 ""))*, we define the state vector $\boldsymbol{s}$ as:

    |  | $\boldsymbol{s}_{t}\=(\boldsymbol{e}_{t},\boldsymbol{e}_{target}-\boldsymbol{e}_{t}),$ |  | (1) |
    | --- | --- | --- | --- |

    where $\boldsymbol{e}_{t}$ and $\boldsymbol{e}_{target}$ are the embedding vectors of the current entity and the target entity. To get the initial node embeddings for entities extracted from the background KG, we adopt the approach proposed by the previous study*Feng et al. ([2020](#bib.bib10 ""))*. Specifically, we transform knowledge triples from the KG into sentences and feed them into pre-trained LM to get node embeddings.

* •

    Action: The action space encompasses all the neighboring entities of the current entity, enabling the agent to explore the KG flexibly. By taking an action, the agent will move from the current entity to the chosen neighboring entity.

* •

    Transition: The transition model $\mathrm{P}$ measures the probability of moving to a new state ($s^{\prime}$) given existing state ($s$) and the undertaken action ($a$). In KGs, the transition model takes on the form $\mathrm{P}(s^{\prime}|s,a)\=1$ if $s$ is directed to $s^{\prime}$ through action $a$; Otherwise, $\mathrm{P}(s^{\prime}|s,a)\=0$.

* •

    Reward: To determine the quality of the formed path, we define the reward based on reachability:

    |  | $r_{reach}\=\begin{cases}+1,\&\textit{if }target;\\ -1,\&\textit{otherwise},\end{cases}$ |  | (2) |
    | --- | --- | --- | --- |

    which represents whether the path eventually reaches the target within limited steps. Specifically, the agent receives a reward of $+1$ if it can attain the target within $K$ actions. Otherwise, it will receive $-1$ as the reward.

Reaching a target entity is not our sole focus. To avoid overlong and rigmarole paths, we also design two auxiliary rewards to promote context relatedness and path conciseness.

#### 3.1.1 Context-relatedness Auxiliary Reward

The key motivation is to encourage paths closely related to the given question context. Specifically, we evaluate the semantic relevance of a path $\mathcal{P}_{i}$ to the context $\mathcal{Q}$. Inspired by the prevailing study*Yasunaga et al. ([2021](#bib.bib46 ""))*, a fixed but well-trained matrix $\boldsymbol{W}$ is applied to map the path embedding $\boldsymbol{\mathcal{P}}$ to the same semantic space with context embedding $\boldsymbol{c}$. To this end, this auxiliary reward is formulated as:

|  | $r_{\text{cr}}\=\frac{1}{|i|}\sum_{source}^{i}cos(\boldsymbol{W\times\mathcal{P}}_{i},\boldsymbol{c}),$ |  | (3) |
| --- | --- | --- | --- |

where $\boldsymbol{c}$ is the embedding of context $\mathcal{Q}$ we obtained from a pre-trained LM*Devlin et al. ([2018](#bib.bib7 ""))* and the embedding of path $\mathcal{P}_{i}$ is the average of the embeddings of all the entities and relations we have walked through till $i$, i.e., $Avg(\boldsymbol{e}_{source}+\boldsymbol{re}_{1}...+\boldsymbol{e}_{i})$, where $i\leq length(\mathcal{P}_{target})$. This step-by-step reward scheme provides rewards before the target is reached.

#### 3.1.2 Conciseness Auxiliary Reward

There are two additional significant challenges for the candidate reasoning background. $(i)$ The natural limitation of black-box LLMs for over-long context understanding gives constrained budgets for prompts, where the extracted path is expected to be concise enough to ensure the full understanding by black-box LLMs. $(ii)$ The prohibitive cost of calling black-box LLMs’ API guides the prompt to be more concise. By limiting the step size, we encourage the policy to find as much valuable information as possible within the shortest path length.

Considering the inevitable homogeneity in the large-scale real-world KG constructed from the online corpus, each step in the final path is ideally a necessity. Specifically, we evaluate the conciseness of a path to reduce twists and turns on redundant entities, e.g., synonyms. Thus, the reward for the conciseness of a path $\mathcal{P}_{i}$ is formulated as follows.

|  | $r_{\text{cs}}\=\frac{1}{|\mathcal{P}_{i}|}.$ |  | (4) |
| --- | --- | --- | --- |

Finally, we use the trade-off parameters to balance the significance of each reward: $r_{\text{total}}\=\lambda_{1}r_{reach}+\lambda_{2}r_{cr}+\lambda_{3}r_{cs}.$, where $\lambda_{1}$, $\lambda_{2}$, and $\lambda_{3}$ are hyperparameters.

#### 3.1.3 Training Policy Network

To solve the MDP defined above, a tailored policy network $\pi_{\theta}(s,a)\=p(a|s;\theta)$ is trained to extract a reasoning path in the KG. We optimize the network with policy gradient*Xiong et al. ([2017](#bib.bib45 ""))*.
The optimal policy navigates the agent from the source entity to the target entity while maximizing the accumulated rewards.

### 3.2 Prompt Construction with Multi-armed Bandit

In this subsection, we design a tailored prompt construction strategy based on Multi-Armed Bandit (MAB). The key idea is to learn to select the best path extraction and prompt templates at a meta-level. We will begin by outlining the overall strategy, followed by detailing its instantiation with two path extraction methodologies and three templates.

Suppose we have several path extraction strategies ${\mathcal{P}_{1},\mathcal{P}_{2},...,\mathcal{P}_{m}}$ and several candidate formats $\mathcal{F}\={\mathcal{F}_{1},\mathcal{F}_{2},...,\mathcal{F}_{n}}$. Each path extraction strategy $\mathcal{P}_{i}$ is a method for selecting a subgraph given a question context, such as the RL-based strategy discussed above. Every prompt template $\mathcal{F}_{j}$ represents a mechanism to transform the triples within the subgraph into a prompt for an LLM prediction.

The prompt construction problem is to identify the best combination of $\mathcal{P}$ and $\mathcal{F}$ for a given question. We define the overall process of selection as a reward maximization problem, $\max\sum r_{pf}$, where $r_{pf}$ is obtained as:

|  | $\sigma(f(\mathcal{P}\mathcal{F}_{(i)}))\=\begin{cases}1\&\textit{if }accurate;\\ 0\&\textit{otherwise}.\end{cases}$ |  | (5) |
| --- | --- | --- | --- |

Specifically, $\mathcal{P}\mathcal{F}_{(i)}$, $i\in{0,1,\cdots,m\times n}$ is one of the combination, and $r_{pf}\in{0,1}$ indicates the performance of the output of LLM in answering the current question.

To capture the context-aware correlation between questions and different combinations of knowledge and prompt formats, we formulate the selection mechanism of MAB with an expectation function $E(\cdot)$. It adaptively measures the potential expectation of a combination for different questions.

|  | $\small E(\mathcal{Q}|\mathcal{P}\mathcal{F}_{(i)})\=\boldsymbol{c}\times\bm{\alpha}_{(i)}+\beta_{(i)}.\vspace{1mm}$ |  | (6) |
| --- | --- | --- | --- |

Here, $\boldsymbol{c}$ represents the embedding of $\mathcal{Q}$. The vector $\bm{\alpha}{(i)}$ corresponds to a set of non-negative parameters associated with $\mathcal{P}\mathcal{F}{(i)}$, which have been learned during the previous $k$-$1$ iterations. Additionally, $\beta_{(i)}$ stands for a balancing factor introducing noise according to a Gaussian distribution.

Empirically maximizing $\boldsymbol{c}\times\bm{\alpha}_{i}$ could encourage exploitation*Chen et al. ([2019](#bib.bib5 "")); Dong et al. ([2023b](#bib.bib9 ""))* for the best combination, we could effectively update $\bm{\alpha}_{(i)}$ via modeling the correlations between the context embedding of the anchor question ${\bf c}_{i}$ and all the previously selected contexts ${\bf C}_{(i)}$ for particular combination $\mathcal{P}\mathcal{F}_{(i)}$ in former $k$ steps, and the rewards $\textbf{r}_{pf}^{(i)}$ obtained from the selection of the current combination. Concretely, the $\bm{\beta}^{(b)}$ is updated as:

|  |  | $\displaystyle J({\bf C}_{(i)}^{(k)},\textbf{r}_{pf}^{(i)(k)})\=\sum\limits_{k\=1}^{K}(\textbf{r}_{pf}^{(i)(k)}-{\bf C}_{(i)}^{(k)}\bm{\alpha}^{(i)})^{2}+{\lambda}^{i}\parallel\bm{\alpha}^{(i)}\parallel_{2}^{2}.$ |  | (7) |
| --- | --- | --- | --- | --- |
| | | $\displaystyle\to\bm{\alpha}^{(i)}\=\left(({\bf C}_{(i)}^{(k)})^{\top}{\bf C}_{(i)}^{(k)}+{\lambda}^{i}\textbf{I}\right)^{-1}({\bf C}_{(i)}^{(k)})^{\top}\textbf{r}_{pf}^{(i)(k)}.$ | | |

Here, $J$ denotes the OLS training loss. $\textbf{I}\in\mathbb{R}^{d\times d}$ is an identity matrix and $\lambda^{i}$ is a regularization factor that controls the complexity of the model.

Similarly, in order to encourage exploration within less frequently selected pairings, we employ an upper confidence bound approach to balance exploration and exploitation. This is achieved through the introduction of the parameter $\beta^{(i)}$. Inspired by prevailing studies*Walsh et al. ([2012](#bib.bib42 "")); Dong et al. ([2023b](#bib.bib9 ""))*, we can derive the following exploration term $\beta^{(i)}$:

|  | $\displaystyle\beta^{(i)}$ | $\displaystyle\=\$ | $\displaystyle\gamma\times\sqrt{{\bf c}_{i}\left(({\bf C}_{(i)}^{(k)})^{\top}{\bf C}_{(i)}^{(k)}+{\lambda}^{i}\textbf{I}\right)^{-1}({\bf c}_{(i)})^{\top}},$ |  | (8) |
| --- | --- | --- | --- | --- | --- |

where $\gamma$ is a fixed constant, i.e., $\gamma\=1+\sqrt{ln(2/\delta)/2}$.

When the model picks a combination with a large $\boldsymbol{c}\times\bm{\alpha}_{i}$, it signifies an exploitation process.
Likewise, when the model selects a combination with larger $\beta^{(i)}$, this variance indicates an exploration process due to the model making fewer selections of the current combination. Thus, jointly maximizing $\boldsymbol{c}\times\bm{\alpha}_{i}+\beta_{(i)}$ could help us get rid of the dilemma of exploration and exploitation.

Consequently, our MAB design can leverage the feedback from the LLM to optimize the selection policy. By maximizing the expectation function $E(\cdot)$, it learns to balance the exploitation and exploration to prioritize the most promising prompts for specific question contexts.

#### 3.2.1 Implementation

We implement the above MAB strategies with two path extraction strategies and three templates. Note that our MAB design is general and can be implemented with more path extraction strategies and prompt templates for better performance. The path extraction strategies include:

* •

    $\mathcal{P}_{\text{RL}}$: The RL-based path extraction strategy presented in the previous subsection.

* •

    $\mathcal{P}_{\text{sub}}$: A heuristic sub-graph extraction strategy that extracts a 2-hop subgraph around both the source and target entities. Detailed implementation can be found in Appendix A.1. Since RL is notoriously unstable*Sutton \& Barto ([2018](#bib.bib38 ""))*, we introduce $\mathcal{P}_{\text{sub}}$ as an alternative candidate strategy for the MAB selection, ensuring a fallback option if the RL-based approach does not perform well.

The prompt templates include:

* •

    Triples, denoted as $\mathcal{F}_{t}$, are indeed the originally extracted knowledge and empirically tested that could be understood by the black-box LLMs, e.g., (Sergey_Brin, founder_of, Google),(Sundar_Pichai, ceo_of, Google), (Google, is_a, High-tech Company).

* •

    Sentences is a following solution to transform the knowledge into a colloquial $\mathcal{F}_{s}$, e.g., ‘Sergey Brin, who is a founder of Google, a high-tech company, has now passed the reigns to Sundar Pichai, who is currently serving as the CEO of the company.’

* •

    Graph Description, $\mathcal{F}_{g}$ prompts the LLM by treating the knowledge as a structured graph. We preprocess the extracted knowledge with black-box LLM itself to generate the description by highlighting the center entity, e.g., ‘Google, a high-tech company, stands central in the network. The entity is strongly associated with significant individuals in the tech industry. Sergey Brin, one of the founders, established Google, underscoring its historical beginnings. In the present graph context, Sundar Pichai is recognized as the CEO of Google, symbolizing the company’s current leadership. Thus, Google serves as a vital link between these key figures.’

Considering two path extraction methods: $\mathcal{P}_{\text{sub}}$
and $\mathcal{P}_{\text{RL}}$
, as well as three prompt translation methods: $\mathcal{F}_{t}$, $\mathcal{F}_{s}$ and $\mathcal{F}_{g}$, the MAB is trained to learn from the feedback from LLMs to prioritize the most appropriate combination among two extraction methods and three predefined prompt formats for different real-world question contexts, i.e.,
$\mathcal{P}\mathcal{F}\={(\mathcal{P}_{sub}\mathcal{F}_{t}),$
$(\mathcal{P}_{sub}\mathcal{F}_{s}),(\mathcal{P}_{sub}\mathcal{F}_{g}),(\mathcal{P}_{RL}\mathcal{F}_{t}),(\mathcal{P}_{RL}\mathcal{F}_{s}),(\mathcal{P}_{RL}\mathcal{F}_{g})}$.

*Table 1: Performance comparison among baselines and KnowGPT on three benchmark datasets.*

| Catagory | Model | CommonsenseQA | | OpenBookQA | | MedQA | |
| --- | --- | --- | --- | --- | --- | --- | --- |
| | | IHdev-Acc. | IHtest-Acc. | Dev-Acc. | Test-Acc. | Dev-Acc. | Test-Acc. |
| LM + Fine-tuning | Bert-base | 0.573 | 0.535 | 0.588 | 0.566 | 0.359 | 0.344 |
| | Bert-large | 0.611 | 0.554 | 0.626 | 0.602 | 0.373 | 0.367 |
| RoBert-large | 0.731 | 0.687 | 0.668 | 0.648 | 0.369 | 0.361 |
| KG-enhanced LM | MHGRN | 0.745 | 0.713 | 0.786 | 0.806 | - | - |
| | QA-GNN | 0.765 | 0.733 | 0.836 | 0.828 | 0.394 | 0.381 |
| HamQA | 0.769 | 0.739 | 0.858 | 0.846 | 0.396 | 0.385 |
| JointLK | 0.777 | 0.744 | 0.864 | 0.856 | 0.411 | 0.403 |
| GreaseLM | 0.785 | 0.742 | 0.857 | 0.848 | 0.400 | 0.385 |
| GrapeQA | 0.782 | 0.749 | 0.849 | 0.824 | 0.401 | 0.395 |
| LLM | ChatGLM | 0.473 | 0.469 | 0.352 | 0.360 | 0.346 | 0.366 |
| | ChatGLM2 | 0.440 | 0.425 | 0.392 | 0.386 | 0.432 | 0.422 |
| Baichuan-7B | 0.491 | 0.476 | 0.411 | 0.395 | 0.334 | 0.319 |
| InternLM | 0.477 | 0.454 | 0.376 | 0.406 | 0.325 | 0.348 |
| GPT-3 | 0.539 | 0.520 | 0.420 | 0.482 | 0.312 | 0.289 |
| ChatGPT | 0.735 | 0.710 | 0.598 | 0.600 | 0.484 | 0.487 |
| GPT-4 | 0.776 | 0.786 | 0.878 | 0.910 | 0.739 | 0.763 |
| Ours | KnowGPT | 0.827 | 0.818 | 0.900 | 0.924 | 0.776 | 0.781 |
| KnowGPT vs. ChatGPT | + 23.7% (Avg.) | + 9.2% | + 10.8% | + 31.2% | + 32.4% | + 29.2% | + 29.4% |
| KnowGPT vs. GPT-4 | +2.9% (Avg.) | + 5.1% | + 3.3% | + 2.2% | + 1.4% | + 3.7% | + 1.8% |

*We used ‘text-davinci-002’ provided by OpenAI as the implementation of GPT-3, and ‘gpt-3.5-turbo’ for ChatGPT.

4 Experiments
-------------

We conduct extensive experiments to evaluate KnowGPT on three benchmark question-answering datasets, covering both commonsense and domain-specific QA.
Our experiments are designed to answer the following research questions:

* •

    RQ1: How does KnowGPT perform when compared with the state-of-the-art LLMs and KG-enhanced QA baselines?

* •

    RQ2: Does the proposed MAB-based prompt construction strategy contribute to the performance?

* •

    RQ3: Can KnowGPT solve complex reasoning tasks, and is KG helpful in this reasoning process?

### 4.1 Experimental Setup

#### 4.1.1 QA Datasets

We evaluate KnowGPT on three QA datasets spanning two fields: CommonsenseQA*Talmor et al. ([2019](#bib.bib39 ""))* and OpenBookQA*Mihaylov et al. ([2018](#bib.bib26 ""))* serve as benchmarks for commonsense reasoning, while MedQA-USMLE*Jin et al. ([2021](#bib.bib15 ""))* acts as a domain-specific QA benchmark. The statistics of these three datasets can be found in TableLABEL:tab:DatasetStatistics in the Appendix.

CommonsenseQA is a multiple-choice question-answering dataset, each question accompanied by five potential answers. Answering its 12,102 questions necessitates a foundation in commonsense knowledge. While the official test set serves primarily for leaderboard rankings, we initially assess model efficacy using the in-house (IH) data split introduced in*Lin et al. ([2019](#bib.bib20 ""))*. The official dataset is denoted as CSQA, while the IH split is represented by CSQA(IH)*.

OpenBookQA, commonly abbreviated as OBQA, comprises 5,957 multiple-choice questions, each offering four possible answers. To successfully answer these questions, one must have a comprehensive understanding of fundamental scientific facts and its applications.

MedQA-USMLE, abbreviated as MedQA, is a dataset consisting of 4-option multiple-choice questions that demand a grasp of biomedical and clinical understanding. These questions are sourced from preparatory tests for the United States Medical Licensing Examinations, and the dataset encompasses 12,723 questions. We adhere to the original data divisions as outlined in*Jin et al. ([2021](#bib.bib15 ""))*.

#### 4.1.2 Background Knowledge

To facilitate common sense reasoning, we employ ConceptNet *Speer et al. ([2017](#bib.bib32 ""))*, an extensive commonsense knowledge graph comprising more than 8 million interconnected entities through 34 concise relationships. For tasks specific to the medical domain, we leverage USMLE *Yasunaga et al. ([2021](#bib.bib46 ""))* as our foundational knowledge source. USMLE is a biomedical knowledge graph that amalgamates the Disease Database segment of the Unified Medical Language System (UMLS)*Bodenreider ([2004](#bib.bib2 ""))* and DrugBank*Wishart et al. ([2017](#bib.bib44 ""))*. This repository encompasses 9,958 nodes and 44,561 edges.

#### 4.1.3 State-of-the-art Baselines

We carefully select baseline models from three categories for a comprehensive evaluation.

LM + Fine-tuning We compare our method with vanilla fine-tuned LMs. Specifically, we choose Bert-base, Bert-large*Devlin et al. ([2018](#bib.bib7 ""))*, and RoBert-large*Liu et al. ([2019](#bib.bib25 ""))* as representative fine-tune LM methods. To conduct commonsense and biomedical QA, we fine-tune these three LMs via an additional linear layer.

KG-enhanced LM We have also implemented several recently released models for integrating KGs into question answering, which encompass MHGRN*Feng et al. ([2020](#bib.bib10 ""))*, QA-GNN*Yasunaga et al. ([2021](#bib.bib46 ""))*, HamQA*Dong et al. ([2023a](#bib.bib8 ""))*, JointLK*Sun et al. ([2021b](#bib.bib37 ""))*, GreaseLM*Zhang et al. ([2022](#bib.bib49 ""))* and GrapeQA*Taunk et al. ([2023](#bib.bib40 ""))*.
To ensure a fair comparison, we implement these baselines with advanced language models that are optimized for particular datasets.
Specifically, RoBert-large*Liu et al. ([2019](#bib.bib25 ""))* is used for CommenseQA, while AristoRoBERTa*Clark et al. ([2020](#bib.bib6 ""))* is designated for OpenBookQA. For MedQA, we opt for the top-tier biomedical language model, SapBERT*Liu et al. ([2020a](#bib.bib21 ""))*. Note that due to the white-box nature of these methods and their high computation overheads, it is infeasible to apply them to state-of-the-art LLMs, like ChatGPT and GPT-4.

LLM We also add several representative generative LLMs, including ChatGLM, ChatGLM2, Baichuan-7B, InternLM, GPT-3, ChatGPT and GPT-4 as knowledge-agnostic alternatives. Specifically, we used the model ‘text-davinci-002’ provided by OpenAI as the implementation of GPT-3, and ‘gpt-3.5-turbo’ and ‘gpt-4’ as the implementations of ChatGPT and GPT-4, respectively (we have provided more implementation details of all LLMs in Appendix A.4). The question-answering task is conducted under the zero-shot setting with the question query from the test set as input.

*Table 2: OpenBookQA Official Leaderboard records of three groups of related models (sorted by rankings).*

| OpenBookQA Leaderboard | |
| --- | --- |
| Human Performance | 0.917 |
| w/o KG | 0.778 |
| MHGRNFeng et al. ([2020](#bib.bib10 "")) | 0.806 |
| QA-GNNYasunaga et al. ([2021](#bib.bib46 "")) | 0.828 |
| GreaseLMZhang et al. ([2022](#bib.bib49 "")) | 0.848 |
| HamQADong et al. ([2023a](#bib.bib8 "")) | 0.850 |
| JointLKSun et al. ([2021b](#bib.bib37 "")) | 0.856 |
| GSCWang et al. ([2021](#bib.bib43 "")) | 0.874 |
| UnifiedQAKhashabi et al. ([2020](#bib.bib16 "")) | 0.872 |
| DRAGONYasunaga et al. ([2022](#bib.bib47 "")) | 0.878 |
| GenMCHuang et al. ([2022](#bib.bib13 "")) | 0.898 |
| GenMC EnsembleHuang et al. ([2022](#bib.bib13 "")) | 0.920 |
| MVP-Tuning EnsembleHuang et al. ([2023](#bib.bib12 "")) | 0.952 |
| KnowGPT | 0.916 |

### 4.2 Main Results (RQ1)

To address RQ1, we evaluate KnowGPT by comparing it to state-of-the-art baselines on the three benchmark datasets. KnowGPT is based on the original ChatGPT. We measure the performance using accuracy, which calculates the percentage of questions correctly predicted by the model out of the total questions in the test set. We make the following observations:

* •

    KnowGPT outperforms all categories of methods, including sixteen different baselines, across all datasets and model architectures. This suggests that KnowGPT can effectively inject the knowledge from KGs to LLMs.

* •

    KnowGPT surpasses the performance of ChatGPT and even GPT-4. On average, KnowGPT achieves a 23.7% higher testing accuracy than ChatGPT. Specifically, KnowGPT outperforms ChatGPT by 10.8%, 32.4%, and 29.4% on the CommonsenseQA, OpenBookQA, and MedQA datasets, respectively. More importantly, despite being based on ChatGPT, KnowGPT outperforms the state-of-the-art LLM GPT-4 by 3.3%, 1.4%, and 1.8% on the CommonsenseQA, OpenBookQA, and MedQA datasets, respectively. These results confirm that black-box knowledge injecting can effectively enhance the capabilities of LLMs.

* •

    KnowGPT outperforms all KG-enhanced LMs significantly. This implies our black-box knowledge injection method proficiently encodes knowledge into LLMs. Furthermore, it showcases the superiority of our black-box approach, given its adaptable application to ChatGPT using only the model API, a feat not achievable by white-box methods.

#### 4.2.1 Leaderboard Ranking

We submit our testing results on the official leaderboard maintained by the authors of OpenbookQA. The full records on the leaderboard are shown on the website111<https://leaderboard.allenai.org/open_book_qa/submissions/public>., while our result can be found from here222<https://leaderboard.allenai.org/open_book_qa/submission/cj9game4arcuacugbrj0>..

We summarize the related submissions in Table[2](#S4.T2 "Table 2 ‣ 4.1.3 State-of-the-art Baselines ‣ 4.1 Experimental Setup ‣ 4 Experiments ‣ KnowGPT: Black-Box Knowledge Injection for Large Language Models"), including three categories: traditional KG-enhanced LM, fine-tuning of LLMs, e.g., T5-11B used in UnifiedQA, and ensemble of multiple predictors. KnowGPT significantly outperforms traditional KG-enhanced LMs with 4.2% improvements when compared to the best baseline.
The third group of methods occupies the leaderboard by leveraging ensemble learning. Nevertheless, KnowGPT can still obtain competitive performance without ensembling with merely 0.4% below GenMC Ensemble*Huang et al. ([2022](#bib.bib13 ""))*. Additionally, our KnowGPT is also comparable to human performance.

*Table 3: Ablation study on the effectiveness of two path extraction methods.*

| Path Extraction | Model | CSQA | | OBQA | MedQA |
| --- | --- | --- | --- | --- | --- |
| | | IHdev | IHtest | Test | Test |
| w/o KG | GPT-3 | 0.539 | 0.520 | 0.482 | 0.289 |
| | ChatGPT | 0.735 | 0.710 | 0.598 | 0.487 |
| GPT-4 | 0.776 | 0.786 | 0.910 | 0.763 |
| $\mathcal{P}_{\text{sub}}$ | ChatGPT | 0.750 | 0.739 | 0.865 | 0.695 |
| $\mathcal{P}_{\text{RL}}$ | ChatGPT | 0.815 | 0.800 | 0.889 | 0.755 |
| Ours | KnowGPT | 0.827 | 0.818 | 0.924 | 0.781 |

### 4.3 Ablation Studies (RQ2)

To answer RQ2, we conduct two ablation studies. First, in Table[3](#S4.T3 "Table 3 ‣ 4.2.1 Leaderboard Ranking ‣ 4.2 Main Results (RQ1) ‣ 4 Experiments ‣ KnowGPT: Black-Box Knowledge Injection for Large Language Models"), we measure the importance of the tailored reinforcement learning-based path extraction module, i.e., $\mathcal{P}_{\text{RL}}$. Specifically, we compare it with the direct path extraction method, i.e., $\mathcal{P}_{\text{sub}}$. The performance is evaluated by directly feeding the extracted knowledge with the prompt format of ‘Sentence’, i.e., $\mathcal{F}_{s}$, to ChatGPT. We also include ‘w/o KG’ as the baseline where ChatGPT is asked to independently answer the given question with no reasoning background provided. The results clearly indicate the vital role of our proposed path extraction strategies. Second, we compare each of the three prompt formats subject to the same extracted knowledge. The detailed results are shown in Table[4](#S4.T4 "Table 4 ‣ 4.3 Ablation Studies (RQ2) ‣ 4 Experiments ‣ KnowGPT: Black-Box Knowledge Injection for Large Language Models"). Though different formats perform similarly within the difference of 2.2% - 3.3%, they are particularly suitable for different kinds of questions. We illustrate this observation in the following case study section. Both ablation studies support the indispensability of each module, armed with a tailored deep reinforcement learning-based path extraction and a context-aware prompt translation, our KnowGPT performs best on all three benchmark datasets.

*Table 4: Ablation study on the effectiveness of prompt translation formats.*

| Path Extraction | Prompts | CSQA | | OBQA | MedQA |
| --- | --- | --- | --- | --- | --- |
| | | IHdev | IHtest | Test | Test |
| $\mathcal{P}_{\text{sub}}$ | $\mathcal{F}_{t}$ | 0.728 | 0.701 | 0.832 | 0.589 |
| | $\mathcal{F}_{s}$ | 0.750 | 0.739 | 0.865 | 0.695 |
| $\mathcal{F}_{g}$ | 0.737 | 0.715 | 0.871 | 0.680 |
| $\mathcal{P}_{\text{RL}}$ | $\mathcal{F}_{t}$ | 0.782 | 0.769 | 0.853 | 0.739 |
| | $\mathcal{F}_{s}$ | 0.815 | 0.800 | 0.889 | 0.755 |
| $\mathcal{F}_{g}$ | 0.806 | 0.793 | 0.906 | 0.762 |
| Full KnowGPT | | 0.827 | 0.818 | 0.924 | 0.781 |

### 4.4 Case Studies (RQ3)

For RQ3, we provide insights into how KnowGPT facilitates the prompt translation with a real case from CommonsenseQA. We visualize both the extracted knowledge and the textual inputs to ChatGPT in Figure[3](#S4.F3 "Figure 3 ‣ 4.4 Case Studies (RQ3) ‣ 4 Experiments ‣ KnowGPT: Black-Box Knowledge Injection for Large Language Models"). In this example, given the same extracted knowledge, ChatGPT answers correctly based on the sentence format that we provide. In contrast, it fails to answer the question with triples and graph descriptions. They clearly indicate the superiority of KnowGPT in an automatic context-aware prompt translation. We make the following observations: $(i)$ Triple format $\mathcal{F}_{t}$ is intuitively suitable for all the simple questions by directly indicating the one-hop knowledge. $(ii)$ Graph description may inevitably introduce noise to ensure the completeness and contextual fluency of the directed graph. In this example, since ‘vacation’ appears in both question and answer choices, over-emphasizing and connecting the knowledge about ‘vacation’ with other concepts in the graph misleads the model to make a prediction with an oblique focus. $(iii)$ Our KnowGPT has shown superior performance in automatically constructing suitable prompts for particular questions.

<img src='x3.png' alt='Refer to caption' title='' width='257' height='246' />

*Figure 3: A case study on exploring the effectiveness of different prompt formats for particular questions.
The extracted knowledge is shown in the middle of this figure in the form of a graph, where the nodes in blue are the key topic entities and the red is the target answer. The text boxes at the bottom are the final prompts generated based on three different formats.*

5 Related Work
--------------

### 5.1 Integration of KGs and LLMs

One heuristic way to incorporate KGs and LLMs is to inject triples as the input*Pan et al. ([2023](#bib.bib28 ""))*. ERNIE3.0*Sun et al. ([2021a](#bib.bib36 ""))* takes triples as token sequences and straightforwardly appends them to the given sentences.
Colake*Sun et al. ([2020](#bib.bib35 ""))* presents a unified graph that combines a word graph with the given context and KG.
QA-GNN*Yasunaga et al. ([2021](#bib.bib46 ""))* takes this further and embeds the question context as an entity in the joint graph. Recent studies utilize the attention mechanism to incorporate KGs into LMs, thereby enhancing comprehension and reasoning processes.
HamQA*Dong et al. ([2023a](#bib.bib8 ""))* proposes a hyperbolic-based graph attention network to learn from the ubiquitous hyponymy in real-world questions.
However, these methods assume they know everything about LMs, including model architectures and parameters. However, many SOTA LLMs are confined to a *black-box* role in practice. In this work, we develop a *black-box* knowledge injection framework that can efficiently integrate KGs into LLMs with APIs only.

### 5.2 Prompt-based Learning

Prompting texts to LLMs for prediction has been a new learning paradigm in natural language processing*Liu et al. ([2023](#bib.bib22 ""))*. In this paper, we refer to fixed-LM prompt tuning*Li \& Liang ([2021](#bib.bib19 "")); Lester et al. ([2021](#bib.bib18 ""))*, where the model only optimizes the prompt construction and keeps the LM frozen. To form effective prompts with suitable templates, researchers propose to shape the contexts into clozes with unfilled slots*Jiang et al. ([2020](#bib.bib14 "")); Petroni et al. ([2019](#bib.bib29 ""))*, or manually define the prompt format for different scenarios*Brown et al. ([2020](#bib.bib4 ""))*. In this paper, we propose a knowledge injection framework that $(i)$ replaces the cloze and examples with knowledgeable reasoning background and $(ii)$ employs a tailored Multi-armed Bandit algorithm to automatically select the suitable prompt formats.

6 Conclusion and Future Work
----------------------------

In this work, we formally define the problem of black-box knowledge injection for LLMs in complex question answering. A novel framework, namely KnowGPT, is presented to integrate KGs into LLMs effectively with model APIs only. We first train a deep RL policy to extract informative and concise reasoning paths from the KG. Then we learn an MAB to select the most effective path extraction method and prompt template for each question.
Extensive experiments on both general and domain-specific QA datasets show the superior performance of KnowGPT and the effectiveness of each component. In the future, we will study more advanced path extraction strategies and prompt templates to improve the performance further.

References
----------

* Amaro et al. (2023)Ilaria Amaro, Attilio Della Greca, Rita Francese, Genoveffa Tortora, and Cesare Tucci.Ai unreliable answers: A case study on chatgpt.In *ICHCI*, pp. 23–40. Springer, 2023.
* Bodenreider (2004)Olivier Bodenreider.The Unified Medical Language System (UMLS): integrating biomedical terminology.*Nucleic Acids Research*, 32:D267–D270, 01 2004.ISSN 0305-1048.
* Bollacker et al. (2008)Kurt Bollacker, Colin Evans, Praveen Paritosh, Tim Sturge, and Jamie Taylor.Freebase: a collaboratively created graph database for structuring human knowledge.In *Proceedings of the 2008 ACM SIGMOD international conference on Management of data*, pp. 1247–1250. ACM, 2008.
* Brown et al. (2020)Tom Brown, Benjamin Mann, Nick Ryder, Ilya Subbiah, and Dario Amodei.Language models are few-shot learners.In H. Larochelle, M. Ranzato, R. Hadsell, M.F. Balcan, and H. Lin (eds.), *Advances in Neural Information Processing Systems*, volume 33, pp. 1877–1901, 2020.
* Chen et al. (2019)Yifang Chen, Chung-Wei Lee, Haipeng Luo, and Chen-Yu Wei.A new algorithm for non-stationary contextual bandits: Efficient, optimal and parameter-free.In *COLT*. PMLR, 2019.
* Clark et al. (2020)Peter Clark, Oren Etzioni, Tushar Khot, Daniel Khashabi, Bhavana Mishra, Kyle Richardson, Ashish Sabharwal, Carissa Schoenick, Oyvind Tafjord, Niket Tandon, et al.From ‘f’to ‘a’on the ny regents science exams: An overview of the aristo project.*AI Magazine*, 41(4):39–53, 2020.
* Devlin et al. (2018)Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.BERT: Pre-training of deep bidirectional transformers for language understanding.*arXiv preprint arXiv:1810.04805*, 2018.
* Dong et al. (2023a)Junnan Dong, Qinggang Zhang, Xiao Huang, Keyu Duan, Qiaoyu Tan, and Zhimeng Jiang.Hierarchy-aware multi-hop question answering over knowledge graphs.In *WWW*, pp. 2519–2527, 2023a.
* Dong et al. (2023b)Junnan Dong, Qinggang Zhang, Xiao Huang, Qiaoyu Tan, Daochen Zha, and Zhao Zihao.Active ensemble learning for knowledge graph error detection.In *WSDM*, pp. 877–885, 2023b.
* Feng et al. (2020)Yanlin Feng, Xinyue Chen, Bill Yuchen Lin, Peifeng Wang, Jun Yan, and Xiang Ren.Scalable multi-hop relational reasoning for knowledge-aware question answering.In *EMNLP*, pp. 1295–1309, 2020.
* Gravel et al. (2023)Jocelyn Gravel, Madeleine D’Amours-Gravel, and Esli Osmanlliu.Learning to fake it: limited responses and fabricated references provided by chatgpt for medical questions.*Mayo Clinic Proceedings: Digital Health*, 1(3):226–234, 2023.
* Huang et al. (2023)Yongfeng Huang, Yanyang Li, Yichong Xu, Lin Zhang, Ruyi Gan, Jiaxing Zhang, and Liwei Wang.Mvp-tuning: Multi-view knowledge retrieval with prompt tuning for commonsense reasoning.In *Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 13417–13432, 2023.
* Huang et al. (2022)Zixian Huang, Ao Wu, Jiaying Zhou, Yu Gu, Yue Zhao, and Gong Cheng.Clues before answers: Generation-enhanced multiple-choice qa.*arXiv preprint arXiv:2205.00274*, 2022.
* Jiang et al. (2020)Zhengbao Jiang, Antonios Anastasopoulos, Jun Araki, Haibo Ding, and Graham Neubig.X-factr: Multilingual factual knowledge retrieval from pretrained language models.*arXiv preprint arXiv:2010.06189*, 2020.
* Jin et al. (2021)Di Jin, Eileen Pan, Nassim Oufattole, Wei-Hung Weng, Hanyi Fang, and Peter Szolovits.What disease does this patient have? a large-scale open domain question answering dataset from medical exams.*Applied Sciences*, 11(14), 2021.ISSN 2076-3417.doi: 10.3390/app11146421.URL [https://www.mdpi.com/2076-3417/11/14/6421](https://www.mdpi.com/2076-3417/11/14/6421 "").
* Khashabi et al. (2020)Daniel Khashabi, Sewon Min, Tushar Khot, Ashish Sabharwal, Oyvind Tafjord, Peter Clark, and Hannaneh Hajishirzi.Unifiedqa: Crossing format boundaries with a single qa system.In *EMNLP 2020*, pp. 1896–1907, 2020.
* Kung et al. (2023)Tiffany H Kung, Morgan Cheatham, Arielle Medenilla, Czarina Sillos, Lorie De Leon, Camille Elepaño, Maria Madriaga, Rimel Aggabao, Giezel Diaz-Candido, James Maningo, et al.Performance of chatgpt on usmle: Potential for ai-assisted medical education using large language models.*PLoS digital health*, 2(2):e0000198, 2023.
* Lester et al. (2021)Brian Lester, Rami Al-Rfou, and Noah Constant.The power of scale for parameter-efficient prompt tuning.*arXiv preprint arXiv:2104.08691*, 2021.
* Li \& Liang (2021)Xiang Lisa Li and Percy Liang.Prefix-tuning: Optimizing continuous prompts for generation.*arXiv preprint arXiv:2101.00190*, 2021.
* Lin et al. (2019)Bill Yuchen Lin, Xinyue Chen, Jamin Chen, and Xiang Ren.Kagnet: Knowledge-aware graph networks for commonsense reasoning.In *EMNLP-IJCNLP*, pp. 2829–2839, 2019.
* Liu et al. (2020a)Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier.Self-alignment pretraining for biomedical entity representations.*arXiv preprint arXiv:2010.11784*, 2020a.
* Liu et al. (2023)Pengfei Liu, Weizhe Yuan, Jinlan Fu, Zhengbao Jiang, Hiroaki Hayashi, and Graham Neubig.Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing.*ACM Computing Surveys*, 55(9):1–35, 2023.
* Liu et al. (2020b)Weijie Liu, Peng Zhou, Zhe Zhao, Zhiruo Wang, Qi Ju, Haotang Deng, and Ping Wang.K-bert: Enabling language representation with knowledge graph.In *Proceedings of the AAAI Conference on Artificial Intelligence*, volume 34, pp. 2901–2908, 2020b.
* Liu et al. (2022)Xiao Liu, Kaixuan Ji, Yicheng Fu, Weng Tam, Zhengxiao Du, Zhilin Yang, and Jie Tang.P-tuning: Prompt tuning can be comparable to fine-tuning across scales and tasks.In *ACL*, pp. 61–68, 2022.
* Liu et al. (2019)Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov.Roberta: A robustly optimized bert pretraining approach.*arXiv preprint arXiv:1907.11692*, 2019.
* Mihaylov et al. (2018)Todor Mihaylov, Peter Clark, Tushar Khot, and Ashish Sabharwal.Can a suit of armor conduct electricity? a new dataset for open book question answering.In *EMNLP*, pp. 2381–2391, 2018.
* OpenAI (2023)OpenAI.Gpt-4 technical report, 2023.
* Pan et al. (2023)Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu.Unifying large language models and knowledge graphs: A roadmap.*arXiv preprint arXiv:2306.08302*, 2023.
* Petroni et al. (2019)Fabio Petroni, Tim Rocktäschel, Patrick Lewis, Anton Bakhtin, Yuxiang Wu, Alexander H Miller, and Sebastian Riedel.Language models as knowledge bases?*arXiv preprint arXiv:1909.01066*, 2019.
* Scao et al. (2022)Teven Le Scao, Angela Fan, Christopher Akiki, Ellie Pavlick, Suzana Ilić, Daniel Hesslow, Roman Castagné, Alexandra Sasha Luccioni, François Yvon, Matthias Gallé, et al.Bloom: A 176b-parameter open-access multilingual language model.*arXiv preprint arXiv:2211.05100*, 2022.
* Shen et al. (2023)Xinyue Shen, Zeyuan Chen, Michael Backes, and Yang Zhang.In chatgpt we trust? measuring and characterizing the reliability of chatgpt.*arXiv preprint arXiv:2304.08979*, 2023.
* Speer et al. (2017)Robyn Speer, Joshua Chin, and Catherine Havasi.Conceptnet 5.5: An open multilingual graph of general knowledge.In *Proceedings of the AAAI conference on artificial intelligence*, volume 31, 2017.
* Su et al. (2021)Yusheng Su, Xu Han, Zhengyan Zhang, Yankai Lin, Peng Li, Zhiyuan Liu, Jie Zhou, and Maosong Sun.Cokebert: Contextual knowledge selection and embedding towards enhanced pre-trained language models.*AI Open*, 2:127–134, 2021.
* Suchanek et al. (2007)Fabian M Suchanek, Gjergji Kasneci, and Gerhard Weikum.YAGO: a core of semantic knowledge.In *Proceedings of the 16th international conference on World Wide Web*, pp. 697–706. ACM, 2007.
* Sun et al. (2020)Tianxiang Sun, Yunfan Shao, Xipeng Qiu, Qipeng Guo, Yaru Hu, Xuanjing Huang, and Zheng Zhang.Colake: Contextualized language and knowledge embedding.*arXiv preprint arXiv:2010.00309*, 2020.
* Sun et al. (2021a)Yu Sun, Shuohuan Wang, Shikun Feng, Siyu Ding, Chao Pang, Junyuan Shang, Jiaxiang Liu, Xuyi Chen, Yanbin Zhao, Yuxiang Lu, et al.Ernie 3.0: Large-scale knowledge enhanced pre-training for language understanding and generation.*arXiv preprint arXiv:2107.02137*, 2021a.
* Sun et al. (2021b)Yueqing Sun, Qi Shi, Le Qi, and Yu Zhang.Jointlk: Joint reasoning with language models and knowledge graphs for commonsense question answering.*arXiv preprint arXiv:2112.02732*, 2021b.
* Sutton \& Barto (2018)Richard S Sutton and Andrew G Barto.*Reinforcement learning: An introduction*.MIT press, 2018.
* Talmor et al. (2019)Alon Talmor, Jonathan Herzig, Nicholas Lourie, and Jonathan Berant.Commonsenseqa: A question answering challenge targeting commonsense knowledge.In *Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers)*, pp. 4149–4158, 2019.
* Taunk et al. (2023)Dhaval Taunk, Lakshya Khanna, Siri Venkata Pavan Kumar Kandru, Vasudeva Varma, Charu Sharma, and Makarand Tapaswi.Grapeqa: Graph augmentation and pruning to enhance question-answering.In *Companion Proceedings of the ACM Web Conference 2023*, pp. 1138–1144, 2023.
* Touvron et al. (2023)Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.Llama: Open and efficient foundation language models.*arXiv preprint arXiv:2302.13971*, 2023.
* Walsh et al. (2012)Thomas J Walsh, István Szita, Carlos Diuk, and Michael L Littman.Exploring compact reinforcement-learning representations with linear regression.*arXiv*, 2012.
* Wang et al. (2021)Kuan Wang, Yuyu Zhang, Diyi Yang, Le Song, and Tao Qin.Gnn is a counter? revisiting gnn for question answering, 2021.
* Wishart et al. (2017)David S Wishart, Yannick D Feunang, An C Guo, Elvis J Lo, Ana Marcu, Jason R Grant, Tanvir Sajed, Daniel Johnson, Carin Li, Zinat Sayeeda, Nazanin Assempour, Ithayavani Iynkkaran, Yifeng Liu, Adam Maciejewski, Nicola Gale, Alex Wilson, Lucy Chin, Ryan Cummings, Diana Le, Allison Pon, Craig Knox, and Michael Wilson.DrugBank 5.0: a major update to the DrugBank database for 2018.*Nucleic Acids Research*, 46:D1074–D1082, 2017.
* Xiong et al. (2017)Wenhan Xiong, Thien Hoang, and William Yang Wang.Deeppath: A reinforcement learning method for knowledge graph reasoning.In *EMNLP*, pp. 564–573, 2017.
* Yasunaga et al. (2021)Michihiro Yasunaga, Hongyu Ren, Antoine Bosselut, Percy Liang, and Jure Leskovec.Qa-gnn: Reasoning with language models and knowledge graphs for question answering.*NAACL*, 2021.
* Yasunaga et al. (2022)Michihiro Yasunaga, Antoine Bosselut, Hongyu Ren, Xikun Zhang, Christopher D Manning, Percy Liang, and Jure Leskovec.Deep bidirectional language-knowledge graph pretraining, 2022.
* Zha et al. (2023)Daochen Zha, Zaid Pervaiz Bhat, Kwei-Herng Lai, Fan Yang, Zhimeng Jiang, Shaochen Zhong, and Xia Hu.Data-centric artificial intelligence: A survey.*arXiv preprint arXiv:2303.10158*, 2023.
* Zhang et al. (2022)Xikun Zhang, Antoine Bosselut, Michihiro Yasunaga, Hongyu Ren, Percy Liang, Christopher D Manning, and Jure Leskovec.Greaselm: Graph reasoning enhanced language models for question answering.*arXiv preprint arXiv:2201.08860*, 2022.
* Zhang et al. (2019)Zhengyan Zhang, Xu Han, Zhiyuan Liu, Xin Jiang, Maosong Sun, and Qun Liu.Ernie: Enhanced language representation with informative entities.*arXiv preprint arXiv:1905.07129*, 2019.
