# EVERYTHING OF THOUGHTS  $\triangleleft$ : DEFYING THE LAW OF PENROSE TRIANGLE FOR THOUGHT GENERATION

Ruomeng Ding, $^{1,2}$  Chaoyun Zhang $^{1}$ , Lu Wang $^{1}$ , Yong Xu $^{1}$ , Minghua Ma $^{1}$ , Wei Zhang $^{3}$ , Si Qin $^{1}$ , Saravan Rajmohan $^{1}$ , Qingwei Lin $^{1}$  & Dongmei Zhang $^{1}$

Microsoft  
2Georgia Institute of Technology  
$^{3}$ East China Normal University

# ABSTRACT

Recent advancements in Large Language Models (LLMs) have revolutionized decision-making by breaking down complex problems into more manageable language sequences referred to as "thoughts". An effective thought design should consider three key perspectives: performance, efficiency, and flexibility. However, existing thought can at most exhibit two of these attributes. To address these limitations, we introduce a novel thought prompting approach called "Everything of Thoughts" (XoT) to defy the law of "Penrose triangle  $\triangleleft$  of existing thought paradigms. XoT leverages pretrained reinforcement learning and Monte Carlo Tree Search (MCTS) to incorporate external domain knowledge and planning capability into thoughts, thereby enhancing LLMs' capabilities and enabling them to generalize to unseen problems efficiently. Through the utilization of the MCTS-LLM collaborative thought revision framework, this approach autonomously produces high-quality comprehensive cognitive mappings with minimal LLM interactions. Additionally, XoT empowers LLMs to engage in unconstrained thinking, allowing for flexible cognitive mappings for problems with multiple solutions.

We evaluate XoT on several challenging problem-solving tasks, including Game of 24, 8-Puzzle, and Pocket Cube. Our results demonstrate that XoT significantly outperforms existing approaches in various dimensions, showcasing its remarkable proficiency in addressing complex problems across diverse domains. The code and dataset to reproduce the results in the paper are available at https: //github.com/microsoft/Everything-of-Thoughts-XoT-.

# 1 INTRODUCTION

Recent advancements in Large Language Models (LLMs) have greatly advanced problem solving in diverse domains such as mathematical reasoning, Frieder et al. (2023), knowledge reasoning, Omar et al. (2023), root cause analysis Chen et al. (2023) and causal inference Kiciman et al. (2023), etc.. This progress can be largely attributed to the technique of decomposing intricate problems into smaller language se

quences referred to as "thoughts". Through a step-by-step inference process involving the use of prompts, each thought functions as an intermediate stage, contributing to the simplification of tackling complex problems to fulfill the problem's ultimate objective.

Effective design of thought steps toward complex problem-solving and reasoning, whether for humans or LLMs, should prioritize three crucial aspects, namely:

Table 1: Comparisons of different prompting paradigms.  

<table><tr><td>Paradigm</td><td>Performance</td><td>Efficiency</td><td>Flexibility</td></tr><tr><td>IO</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>CoT</td><td>X</td><td>✓</td><td>X</td></tr><tr><td>CoT-SC</td><td>X</td><td>X</td><td>X</td></tr><tr><td>ToT</td><td>✓</td><td>X</td><td>✓</td></tr><tr><td>GoT</td><td>✓</td><td>X</td><td>✓</td></tr><tr><td>XoT</td><td>✓</td><td>✓</td><td>✓</td></tr></table>

- Performance. Performance is the accuracy of the solution to a problem, including the precision of each thought at intermediate stages. This metric holds paramount importance for problem-solving.  
- Efficiency. Efficiency relates to the number of LLM inference calls required to solve a single problem. Minimizing this aspect is crucial due to the high computational cost associated with LLM inference, thereby reducing the overall number of cost.  
- Flexibility. Flexibility in thought topology refers to the diverse structures that can be employed by LLMs when organizing thoughts for problem-solving. These structures may include chains, trees, or even graphs, mirroring human thought processes. Enabling more flexible thought structures enhances the capacity of LLMs for divergent and creative thinking, which is particularly advantageous in addressing complex problems, especially those with multiple potential solutions.

There exist several thought generation paradigms, such as Chain-of-Thought (CoT) Wei et al. (2022), Tree-of-Thought (ToT) Yao et al. (2023), and Graph-of-Thought (GoT) Besta et al. (2023), etc.. However, these paradigms each have their limitations and cannot simultaneously achieve all the three desired attributes, as illustrated in Table 1. Specifically, direct Input-Output (IO) prompting is suitable primarily for simple problem-solving scenarios with single-step processes, lacking both in performance and flexibility. CoT and self-consistency CoT (CoT-SC) enable step-by-step problem solving, resulting in modest performance improvements, but they are confined to linear thought structures, limiting their flexibility. In contrast, ToT and GoT permit more versatile thought topologies, accommodating tree-like or graph-like structures. However, these paradigms require the evaluation of intermediate thought steps through LLM itself, incurring significant computational costs and inefficiencies due to multiple LLM calls. These paradigms are constrained by a law analogous to the "Penrose triangle", wherein they can achieve a maximum of two out of the three attributes, and none of them can simultaneously attain all three.

We propose a novel solution called "Everything of Thoughts" (XoT) to address the limitations of conventional thought frameworks, enhancing essential attributes of thought generation, including performance, efficiency, and flexibility for LLM inference. XoT leverages reinforcement learning (RL) Li (2017) and Monte Carlo Tree Search (MCTS) Silver et al. (2017), in conjunction with lightweight policy and value networks, to pretrain on specific tasks for thought searching and subsequently generalize to new problems. This pretraining effectively integrates external domain knowledge and planning capability into the "thoughts" provided to LLMs, expanding their problem-solving capabilities, and thereby significantly improving Performance. Once trained, XoT efficiently performs thought searching using MCTS with cost-effective policy and value networks for exploration and autonomously generates complete cognitive mappings for LLMs. It then employs a MCTS-LLM collaborative thought revision process to further improve the thought quality while minimizing LLM interactions. This eliminates the need for LLMs to explore and evaluate thoughts themselves, as required by ToT and GoT, enhancing XoT's Efficiency. Furthermore, MCTS demonstrates remarkable Flexibility as it can explore various thought topologies, including graph structures akin to those employed in human mind mapping processes Faste & Lin (2012); Jamieson (2012). This enables diverse and creative thinking for LLMs, making it particularly valuable when dealing with complex thought structures or tasks featuring multiple potential solutions. By concurrently achieving superior performance, efficiency, and flexibility, XoT challenges the constraints posed by the "Penrose triangle  $\triangle$  law, significantly surpassing the capabilities of other thought generation paradigms.

We comprehensively evaluate XoT across a diverse range of challenging problem-solving tasks, namely Game of 24, 8-Puzzle, and Pocket Cube. Our experimental results consistently showcase XoT's superior performance, and its capacity to provide multiple solutions to problems efficiently with just a few LLM calls. These findings establish XoT as an effective thought generation approach, paving the way for new avenues in LLMs' problem-solving capabilities.

# 2 BACKGROUND

Thought for LLMs. Addressing complex problems often entails breaking down the overarching objective into multiple intermediary steps. The outcomes or cognitive processes associated with each step are thoughts, which can be expressed as linguistic prompt sequences for LLMs to facilitate

Figure 1: Comparison of XoT versus other prompting paradigms.

problem-solving. Structures of these thought may take various forms, including linear chains, hierarchical trees, or interconnected graphs, depending on how the thoughts are organized to advance towards a solution.

Input-Output (IO) Prompting (Fig. 1 (a)). The IO method is the most straightforward approach to instruct LLMs to address a problem without the provision of any intermediate thought processes.

Chain-of-thought (CoT) Wei et al. (2022) (Fig. 1 (b)). CoT decomposes problem-solving into a sequential chain of thoughts, allowing LLMs to approach complex problems step by step.

Self-consistency CoT (CoT-SC) Wang et al. (2023a) (Fig. 1 (c)). CoT-SC employs multiple instances of the CoT to generate multiple outputs from LLMs. It selects the best results from multiple LLM outputs, offering more robust and consistent inference compared to the vanilla CoT.

Tree-of-thought (ToT) Yao et al. (2023) (Fig. 1 (d)). ToT organizes thoughts in a tree-like structure and utilizes search algorithms (e.g., Breadth-First Search, Depth-First Search) to expand the tree in pursuit of an optimal solution. However, thought evaluation in ToT relies on LLMs themselves, necessitating multiple costly and inefficient LLM inference calls.

Graph-of-thought (GoT) Besta et al. (2023) (Fig. 1 (e)). GoT extends the ToT approach by enabling the generation of graph-like thought structures through thought aggregation and refinement during intermediate search phases. Although this method permits more flexible thought structures, it still demands multiple LLM inference calls for evaluation, incurring significant computational costs.

# 3 XOT: EVERYTHING OF THOUGHTS

XoT serves as an LLM-MCTS collaborative framework designed to enhance the thought generation process, thereby assisting LLMs in resolving complex problems. It leverages MCTS for proficient and efficient thought exploration while harnessing the capabilities of LLMs to refine and amend the thoughts derived from MCTS. This synergistic interaction creates a mutually beneficial arrangement, ultimately enabling the successful resolution of intricate problems characterized by high levels of performance, efficiency, and flexibility.

# 3.1 XOT IN A NUTSHELL

We present an overview of the architecture of XoT in Fig. 1 (f). XoT comprises two key components: (i) a MCTS module guided by policy/value networks; and (ii) an LLM solver for thought revision and inference. The MCTS and policy/value networks need to be trained and then generalize to the inference process.

During the training phase, MCTS is harnessed to explore potential thought structures for a specific task through simulated scenarios. This process entails the recording of states, values, and the visitation frequencies of thought nodes in each simulation. These recorded data are subsequently employed to iteratively train the policy and value estimation model, enabling it to assimilate domain knowledge and comprehend the world model.

Once trained, the estimated policy and value are utilized to guide the MCTS to systematically search for a thought trajectory provided to aid LLMs in problem-solving. Note that thoughts extracted only

Figure 2: An illustration of iterative phases in MCTS for thought searching ((a)-(c)) and thought inference in problem resolution (d).

play a supporting role, assisting LLMs in gathering knowledge from external sources and improving its planning capability. These thoughts do not provide LLMs with definitive or error-free answers, as they may contain inaccuracies or suboptimal solutions. LLMs are responsible for review and refining these thoughts when they seem erroneous or require adjustments. They continue MCTS the search process if needed and eventually formulate the final answers by integrating these external thoughts with their internal knowledge.

# 3.2 THOUGHT SEARCHING FORMULATION

The fundamental objective of employing the thought generation paradigm for LLMs is to identify the optimal decomposition of a complex problem into several manageable sub-steps. Each sub-step aims to alter the current status of the problem, eventually culminating in the successful resolution of the overarching problem. This approach, as seen in ToT and GoT, hinges on well-defined state transitions and clear final objectives. Consequently, it is natural to conceptualize the thought-searching process as a Markov Decision Process (MDP) Puterman (1990), in which:

- State  $s_t$ : Represents the current status of the problem. The initial state  $s_0$  corresponds to the original problem, while intermediate states are characterized by either decomposed sub-problems or the results stemming from their resolution.  
- Action  $a_{t}$ : Signifies the one-step solution or action associated with tackling a problem, leading to a transition to a new state, by incorporating their outcomes.  
- Reward  $r$ : Reflects the comprehensive evaluation of the solution to the original problem, assessing whether it has been effectively resolved through the process of problem decomposition.  
- Thought  $\tau$ : A one-step thought is a combination of one-step state and action, i.e.,  $\tau = \{s, a\}$ . This formulation naturally encapsulates the process of decomposing a complex problem into multiple sub-tasks, each accompanied by their respective outcomes.

The detailed definitions of state, action, reward and thought for each task are shown in Table 1. The generation of complete thoughts  $\mathcal{T} = \{\tau_1,\dots ,\tau_N\}$ , can be construed as the endeavor to discover a thought trajectory to maximize the accumulated reward to address the overall problem.

# 3.3 THOUGHTS SEARCHING WITH MCTS

The formulation above naturally aligns the thought within LLM as a state-action pair. This approach facilitates the effective exploration of its optimal trajectory using a combination of MCTS and RL. This adheres to an iterative simulation cycle that encompasses three key phases: selection, expansion & evaluation, and backpropagation. It heavily depends on the utilization of neural networks  $f_{\theta}$ , which simultaneously estimate the value and action probability for a given state  $s_t$ . The aim is to reduce the number of rollouts and accelerate the search process, similar to the approach employed in AlphaGo Zero Silver et al. (2017). We provide a visual representation of an iteration of the MCTS in Fig. 2 (a)-(c) by taking Pocket Cube as an example and detail each process below.

Selection. In the selection phase, the algorithm initiates at the root node and proceeds to choose an action  $a^*$  from the available set  $\mathcal{A}(s)$  for single-step thought generation in the current state  $s$ . This process continues until a leaf node within the current tree is reached. The selection is guided by the

PUCT algorithm Rosin (2011), aiming to maximize the Upper Confidence Bound (UCB) Garivier & Moulines (2011), as follows:

$$
a ^ {*} = \arg \max  _ {a \in \mathcal {A} (s)} \left[ Q (s, a) + w \cdot P _ {\theta} (s, a) \sqrt {\frac {N (s)}{1 + N (s , a)}} \right]. \tag {1}
$$

Here,  $Q(s,a)$  denotes the Q-value of a state-action pair  $(s,a)$ , which estimates the quality of a particular action in a given state. The higher the Q-value, the better the action is considered to be.  $P_{\theta}(s,a)$  denotes the predicted prior probability of selecting action  $a$  given the state  $s$  obtained from a neural network  $f_{\theta}$ , and  $N(s,a)$  represents the count of times action  $a$  has been chosen in state  $s$ . The parameter  $w$  controls the trade-off between exploration and exploitation. The selection process will continue until an unexplored node is encountered.

Evaluation and Expansion. Upon reaching a previously unselected leaf node, we expand to the state  $s$  for the next step for new thought exploration. This expansion involves the evaluation of its value and action probability on the state, which are modeled by neural networks parameterized by  $\theta$ , i.e.,  $(P_{\theta}(s), v_{\theta}(s)) = f_{\theta}(s)$ . Here  $P_{\theta}(s)$  is the prior probabilities for all actions on  $s$ , and  $v_{\theta}(s)$  denotes its predicted state value. These two values are retained and stored for backup purposes, and state  $s$  is masked as "visited".

Backpropagation. Following the expansion of a leaf node in the above phases, which could be either an unexplored or terminal state, the algorithm proceeds to update all the  $Q(s,a)$  values via backpropagation. For unexplored nodes, this update involves computing the mean of its estimated value  $v_{\theta}$ , while for terminated nodes, it's based on the true reward  $r$ . These updates occur as information is backpropagated along the trajectory to subsequent nodes. Additionally, the visit count for each state-action pair is also incremented as follows:  $N(s,a) = N(s,a) + 1$ .

A simulation is completed after a sequence of selection, evaluation, expansion, and backpropagation steps. After conducting multiple simulations, we proceed to the next step by selecting an action at state  $s$  using a probability distribution defined as  $\varepsilon_{a} \propto N(s,a)^{1 / \gamma}$ , where  $\gamma$  is a temperature constant that regulates the level of exploration.

Policy and Value Networks Training. The simulations described above allow us to compile a dataset for each sample state  $s$  containing  $(s, \varepsilon(s), v(s))$ , where  $\varepsilon(s) = \{\varepsilon_a \mid a \in \mathcal{A}(s)\}$ , and  $v(s)$  represents the ground truth value obtained by accumulating rewards along the trajectory starting from state  $s$ . Subsequently, we can train a combined policy and value network  $f_{\theta}$  to minimize the discrepancy between the predicted value  $v_{\theta}(s)$  and the actual value  $v(s)$ , while also maximizing the alignment between the action probabilities produced by the neural network  $P_{\theta}(s)$  and the search probabilities  $\varepsilon(s)$ . This can be achieved by minimizing the following loss function:

$$
\mathcal {L} = (v (s) - v _ {\theta} (s)) ^ {2} + \varepsilon (s) ^ {T} \log P _ {\theta} (s)). \tag {2}
$$

This training iterates alongside the simulation process to continually enhance the performance of  $f_{\theta}$ , resulting in progressive improvements in thought searching capabilities.

# 3.4 THOUGHT INFERENCE WITH MCTS

Once trained, we utilize the  $f_{\theta}$  to guide the MCTS in generating a thought for a new problem, which assists the LLM in solving it. Specifically, MCTS is utilized to perform  $K$  simulations aimed at thought searching and problem-solving, as illustrated in Fig.2 (d). In each simulation,  $f_{\theta}$  is employed to guide the MCTS in its search for a thought trajectory. Throughout the training process,  $f_{\theta}$  incorporates external information related to the state and action quality. This information helps LLMs understand the world model, enhancing their long-term reasoning and planning abilities, which are areas they may not excel in Stechly et al. (2023); Valmeekam et al. (2023), thereby ensuring the performance of thought generation. Once the simulation concludes, we record the visiting count  $N(s,a)$  and the thought trajectory is obtained based on the number of solutions required:

- Single solution. starting from each state  $s$ , the action with the highest visiting count  $N(s, a)$  is selected.  
- Multiple solution. we sample  $M$  thought trajectories following the probability distribution  $\varepsilon_{a} \propto N(s, a)$  and remove duplicates.

Figure 3: An illustration of thought revision process in XoT.

This results in one or multiple thought trajectories  $\mathcal{T}^*$  that consist of a sequence of state-action pairs for problem-solving. The trajectories for multi-solution problems may intertwine and converge at the same goal state, resulting in a graph-like thought structure. This demonstrates that XoT is capable of generating thought structures with flexibility. These trajectories are then transformed into text sequences that are concatenated to form a prompt sequence provided to LLMs. Note that the thought trajectory is concatenated into a single prompt, even in the case of problems with multiple solutions. Therefore, we only require a single LLM inference call at this stage. Given that the  $f_{\theta}$  network is relatively lightweight, this ensures the efficiency of XoT.

Thought-to-Prompt Parsing. Once the thought trajectories  $\mathcal{T}^*$  are extracted from MCTS, we convert them into a textual format necessary for LLM inference. In this conversion process, we transform both the state and action at each step of the thought, i.e.,  $\tau = \{s,a\}$  in  $\mathcal{T}^*$ , into text. This conversion aims to provide a comprehensive state transition, facilitating LLMs in better understanding the task step by step. In the case of multi-solution scenarios, multiple trajectories are concatenated. This format remains consistent across all baselines, and the resulting prompt text is then fed to LLMs for inference or thought revision.

Thought Revision. It is important to acknowledge that that MCTS may not always provide the globally optimal thought trajectory to directly solve the problem flawlessly. Therefore, the thoughts extracted from MCTS serve as a reference thinking process for the problem, aiding LLMs in a supportive capacity. The LLMs will leverage their internal knowledge to review the extracted thought, identify errors in the thought trajectory, and then ground its knowledge in collaboration with the MCTS to revise and refine the thought. In this context, LLM plays a role akin to a participant in the collaborative framework, guiding MCTS to enhance its performance.

The revision process is iterative in nature, as shown in Fig. 3. Initially, upon obtaining the extracted thought, we instruct the LLM to detect any errors in the thought generated by MCTS using its internal knowledge. If the LLM identifies an error, it results in an error state denoted as  $s_e$  within the thought. If no error is found, the thought remains unchanged. Starting from the parent state of  $s_e$ , MCTS conducts an additional set of  $L$  simulations, ultimately yielding a revised thought for the LLM. In scenarios involving multiple solutions, each solution undergoes this process individually. Upon the completion of the revision, we supply the LLMs with the revised thoughts for problem-solving. The revision process can be repeated several times to enhance the reliability of the answer. This collaborative MCTS-LLM framework nurtures a mutually beneficial process for both components, ultimately contributing to the overall performance of problem-solving. Since LLMs are solely utilized for identifying errors during the revision process with only one call, the efficiency of XoT is effectively maintained.

The collaborative revision framework harnesses the strengths of both MCTS and LLMs. MCTS efficiently and flexibly generates candidate thoughts for LLMs through simulations, while LLMs use their internal knowledge to revise and ground these thoughts within the MCTS framework, effectively turning MCTS into a world model for LLMs. This process ensures the generation of high-quality thoughts for problem-solving.

# 4 EXPERIMENT

We conduct an extensive evaluation of our XoT approach in comparison to several baseline methods across three challenging tasks: the Game of 24, the 8-Puzzle (with a  $3 \times 3$  grid), and the  $2 \times 2$

Table 2: An overview of tasks employed in this study.  

<table><tr><td></td><td>Game of 24</td><td>8-Puzzle</td><td>Pocket Cube</td></tr><tr><td>Objective</td><td>Use four numbers on playing cards to make the number 24 through +,-,×, or ÷.</td><td>Rearrange the tiles in the 3 × 3 puzzle from an scrambled state to a goal state</td><td>Rotating the faces of a 2 × 2 pocket cube until each face of the cube is a uniform color</td></tr><tr><td>Input</td><td>4 numbers ranging from 1 to 13, e.g., (4, 6, 10, 10).</td><td>A scrambled 3 × 3 digital puzzle, e.g.,</td><td>A scrambled 2 × 2 pocket cube, e.g., Colors represented as numbers for LLMs.</td></tr><tr><td>Output</td><td>An equation to reach 24, e.g., 4 × 6 + 10 - 10 = 24.</td><td>The slide sequence of the “-” tile, e.g., (Up, Down, Left, Right …).</td><td>The rotation move sequence of the cube, e.g., (F, R2, U&#x27;…).</td></tr><tr><td>Thought</td><td>3 intermediate equations.</td><td>The step-by-step sliding, and the puzzle state after the move.</td><td>The step-by-step rotation, and the cube state after the move.</td></tr><tr><td>State</td><td>The remaining 1-4 numbers.</td><td>The current number layout of the puzzle.</td><td>Colors of each face of the pocket cube.</td></tr><tr><td>Action</td><td>Picking two number and a operation to compose an equation.</td><td>The one-step moving action of the “-” tile.</td><td>The one-step rotation action of cube.</td></tr><tr><td>Reward</td><td>1 if the number of the final number is equal to 24 otherwise -1.</td><td>The negative minimum step on solving the current puzzle state toward the goal state.</td><td>The negative minimum moving step on solving current cube state toward the goal state.</td></tr></table>

Pocket Cube. An overview of these tasks is provided in Table 2. These tasks are characterized by their complexity, requiring multiple steps for completion and potentially having multiple solutions. To assess the effectiveness of XoT, we compare it against IO, CoT, CoT-SC, ToT, GoT, and single MCTS without LLMs for inference and revision. We also finetune LLaMA-2-13B Touvron et al. (2023) for comparison, using the same training data and ground truth labels. The setup of LLaMA-2-13B can be found in Appendix A. We employ both GPT-3.5 Ouyang et al. (2022) and GPT-4 OpenAI (2023) for these evaluations. Note that temperature and top-p are set to 0.0 for all LLM invoked. We further conduct ablation study to assess the impact of thought revisions, the revision success rate, and the sensitivity to the completeness of the provided thoughts, presented in Section 4.4. We conduct case study in Multi-Solution Scenarios in Section 4.5 to illustrate the thought structures. The computational training costs of MCTS are discussed in Appendix B. The discussion on generalizing XoT to other NLP tasks, such as Document Merging Besta et al. (2023), can be found in Appendix C.

Policy/Value Networks Configurations. The policy and value networks in our model utilize a shared multi-layer perceptron (MLP) architecture with two layers and hidden units arranged as (128, 256). Two heads connected to the MLP are responsible for predicting  $v_{\theta}(s)$  and  $P_{\theta}(s)$  separately. The total number of parameters in the Policy/Value Network for all three tasks is approximately  $10^{6}$ . This design results in a considerably smaller model compared to LLM, making it much more efficient. We train this model through three iterations, with each iteration comprising 10 self-play episodes for MCTS.

Evaluation Metric. For each task, we assess the accuracy of each approach on the test set. Additionally, we track the number of LLM invocations required for all approaches to solve a problem, as well as the number of times  $f_{\theta}$  is invoked in the case of XoT. It's important to note that  $f_{\theta}$  is a considerably smaller model compared to LLMs. In the context of multi-solution scenarios, accuracy is computed as the percentage of problems for which any of the answers provided by each approach is correct. Multi-solution Accuracy (MultiAcc) is calculated as the average percentage of correctness across all solutions offered. Furthermore, we capture the total count of distinct solutions provided by each approach, regardless of their correctness, represented as #Sol. Note that we set the

Table 3: Performance comparison on Game of 24.  

<table><tr><td rowspan="2">Model</td><td colspan="3">GPT-3.5</td><td colspan="3">GPT-4</td></tr><tr><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td></tr><tr><td>IO</td><td>6.57</td><td>1.00</td><td>-</td><td>10.22</td><td>1.00</td><td>-</td></tr><tr><td>CoT</td><td>2.19</td><td>1.00</td><td>-</td><td>4.38</td><td>1.00</td><td>-</td></tr><tr><td>CoT-SC</td><td>2.19</td><td>10.00</td><td>-</td><td>4.38</td><td>10.00</td><td>-</td></tr><tr><td>ToT (b=1)</td><td>5.84</td><td>22.11</td><td>-</td><td>34.31</td><td>23.50</td><td>-</td></tr><tr><td>ToT (b=3)</td><td>10.22</td><td>43.96</td><td>-</td><td>60.58</td><td>39.83</td><td>-</td></tr><tr><td>GoT (k=1)</td><td>2.92</td><td>7.00</td><td>-</td><td>10.95</td><td>7.00</td><td>-</td></tr><tr><td>LLaMA-2-13B</td><td>2.19</td><td>-</td><td>-</td><td>2.19</td><td>-</td><td>-</td></tr><tr><td>MCTS</td><td>62.77</td><td>-</td><td>-</td><td>62.77</td><td>-</td><td>-</td></tr><tr><td>XoT (w/ 1 r)</td><td>79.56</td><td>1.39</td><td>92.15</td><td>74.45</td><td>1.38</td><td>88.20</td></tr><tr><td>XoT (w/ 2 r)</td><td>88.32</td><td>1.58</td><td>93.87</td><td>83.94</td><td>1.57</td><td>89.63</td></tr><tr><td>XoT (w/ 3 r)</td><td>90.51</td><td>1.72</td><td>95.94</td><td>85.40</td><td>1.78</td><td>92.48</td></tr></table>

Table 4: Performance comparison on Game of 24 in the multi-solution scenario.  

<table><tr><td rowspan="2">Model</td><td colspan="4">GPT-3.5</td><td colspan="4">GPT-4</td></tr><tr><td>Multi Acc.</td><td>#Sol</td><td>LLM invoked</td><td>fθ invoked</td><td>Multi Acc.</td><td>#Sol</td><td>LLM invoked</td><td>fθ invoked</td></tr><tr><td>IO</td><td>4.87</td><td>2.88</td><td>1.00</td><td>-</td><td>8.27</td><td>2.99</td><td>1.00</td><td>-</td></tr><tr><td>CoT</td><td>1.22</td><td>2.77</td><td>1.00</td><td>-</td><td>7.79</td><td>2.94</td><td>1.00</td><td>-</td></tr><tr><td>CoT-SC</td><td>1.70</td><td>2.76</td><td>10.00</td><td>-</td><td>8.03</td><td>2.99</td><td>10.00</td><td>-</td></tr><tr><td>ToT (b=3)</td><td>3.41</td><td>2.99</td><td>43.96</td><td>-</td><td>39.90</td><td>2.78</td><td>39.83</td><td>-</td></tr><tr><td>GoT (k=3)</td><td>8.03</td><td>1.93</td><td>7.00</td><td>-</td><td>10.46</td><td>1.39</td><td>7.00</td><td>-</td></tr><tr><td>XoT (w/ 1 r)</td><td>62.90</td><td>2.29</td><td>3.51</td><td>116.34</td><td>76.25</td><td>2.36</td><td>2.31</td><td>109.64</td></tr></table>

maximum solution number to 3 for all problems in multi-solution scenarios. In Table 3 to Table 8, the number of thought revision is denoted by  $\mathbf{r}$ .

# 4.1 GAME OF 24

The Game of 24 presents a arithmetic challenge wherein the goal is to employ four numbers within the range of 1 to 13, in conjunction with basic arithmetic operations, (i.e.,  $+$ ,  $-$ ,  $\times$ ,  $\div$ ), to attain a final result of 24. This game may possess multiple valid solutions.

# 4.1.1 TASK SETUP

We collect a dataset from 4nu, comprising 1,362 games ranked by human solving time, spanning a range of difficulty levels from easy to hard. For our testing phase, we randomly selected 137 games, ensuring coverage of various difficulty intervals. The remaining 1,225 problems were used to train the policy/value networks with MCTS. In the context of this task, as outlined in Table 1, the thoughts refer to the three intermediate equations, while the state encompasses the available numbers (ranging from 1 to 4) for creating the equations. Actions involve the selection of two numbers and an operator to form an equation, and the reward is set to 1 if the final equation is both valid and results in the number 24, utilizing each of the input numbers exactly once, otherwise it is set to -1. Performance is measured by calculating the success rate across the 137 test games.

# 4.1.2 BASELINES & XOT SETUP

The IO prompt is supported by five in-context examples. In the case of CoT, we augment each input-output pair by including three intermediate equations. As for ToT, we solicit one-step thought candidates from the LLM at each step, subsequently instructing the LLM to categorize each thought candidate for intermediate selection. For experimental comparison, we conduct experiments on both the top-1 candidate (with  $b = 1$ ) and the top-3 candidates (with  $b = 3$ ) being retained, where  $b$  indicates the branches retained for exploration at each step. For GoT, we employ LLM to generate one-step thought candidates in the same manner as ToT, then we direct the LLM to select the top-1 thought from all candidates for merging the thoughts. We also examine a CoT-SC baseline, which derives the majority output from 10 CoT samples. For XoT, we perform 200 simulations for each action taken, and this count is increased to 500 during the thought revision process.

In the multi-solution scenario, the IO, CoT, and CoT-SC prompts each include 5 examples, with each problem having 1 to 3 different solutions. For ToT, the top-3 candidates (with  $b = 3$ ) at the final step are considered as different solutions. Rather than keeping only the top-1 thought, GoT is instructed to select between 1 to 3 thoughts from all candidates at each step to generate a wider range of solutions. As for XoT, after performing simulations on MCTS, we sample 500 thought trajectories as for exploration and remove duplicates. The top-3 thoughts with the highest counts are preserved.

# 4.1.3 RESULTS

Table 3 displays the overall performance of all methods on this task. Notably, XoT consistently outperforms other baselines on both GPT-3.5 and GPT-4, achieving an accuracy of  $79.56\%$  and  $74.45\%$  respectively, with 1-time revision. However, after 3-time revision process, XoT's accuracy substantially improves to  $90.51\%$  and  $85.40\%$  for GPT-3.5 and GPT-4 respectively. This underscores the impressive performance of XoT, and demonstrates that the revision process significantly enhances performance, with only a limited increase in the utilization of LLM and  $f_{\theta}$ . Interestingly, the revision process in XoT mitigates the performance gap attributable to the modeling ability in this task. As we observe that XoT with GPT-3.5 achieves higher accuracy after revision compared to GPT-4.

Moreover, XoT consistently outperforms the use of MCTS solely. The performance advantages exhibit growth with the number of revision iterations, underscoring the complementary roles of LLM and MCTS, emphasizing their joint necessity in achieving superior results. The fine-tuned LLaMA-2-13B is only successful on  $2.19\%$  of the test data. This performance is lower than the IO method, indicating that the finetuning method is not be suitable for planning tasks like the Game of 24. The best-performing prompting baseline, ToT  $(b = 3)$  on GPT-4, attains an accuracy of  $60.58\%$ . However, it demands a substantial number of LLM invocations (39.83), which results in inefficiency. In contrast, XoT only requires less than 1.8 calls with revision. Although XoT requires some inference calls for  $f_{\theta}$ , the model is significantly less complex than LLM, making it a much more efficient approach.

Table 4 presents the performance of different methods in the multi-solution scenario. Overall, XoT remains the best-performing approach in terms of MultiAcc, significantly outperforming other baselines. Although XoT does not generate the most number of answers compared to other baselines, it generates more accurate answers, as its MultiAcc significantly outperforms other approaches. Notably, generating multiple solutions does not significantly increase XoT's complexity, as it only requires 2.31 LLM calls with GPT-4 and around 100 calls for a smaller  $f_{\theta}$ , making it remain efficient. Overall, the remarkable performance of XoT in the multi-solution scenario demonstrates its ability to generate complex thoughts.

# 4.2 8-PUZZLE

The 8-Puzzle is a classic sliding puzzle game that consists of a  $3 \times 3$  grid with eight numbered tiles and one empty space denoted as “-”. Its objective is to rearrange the tiles from a given initial configuration into a target configuration. The maximum number of steps necessary for the optimal solution of the 8-Puzzle is 31. This problem falls within the category of NP-complete problems Ratner & Warmuth (1986) and may have multiple solutions.

# 4.2.1 TASK SETUP

We randomly generated 419 solvable 8-puzzle problems, with 300 instances allocated for training and 119 instances for testing. All generated problems are solvable within 9 steps. The action space encompasses four directions: [Up, Down, Left, Right]. Note that the legal action space for each problem state may vary due to the dynamic position of the empty space. As shown in Table 1, the thoughts refer to the step-by-step move, and the puzzle state after the move.

# 4.2.2 BASELINES & XOT SETUP

The IO prompt is extended with three in-context examples. In the CoT approach, each input-output pair is enriched by incorporating intermediate legal action sets, the current action, and the current state. In ToT, at each stage, a set of one-step thought candidates are derived from the LLM, from the

Table 5: Performance comparison on 8-Puzzle.  

<table><tr><td rowspan="2">Model</td><td colspan="3">GPT-3.5</td><td colspan="3">GPT-4</td></tr><tr><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td></tr><tr><td>IO</td><td>0.00</td><td>1.00</td><td>-</td><td>1.68</td><td>1.00</td><td>-</td></tr><tr><td>CoT</td><td>0.00</td><td>1.00</td><td>-</td><td>7.56</td><td>1.00</td><td>-</td></tr><tr><td>CoT-SC</td><td>0.84</td><td>10.00</td><td>-</td><td>8.40</td><td>10.00</td><td>-</td></tr><tr><td>ToT (b=1)</td><td>5.88</td><td>31.76</td><td>-</td><td>3.36</td><td>27.49</td><td>-</td></tr><tr><td>ToT (b=3)</td><td>6.72</td><td>55.86</td><td>-</td><td>13.45</td><td>54.13</td><td>-</td></tr><tr><td>GoT (k=1)</td><td>3.36</td><td>19.00</td><td>-</td><td>3.36</td><td>19.00</td><td>-</td></tr><tr><td>LLaMA-2-13B</td><td>0.00</td><td>-</td><td>-</td><td>0.00</td><td>-</td><td>-</td></tr><tr><td>MCTS</td><td>51.26</td><td>-</td><td>-</td><td>51.26</td><td>-</td><td>-</td></tr><tr><td>XoT (w/ 1 r)</td><td>59.66</td><td>1.50</td><td>41.09</td><td>93.28</td><td>1.48</td><td>55.66</td></tr><tr><td>XoT (w/ 2 r)</td><td>59.66</td><td>1.92</td><td>42.18</td><td>94.96</td><td>1.55</td><td>58.91</td></tr><tr><td>XoT (w/ 3 r)</td><td>63.03</td><td>2.29</td><td>42.60</td><td>95.80</td><td>1.61</td><td>62.22</td></tr></table>

Table 6: Performance comparison on 8-Puzzle in the multi-solution scenario.  

<table><tr><td rowspan="2">Model</td><td colspan="4">GPT-3.5</td><td colspan="4">GPT-4</td></tr><tr><td>Multi Acc.</td><td>#Sol</td><td>LLM invoked</td><td>fθ invoked</td><td>Multi Acc.</td><td>#Sol</td><td>LLM invoked</td><td>fθ invoked</td></tr><tr><td>IO</td><td>0.00</td><td>2.47</td><td>1.00</td><td>-</td><td>0.84</td><td>2.97</td><td>1.00</td><td>-</td></tr><tr><td>CoT</td><td>1.43</td><td>2.05</td><td>1.00</td><td>-</td><td>7.84</td><td>1.21</td><td>1.00</td><td>-</td></tr><tr><td>CoT-SC</td><td>1.54</td><td>1.90</td><td>10.00</td><td>-</td><td>6.58</td><td>2.08</td><td>10.00</td><td>-</td></tr><tr><td>ToT (b=3)</td><td>2.52</td><td>2.98</td><td>55.86</td><td>-</td><td>5.60</td><td>2.97</td><td>54.13</td><td>-</td></tr><tr><td>GoT (k=3)</td><td>3.36</td><td>2.96</td><td>24.18</td><td>-</td><td>16.61</td><td>2.70</td><td>22.76</td><td>-</td></tr><tr><td>XoT (w/ 1 r)</td><td>27.45</td><td>2.85</td><td>4.19</td><td>52.06</td><td>76.33</td><td>1.52</td><td>4.30</td><td>66.66</td></tr></table>

current set of legal actions. We impose a maximum step limit of 9 since all generated problems can be solved within this range. The 8-puzzle's rules are conveyed through a system message, including detailed explanations of each action's execution. Similarly, we perform 20 simulations for each action taken with XoT, and increase this number to 50 for thought revision processes.

In the multi-solution scenario, all of the IO, CoT, and CoT-SC prompts consist of four examples. Each problem is presented with one to three distinct solutions. For ToT  $(\mathrm{b} = 3)$  and GoT  $(\mathrm{k} = 3)$ , the maximum number of steps is increased to 12, as correct solutions may not always be optimal and could exceed 9 steps. In the case of XoT, after conducting simulations with MCTS, we sample 50 thought trajectories for exploration and select the top-3 thoughts with the highest counts.

# 4.2.3 RESULTS

The inherent spatial complexity of the 8-Puzzle, the need for long-term planning, and the presence of invalid actions create a significant challenge for LLMs, which rely solely on textual data as input. This challenge is starkly evident in the poor performance of the baselines on both GPT-3.5, where its IO prompting achieve a mere  $0\%$  success rate. XoT successfully addresses this issue by supplying thoughts acquired from MCTS, thereby infusing external knowledge into the problem-solving process. This augmentation empowers LLMs to tackle problems that were previously insurmountable. In summary, when using GPT-4, XoT achieves an accuracy of  $93.28\%$  with 1 revision and  $95.80\%$  with 3 revisions in the 8-Puzzle task, outperforming the best prompting baseline, ToT  $(b = 3)$ , which only achieves  $13.45\%$  accuracy. Additionally, XoT demonstrates efficiency, as it only requires approximately 1.6 LLM calls for 3-time revision setting. The poor performance of finetuned LLaMA-2-13B  $(0\%)$  revealed a significant issue with hallucination. This underscores the inefficiency and ineffectiveness of finetuning approaches for tasks necessitating long-term planning, while also bringing to light the heightened costs associated with its use.

The multi-solution performance presented in Table 6 confirms that the XoT method continues to outperform other baselines for both GPT-3.5 and GPT-4 models in terms of MultiAcc, whether or not revision is applied. The revision process of XoT is particularly beneficial for GPT-4, as it improves the MultiAcc from  $51.26\%$  to  $76.33\%$ , compared to single MCTS. These results again demonstrate that XoT can effectively generate complex thought structures for multi-solutions with high performance and efficiency, making it particularly suitable for this task.

Table 7: Performance comparison on Pocket Cube.  

<table><tr><td rowspan="2">Model</td><td colspan="3">GPT-3.5</td><td colspan="3">GPT-4</td></tr><tr><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td></tr><tr><td>IO</td><td>1.09</td><td>1.00</td><td>-</td><td>1.09</td><td>1.00</td><td>-</td></tr><tr><td>CoT</td><td>0.00</td><td>1.00</td><td>-</td><td>1.09</td><td>1.00</td><td>-</td></tr><tr><td>CoT-SC</td><td>0.00</td><td>10.00</td><td>-</td><td>1.09</td><td>10.00</td><td>-</td></tr><tr><td>ToT (b=1)</td><td>7.65</td><td>16.50</td><td>-</td><td>11.48</td><td>16.39</td><td>-</td></tr><tr><td>ToT (b=3)</td><td>17.49</td><td>58.72</td><td>-</td><td>19.57</td><td>56.58</td><td>-</td></tr><tr><td>GoT (k=1)</td><td>1.64</td><td>8.93</td><td>-</td><td>18.03</td><td>8.55</td><td>-</td></tr><tr><td>LLaMA-2-13B</td><td>0.00</td><td>-</td><td>-</td><td>0.00</td><td>-</td><td>-</td></tr><tr><td>MCTS</td><td>46.44</td><td>-</td><td>-</td><td>46.44</td><td>-</td><td>-</td></tr><tr><td>XoT (w/ 1 r)</td><td>74.32</td><td>1.55</td><td>64.63</td><td>77.60</td><td>1.54</td><td>75.51</td></tr><tr><td>XoT (w/ 2 r)</td><td>80.33</td><td>1.81</td><td>96.46</td><td>79.32</td><td>1.79</td><td>146.52</td></tr><tr><td>XoT (w/ 3 r)</td><td>84.70</td><td>2.01</td><td>103.22</td><td>83.61</td><td>2.00</td><td>84.63</td></tr></table>

Table 8: Performance comparison on Pocket Cube in the multi-solution scenario.  

<table><tr><td rowspan="2">Model</td><td colspan="4">GPT-3.5</td><td colspan="4">GPT-4</td></tr><tr><td>Multi Acc.</td><td>#Sol</td><td>LLM invoked</td><td>fθ invoked</td><td>Multi Acc.</td><td>#Sol</td><td>LLM invoked</td><td>fθ invoked</td></tr><tr><td>IO</td><td>0.27</td><td>2.00</td><td>1.00</td><td>-</td><td>1.09</td><td>1.98</td><td>1.00</td><td>-</td></tr><tr><td>CoT</td><td>0.55</td><td>1.05</td><td>1.00</td><td>-</td><td>0.82</td><td>1.91</td><td>1.00</td><td>-</td></tr><tr><td>CoT-SC</td><td>0.18</td><td>2.90</td><td>10.00</td><td>-</td><td>0.82</td><td>2.92</td><td>1.00</td><td>-</td></tr><tr><td>ToT (b=3)</td><td>5.83</td><td>2.99</td><td>58.72</td><td>-</td><td>6.52</td><td>2.99</td><td>56.58</td><td>-</td></tr><tr><td>GoT (k=3)</td><td>1.09</td><td>2.99</td><td>14.76</td><td>-</td><td>16.85</td><td>2.77</td><td>13.36</td><td>-</td></tr><tr><td>XoT (w/ 1 r)</td><td>48.72</td><td>2.20</td><td>4.13</td><td>115.73</td><td>77.41</td><td>1.72</td><td>4.08</td><td>122.54</td></tr></table>

# 4.3 POCKET CUBE

The  $2 \times 2$  Pocket Cube is a simplified variant of the classic Rubik's Cube puzzle. Its primary objective is to restore all of its faces to a uniform color by executing various face rotations. The maximum number of steps required to optimally solve the cube is 11, and it is also a NP-complete problem Demaine et al. (2017) and may possess multiple solutions. This task is known to be challenging to LLMs cub.

# 4.3.1 TASK SETUP

We initially set all faces of the cube to a uniform color and then randomly apply 5 actions sequentially selected from the 27 legal actions of the Rubik's Cube. This process resulted in the creation of 1,000 training samples and 183 testing samples. All generated problems can be solved within 4 steps. To simplify the action space, we reduced the 27 legal operations to 9 actions, namely:  $\{\mathrm{U},\mathrm{U}^{\prime},\mathrm{U}2,\mathrm{R},\mathrm{R}^{\prime},\mathrm{R}2,\mathrm{F},\mathrm{F}^{\prime},\mathrm{F}2\}$ , which are used in our experiments with both baselines and XoT. As shown in Table 1, the thoughts pertain to the step-by-step rotation, and the cube state after the move.

# 4.3.2 BASELINES & XOT SETUP

The IO prompt is augmented with a single in-context example. In CoT, we enrich each input-output pair by including intermediate actions and states. In ToT, we retrieve one-step thought candidates from the LLM at each stage and instruct the LLM to classify each candidate for intermediate selection. A maximum step limit of 4 is imposed, as all generated problems can be resolved within this range. The cube's rules are conveyed through a system message, which includes the definition of the action space and illustrations of the execution of each action. For XoT, we conduct 20 simulations for each action taken and increase it to 500 for revision.

In the multi-solution setup, the IO, CoT, and CoT-SC prompts each include 3 examples, and each problem within these prompts offers 3 unique solutions. As for ToT  $(\mathrm{b} = 3)$  and GoT  $(\mathrm{k} = 3)$ , the maximum number of steps allowed is extended to 7. In the case of XoT, after conducting MCTS simulations, we gather 50 thought trajectories, and we keep the top 3 thoughts with the highest counts.

# 4.3.3 RESULTS

The Pocket Cube task, similar to the 8-Puzzle, poses a challenge that demands spatial imagination skills, making it difficult for LLMs to excel. As expected, most of the baselines show very poor performance in this task, with some baselines achieving  $0\%$  accuracy. The best prompting baseline, ToT  $(b = 3)$  with GPT-4, only attains a success rate of  $19.57\%$ . In contrast, XoT can achieve over  $77.60\%$  accuracy with 1-time revision and over  $80\%$  accuracy with 3-time revision, establishing itself as an expert in solving this task. This is attributed to the injection of external knowledge from MCTS, enabling LLMs to solve problems that they would struggle with on their own. On the other hand, XoT improves accuracy by  $30\%$  compared to a single MCTS with one-time revision. This demonstrates the effectiveness of integrating MCTS and LLMs. Notably, XoT maintains high efficiency in this task, requiring only approximately 2 LLM inference calls for both GPT-3.5 and GPT-4. Again, the finetuned LLaMA-2-13B struggles with the Pocket Cube task  $(0\%)$ , due to significant hallucination issues. This comparison further validates the potential of XoT in contexts demanding extensive planning and decision-making accuracy.

In the case of the multi-solution scenario, the performance of the XoT method remains remarkable, achieving over  $77\%$  MultiAcc with GPT-4. The revision process continues to play an important role, significantly improving the performance of XoT with both GPT models. The closest competitor in this setting is GoT  $(k = 3)$  with GPT-4, which achieves a MultiAcc of  $16.85\%$ , but it requires a significantly higher number of LLM invocations compared to XoT (13.36 vs. 4.08) and much lower MultiAcc. Overall, XoT retains its position as the best solution for the Pocket Cube.

# 4.4 ABLATION STUDY

In our ablation study, we consider two aspects: the impact of the number of revisions on the performance and efficiency of XoT and the sensitivity of performance to the completeness of the provided thoughts. These angles allow us to gain insights into how XoT's performance can be improved and understand the importance of providing complete thoughts in complex problem-solving tasks.

# 4.4.1 NUMBER OF REVISIONS

It's important to highlight that the performance of each task can be further improved through multiple revisions of the thought using the MCTS-LLM collaborative framework. In Fig. 4, we compare the performance of GPT-3.5 and GPT-4 models using the XoT method with varying numbers of revisions, ranging from 0 to 3, across all three tasks.

In the Game of 24 task, as the number of revisions increases, both models exhibit improved performance. Notably, GPT-3.5 consistently outperforms GPT-4 in terms of accuracy. After three revisions, GPT-3.5 achieves an accuracy of  $90.51\%$ , while GPT-4 reaches  $85.40\%$ . This improved performance comes at the cost of increased inference times and model calls, primarily driven by the need for more interactions to generate revised thoughts. For the 8-Puzzle task, the trend of increasing accuracy with more revisions remains valid. However, in this task, GPT-4 significantly outperforms GPT-3.5. After one revision, GPT-4 achieves an accuracy of  $93.28\%$ , which increases to  $95.80\%$  after the third revision. In contrast, GPT-3.5 only attains an accuracy of  $63.03\%$  after the third revision. In the Pocket Cube task, the performance trend is similar. The accuracy of both models improves with an increase in the number of revisions. GPT-3.5 starts at an accuracy of  $45.36\%$  without revision and improves to  $84.70\%$  after three revisions. GPT-4 begins with an accuracy of  $45.90\%$  and reaches  $83.61\%$  after three revisions. Inference times and model calls are comparable between the two models, with GPT-4 showing a substantial increase in model calls after the third revision.

Note that the number of LLM invocations does not increase dramatically with additional revisions, even though  $f_{\theta}$  is called more times to guide simulations. Considering the significant disparity in inference costs between LLM and  $f_{\theta}$ , increasing the number of revisions to achieve better performance appears to be a favorable trade-off.

We also focus on the efficacy of the revision process within the XOT framework across three distinct tasks. The Revision Success Rate is calculated as the ratio of successfully detected errors to the number of failed cases without revision, thereby providing insight into the effectiveness of revisions. The results for both GPT-3.5 and GPT-4 are presented in Table 9 and Table 10. Our observations


(a) Game of 24



(b) 8-Puzzle



(c) Pocket Cube

Figure 4: Accuracy, LLM and  $f_{\theta}$  invoked comparison on XOT w.r.t. the number of revisions.

Table 9: Revision Success Rate for GPT-3.5.  

<table><tr><td>Revisions</td><td>Game of 24</td><td>8-Puzzle</td><td>Pocket Cube</td></tr><tr><td>XoT (w/ 1 r)</td><td>47.17%</td><td>20.00%</td><td>53.00%</td></tr><tr><td>XoT (w/ 2 r)</td><td>69.81%</td><td>21.31%</td><td>63.64%</td></tr><tr><td>XoT (w/ 3 r)</td><td>75.93%</td><td>26.67%</td><td>72.00%</td></tr></table>

Table 10: Revision Success Rate for GPT-4.  

<table><tr><td>Revisions</td><td>Game of 24</td><td>8-Puzzle</td><td>Pocket Cube</td></tr><tr><td>XoT (w/ 1 r)</td><td>32.69%</td><td>85.96%</td><td>58.59%</td></tr><tr><td>XoT (w/ 2 r)</td><td>55.10%</td><td>89.47%</td><td>60.00%</td></tr><tr><td>XoT (w/ 3 r)</td><td>60.00%</td><td>91.38%</td><td>70.00%</td></tr></table>

reveal a high revision success rate in the XoT framework, which increases with the number of revisions. This underscores the effectiveness of LLMs in the revision process, positioning it as a highly efficient approach to thoughts revision.

# 4.4.2 INCOMPLETE THOUGHT

In this ablation study, we explore the performance of LLMs when provided with incomplete thoughts, specifically omitting the last step of the thought trajectory. This simulates scenarios where MCTS might supply inaccurate or incomplete thoughts. The aim is to test whether LLMs can independently solve problems or rely on their own reasoning, rather than solely relying on the thought from MCTS as answers. We present the performance comparison for all three tasks in Table 11.

Table 11: Performance comparison on three tasks with incomplete thoughts.  

<table><tr><td rowspan="2">Task</td><td rowspan="2">Model</td><td colspan="3">GPT-3.5</td><td colspan="3">GPT-4</td></tr><tr><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td><td>Acc. [%]</td><td>LLM invoked</td><td>fθ invoked</td></tr><tr><td rowspan="3">Game of 24</td><td>ToT (b=1)</td><td>3.65</td><td>17.15</td><td>-</td><td>40.88</td><td>18.55</td><td>-</td></tr><tr><td>GoT (k=1)</td><td>2.19</td><td>5.00</td><td>-</td><td>9.49</td><td>5.00</td><td>-</td></tr><tr><td>XoT (w/o revise)</td><td>17.52</td><td>1.00</td><td>68.73</td><td>43.07</td><td>1.00</td><td>68.70</td></tr><tr><td rowspan="3">8-Puzzle</td><td>ToT (b=1)</td><td>0.00</td><td>32.60</td><td>-</td><td>6.72</td><td>26.98</td><td>-</td></tr><tr><td>GoT (k=1)</td><td>0.00</td><td>18.63</td><td>-</td><td>3.36</td><td>19.00</td><td>-</td></tr><tr><td>XoT (w/o revise)</td><td>2.52</td><td>1.00</td><td>36.66</td><td>40.34</td><td>1.00</td><td>36.24</td></tr><tr><td rowspan="3">Pocket Cube</td><td>ToT (b=1)</td><td>0.55</td><td>16.48</td><td>-</td><td>2.19</td><td>16.39</td><td>-</td></tr><tr><td>GoT (k=1)</td><td>0.00</td><td>8.96</td><td>-</td><td>1.64</td><td>8.68</td><td>-</td></tr><tr><td>XoT (w/o revise)</td><td>5.46</td><td>1.00</td><td>18.85</td><td>6.01</td><td>1.00</td><td>18.89</td></tr></table>

Figure 5: Examples of thought structures generated by XOT for all three tasks in the multi-solution scenario.



Note that we only compare ToT and GoT since other baselines do not support this comparison by their nature.

The results clearly show that incomplete thoughts lead to a significant performance drop in all three tasks. GPT-3.5 is more affected than GPT-4, with GPT-3.5 achieving  $0\%$  accuracy on several baselines. In contrast, XOT with GPT-4 attains satisfactory performance on the Game of 24 and 8-Puzzle, achieving over  $40\%$  accuracy. However, the performance of XOT is dramatically affected in the Pocket Cube task, with accuracy dropping to  $6\%$ . This demonstrates that for very complex tasks, LLMs are highly sensitive to the completeness of the thoughts provided. Missing steps in the thought can lead to a substantial drop in performance, highlighting the importance of providing complete thoughts for such tasks.

# 4.5 CASE STUDY

Finally, in Fig. 5, we provide examples of thought structures generated by XoT for all three tasks in the multi-solution scenario. It is noteworthy that, owing to the multiple solutions required, the generated thoughts intertwine during intermediate steps and converge towards the final goal state. This results in a naturally woven thought structure resembling a graph, showcasing the remarkable flexibility achieved by XoT. Upon closer examination of each example, in the case of the Game of 24, there are multiple solutions to reach the goal of 24 from the initial state. XoT effectively predicts these trajectories, indicating its ability to grasp complex thought structures. In the 8-Puzzle example, we observe instances of reflection in the thought structure, with back-and-forth recurrent state transitions. This demonstrates XoT's capacity for self-reflection, a crucial attribute for LLMs, as discussed in previous work Shinn et al. (2023). In the case of the Pocket Cube, XoT identifies four distinct pathways to reach the goal state, leading to successful problem-solving across multiple solutions.

Overall, these cases highlight how XOt encapsulates the flexibility required in thought generation, fostering diverse and creative thinking for LLMs. This enables them to produce multiple high-quality answers to a single problem effectively.

# 4.6 EXPERIMENT SUMMARY

In summary, our approach XoT significantly improves the performance of LLMs by introducing a streamlined thought trajectory revision process. This represents a fundamental shift from traditional problem-solving approaches, resulting in substantial performance enhancements across a range of tasks. Notably, XoT excels in solving the Game of 24 and demonstrates its ability to overcome challenges requiring spatial reasoning, such as the 8-Puzzle and Pocket Cube, which were previously challenging for LLMs. The remarkable synergy of improved performance, efficiency, and flexibility exhibited by XoT positions it as an exemplary and superior method for eliciting optimal responses from LLMs.

# 5 RELATED WORK

Decision Making & Planning with LLMs. The utilization of LLMs for decision-making and planning has become a prominent area of research. Similar to human problem-solving, the process involves breaking down complex problems into sub-tasks. Various frameworks, such as CoT Wei et al. (2022), ToT Yao et al. (2023), and GoT Besta et al. (2023), have been designed to facilitate problem decomposition in different structural forms, leading to enhanced solutions derived from LLMs. Extensions of these frameworks have also been explored across different domains and modalities Zhang et al. (2022; 2023); Ning et al. (2023); Turpin et al. (2023); Long (2023). Our approach XoT distinguishes itself from the aforementioned work by concurrently achieving superior performance, efficiency, and flexibility, embodying the concept of comprehensive thought generation.

Furthermore, the "Describe, Explain, Plan, and Select" framework introduced in Wang et al. (2023b) presents an interactive planning approach for LLMs, significantly enhancing planning performance for multi-task agents. Research conducted in Singh et al. (2023) leverages LLMs to suggest next actions or sequences during task planning for robotics, leading to improved task performance across various metrics. Additionally, work presented in Xie et al. (2023) employs LLMs to translate natural language into planning goals, demonstrating their capacity to harness commonsense knowledge and reasoning to provide missing details for under-specified goals. These studies underscore the growing potential of LLMs in the field of planning, with research efforts expanding rapidly.

Augmenting LLMs with RL. Enhancing the capabilities of LLMs through the incorporation of external models constitutes an effective strategy for improving their overall quality. The foundational work of ChatGPT Ouyang et al. (2022) leverages RL from human feedback to enable LLMs to adhere to human guidance, resulting in a substantial enhancement of their truthfulness and a reduction in toxic output. Similarly, GLAM Carta et al. (2023) employs online RL to establish alignment between LLMs' knowledge and the broader environment, thus enhancing their ability to generalize to new objects or tasks and ultimately improving their performance. Additionally, an interesting study in Yuan et al. (2023) utilizes RL to acquire basic skills in the context of Minecraft Cipollone et al. (2014), with subsequent high-level planning carried out by LLMs. This approach demonstrates promising performance across various Minecraft tasks. Furthermore, the ESPER framework Yu et al. (2023) harnesses RL to achieve alignment between multimodal inputs and language model generations, all without the need for direct supervision. This empowers LLMs to effectively tackle multimodal tasks and provides robust visual alignment and rapid inference speeds while preserving the textual domain. Collectively, these research endeavors underscore the considerable potential in augmenting LLMs with reinforcement learning techniques.

MCTS is also integrated with LLMs to enhance both training and inference processes. Hao et al., propose "Reasoning via Planning", utilizing LLMs as a world model and reasoning agent, while combining MCTS as a strategic explorer to enhance LLMs' reasoning and planning abilities Hao et al. (2023). Liu et al., incorporate MCTS and PPO Schulman et al. (2017) to devise a value-guided decoding algorithm, thereby enhancing the preferability of generated text by LLMs Liu et al. (2023). Additionally, Feng et al., employ MCTS to augment LLMs' decoding and, consequently,

their reasoning and planning capabilities Feng et al. (2023). These studies underscore the significant potential of integrating MCTS with LLMs to improve their overall capabilities.

# 6 DISCUSSION

Generalization While XoT is presently utilized for reasoning and search problems, its applicability can be extended to a broader spectrum of problem domains characterized by decomposable tasks with well-defined objectives. The MCTS utilized in XoT is particularly suitable for such tasks and can therefore generalize to more complex problems. We also note that MCTS is functioning in a supportive role and can be substituted with alternative supervised or RL models for thought exploration and generation, which can serve as a copilot to inject domain knowledge of the real-world model to LLMs. This opens up a promising avenue for future research, enabling LLMs to engage in more effective planning and problem solving processes.

**Limitation** We also note that the implementation of XoT necessitates the training of additional policy and value models to expedite the inference process. This training process requires the acquisition of datasets from real-world environments, introducing supplementary costs and efforts. However, note that these policy and value models are considerably smaller and more computationally efficient than the underlying LLMs. Consequently, the incurred costs are deemed low, particularly in the context of tasks featured in this study, where the thought steps and objectives are well-defined. In future research endeavors, we intend to explore methods to enhance the efficiency of the training process for XoT in scenarios where the objectives are less straightforward, such as multi-agent planning and code generation tasks Talebirad & Nadiri (2023); Vaithilingam et al. (2022). This endeavor will expand the applicability of the proposed XoT framework to a broader range of applications.

In terms of potential risks, XOT is susceptible to the MCTS module providing incorrect intermediate thoughts, which may result in an inaccurate final answer or hallucination. Changes in the environment could lead to inaccuracies in MCTS and subsequently in the thoughts provided to LLMs. However, LLMs have proven effective in revising thoughts by leveraging their internal knowledge, mitigating the risk associated with inaccuracies in the initial thought generation. Additionally, LLMs may make mistakes and sometimes deviate from the thoughts generated by the MCTS module, leading to errors. This aspect should be taken into consideration when employing the approach.

Conclusion The XoT framework presented in this paper signifies a significant progression in thought generation for LLMs aimed at solving complex tasks. It challenges the constraints of the "Penrose Triangle  $\triangleleft$  by concurrently achieving performance, efficiency, and flexibility, a feat unattainable by existing prompting paradigms. This accomplishment is achieved through the integration of MCTS with pretrained low-cost policy and value networks, by injecting domain knowledge and planning capability into LLMs, offloading thought searching, and facilitating unconstrained free-style thought exploration. The collaborative thought revision framework involving MCTS and LLM further enhances the quality of thought generation. Experimental evaluations conducted across three intricate real-world problems, namely the Game of 24, 8-Puzzle, and Pocket Cube, provide empirical evidence that our XoT framework significantly outperforms existing prompting paradigms, particularly in scenarios involving multi-solution problems.

# REFERENCES

4 Numbers. https://www.4nums.com/game/difficulties/. [Online; accessed 21-Sep-2023].  
I Calculated ChatGPT's IQ. https://www.youtube.com/watch?v=HXb9Azzhr1k. Accessed: 2023-10-30.  
Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, et al. Graph of thoughts: Solving elaborate problems with large language models. arXiv preprint arXiv:2308.09687, 2023.  
Thomas Carta, Clément Romac, Thomas Wolf, Sylvain Lamprier, Olivier Sigaud, and Pierre-Yves Oudeyer. Grounding large language models in interactive environments with online reinforcement learning. arXiv preprint arXiv:2302.02662, 2023.

Yinfang Chen, Huaibing Xie, Minghua Ma, Yu Kang, Xin Gao, Liu Shi, Yunjie Cao, Xuedong Gao, Hao Fan, Ming Wen, et al. Empowering practical root cause analysis by large language models for cloud incidents. arXiv preprint arXiv:2305.15778, 2023.  
Maria Cipollone, Catherine C Schifter, and Rick A Moffat. Minecraft as a creative tool: A case study. International Journal of Game-Based Learning (IJGBL), 4(2):1-14, 2014.  
Erik D Demaine, Sarah Eisenstat, and Mikhail Rudoy. Solving the rubik's cube optimally is np-complete. arXiv preprint arXiv:1706.06708, 2017.  
Haakon Faste and Honray Lin. The untapped promise of digital mind maps. In Proceedings of the SIGCHI conference on human factors in computing systems, pp. 1017-1026, 2012.  
Xidong Feng, Ziyu Wan, Muning Wen, Ying Wen, Weinan Zhang, and Jun Wang. Alphazerolike tree-search can guide large language model decoding and training. arXiv preprint arXiv:2309.17179, 2023.  
Simon Frieder, Luca Pinchetti, Ryan-Rhys Griffiths, Tommaso Salvatori, Thomas Lukasiewicz, Philipp Christian Petersen, Alexis Chevalier, and Julius Berner. Mathematical capabilities of chatgpt. arXiv preprint arXiv:2301.13867, 2023.  
Aurelien Garivier and Eric Moulines. On upper-confidence bound policies for switching bandit problems. In International Conference on Algorithmic Learning Theory, pp. 174-188. Springer, 2011.  
Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. arXiv preprint arXiv:2305.14992, 2023.  
Peter Jamieson. Using modern graph analysis techniques on mind maps to help quantify learning. In 2012 Frontiers in Education Conference Proceedings, pp. 1-6. IEEE, 2012.  
Emre Kiciman, Robert Ness, Amit Sharma, and Chenhao Tan. Causal reasoning and large language models: Opening a new frontier for causality. arXiv preprint arXiv:2305.00050, 2023.  
Minae Kwon, Sang Michael Xie, Kalesha Bullard, and Dorsa Sadigh. Reward design with language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=10uNUgI5Kl.  
Yuxi Li. Deep reinforcement learning: An overview. arXiv preprint arXiv:1701.07274, 2017.  
Jiacheng Liu, Andrew Cohen, Ramakanth Pasunuru, Yejin Choi, Hannaneh Hajishirzi, and Asli Celikyilmaz. Making ppo even better: Value-guided monte-carlo tree search decoding. arXiv preprint arXiv:2309.15028, 2023.  
Jieyi Long. Large language model guided tree-of-thought. arXiv preprint arXiv:2305.08291, 2023.  
Xuefei Ning, Zinan Lin, Zixuan Zhou, Huazhong Yang, and Yu Wang. Skeleton-of-thought: Large language models can do parallel decoding. arXiv preprint arXiv:2307.15337, 2023.  
Reham Omar, Omij Mangukiya, Panos Kalnis, and Essam Mansour. Chatgpt versus traditional question answering for knowledge graphs: Current status and future directions towards knowledge graph chatbots. arXiv preprint arXiv:2302.06466, 2023.  
OpenAI. Gpt-4 technical report, 2023.  
Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35: 27730-27744, 2022.  
Martin L Puterman. Markov decision processes. *Handbooks in operations research and management science*, 2:331-434, 1990.

Daniel Ratner and Manfred Warmuth. Finding a shortest solution for the  $n \times n$  extension of the 15-puzzle is intractable. In Proceedings of the Fifth AAAI National Conference on Artificial Intelligence, pp. 168-172, 1986.  
Christopher D Rosin. Multi-armed bandits with episode context. Annals of Mathematics and Artificial Intelligence, 61(3):203-230, 2011.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.  
Noah Shinn, Beck Labash, and Ashwin Gopinath. Reflexion: an autonomous agent with dynamic memory and self-reflection. arXiv preprint arXiv:2303.11366, 2023.  
David Silver, Julian Schrittwieser, Karen Simonyan, Ioannis Antonoglou, Aja Huang, Arthur Guez, Thomas Hubert, Lucas Baker, Matthew Lai, Adrian Bolton, et al. Mastering the game of go without human knowledge. nature, 550(7676):354-359, 2017.  
Ishika Singh, Valts Blukis, Arsalan Mousavian, Ankit Goyal, Danfei Xu, Jonathan Tremblay, Dieter Fox, Jesse Thomason, and Animesh Garg. Progprompt: Generating situated robot task plans using large language models. In 2023 IEEE International Conference on Robotics and Automation (ICRA), pp. 11523-11530. IEEE, 2023.  
Kaya Stechly, Matthew Marquez, and Subbarao Kambhampati. Gpt-4 doesn't know it's wrong: An analysis of iterative prompting for reasoning problems. arXiv preprint arXiv:2310.12397, 2023.  
Yashar Talebirad and Amirhossein Nadiri. Multi-agent collaboration: Harnessing the power of intelligent llm agents. arXiv preprint arXiv:2306.03314, 2023.  
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023.  
Miles Turpin, Julian Michael, Ethan Perez, and Samuel R Bowman. Language models don't always say what they think: Unfaithful explanations in chain-of-thought prompting. arXiv preprint arXiv:2305.04388, 2023.  
Priyan Vaithilingam, Tianyi Zhang, and Elena L Glassman. Expectation vs. experience: Evaluating the usability of code generation tools powered by large language models. In *Chi conference on human factors in computing systems extended abstracts*, pp. 1-7, 2022.  
Karthik Valmeekam, Matthew Marquez, and Subbarao Kambhampati. Can large language models really improve by self-critiquing their own plans? arXiv preprint arXiv:2310.08118, 2023.  
Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2023a.  
Zihao Wang, Shaofei Cai, Anji Liu, Xiaojian Ma, and Yitao Liang. Describe, explain, plan and select: Interactive planning with large language models enables open-world multi-task agents. arXiv preprint arXiv:2302.01560, 2023b.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824-24837, 2022.  
Yaqi Xie, Chen Yu, Tongyao Zhu, Jinbin Bai, Ze Gong, and Harold Soh. Translating natural language to planning goals with large-language models. arXiv preprint arXiv:2302.05128, 2023.  
Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. arXiv preprint arXiv:2305.10601, 2023.

Youngjae Yu, Jiwan Chung, Heeseung Yun, Jack Hessel, Jae Sung Park, Ximing Lu, Rowan Zellers, Prithviraj Ammanabrolu, Ronan Le Bras, Gunhee Kim, et al. Fusing pre-trained language models with multimodal prompts through reinforcement learning. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 10845-10856, 2023.  
Haoqi Yuan, Chi Zhang, Hongcheng Wang, Feiyang Xie, Penglin Cai, Hao Dong, and Zongqing Lu. Plan4mc: Skill reinforcement learning and planning for open-world minercraft tasks. arXiv preprint arXiv:2303.16563, 2023.  
Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. arXiv preprint arXiv:2210.03493, 2022.  
Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola. Multimodal chain-of-thought reasoning in language models. arXiv preprint arXiv:2302.00923, 2023.

# A LLAMA-2-13B SETUP

LLaMA-2-13B (finetuned). To evaluate the potential of directly distilling knowledge from simulations into a smaller model to possibly avoid using a large model like GPT-4 during testing, we fine-tuned the LLaMA-2-13B model. Our experiments were carried out on eight V100 GPUs, each with 80GB of memory, and lasted approximately 5 hours. The training setup involved 5 epochs, a train batch size of 32, an evaluation batch size of 1, and a single step for gradient accumulation. The evaluation and save strategies were set to "no" and "steps" respectively, with saving occurring every 20 steps and a limit of one saved model. The learning rate was 2e-5, with no warmup steps and logging every 2 steps. We employed a cosine learning rate scheduler. By using ground truth labels—considered more accurate than labels from MCTS simulations—we aimed to convert an optimization or search problem into a more straightforward prediction or supervised learning challenge, using a training dataset of (question, answer) pairs.

# B COMPUTATIONAL TRAINING COSTS OF MCTS

The number of training and testing policy/value model calls for XoT are listed in Table 12. We train this model through three iterations, each comprising 10 self-play episodes for MCTS. Offline pretraining serves as a one-time solution that reduces the computational burden of testing by integrating external knowledge. Methods like ToT and GoT, which rely solely on the LLMs' internal knowledge, do not require pretraining but necessitate frequent calls to LLM during testing. For example, the average number of LLM invocations for three tasks in ToT are 39.83, 54.13, and 56.58, averaging 50.18 times per test problem. The computational cost of these recurring calls during testing exceeds the pretraining cost of the policy/value model in XoT.

Furthermore, it's worth highlighting that GPT-3.5 boasts 175 billion parameters, and GPT-4 is estimated to have an astonishing over 1 trillion parameters. In contrast, the total number of parameters in the Policy/Value Network for all three tasks is approximately 1e6. This deliberate design choice results in a model significantly smaller than LLMs, ensuring efficiency even with additional calls during training.

# C EXPERIMENT RESULTS ON OTHER NLP TASKS

In addition to the tasks employed in this paper, many other NLP tasks can be formulated as MCTS searching problems, using LLMs to get rewards and rendering XoT applicable to a broader range of scenarios. For example, in ToT Yao et al. (2023), the task of Creative Writing uses LLMs to evaluate the quality of generated paragraphs. In a similar vein, GoT Besta et al. (2023) utilizes LLMs to rate the outcomes of Document Merging tasks. This strategy of employing LLMs for reward design is gaining traction and is currently a subject of active research Kwon et al. (2023).

To illustrate, we present preliminary results for GPT-3.5 on the Document Merging task in Table 13, where the scores are indicative of a weighted combination of duplication and information intact in the merged document (the higher the better). The objective of this task is to create a new

Table 12: Number of policy/value model calls in training and testing per iteration for different tasks.  

<table><tr><td></td><td>Game of 24</td><td>8-Puzzle</td><td>Pocket Cube</td></tr><tr><td>Training</td><td>1044.70</td><td>834.70</td><td>787.00</td></tr><tr><td>Testing</td><td>88.20</td><td>55.66</td><td>75.51</td></tr></table>

Table 13: Performance comparison on Document Merging.  

<table><tr><td>Method</td><td>Score (0-10)</td><td>Cost (Avg num of tokens)</td></tr><tr><td>IO</td><td>6.390</td><td>2292.60</td></tr><tr><td>CoT</td><td>6.524</td><td>3152.90</td></tr><tr><td>ToT</td><td>7.715</td><td>51486.00</td></tr><tr><td>GoT</td><td>7.559</td><td>27685.28</td></tr><tr><td>XoT</td><td>8.168</td><td>15270.80</td></tr></table>

Non-Disclosure Agreement (NDA) document by combining several input documents that partially overlap in content. The aim is to minimize duplication while maximizing information retention. The experimental setting is aligned with in the GoT Besta et al. (2023) paper. We utilized the same dataset provided in their repository.

Remarkably, XoT emerges as the most effective approach, achieving the highest score of 8.168. Notably, XoT maintains a balance in resource efficiency, with an average token cost of 15270.80, surpassing both ToT and GoT. These outcomes underscore XoT's advanced capabilities in handling general textual tasks, extending beyond gaming problems.

# D PROMPT EXAMPLE

Prompts 1-3 display example CoT prompts utilized for Game of 24, 8-Puzzle, and Pocket Cube. These templates are applicable to CoT, ToT, GoT, and our XoT in the final inference process. Each thought step includes the action taken and the resulting new state.

# Instruction: Game of 24

Use numbers and basic arithmetic operations  $(+ - * /)$  to obtain 24.

# Prompt: Game of 24

Input: 291012

Steps:

$12*2 = 24$  (left:91024) Expression:9,10,(12)\* (2)

$10 - 9 = 1$  (left: 24 1) Expression: (12)  $*$  (2), (10) - (9)

$1*24 = 24$  (left: 24) Expression: ((10) - (9))  $\ast$  ((12)  $\ast$  (2))

Answer:  $(12*2)*(10 - 9) = 24$

# Revision: Game of 24

Using the given [input] numbers and basic arithmetic operations  $(+, -, *, /)$ , follow the steps strictly to achieve a result of 24.

All the [input] numbers can reach 24 by basic arithmetic operations  $(+, -, *, /)$ .

If the final answer is not exactly 24, then the corresponding [Steps] is considered [wrong]. Please help me identify the exact wrong step based on its left number, among [Step 1, Step 2, Step 3]. If you are uncertain about which step is wrong, please begin your analysis with [Step 1] for better understanding.

Input: 291012

# Steps:

[Steps 1]  $12 * 2 = 24$  (left: 9 10 24) Expression: 9, 10, (12) * (2)

[Steps 2] 24 - 10 = 14 (left: 9 14) Expression: 9, ((12) * (2)) - (10)

[Steps 3]  $9 + 14 = 23$  (left: 23) Expression:  $(9) + ((12)^{*}(2)) - (10)$

The Steps are wrong. Because it can not reach 24 in the end. To be specific,

23 is not equal to 24. [Steps 2] is wrong. Because it is impossible to reach 24 from the step 2. After Step 2, left numbers are 9, 14.

$9 + 14 = 23$

$9*14 = 126$

9-14=-5

It is impossible to reach 24 from [Steps 2].

# Instruction: 8-Puzzle

You are a virtual expert in solving a 8-puzzle problem. Please follow the instructions and rules below to complete the solving. Your goal is to reach the goal state with valid moves.

[The goal state]

012

345

678

[Instructions]

The 8-puzzle consists of a  $3 \times 3$  grid containing 8 numbered tiles (from 1 to 8) and one empty space (denoted by 0). Only 0 can be moved horizontally or vertically, and the objective is to reach the goal state from a given initial state. The goal state is typically the numbers ordered sequentially, with the 0 in the first position:

[The goal state]

012

345

678

[Rules]

1. Only 0 can be moved horizontally or vertically.

2. Each move is chosen from the following set of options:

- 'Left': move 0 to the left

- 'Down': move 0 downward

- 'Right': move 0 to the right

- 'Up': move 0 upward

For example:

Before move:

123

406

785

After move 'Left':

123

046

785

Before move:

123

406

785

After move 'Down':

123

486

705

Before move:

123

406

785

After move 'Right':

123

460

785

Before move:

123

406

785

After move 'Up':

103

426

785

3. The next move must be chosen from the valid move set depending on the position of '0'.

For example:

p1 p2 p3

p4 p5 p6

p7 p8 p9

(1) If '0' is located at position 'p1', the valid move set is ['Right', 'Down'].  
(2) If '0' is located at position 'p2', the valid move set is ['Left', 'Right', 'Down'].  
(3) If '0' is located at position 'p3', the valid move set is ['Left', 'Down'].  
(4) If '0' is located at position 'p4', the valid move set is ['Right', 'Up', 'Down'].  
(5) If '0' is located at position 'p5', the valid move set is ['Left', 'Right', 'Up', 'Down'].  
(6) If '0' is located at position 'p6', the valid move set is ['Left', 'Up', 'Down'].  
(7) If '0' is located at position 'p7', the valid move set is ['Right', 'Up'].  
(8) If '0' is located at position 'p8', the valid move set is ['Left, 'Right', 'Up'].  
(9) If '0' is located at position 'p9', the valid move set is ['Left', 'Up'].  
4. Diagonal moves are not allowed.  
5. The objective is to return the moves which can reach the goal state.

# Prompt: 8-Puzzle

All given problems can be solved within 1 to 9 steps. The next move must be chosen from the valid move set. The maximum step number you can take is 9. Try to reach the goal state using the least number of steps ( $\leq 9$ ). **DO NOT exceed 9 steps.**

[Initial State]:

312  
645  
780  
[Process]:  
312  
645  
780  
Step 1: Choose one valid move from: [Left, Up]  
Move: Left  
Current State:  
312  
645  
708

Step 2: Choose one valid move from: [Left, Right, Up]

Move: Left

Current State:

312

645

078

Step 3: Choose one valid move from: [Right, Up]

Move: Up

Current State:

312

045

678

Step 4: Choose one valid move from: [Right, Up]

Move: Up

Current State:

012

345

678

Finished.

[Moves]:

Left, Left, Up, Up

# Revision: 8-Puzzle

The given [Process] is not correct since it does not reach the goal state in the end.

If the final answer does not reach the goal state, then the corresponding [Process] is considered [wrong]. Please help me identify the exact wrong step based on its left number, among [Step 1, Step 2, Step 3, ...]. If you are uncertain about which step is wrong, please begin your analysis with [Step 1] for better understanding.

Please help me identify the exact step number that is wrong. You must provide one wrong step.

[Initial State]:

312

645

780

[Process]

312

645

780

Step 1: Choose one valid move from: [Left, Up]

Left

312

645

708

Step 2: Choose one valid move from: [Left, Right, Up]

Left

312

645

078

Step 3: Choose one valid move from: [Right, Up]

Up

312

045

678

Step 4: Choose one valid move from: [Right, Up]

Right

312

405

678

Finished.

The given [Process] is not correct because number 3, 4, 0, 5 are not their goal positions in the end. The puzzle has failed on reaching its goal state.

Now please help me identify the exact step number that is wrong. You must provide one wrong step. If you can not provide an exact step number, please consider that it could be "all steps are wrong".

[Step 4] is wrong, with Move: Right.

# Instruction: Pocket Cube

You are a virtual expert in solving a 2x2 Pocket Cube. Your task is to restore a scrambled 2x2 Rubik's Cube to its original state. All the given problems can be solved in 1 to 4 moves. You cannot exceed more than 11 moves. Provide the sequence of moves required for the restoration. Please follow the instructions and rules below to complete the solving:

1. A 2x2 Pocket Cube has six faces, namely: [Upper, Front, Bottom, Left, Right, Back] Each consisting of a 2x2 grid of squares, with each square having its own color.  
2. Colors in the Cube are represented in numbers: [0, 1, 2, 3, 4, 5]  
3. The Cube's state is represented into a facelets expanding graph, for instance:

Upper:

00

00

Front:

55

22

Down:

33

33

Left:

11

44

Right:

44

11

Back:

22

55

4. A restoration of a Pocket Cube is to move squares in each face to have same numbers. Some example Restored States are:

[Restored State]

Upper:

00

00

Front:

22

22

Down:

33

33

Left:

44

44

Right:

11

11

Back:

55

55

Or

[Restored State]

Upper:

22

22

Front:

00

00

Down:

33

33

Left:

11

11

Right:

44

44

Back:

55

55

You must make move to the Cube to achieve a Restored State, not limited to the above one.

Note that we just need each face to have same numbers, no matter which face has which color.

5. You are only allowed to use following moves [U, U', U2, R, R', R2, F, F', F2].

"[U": Turn the Upper face of the cube 90 degrees clockwise. For instance, after taking move

U:

Upper:

00

00

Front:

22

22

Down:

33

33

Left:

44

44

Right:

11

11

Back:

55

55

will become

Up:

00

00

Front:

11

22

Down:

33

33

Left:

2244

Right:

55

11

Back:

4455

"U": Turn the Upper face of the cube 90 degrees counterclockwise (or anti-clockwise). For instance, after taking move U':

Upper:

00

00

Front:

22

22

Down:

33

33

Left:

44

44

Right:

11

11

Back:

55

55

will become

Upper:

00

00

Front:

44

22

Down:

33

33

Left:

55

44

Right:

22

11

Back:

11

55

"U2": Turn the Upper face of the cube 180 degrees (a half turn). For instance, after taking move U2:

Upper:

00

00

Front:

22

22

Down:

33

33

Left:

44

44

Right:

11

11

Back:

55

55

will become

Up:

00

00

Front:

55

22

Down:

33

33

Left:

11

44

Right:

44

11

Back:

22

55

"R": Turn the Right face of the cube 90 degrees clockwise. For instance, after taking move R:

Upper:

00

00

Front:

22

22

Down:

33

33

Left:

44

44

Right:

11

11

Back:

55

55

will become

Upper:

02

02

Front:

23

23

Down:

35

35

Left:

44

44

Right:

11

11

Back:

05

05

"R": Turn the Right face of the cube 90 degrees counterclockwise. For instance, after taking move R':

Upper:

00

00

Front:

22

22

Down:

33

33

Left:

4.4

4.4

Right:

11

11

Back:

55

55

will become

Upper:

05

05

Front:

20

20

Down:

32

32

Left:

44

44

Right:

11

11

Back:

35

35

"R2": Turn the Right face of the cube 180 degrees. For instance, after taking move R':

Upper:

00

00

Front:

22

22

Down:

3.3

33

Left:

44

4.4

Right:

11

11

Back:

55

55

will become

Up:

03

03

Front:

25

25

Down:

30

30

Left:

44

44

Right:

11

11

Back:

25

25

"F": Turn the Front face of the cube 90 degrees clockwise. For instance, after taking move

F:

Upper:

00

00

Front:

22

22

Down:

33

33

Left:

44

44

Right:

11

11

Back:

55

55

will become

Up:

00

44

Front:

22

22

Down:

11

33

Left:

43

43

Right:

01

01

Back:

55

55

"F": Turn the Front face of the cube 90 degrees counterclockwise. For instance, after taking move F': Upper:

00

00

Front:

22

22

Down:

33

33

Left:

44

44

Right:

11

11

Back:

55

55

will become

Upper:

00

11

Front:

22

22

Down:

44

33

Left:

40

40

Right:

31

31

Back:

55

55

"F2": Turn the Front face of the cube 180 degrees. For instance, after taking move F2:

Upper:

00

00

Front:

22

22

Down:

33  
33  
Left:  
44  
44  
Right:  
11  
11  
Back:  
55  
55  
will become  
Upper:  
00  
33  
Front:  
22  
22  
Down:  
00  
33  
Left:  
41  
41  
Right:  
41  
41  
Back:  
55  
55

# Prompt: Pocket Cube

# [Initial Cube State]:

All the given problems can be solved in 1 to 4 moves. **You cannot exceed more than 11 moves.** Please complete [Process] and return the [Restoration Moves].  
Upper:  
45  
4.4  
Front:  
51  
50  
Down:  
00  
20  
Left:  
11  
32  
Right:  
22  
43  
Back:  
33  
15  
[Process]:  
[Step 1]  
[Move] R

[Current Cube State]

Upper:

40

40

Front:

55

01

Down:

01

22

Left:

11

33

Right:

22

43

Back:

43

55

[Step 2]

[Move] U

[Current Cube State]

Upper:

00

44

Front:

01

01

Down:

22

22

Left:

11

33

Right:

43

43

Back:

55

55

[Step 3]

[Move] F'

[Current Cube State]

Upper:

00

00

Front:

11

11

Down:

22

22

Left:

33

33

Right:

44

44  
Back:  
55  
55  
Finished.  
Now strictly follow the above process to form Restoration Moves.  
[Restoration Moves]:  
R U' F

# Revision: Pocket Cube

The given [Process] is not correct since it does not reach the goal state in the end.

If the final answer does not reach the goal state, then the corresponding [Process] is considered [wrong]. Please help me identify the exact wrong step based on its left number, among [Step 1, Step 2, Step 3, ...]. If you are uncertain about which step is wrong, please begin your analysis with [Step 1] for better understanding.

Please help me identify the exact step number that is wrong. You must provide one wrong step.

[Initial Cube State]:

Upper:

45

44

Front:

51

50

Down:

00

20

Left:

11

32

Right:

22

43

Back:

33

15

[Process]:

[Step 1]

[Move] R

[Current Cube State]

Upper:

40

40

Front:

55

01

Down:

01

22

Left:

11

33

Right:

22

43

Back:

43

55

[Step 2]

[Move] U

[Current Cube State]

Upper:

00

44

Front:

01

01

Down:

22

22

Left:

11

33

Right:

43

43

Back:

55

55

[Step 3]

[Move] F2

[Current Cube State]

Upper:

00

11

Front:

22

22

Down:

4.4

33

Left:

40

40

Right:

31

31

Back:

55

55

Finished.

After finishing all the moves: The Upper face still has 2 differnet colors. The Down face still has 2 differnet colors. The Left face still has 2 differnet colors. The Right face still has 2 differnet colors.

The given [Process] is not correct because not every face has the same numbers in the end. The cube has failed on restoring to its original state. Now please help me identify the exact step number that is wrong. You must provide one wrong step. If you can not provide an exact step number, please consider that it could be "all steps are wrong".

[Step 3] is wrong, with Move: F2.

# Footnotes:

Page 0: *This work was completed during her internship at Microsoft Research Asia. 
Page 1: <sup>1</sup>We named it "Everything of Thoughts" to signify its three comprehensive thought generation capabilities. 
