# Dynamic Dual-Granularity Skill Bank for Agentic RL

Songjun Tu $^{1,2}$ , Chengdong Xu $^{3,2}$ , Qichao Zhang $^{1}$ , Yaocheng Zhang $^{1}$ , Xiangyuan Lan $^{2}$ , Linjing Li $^{1}$ , Dongbin Zhao $^{1,2}$   
Institute of Automation, Chinese Academy of Sciences $^{1}$   
Pengcheng Laboratory $^{2}$  Sun Yat-Sen University $^{3}$

# Abstract

Agentic reinforcement learning (RL) can benefit substantially from reusable experience, yet existing skill-based methods mainly extract trajectory-level guidance and often lack principled mechanisms for maintaining an evolving skill memory. We propose D2Skill, a dynamic dual-granularity skill bank for agentic RL that organizes reusable experience into task skills for high-level guidance and step skills for fine-grained decision support and error correction. D2Skill jointly trains the policy and skill bank through paired baseline and skill-injected rollouts under the same policy, using their performance gap to derive hindsight utility signals for both skill updating and policy optimization. Built entirely from training-time experience, the skill bank is continuously expanded through reflection and maintained with utility-aware retrieval and pruning. Experiments on ALFWORLD and WEBSHOP with QWEN2.5-7B-INSTRUCT and QWEN3-4B-INSTRUCT-2507 show that D2Skill consistently improves success rates over skill-free baselines by 10–20 points. Further ablations and analyses show that both dual-granularity skill modeling and dynamic skill maintenance are critical to these gains, while the learned skills exhibit higher utility, transfer across evaluation settings, and introduce only modest training overhead.

Project Page: https://github.com/TU2021/D2Skill-AgenticRL.

![](images/2603.28716/b35e36e51b3db7974871801b0db0c7c6d01b18eebeeb36e8cb14a5d10597aec4.jpg)  
(a) D2Skill Framework

![](images/2603.28716/5c3cefa563e3b48795c909be547fc8e1d3697d30722161424f9a13577083e516.jpg)  
(b) Main Results

![](images/2603.28716/31e2baece6a5f18498729d0e1a5e0be725d39581a197f1d3eeaa8dec4cffc516.jpg)  
(c) Training Curves of Success Rate

![](images/2603.28716/1fa479b6c702654329ac4415ba9971b2dadb0a97067d6f44a5ce91912a1b8578.jpg)  
(d) Skill Bank Dynamics of Utility  
Figure 1: Overview of D2Skill. (a) The dynamic dual-granularity skill bank with retrieval, reflection-driven generation, and management. (b) Overall results on ALFWORLD and WEBSHOP. (c) ALFWORLD training curves for the D2Skill skill group, paired baseline group, and GRPO. (d) Skill bank dynamics with and without management, shown by average skill utility and retrieval statistics.

# 1 Introduction

Agentic reinforcement learning (RL) has recently emerged as a promising paradigm for training language-based agents to solve long-horizon decision-making tasks, including interactive environments (Jiang et al., 2026), web search (Zhang et al., 2025c), and research scenarios (Tu et al., 2026). In these settings, the policy interacts with the environment through a textual interface and selects actions based on the task description together with a limited history of past observations and actions. However, such history-based context is generally not a sufficient statistic of the underlying state, resulting in severe partial observability and making credit assignment increasingly difficult as the decision horizon grows (Zhang et al., 2025b). Under sparse rewards and large action spaces, learning each task in isolation is highly inefficient (Feng et al., 2025). Effective policies therefore require mechanisms for accumulating reusable knowledge that can be transferred across tasks.

Recent studies alleviate these challenges by introducing additional supervision signals for agentic RL. Some methods employ outcome-based credit assignment to provide process rewards (Feng et al., 2025), while others derive hindsight supervision from completed trajectories (Yu et al., 2025). More recent work focuses on enabling agents to accumulate experience across tasks and iteratively refine it during training (Zhai et al., 2025; Cai et al., 2025b;a). Within this line, reusable skills have emerged as an effective form of past experience and shown strong empirical gains in agentic RL (Wang et al., 2026). For instance, SkillRL (Xia et al., 2026) builds a skill bank from past trajectories and retrieves relevant skills to guide policy interaction, improving exploration efficiency in long-horizon tasks.

However, existing skill-based and reflection-driven frameworks remain limited in two respects. Most methods derive skills from complete trajectories and emphasize task-level reflection, which captures high-level guidance but is less effective for correcting fine-grained errors at individual interaction steps (Xia et al., 2026; Zhang et al., 2026a). In addition, as training progresses, the skill bank expands continuously, making retrieval and management increasingly challenging. Without principled mechanisms for skill evaluation and pruning, redundant or ineffective skills can degrade retrieved guidance and hinder policy optimization (Zhou et al., 2025; 2026).

To address these limitations, we propose Dynamic Dual-Granularity Skill Bank (D2Skill) for agentic RL, which maintains reusable skills at both the task and step granularities throughout training. As illustrated in Figure 1a, D2Skill distinguishes between task skills for high-level task guidance and step skills for fine-grained decision support and local error correction during interaction. During training, it contrasts skill-injected and non-injected trajectories under the same policy to estimate hindsight skill utility, which in turn guides skill maintenance, retrieval, and policy optimization; the overall training behavior is further reflected in the success-rate curves in Figure 1c. Meanwhile, D2Skill continuously expands and refines the skill bank through reflection while pruning redundant or ineffective skills, keeping the memory compact, informative, and beneficial throughout training, as further illustrated by the skill-bank dynamics in Figure 1d. Experiments on representative agentic tasks show that D2Skill consistently outperforms both non-skill baselines and existing skill-augmented methods (Figure 1b), while effectively maintaining a dynamically updated skill bank with high utility throughout training.

The main contributions of this work are as follows:

1. We present D2Skill, a dynamic dual-granularity skill bank framework for agentic RL, which organizes reusable experience into task skills for high-level guidance and step skills for fine-grained interaction support.  
2. We develop a joint training paradigm where the skill bank and policy co-evolve, with skills expanded through reflection and maintained via utility-guided retrieval and pruning, ensuring an efficient memory throughout training.  
3. We evaluate D2Skill on ALFWORLD and WEBSHOP using QWEN2.5-7B-INSTRUCT and QWEN3-4B-INSTRUCT-2507. D2Skill achieves 10-20 point gains in success rate over skill-free baselines (GRPO), and ablations validate its core components.

# 2 Preliminaries

# 2.1 Agentic RL as a History-Augmented Decision Process

We consider agentic RL in long-horizon environments modeled as a Markov decision process (MDP)  $\mathcal{M} = (\mathcal{S},\mathcal{A},P,r,\gamma)$ , where  $s_t\in S$ ,  $a_{t}\in \mathcal{A}$ , and  $(s_{t + 1},r_t,d_t)\sim P(\cdot \mid s_t,a_t)$ . Unlike classical RL, the policy does not directly observe the environment state. Instead, the agent interacts through a textual interface that provides a partial description of the task and the interaction history.

For a task instance  $g$ , let  $\tau_g$  denote the task specification,  $o_t$  the textual observation at step  $t$ , and  $\mathcal{H}_t^L = \{(o_{t-L}, a_{t-L}), \ldots, (o_{t-1}, a_{t-1})\}$  the most recent  $L$  observation-action pairs retained in the prompt. Let  $\mathcal{A}_t^{\mathrm{adm}} \subseteq \mathcal{A}$  be the admissible action set. The policy acts on the effective context  $x_t = (\tau_g, \mathcal{H}_t^L, o_t, \mathcal{A}_t^{\mathrm{adm}})$ , and selects actions according to  $\pi_\theta(a_t \mid x_t)$ . Although the underlying dynamics are Markovian in  $(s_t, a_t)$ , the context  $x_t$  is a fixed-window summary of past interactions and is generally not a sufficient statistic of the latent state, so the resulting MDP can be viewed as a history-augmented partially observable MDP.

# 2.2 Skill Bank as an External Knowledge Store

In addition to the sliding window  $\mathcal{H}_t^L$ , we maintain a persistent skill bank  $\mathcal{M}$ , where each skill  $m \in \mathcal{M}$  stores language guidance for decision making. At step  $t$ , a retrieval operator selects a small set of relevant skills  $m_t \subseteq \mathcal{M}$  conditioned on the current context  $x_t = (\tau_g, \mathcal{H}_t^L, o_t, \mathcal{A}_t^{\mathrm{adm}})$ , and the policy acts on the augmented context  $\tilde{x}_t = (x_t, m_t)$ .

Taking GRPO (Shao et al., 2024) as the RL algorithm for example. For each task  $g$ , a group of  $N$  trajectories is sampled and advantages are computed by normalizing returns within the group. Under the skill-augmented context, the GRPO objective is

$$
\mathcal {L} _ {\mathrm {G R P O}} (\theta) = \mathbb {E} _ {i} \left[ \min \big (r _ {i} (\theta) \hat {A} _ {i}, \operatorname {c l i p} (r _ {i} (\theta), 1 - \epsilon , 1 + \epsilon) \hat {A} _ {i} \big) - \beta D _ {\mathrm {K L}} \big (\pi_ {\theta} (\cdot | \tilde {x} _ {i}) \| \pi_ {\mathrm {r e f}} (\cdot | \tilde {x} _ {i}) \big) \right].
$$

Here  $\tilde{x}_t^i$  denotes the context augmented with retrieved skills,  $\hat{A}_{i,t}$  denote the normalized advantage and  $r_{i,t}(\theta) = \frac{\pi_{\theta}(a_{i,t}|\tilde{x}_{i,t})}{\pi_{\theta_{\mathrm{old}}}(a_{i,t}|\tilde{x}_{i,t})}$  be the likelihood ratio. The policy is optimized under skill-augmented observations, while the objective remains the same as in standard RL.

# 3 Method

# 3.1 Overall Framework

The framework of D2Skill combines RL with a dynamic skill bank that is continuously updated through reflection and reused to guide policy interaction. As illustrated in Fig. 2, the framework consists of three main components.

RL training with skill injection. (Section 3.2) During training, trajectories are sampled in groups that include both baseline rollouts and skill-injected rollouts under the same policy. Retrieved skills are injected into the policy context to guide decision making, and the performance gap between the two groups is used to construct hindsight signals for policy optimization and skill utility updates under the GRPO objective.

Reflection-driven skill generation. (Section 3.3) When performance on a task group falls below a threshold, a reflection module analyzes representative trajectories to produce new reusable skills. Generated skills are associated with retrieval keys and inserted into the skill bank after normalization and deduplication, allowing the agent to accumulate experience across tasks during training.

Skill retrieval and bank management. (Section 3.3) During interaction, relevant skills are retrieved from the skill bank based on the current task and observation and injected into

![](images/2603.28716/8d7ffd3ad371ea16273ae47a8d93e4313d06994ab35d93ddbdcdd05ce8f91f39.jpg)  
Figure 2: Overall framework of D2Skill. D2Skill couples RL with a dynamic dual-granularity skill bank. For each task, training rollouts are divided into a baseline group and a skill group, whose performance gap yields hindsight signals for policy optimization and skill utility estimation. When performance is poor, reflection on representative failed trajectories produces task skills for high-level guidance and step skills for local error correction. Skills are stored with retrieval keys, reused during subsequent interaction, and periodically pruned by utility-based bank management.

the policy context. Skill utilities are updated online according to rollout outcomes, and the skill bank is periodically pruned using utility-based criteria to maintain a bounded memory while preserving effective skills.

# 3.2 RL Training with Skill Injection and Hindsight Optimization

Rollout with skill injection. For each task  $g$ , we sample a group of  $N$  parallel trajectories, denoted by  $\mathcal{G}_g$ . The group is evenly divided into a skill group  $\mathcal{G}_g^{\mathrm{skill}}$  and a baseline group  $\mathcal{G}_g^{\mathrm{base}}$ , each containing  $N/2$  trajectories. Let  $b_i \in \{0,1\}$  denote the group indicator for trajectory  $i \in \mathcal{G}_g$ , where  $b_i = 1$  indicates  $i \in \mathcal{G}_g^{\mathrm{skill}}$  and  $b_i = 0$  indicates  $i \in \mathcal{G}_g^{\mathrm{base}}$ . Trajectories in the skill group retrieve skills from the skill bank during interaction, while those in the baseline group follow the same policy without skill injection.

Let  $Y_{i} \in \{0,1\}$  denote the terminal success indicator of trajectory  $i$ . For each task  $g$ , the baseline success rate and the skill-group success rate are defined as

$$
\bar {Y} _ {g} ^ {\text {b a s e}} = \frac {1}{| \mathcal {G} _ {g} ^ {\text {b a s e}} |} \sum_ {i \in \mathcal {G} _ {g} ^ {\text {b a s e}}} Y _ {i}, \quad \bar {Y} _ {g} ^ {\text {s k i l l}} = \frac {1}{| \mathcal {G} _ {g} ^ {\text {s k i l l}} |} \sum_ {i \in \mathcal {G} _ {g} ^ {\text {s k i l l}}} Y _ {i}. \tag {1}
$$

Hindsight signals and utility updates. We use the performance gap between the skill group and the baseline group to construct hindsight signals for updating skill utilities. For each task  $g$ , the task-level hindsight signal  $\Delta_{g}^{\text{task}}$  and the trajectory-level credit  $c_{i}$  for step skills retrieved along skill-injected trajectory  $i$  are defined as

$$
\Delta_ {g} ^ {\text {t a s k}} = \bar {Y} _ {g} ^ {\text {s k i l l}} - \bar {Y} _ {g} ^ {\text {b a s e}}, \quad c _ {i} = Y _ {i} - \bar {Y} _ {g} ^ {\text {b a s e}}. \tag {2}
$$

Each skill  $m$  maintains a utility  $u_{m}$  updated using an exponential moving average. For a given task  $g$ , all retrieved task skills share the same signal  $\Delta_{g}^{\text{task}}$ , since the task context is identical for the whole group. In contrast, multiple step skills may be retrieved at different steps and from different trajectories, and each retrieved step skill is updated using the credit of the trajectory in which it appears. The updates are defined as

$$
u _ {m} \leftarrow (1 - \beta_ {\mathrm {t a s k}}) u _ {m} + \beta_ {\mathrm {t a s k}} \Delta_ {g} ^ {\mathrm {t a s k}}, \qquad u _ {m} \leftarrow (1 - \beta_ {\mathrm {s t e p}}) u _ {m} + \beta_ {\mathrm {s t e p}} c _ {i}, \tag {3}
$$

where the first rule is applied to task skills retrieved in task  $g$ , and the second rule is applied to each step skill retrieved along skill-injected trajectory  $i$ .

Hindsight intrinsic reward shaping. To encourage effective use of retrieved skills, we introduce a hindsight intrinsic reward for trajectories in the skill group. For each skill-injected trajectory  $i \in \mathcal{G}_g^{\mathrm{skill}}$ , the hindsight intrinsic reward is defined as

$$
R _ {i} ^ {\text {i n t}} = \lambda \left(Y _ {i} - \bar {Y} _ {g} ^ {\text {b a s e}}\right), \tag {4}
$$

where  $\lambda$  controls the strength of the shaping signal. This term measures performance gain over the baseline and encourages effective skill usage. The hindsight intrinsic reward is applied at the end of each skill-injected trajectory and included in the policy optimization.

Policy optimization with skill-augmented returns. The policy is optimized on the full samples. For each task  $g$ , trajectories in the skill group  $\mathcal{G}_g^{\mathrm{skill}}$  are generated under skill-augmented context and receive an additional reward  $R^{\mathrm{int}}$ . Let  $R_i$  denote the origin return of trajectory  $i$ . For skill-injected trajectories, the return is augmented with  $R_i^{\mathrm{int}}$ , and advantages are computed by group normalization over the whole trajectory group:

$$
\tilde {R} _ {i} = \left\{ \begin{array}{l l} R _ {i} + R _ {i} ^ {\text {i n t}}, & i \in \mathcal {G} _ {g} ^ {\text {s k i l l}}, \\ R _ {i}, & i \in \mathcal {G} _ {g} ^ {\text {b a s e}}, \end{array} \quad A _ {i} = \frac {\tilde {R} _ {i} - \operatorname {m e a n} \left(\{\tilde {R} _ {j} \} _ {j \in \mathcal {G} _ {g}}\right)}{\operatorname {s t d} \left(\{\tilde {R} _ {j} \} _ {j \in \mathcal {G} _ {g}}\right)}. \right. \tag {5}
$$

Taking GRPO as an example, the final policy loss is

$$
\mathcal {L} = \mathbb {E} _ {i \in \mathcal {G} _ {\mathrm {g}}} \left[ \min  \left(r _ {i} A _ {i}, \operatorname {c l i p} \left(r _ {i}, 1 - \epsilon , 1 + \epsilon\right) A _ {i}\right) - \beta D _ {\mathrm {K L}} \right]. \tag {6}
$$

# 3.3 Skill Generation, Retrieval, and Bank Management

Reflection and skill generation. Reflection is triggered only for task groups with low performance, i.e., when  $\bar{Y}_g^{\mathrm{skill}} < \tau_{\mathrm{ref}}$ , where  $\tau_{\mathrm{ref}}$  is a reflection threshold. For each such task  $g$ , we sample one failed trajectory  $\tau_g^{-}$  from the skill group and, when available, one successful trajectory  $\tau_g^{+}$  from either the skill or the baseline group, and use them for skill generation. The reflector produces at most one task skill and one step skill for each task group, formalized as

$$
m _ {g} ^ {\text {t a s k}} = f _ {\text {r e f l e c t}} ^ {\text {t a s k}} \left(g, \tau_ {g} ^ {-}, \tau_ {g} ^ {+}\right), \quad \left(m _ {g} ^ {\text {s t e p}}, o _ {j}\right) = f _ {\text {r e f l e c t}} ^ {\text {s t e p}} \left(g, \tau_ {g} ^ {-}, \tau_ {g} ^ {+}\right), \tag {7}
$$

where  $f_{\mathrm{reflect}}$  denotes an external reflector LLM used for skill generation, and  $o_j$  denotes the observation at the earliest failure step  $j$  identified from the sampled failed trajectory.

For each skill  $m$ , we define a retrieval key  $k_{m}$  that determines when the skill is applicable. For  $m \in \mathcal{M}_{\mathrm{task}}$ , the key is defined as  $k_{m} = g$ . For  $m \in \mathcal{M}_{\mathrm{step}}$ , the key is defined as  $k_{m} = (g, o_{j})$ . New skills are inserted into the skill bank after dedduplication and participate in subsequent retrieval and utility updates.

Two-stage skill retrieval. When interacting with environment, skills are retrieved from the skill bank by matching the current query key with the retrieval key  $k_{m}$  of each skill. For task-level retrieval, the query key is  $q = g$ , while for step-level retrieval the query key is  $q_{t} = (g, o_{t})$ , where  $g$  denotes the task identifier and  $o_{t}$  is the observation at step  $t$ .

In the first stage, we retrieve the top- $m$  candidate skills from the pool  $\mathcal{M} \in \{\mathcal{M}_{\mathrm{task}}, \mathcal{M}_{\mathrm{step}}\}$  according to cosine similarity between the embedding of  $q$  and  $k_{m}$ . A minimum similarity threshold  $\tau_{\mathrm{sim}}$  is applied, and only skills satisfying  $\operatorname{sim}(q, k_{m}) \geq \tau_{\mathrm{sim}}$  are retained.

In the second stage, the candidates are ranked using a combination of semantic similarity and utility-based exploration. For each skill  $m \in \mathcal{M}$ , we define the selection score

$$
\operatorname {s c o r e} (m) = \alpha \widehat {\operatorname {s i m}} (m, q) + (1 - \alpha) \left(u _ {m} + \eta \sqrt {\frac {\log \left(1 + N _ {r}\right)}{1 + n _ {m}}}\right), \tag {8}
$$

where  $\widehat{\operatorname{sim}(m,q)}\in [0,1]$  is the normalized cosine similarity,  $u_{m}$  is the utility of skill  $m$ ,  $n_m$  is the number of times the skill has been retrieved, and  $N_{r} = \sum_{m^{\prime}\in \mathcal{M}}n_{m^{\prime}}$  is the total retrieval count in the active pool. The second term corresponds to a UCB-style bonus that encourages exploration of skills with low retrieval counts. The top- $k$  (<top-m) skills ranked by this score are injected into the policy context.

Skill pruning by utility. To prevent unbounded growth of the skill bank, we periodically prune each skill pool  $\mathcal{M}$  after validation intervals. Each pool is associated with a capacity limit  $N_{\mathrm{max}}$ . If  $|\mathcal{M}| > N_{\mathrm{max}}$ , each skill  $m \in \mathcal{M}$  is assigned an eviction score

$$
\operatorname {e v i c t} (m) = u _ {m} + \eta \sqrt {\frac {\log \left(1 + N _ {r}\right)}{1 + n _ {m}}}. \tag {9}
$$

Then, skills are sorted by  $\operatorname{evict}(m)$  in ascending order, and the lowest-scoring skills are removed until  $|\mathcal{M}| \leq N_{\max}$ . Skills created within the last  $T_{\mathrm{prot}}$  training steps, i.e.,  $t - t_m^{\mathrm{create}} < T_{\mathrm{prot}}$ , are excluded from eviction to ensure sufficient evaluation.

# 4 Experiments

We evaluate D2Skill on two representative LLM agentic benchmarks, ALFWORLD (Shridhar et al., 2020) and WEBSHOP (Yao et al., 2022), and compare it against both skill-free RL baselines and prior memory- or skill-augmented methods. Our experiments are designed to answer three questions:

1. Main Performance: Does D2Skill outperform standard RL and existing baselines on agentic tasks? (Section 4.1)  
2. Ablation: What is the contribution of each major component to the overall gains? (Section 4.2)  
3. Analysis: How does the dynamic skill bank affect skill utility, training dynamics, and computational overhead, and does the learned policy retain generalization ability without the skill bank at evaluation time? (Section 4.3)

# 4.1 Main Performance

Table 1 shows that D2Skill consistently outperforms strong skill-free baselines across both Qwen2.5-7B-Instruct and Qwen3-4B-Instruct-2507 (Yang et al., 2025a), while also surpassing prior memory- and skill-based methods where available. During validation stage, the skill bank  $\mathcal{M}$  is fixed, and the agent only performs retrieval from  $\mathcal{M}_{\mathrm{task}}$  and  $\mathcal{M}_{\mathrm{step}}$  to guide policy, without reflection or skill updates.

Under Qwen2.5-7B-Instruct, D2Skill achieves 90.6 overall success on ALFWORLD, exceeding GRPO by 15.6 points and SkillRL by 1.5 points. On WEBSHOP, the best D2Skill variants reach 91.1 in score and 84.4 in success rate, compared with 86.0 / 72.6 for GRPO and 85.2 / 72.7 for SkillRL. It also substantially outperforms memory-augmented GRPO variants such as Mem0+GRPO (Chhikara et al., 2025) and SimpleMem+GRPO (Liu et al., 2026a). Notably, SkillRL constructs skills from validation trajectories and therefore benefits from stronger privileged information. By contrast, D2Skill acquires and maintains its skill bank using only training-time experience, while still achieving better overall performance under this more restrictive setting. Under the smaller Qwen3-4B-Instruct-2507 base model, D2Skill improves ALFWORLD overall success from 53.9 with GRPO to 69.6 and 72.7, yielding gains of 15.7 and 18.8 points using skills generated by Gemini-3-Flash (G3F) and O3, respectively.

We further evaluate D2Skill on a teacher-initialized policy obtained by collecting 300 successful trajectories per environment with O3 (for ALFWORLD) / Gemini-3-Pro (for WEBSHOP) and performing SFT on Qwen3-4B-Instruct-2507 before RL. Even in this strong setting, D2Skill continues to improve both training efficiency and final performance. After 40 training steps, D2Skill reaches 92.2 on ALFWORLD, nearly matching GRPO trained for

Table 1: Performance on ALFWORLD and WEBSHOP. For ALFWORLD, we report the average success rate (\%) for each subtask and the overall success rate. For WEBSHOP, we report the average score and average success rate (\%). Unless otherwise stated, all methods are trained for 160 training steps in each environment, evaluated every 5 training steps on 128 validation tasks by default, and reported by their best performance over the entire training run. Following SkillRL, we use an SFT-initialized model for Qwen2.5-7B-Instruct to ensure reliable instruction-following for skill usage, while for Qwen3-4B-Instruct-2507 we directly use the original instruct model. For Qwen3-4B-Instruct-2507, performance on WEBSHOP remains close to zero both before and after training, and is therefore omitted from the table. * denotes results replicated from (Feng et al., 2025) and (Xia et al., 2026). The best and second-best results are highlighted in red and blue, respectively.

<table><tr><td rowspan="2">Method</td><td colspan="7">ALFWorld</td><td colspan="2">WebShop</td></tr><tr><td>Pick</td><td>Clean</td><td>Cool</td><td>Look</td><td>Heat</td><td>Pick2</td><td>All</td><td>Score</td><td>Success</td></tr><tr><td colspan="10">Closed-source LLMs</td></tr><tr><td>Gemini-3-Flash</td><td>96.4</td><td>57.1</td><td>96.2</td><td>85.7</td><td>72.2</td><td>95.3</td><td>85.2</td><td>14.1</td><td>16.5</td></tr><tr><td>O3</td><td>64.3</td><td>19.1</td><td>23.1</td><td>64.3</td><td>33.3</td><td>61.9</td><td>43.8</td><td>5.8</td><td>4.7</td></tr><tr><td colspan="10">Base Model: Qwen2.5-7B-Instruct</td></tr><tr><td>Origin</td><td>17.9</td><td>4.8</td><td>3.8</td><td>64.3</td><td>0.0</td><td>5.3</td><td>12.5</td><td>16.6</td><td>3.9</td></tr><tr><td>GRPO</td><td>88.3</td><td>73.3</td><td>76.0</td><td>83.3</td><td>81.3</td><td>40.0</td><td>75.0</td><td>86.0</td><td>72.6</td></tr><tr><td>Mem0+GRPO*</td><td>78.1</td><td>56.1</td><td>65.0</td><td>54.8</td><td>31.0</td><td>26.9</td><td>54.7</td><td>58.1</td><td>37.5</td></tr><tr><td>SimpleMem+GRPO*</td><td>89.5</td><td>60.0</td><td>64.9</td><td>36.3</td><td>50.0</td><td>26.3</td><td>62.5</td><td>67.8</td><td>46.9</td></tr><tr><td>SkillRL(O3)*</td><td>94.3</td><td>90.6</td><td>92.0</td><td>83.3</td><td>93.7</td><td>80.0</td><td>89.1</td><td>85.2</td><td>72.7</td></tr><tr><td>D2Skill(Gemini-3-Flash)</td><td>97.1</td><td>100.0</td><td>75.0</td><td>87.5</td><td>100.0</td><td>78.6</td><td>90.6</td><td>91.1</td><td>80.5</td></tr><tr><td>D2Skill(O3)</td><td>93.8</td><td>94.7</td><td>95.5</td><td>77.8</td><td>95.0</td><td>72.0</td><td>87.8</td><td>90.1</td><td>84.4</td></tr><tr><td colspan="10">Base Model: Qwen3-4B-Instruct-2507</td></tr><tr><td>Origin</td><td>50.0</td><td>9.5</td><td>0.0</td><td>2.1</td><td>11.1</td><td>4.8</td><td>17.2</td><td>-</td><td>-</td></tr><tr><td>GRPO</td><td>73.5</td><td>46.6</td><td>48.0</td><td>61.1</td><td>62.5</td><td>20.0</td><td>53.9</td><td>-</td><td>-</td></tr><tr><td>SkillRL(O3)</td><td>90.0</td><td>92.3</td><td>52.0</td><td>63.6</td><td>42.9</td><td>40.9</td><td>67.2</td><td>-</td><td>-</td></tr><tr><td>D2Skill(Gemini-3-Flash)</td><td>88.6</td><td>75.0</td><td>54.2</td><td>66.7</td><td>60.0</td><td>52.6</td><td>69.6</td><td>-</td><td>-</td></tr><tr><td>D2Skill(O3)</td><td>89.4</td><td>72.4</td><td>66.7</td><td>54.5</td><td>60.0</td><td>50.0</td><td>72.7</td><td>-</td><td>-</td></tr><tr><td colspan="10">Base Model: Qwen3-4B-Instruct-2507 + SFT</td></tr><tr><td>Origin</td><td>53.6</td><td>28.6</td><td>46.2</td><td>71.4</td><td>55.5</td><td>38.1</td><td>47.7</td><td>65.6</td><td>53.1</td></tr><tr><td>GRPO(40-Steps)</td><td>89.7</td><td>77.8</td><td>85.7</td><td>91.6</td><td>86.7</td><td>69.6</td><td>83.6</td><td>77.4</td><td>67.2</td></tr><tr><td>GRPO(120-Steps)</td><td>100.0</td><td>95.2</td><td>80.8</td><td>88.9</td><td>78.6</td><td>88.3</td><td>92.9</td><td>88.2</td><td>79.9</td></tr><tr><td>D2Skill(40-Steps)</td><td>92.9</td><td>100.0</td><td>95.2</td><td>80.0</td><td>90.9</td><td>86.7</td><td>92.2</td><td>84.1</td><td>71.9</td></tr><tr><td>D2Skill(120-Steps)</td><td>97.6</td><td>95.8</td><td>100.0</td><td>88.9</td><td>90.0</td><td>91.7</td><td>95.3</td><td>89.2</td><td>81.3</td></tr></table>

120 steps (92.9), and improves WEBSHOP to 84.1 / 71.9 in score / success rate. After 120 steps, it further reaches 95.3 on ALFWORLD and 89.2 / 81.3 on WEBSHOP, consistently outperforming GRPO under the same budget.

An additional finding is that the closed-source teacher models used in our framework are not necessarily strong standalone agents in these environments. Their direct rollout performance is often substantially below that of the final RL-trained policies. However, when deployed as reflectors to critique trajectories and extract reusable skills, they still yield clear gains in both training efficiency and final performance. This indicates that the utility of these models in D2Skill comes less from direct action generation and more from their ability to perform trajectory-level diagnosis and skill abstraction, which in turn provides effective supervision for policy improvement.

# 4.2 Ablation Study

We conduct ablations on ALFWORLD with QWEN3-4B-INSTRUCT-2507 to assess the contribution of each component in D2Skill. During training, we report the peak success rates of the skill and baseline groups, measured by the maximum 10-step moving average, and during validation we report the best held-out success rate. We consider six ablated variants: (i) w/o task skills, removing task-level skills; (ii) w/o step skills, removing step-level skills; (iii) w/o skill management, disabling skill pruning and retaining all accumulated skills; (iv) w/o

baseline group, removing paired baseline rollouts and training with absolute rewards only; (v) w/o utility retrieval, removing utility-aware ranking and using similarity-only retrieval; (vi) w/o utility module, removing the utility mechanism entirely, including baseline-based utility estimation and updates; and (vii) w/o skills (GRPO) as a skill-free reference.

The ablation results in Table 2 reveal three main findings. First, removing either task skills or step skills consistently reduces performance, indicating that both high-level task guidance and fine-grained step support are important to D2Skill. Second, the larger degradation caused by removing skill management highlights the importance of dynamic bank maintenance in discarding ineffective skills and retaining compact, high-utility knowledge for reuse. Third, removing

Table 2: Ablation Study on ALFWORLD.  

<table><tr><td rowspan="2">Method</td><td colspan="2">Train</td><td rowspan="2">Val</td></tr><tr><td>Skill</td><td>Baseline</td></tr><tr><td>D2Skill</td><td>70.9</td><td>65.8</td><td>72.7</td></tr><tr><td>w/o task skills</td><td>59.1</td><td>53.7</td><td>62.7</td></tr><tr><td>w/o step skills</td><td>57.8</td><td>55.8</td><td>60.2</td></tr><tr><td>w/o skill management</td><td>60.0</td><td>57.4</td><td>57.8</td></tr><tr><td>w/o baseline group</td><td>63.9</td><td>-</td><td>68.8</td></tr><tr><td>w/o utility retrieval</td><td>61.4</td><td>51.8</td><td>64.8</td></tr><tr><td>w/o utility module</td><td>60.3</td><td>-</td><td>62.5</td></tr><tr><td>w/o skills (GRPO)</td><td>-</td><td>50.6</td><td>53.9</td></tr></table>

the baseline group or utility estimation results in smaller but still clear drops, suggesting that these components primarily enhance credit assignment and skill valuation, thereby improving optimization and retrieval quality, rather than driving the main gains directly.

# 4.3 Additional Analysis

Utility and transferability of the skill bank. As shown in Figure 1d, enabling skill management yields a skill bank and retrieved skills with consistently higher average utility, indicating that utility-aware maintenance improves memory and retrieval quality by filtering ineffective skills. Figure 3 further shows that the learned skills are transferable. Even without a skill bank at evaluation time, the policy trained with D2Skill remains competitive with, or outperforms, GRPO, suggesting that part of the gain from skill augmentation

![](images/2603.28716/cdaa65360628a0d2964aa71c9b04ce72e42d8dce810c8e2d01c04b9b93ff88c0.jpg)  
Figure 3: Eval with Different Skills.

has been internalized into the policy during training. Moreover, using the Gemini-3-Flash-generated skill bank from the corresponding training setting at evaluation time still yields clear gains over the no-skill variant in both ALFWORLD and WEBSHOP, while the self-generated skill bank remains the most effective. This suggests that D2Skill learns reusable skills that retain utility beyond the specific skill bank used during training.

Training cost. Table 3 reports the wall-clock training time on ALFWORLD with Qwen3-4B-Instruct-2507, measured on  $8 \times \mathrm{H}100$  GPUs. D2Skill takes 25.6 hours, remaining close to GRPO (20.8 hours) while being substantially cheaper than SkillRL (49.2 hours). As shown in Figure 4, D2Skill also reaches strong evaluation performance much earlier in wall-clock time, making it about  $1.7 \times$  faster than SkillRL in practice. This low overhead mainly comes from an efficient retrieval pipeline: retrieval is executed with batched embedding queries and skill embeddings are updated incrementally, so only newly added skills need to be encoded after each bank update. As a result, D2Skill remains close to GRPO in training cost despite introducing skill retrieval and management. Further implementation details are available in our open-source codebase.

Table 3: Training Cost.  

<table><tr><td>Method</td><td>Training Hours</td></tr><tr><td>GRPO</td><td>20.8h (1.0×)</td></tr><tr><td>SkillRL</td><td>49.2h (2.4×)</td></tr><tr><td>D2Skill</td><td>25.6h (1.2×)</td></tr></table>

![](images/2603.28716/34b73112920f6a700385fb976e279985da692a28a83aa440e191482ffa6d0dcb.jpg)  
Figure 4: Val Success Dynamics.

# 5 Related Works

# 5.1 Agent Evolution and Memory Management

Recent work has increasingly studied agent evolution to address the limited post-training adaptability of LLMs. A central mechanism in this line is external memory, which supports continual adaptation beyond parameter updates (Zhang et al., 2024; Gao et al., 2025; Du, 2026). Existing studies explore evolving long-term memory from multiple perspectives, including retention and forgetting (Chhikara et al., 2025), structured updating and organization (Xu et al., 2025; Yan et al., 2025), retrieval-aware optimization (Zhou et al., 2025), and hierarchical or generative memory construction (Zhang et al., 2025a). Beyond storing interaction history, another line of work abstracts experience into reusable knowledge, such as reasoning strategies (Zhao et al., 2024; Ouyang et al., 2025), reusable workflows (Wang et al., 2024), hierarchical experience libraries (Yang et al., 2025b), and continual experience refinement (Cai et al., 2025b). Overall, these studies suggest that agent evolution increasingly relies on structured, reusable memory with effective retrieval and management.

# 5.2 Memory-augmented Agentic RL

Memory serves as a non-parametric complement to RL by storing useful successful or failed experiences in external memory and retrieving them into the policy context to improve agent performance (Liu et al., 2026b; Zhou et al., 2026). By coupling parametric RL updates with evolving experience repositories, recent methods enable LLM agents to accumulate reusable knowledge beyond model weights, improving both reasoning (Tu et al., 2025; Suzgun et al., 2026) and adaptation on complex tasks (Bai et al., 2026; Li et al., 2026). This is particularly important in agentic settings, where long-horizon decision making under partial observability often benefits more from structured reusable guidance than from raw trajectory storage. Accordingly, recent work increasingly organizes cross-task experience into reusable guidance for planning and action selection (Zhai et al., 2025; Cai et al., 2025a), with skills emerging as an especially effective abstraction for improving policy performance on complex agentic tasks (Wang et al., 2026).

# 5.3 Comparison with Contemporaneous Work

Contemporaneous works such as RetroAgent (Zhang et al., 2026b) and Complementary RL (Muhtar et al., 2026) are related in spirit to our approach, showing that self-evolving experience can substantially improve agentic RL performance. However, their results rely on more elaborate prompting pipelines for retrospection and experience extraction, which may increase system complexity and prompt dependence. SkillRL (Xia et al., 2026) is the most closely related prior work to D2Skill. Although SkillRL also distinguishes between two task types, this mainly reflects task categorization rather than different skill granularities, and its guidance remains task-level: each task retrieves skills once and uses them throughout the trajectory. In contrast, D2Skill maintains both task skills and step skills, enabling high-level guidance and fine-grained support with retrieval at each interaction step. Moreover, D2Skill performs skill generation and management during training, rather than relying on privileged validation information for skill construction.

# 6 Conclusion

We presented D2Skill, a dynamic dual-granularity skill bank framework for agentic RL. By combining task and step skill reuse with reflection-driven expansion, utility-aware retrieval, and pruning, D2Skill enables the policy and skill bank to improve jointly during training. Experiments show that this design consistently outperforms strong baselines, while ablations and analyses confirm the importance of both dual-granularity skill modeling and dynamic skill management, as well as the utility and transferability of the learned skill bank. Our evaluation is currently limited to two representative benchmarks, and D2Skill still relies on an external reflector model. Extending it to broader environments while reducing this dependency is an important direction for future work.

# References

Fei Bai, Zhipeng Chen, Chuan Hao, Ming Yang, Ran Tao, Bryan Dai, Wayne Xin Zhao, Jian Yang, and Hongteng Xu. Towards effective experiential learning: Dual guidance for utilization and internalization. arXiv preprint arXiv:2603.24093, 2026.  
Yuzheng Cai, Siqi Cai, Yuchen Shi, Zihan Xu, Lichao Chen, Yulei Qin, Xiaoyu Tan, Gang Li, Zongyi Li, Haojia Lin, et al. Training-free group relative policy optimization. arXiv preprint arXiv:2510.08191, 2025a.  
Zhicheng Cai, Xinyuan Guo, Yu Pei, Jiangtao Feng, Jinsong Su, Jiangjie Chen, Ya-Qin Zhang, Wei-Ying Ma, Mingxuan Wang, and Hao Zhou. Flex: Continuous agent evolution via forward learning from experience. arXiv preprint arXiv:2511.06449, 2025b.  
Prateek Chhikara, Dev Khant, Saket Aryan, Taranjeet Singh, and Deshraj Yadav. Mem0: Building production-ready ai agents with scalable long-term memory. arXiv preprint arXiv:2504.19413, 2025.  
Pengfei Du. Memory for autonomous llm agents: Mechanisms, evaluation, and emerging frontiers. arXiv preprint arXiv:2603.07670, 2026.  
Lang Feng, Zhenghai Xue, Tingcong Liu, and Bo An. Group-in-group policy optimization for llm agent training. arXiv preprint arXiv:2505.10978, 2025.  
Huan-ang Gao, Jiayi Geng, Wenyue Hua, Mengkang Hu, Xinzhe Juan, Hongzhang Liu, Shilong Liu, Jiahao Qiu, Xuan Qi, Yiran Wu, Hongru Wang, Han Xiao, Yuhang Zhou, Shaokun Zhang, Jiayi Zhang, Jinyu Xiang, Yixiong Fang, Qiwen Zhao, Dongrui Liu, Qihan Ren, Cheng Qian, Zhenhailong Wang, Minda Hu, Huazheng Wang, Qingyun Wu, Heng Ji, and Mengdi Wang. A survey of self-evolving agents: What, when, how, and where to evolve on the path to artificial super intelligence. arXiv preprint arXiv:2507.21046, 2025.  
Zhennan Jiang, Shangqing Zhou, Yutong Jiang, Zefang Huang, Mingjie Wei, Yuhui Chen, Tianxing Zhou, Zhen Guo, Hao Lin, Quanlu Zhang, et al. Wovr: World models as reliable simulators for post-training via policies with rl. arXiv preprint arXiv:2602.13977, 2026.  
Yu Li, Rui Miao, Zhengling Qi, and Tian Lan. Arise: Agent reasoning with intrinsic skill evolution in hierarchical reinforcement learning. arXiv preprint arXiv:2603.16060, 2026.  
Jiaqi Liu, Yaofeng Su, Peng Xia, Siwei Han, Zeyu Zheng, Cihang Xie, Mingyu Ding, and Huaxiu Yao. Simplemem: Efficient lifelong memory for llm agents. arXiv preprint arXiv:2601.02553, 2026a.  
Zeyuan Liu, Jeonghye Kim, Xufang Luo, Dongsheng Li, and Yuqing Yang. Exploratory memory-augmented llm agent via hybrid on-and off-policy optimization. arXiv preprint arXiv:2602.23008, 2026b.  
Dilxat Muhtar, Jiashun Liu, Wei Gao, Weixun Wang, Shaopan Xiong, Ju Huang, Siran Yang, Wenbo Su, Jiamang Wang, Ling Pan, and Bo Zheng. Complementary reinforcement learning. arXiv preprint arXiv:2603.17621, 2026.  
Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. Reasoningbank: Scaling agent self-evolving with reasoning memory. arXiv preprint arXiv:2509.25140, 2025.  
Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Xiao Bi, Haowei Zhang, Mingchuan Zhang, YK Li, Yang Wu, et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.  
Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Côté, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. Alfworld: Aligning text and embodied environments for interactive learning. arXiv preprint arXiv:2010.03768, 2020.

Mirac Suzgun, Mert Yuksekgonul, Federico Bianchi, Dan Jurafsky, and James Zou. Dynamic cheatsheet: Test-time learning with adaptive memory. In Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 7080-7106, 2026.  
Songjun Tu, Jiahao Lin, Xiangyu Tian, Qichao Zhang, Linjing Li, Yuqian Fu, Nan Xu, Wei He, Xiangyuan Lan, Dongmei Jiang, et al. Enhancing llm reasoning with iterative dpo: A comprehensive empirical investigation. In Second Conference on Language Modeling, 2025.  
Songjun Tu, Yiwen Ma, Jiahao Lin, Qichao Zhang, Xiangyuan Lan, Junfeng Li, Nan Xu, Linjing Li, and Dongbin Zhao. Paperaudit-bench: Benchmarking error detection in research papers for critical automated peer review. arXiv preprint arXiv:2601.19916, 2026.  
Yinjie Wang, Xuyang Chen, Xiaolong Jin, Mengdi Wang, and Ling Yang. Openclaw-rl: Train any agent simply by talking. arXiv preprint arXiv:2603.10165, 2026.  
Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent workflow memory. arXiv preprint arXiv:2409.07429, 2024.  
Peng Xia, Jianwen Chen, Hanyang Wang, Jiaqi Liu, Kaide Zeng, Yu Wang, Siwei Han, Yiyang Zhou, Xujiang Zhao, Haifeng Chen, et al. Skillrl: Evolving agents via recursive skill-augmented reinforcement learning. arXiv preprint arXiv:2602.08234, 2026.  
Wujiang Xu, Zujie Liang, Kai Mei, Hang Gao, Juntao Tan, and Yongfeng Zhang. A-mem: Agentic memory for llm agents. arXiv preprint arXiv:2502.12110, 2025.  
Sikuan Yan, Xiufeng Yang, Zuchao Huang, Ercong Nie, Zifeng Ding, Zonggen Li, Xiaowen Ma, Jinhe Bi, Kristian Kersting, Jeff Z. Pan, Hinrich Schuetze, Volker Tresp, and Yunpu Ma. Memory-r1: Enhancing large language model agents to manage and utilize memories via reinforcement learning. arXiv preprint arXiv:2508.19828, 2025.  
An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025a.  
Cheng Yang, Xuemeng Yang, Licheng Wen, Daocheng Fu, Jianbiao Mei, Rong Wu, Pinlong Cai, Yufan Shen, Nianchen Deng, Botian Shi, Yu Qiao, and Haifeng Li. Learning on the job: An experience-driven self-evolving agent for long-horizon tasks. arXiv preprint arXiv:2510.08002, 2025b.  
Shunyu Yao, Howard Chen, John Yang, and Karthik Narasimhan. Webshop: Towards scalable real-world web interaction with grounded language agents. Advances in Neural Information Processing Systems, 35:20744-20757, 2022.  
Hongli Yu, Tinghong Chen, Jiangtao Feng, Jiangjie Chen, Weinan Dai, Qiying Yu, Ya-Qin Zhang, Wei-Ying Ma, Jingjing Liu, Mingxuan Wang, et al. Memagent: Reshaping long-context llm with multi-conv rl-based memory agent. arXiv preprint arXiv:2507.02259, 2025.  
Yunpeng Zhai, Shuchang Tao, Cheng Chen, Anni Zou, Ziqian Chen, Qingxu Fu, Shinji Mai, Li Yu, Jiaji Deng, Zouying Cao, et al. Agentevolver: Towards efficient self-evolving agent system. arXiv preprint arXiv:2511.10395, 2025.  
Guibin Zhang, Muxin Fu, Guancheng Wan, Miao Yu, Kun Wang, and Shuicheng Yan. G-memory: Tracing hierarchical memory for multi-agent systems. arXiv preprint arXiv:2506.07398, 2025a.  
Guibin Zhang, Hejia Geng, Xiaohang Yu, Zhenfei Yin, Zaibin Zhang, Zelin Tan, Heng Zhou, Zhongzhi Li, Xiangyuan Xue, Yijiang Li, et al. The landscape of agentic reinforcement learning for llms: A survey. arXiv preprint arXiv:2509.02547, 2025b.  
Shengtao Zhang, Jiaqian Wang, Ruiwen Zhou, Junwei Liao, Yuchen Feng, Zhuo Li, Yujie Zheng, Weinan Zhang, Ying Wen, Zhiyu Li, et al. Memrl: Self-evolving agents via runtime reinforcement learning on episodic memory. arXiv preprint arXiv:2601.03192, 2026a.

Xiaoying Zhang, Zichen Liu, Yipeng Zhang, Xia Hu, and Wenqi Shao. Retroagent: From solving to evolving via retrospective dual intrinsic feedback. arXiv preprint arXiv:2603.08561, 2026b.  
Yaocheng Zhang, Haohuan Huang, Zijun Song, Yuanheng Zhu, Qichao Zhang, Zijie Zhao, and Dongbin Zhao. Criticsearch: Fine-grained credit assignment for search agents via a retrospective critic. arXiv preprint arXiv:2511.12159, 2025c.  
Zeyu Zhang, Xiaohe Bo, Chen Ma, Rui Li, Xu Chen, Quanyu Dai, Jieming Zhu, Zhenhua Dong, and Ji-Rong Wen. A survey on the memory mechanism of large language model based agents. arXiv preprint arXiv:2404.13501, 2024.  
Andrew Zhao, Daniel Huang, Quentin Lin, Matthieu Lin, Yong-Jin Liu, and Gao Huang.  
Expel: Llm agents are experiential learners. arXiv preprint arXiv:2308.10144, 2024.  
Huichi Zhou, Yihang Chen, Siyuan Guo, Xue Yan, Kin Hei Lee, Zihan Wang, Ka Yiu Lee, Guchun Zhang, Kun Shao, Linyi Yang, et al. Memento: Fine-tuning llm agents without fine-tuning llms. arXiv preprint arXiv:2508.16153, 2025.  
Huichi Zhou, Siyuan Guo, Anjie Liu, Zhongwei Yu, Ziqin Gong, Bowen Zhao, Zhixun Chen, Menglong Zhang, Yihang Chen, Jinsong Li, et al. Memento-skills: Let agents design agents. arXiv preprint arXiv:2603.18743, 2026.