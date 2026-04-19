# ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding

Zhongxiang Sun

Qipeng Wang

Gaoing School of Artificial Intelligence

Renmin University of China

Beijing, China

{sunzhongxiang,wqp}@ruc.edu.cn

Weijie Yu

School of Information Technology

and Management

University of International Business

and Economics

Beijing, China

yu@uibe.edu.cn

Xiaoxue Zang

Kai Zheng

Kuaishou Technology Co., Ltd.

Beijing, China

xxic666@126.com

zhengkai@kuaishou.com

Jun Xu*

Xiao Zhang

Gaoing School of Artificial Intelligence

Renmin University of China

Beijing, China

{junxu,zhangx89}@ruc.edu.cn

Yang Song

Han Li

Kuaishou Technology Co., Ltd.

Beijing, China

lihan08@kuaishou.com

ys@sonyis.me

# ABSTRACT

Retrieval-Augmented Generation (RAG) systems for Large Language Models (LLMs) have shown promise in knowledge-intensive tasks, yet their reasoning capabilities, particularly for complex multi-step reasoning, remain limited. Although recent approaches have explored integrating RAG with chain-of-thought reasoning or incorporating test-time search with process reward model (PRM), these methods face several untrustworthy challenges, including lack of explanations, bias in PRM training data, early-step bias in PRM scores, and ignoring post-training that fails to fully optimize reasoning potential. To address these issues, we propose Retrieval-Augmented Reasoning through Trustworthy Process Rewarding (ReARTeR), a framework that enhances RAG systems' reasoning capabilities through both post-training and test-time scaling. At test time, ReARTeR introduces Trustworthy Process Rewarding via a Process Reward Model for accurate scalar scoring and a Process Explanation Model (PEM) for generating natural language explanations, enabling step refinement. During post-training, we leverage Monte Carlo Tree Search guided by Trustworthy Process Rewarding to collect high-quality step-level preference data, which is used to optimize the model through Iterative Preference Optimization. ReARTeR tackles three key challenges: (1) misalignment between PRM and PEM, addressed through off-policy preference learning; (2) bias in PRM training data, mitigated by a balanced annotation

method and incorporating stronger annotations for difficult examples; and (3) early-step bias in PRM, resolved via a temporal-difference-based look-ahead search strategy. Experimental results on multi-step reasoning benchmarks demonstrate that ReARTeR significantly improves reasoning performance, highlighting its potential to advance the reasoning capability of RAG systems.

# CCS CONCEPTS

- Information systems  $\rightarrow$  Question answering.

# KEYWORDS

Retrieval Augment Generation, Reasoning, Trustworthy

# ACM Reference Format:

Zhongxiang Sun, Qipeng Wang, Weijie Yu, Xiaoxue Zang, Kai Zheng, Jun Xu, Xiao Zhang, Yang Song, and Han Li. 2025. ReARTeR: Retrieval-Augmented Reasoning with Trustworthy Process Rewarding. In Proceedings of Make sure to enter the correct conference title from your rights confirmation email (Conference acronym 'XX). ACM, New York, NY, USA, 11 pages. https://doi.org/XXXXXXXXX.XXXXXXXXXX

# 1 INTRODUCTION

Retrieval-augmented generation (RAG) for Large Language Models (LLMs) is widely utilized to address knowledge-intensive tasks, typically comprising a generator (LLM) and a retriever (for external knowledge retrieval) [6, 11, 16, 44]. However, complex multi-step reasoning tasks remain challenging even for the most advanced RAG systems. Existing works have explored integrating RAG with chain-of-thought (CoT) reasoning [36, 46], but these approaches have yet to fully leverage the reasoning potential of LLMs.

Recently, Process Reward Models (PRMs) have been introduced to enhance the reasoning capability of RAG systems through test-time scaling [3, 18]. However, these methods often face untrustworthy challenges: (1) Lack of Explanations: Existing PRMs often generate unexplainable scalar scores and cannot incorporate natural language critiques, which limits interpretability and hinders

Figure 1: An example of how RARTPR tackles complex multi-step questions. The right part highlights test-time search with PRM and refinement via PEM explanations, while the left part details the reasoning step, including sub-query, adaptive retrieval, and reasoning thought.

their effectiveness in enhancing refinement during test-time reasoning [21, 39]; (2) Bias in PRM training data: Traditional Monte Carlo methods for collecting Process Supervision Datasets often result in a distributional bias, where most questions receive disproportionately high scores [18, 39, 41]. Consequently, the PRM struggles to identify erroneous steps and fails to provide meaningful feedback on difficult examples; (3) Early-Step Bias in PRM: PRMs exhibit reduced accuracy in predicting rewards for earlier reasoning steps compared to those closer to the reasoning endpoint, due to the increased randomness and uncertainty in earlier steps; (4) Lack of Reasoning Optimization: Additionally, these approaches rely on off-the-shelf LLMs as generators without incorporating reasoning-specific optimization during the post-training phase [22, 43, 48].

To address the above limitations and improve the reasoning capabilities of RAG systems, we explore enhancing Retrieval-Augmented Reasoning through Trustworthy Process Rewarding (ReARTeR) in both test-time and post-training scaling. Specifically, as shown in Figure 1, the testing phase is guided by the Trustworthy Process Rewarding. The Trustworthy Process Rewarding is implemented through two models: (1) a Process Reward Model (PRM), which provides scalar scores that, while accurate, lack interpretability; and (2) a Process Explanation Model (PEM), which generates natural language explanations for the process reward model's scores, facilitating refinement of steps with lower scores. During the posttraining phase, we introduce step-level offline reinforcement finetuning to enhance the reasoning capabilities of the RAG system. Specifically, recognizing the dynamic interaction between the generator and retriever in RAG, on each iteration we employ Monte Carlo Tree Search (MCTS) [2] guided by Trustworthy Process Rewarding to generate high-quality, step-level preference data. This data is subsequently utilized to optimize the model, resulting in a substantial improvement in the system's reasoning performance.

As the core component of ReARTeR, the Trustworthy Process Rewarding solving the following challenges: (1) Misalignment between the PEM and PRM: Off-the-shelf LLMs used as PEM often generate explanations that are not aligned with the PRM's scalar scores, hindering the generator's ability to refine outputs based on

external feedback. To address this issue, we propose aligning the PEM with the PRM through Off-policy Preference Learning, which leverages preference labels derived from PRM scores before and after the generator refines the reasoning step based on PEM explanations. If the explanation improves the PRM score, it is treated as a positive example; otherwise, it is treated as a negative example; (2) Bias in PRM training data: To mitigate this, we leverage OmegaPRM [21], which emphasizes identifying errors in reasoning steps and balances positive and negative examples. For challenging samples, we incorporate annotations from stronger models or human experts to provide accurate reasoning steps, thereby enhancing the PRM's ability to discern correct reasoning paths in difficult scenarios; (3) Early-Step Bias in PRM: To resolve this, we propose a temporal-difference (TD)-based look-ahead search strategy, where simulated future reasoning steps are used to compute expected rewards, enabling updates to the current step's reward estimation. Compared to previous approaches [31], this method effectively achieves a balance between bias and variance.

We summarize the major contributions of this paper as follows:

(1) We pioneer the exploration of combining post-training and test-time scaling to enhance reasoning capabilities in Retrieval-Augmented Generation scenarios. By integrating Trustworthy Process Rewarding, ReARTeR significantly improves the quality of reasoning paths discovered during the post-training phase, as well as the accuracy of search and the generator's refinement ability during the test phase.  
(2) We tackle key challenges in implementing Trustworthy Process Rewarding by aligning the PEM and PRM through off-policy preference learning, balancing the training data of PRM, and employing a TD-based look-ahead search strategy to reduce Early-Step Bias of PRM.  
(3) Experimental results demonstrate that ReARTeR achieves significant improvements on multiple public multi-step reasoning RAG datasets, validating the feasibility of enhancing RAG systems' reasoning capabilities through post-training and inference-time scaling with ReARTeR.

# 2 RELATED WORK

# 2.1 Learning and Search for Reasoning

Advanced reasoning models often follow the learning and search principle [32] to enhance reasoning capabilities through post-training and test-time scaling strategies.

Post-training Scaling. ReFT [22] employs reinforcement finetuning, where LLMs explore reasoning paths and optimize based on feedback, using PPO for training. While PPO achieves better results than DPO due to interactive updates, it suffers from instability. Iterative training methods [30] offer more stability and efficiency, with Iterative Preference Optimization [25] improving reasoning by constructing preference CoT data and using iterative DPO. However, these approaches face challenges in collecting step-level reasoning preferences and rely on difficult-to-collect pairwise data. To address these limitations, we propose using MCTS to collect step-level preference data and employ KTO [5] for stable optimization, leveraging process supervision to enhance reasoning.

Test-time Scaling. Test-time scaling typically relies on (1) Self-Refinement, where models iteratively improve outputs [23], but

this approach is limited by the lack of external feedback [10]; and (2) Search with Verifier, which generates multiple outputs and selects the best using a verifier, such as a Process Reward Model (PRM). While PRM scores have been used as feedback for Self-Refinement [42], they often fail to guide effective improvements in RAG scenarios. To overcome this, we combine PRM-aligned PEM explanations with step-level Self-Refinement for better reasoning performance. PRMs are critical during search, but their training data collection significantly affects performance. Existing methods [3, 18] use Monte Carlo methods to generate process supervision signals, discarding reasoning steps after rollouts, resulting in inefficiency. OmegaPRM [21] improves this by storing rollouts for reuse and using binary search to identify errors, balancing positive and negative examples. Building on OmegaPRM, we incorporate stronger generators for difficult problems and propose a TD-based look-ahead search strategy to enhance PRM accuracy for shallow reasoning nodes, achieving trustworthy process rewards.

# 2.2 Retrieval-Augmented Reasoning

Retrieval-augmented generation for Large Language Models is widely used for knowledge-intensive tasks [6, 11, 16, 44], but remains limited in handling complex multi-step reasoning. Facing this challenge, existing works integrate RAG with CoT reasoning [36, 46]. For instance, Self-Ask [26] uses CoT to explicitly reason through follow-up questions before addressing the query, while IRCoT [36] interleaves retrieval with reasoning steps to iteratively refine reasoning using CoT and retrieved results. Recently, Yue et al. [47] proposed an iterative demonstration-based RAG method that performs multiple iterations to achieve test-time scaling. However, these approaches primarily leverage the long context capabilities of LLMs and directly combine CoT with retrieval without effectively utilizing learning and search to enhance the reasoning capabilities of RAG systems. CR-Planner [18] attempts to directly use Process Reward Models (PRMs) to assist search and improve the reasoning capability of RAG systems through test-time scaling. However, it fails to address the untrustworthy challenges inherent in PRMs. We pioneer the use of trustworthy process rewarding to guide both post-training scaling and test-time scaling, significantly enhancing the multi-step reasoning capabilities of RAG systems.

# 3 METHOD

# 3.1 Overview

In this section, we present the overview of ReARTeR, which enhances Retrieval-Augmented Reasoning through Trustworthy Process Rewarding in both test-time and post-training scaling. The policy model  $\pi_{\theta}$  of ReARTeR includes a generator  $G$ , which can either be an off-the-shelf LLM such as the proprietary model GPT40 [1] or an open-source model such as LLaMA3 [4] which can be post-trained for enhancing reasoning, and a retriever  $E$ . Additionally, ReARTeR incorporates a Process Reward Model (PRM)  $R$  and a Process Explanation Model (PEM)  $C$ .

Given a complex multi-step question  $q$  and a retrieval corpus  $\mathcal{D}$ , ReARTeR generates a reasoning process (CoT)  $\mathbf{e}$  before producing an answer  $a$  to  $q$ . The CoT of ReARTeR consists of a sequence of reasoning steps:

$$
\mathbf {e} = \left[ e _ {1}, e _ {2}, \dots , e _ {T} \right], \tag {1}
$$

where  $T$  represents the maximum length of the reasoning steps.

As illustrated in Figure 2, each reasoning step  $e_t$  comprises a subquery  $q_t$ , a retrieval indicator  $j_t$ , external knowledge  $d_t$  retrieved by  $E$  from the corpus  $\mathcal{D}$  if  $j_t = \text{"Yes"}$ , and a thought  $r_t$  generated by the generator based on the context:

$$
e _ {t} = \left[ q _ {t}, j _ {t}, d _ {t}, r _ {t} \right]. \tag {2}
$$

At timestep  $t$ , the reasoning step  $e_t$  is sampled from the policy  $\pi_{\theta}(s_t)$ , where the state  $s_t$  represents the combination of the question  $q$  and the sequence of reasoning steps up to  $e_{t-1}$ .

For the sampling process of  $e_t$ , we first sample  $M$  different reasoning steps:

$$
\mathcal {E} _ {t} = \left[ e _ {t} ^ {1}, e _ {t} ^ {2}, \dots , e _ {t} ^ {M} \right].
$$

Subsequently, the PRM  $R$  predicts scores to the reasoning steps in  $\mathcal{E}_t$ :

$$
r _ {t} = R \left(s _ {t}, e _ {t}\right), \quad r _ {t} \in (0, 1).
$$

The reasoning step with the highest reward score is selected:

$$
\hat{e}_{t} = \arg \max_{e_{t}^{m}\in \mathcal{E}_{t}}R(s_{t},e_{t}^{m}),
$$

if  $\hat{r}_t > \tau$ , then  $e_t \gets \hat{e}_t$  is directly added to  $s_t$ . Otherwise, a refinement phase is initiated, where the process critic model  $C$  provides an explanation  $c_t$  for the low process reward score of  $\hat{e}_t$ :

$$
c _ {t} = C \left(s _ {t}, \hat {e} _ {t}, \hat {r} _ {t}\right).
$$

The policy model then utilizes external feedback to correct  $\hat{e}_t$ :

$$
e _ {t} = \pi_ {\theta} (s _ {t} | \hat {e} _ {t}, c _ {t}, \hat {r} _ {t}).
$$

Finally,  $e_t$  is added to the reasoning process:

$$
s _ {t + 1} = \left[ s _ {t}, e _ {t} \right].
$$

Ultimately, the policy  $\pi_{\theta}$  generates the final answer  $a$  based on the question  $q$  and the complete reasoning process  $\mathbf{e}$ .

In the following sections, we will introduce the implementation of the PRM (\$ 3.2), including its training process and the method for reducing early-step bias for PRM. Additionally, we describe the training process of the PEM (\$ 3.4) and the post-training scaling strategy for ReARTeR (\$ 3.5).

# 3.2 Process Reward Model Training

The Process Reward Model of ReARTeR is trained to truthfully predict the process reward score of each intermediate step  $e_t$ .

Training data collection: Considering the training data requires process supervision labels which are hard to annotate, to reduce human annotation costs, existing methods propose an automatic annotation approach using the Monte Carlo method to generate process supervision signals [3, 18]. For each step of a CoT e, multiple complete reasoning paths and final answers are obtained via rollouts. By evaluating the accuracy of the final answers, process supervision signals for the current reasoning step can be derived.

However, as shown in Figure 2(a), we observed that this method often introduces distributional bias, where most questions receive disproportionately high scores. Additionally, for difficult questions, the sampled process supervision signals frequently result in a value of zero, leaving the PRM unable to identify erroneous steps or provide meaningful feedback on challenging examples.

To address this issue, as illustrated in Figure 2(a), we first perform  $N$  rollouts for the question  $q$  to obtain  $\{(q_1,\mathbf{e}_1,a_1),\ldots ,(q_N,\mathbf{e}_N,a_N)\}$

Figure 2: Test-Time Scaling of ReARTeR, which includes collecting unbiased PRM training data for PRM, reducing early-step bias in PRM, and alignment between PEM and PRM.

Figure 3: Post-Training Scaling of ReARTeR, which includes Warm-Up and Step-Level Offline Reinforcement Stages.

The accuracy of the final answers across all rollouts is used to compute the Monte Carlo (MC) score:

$$
M C = \frac {\sum_ {n = 1} ^ {N} \operatorname {c o r r e c t} \left(a _ {n}\right)}{N}. \tag {3}
$$

For questions where  $0 < \mathrm{MC} < 1$ , we employ the OmegaPRM [21] annotation scheme, which efficiently identifies the first error in e using binary search and balances positive and negative examples, thereby ensuring both efficiency and quality. For questions where  $\mathrm{MC} = 0$ , we switch to a stronger generator for reasoning. Questions with final  $\mathrm{MC} = 1$  or  $\mathrm{MC} = 0$  (even when using a stronger generator) are discarded, as they lack discriminative value and do not enable the model to identify correct or incorrect reasoning steps for specific questions.

For the selected questions, following the above process, we construct the process supervision data  $\mathcal{D}_{\mathrm{prm}} = \{(s_i,e_i,\mathrm{MC}_i)\}_{i = 1}^{M_r}$ , where  $M_{r}$  represents the number of samples in  $\mathcal{D}_{\mathrm{prm}}$ , and  $\mathrm{MC}_i$  is the MC score computed for  $[s_i,e_i]$  after  $N$  rollouts using Eq. 3.

PRM Training: For each process supervision data  $(s_i, e_i, \mathrm{MC}_i)$  in  $\mathcal{D}_{\mathrm{prm}}$ , we define binary labels  $y_i = 1$  if  $\mathrm{MC}_i > 0.5$ , otherwise  $y_i = 0$ . We utilize the Cross-Entropy (CE) loss to train the PRM:

$$
\mathcal {L} _ {\mathrm {p r m}} = - \frac {1}{M _ {r}} \sum_ {i = 1} ^ {M _ {r}} \left[ y _ {i} \log R \left(s _ {i}, e _ {i}\right) + \left(1 - y _ {i}\right) \log \left(1 - R \left(s _ {i}, e _ {i}\right)\right) \right],
$$

where  $R$  denotes the process reward model.

# 3.3 Reducing Early-Step Bias for PRM

At the inference stage of PRM, we observe that PRMs exhibit reduced accuracy in predicting rewards for earlier reasoning steps

(shallow nodes) compared to those closer to the reasoning end-point (deep nodes), as shown in Figure 2(b). This phenomenon, attributed to the increased randomness and uncertainty in earlier steps, is referred to as early-step bias.

Some existing works adopt a Lookahead Search strategy [31], which performs a simulation by rolling out up to  $H$  steps further, stopping early if the solution end-point is reached. The PRM's score at the end of this rollout is then used to evaluate the current step during beam search. While this approach mitigates bias, it introduces significant variance [33]. To achieve a bias-variance trade-off, inspired by Temporal Difference (TD) learning [34], we propose a TD-based Lookahead Search to update the PRM scores for shallow nodes:

$$
r _ {t} \leftarrow r _ {t} + \alpha \Delta_ {t},
$$

where

$$
\Delta_ {t} = \left(r _ {t + 1} - r _ {t}\right),
$$

$r_t = R(s_t, e_t)$ , and  $\alpha$  is the discount factor.

In our approach, the termination of the Lookahead Search simulation is adaptively determined by whether  $\Delta_t$  falls below a threshold  $\beta$  (indicating diminishing returns in further rollouts when reward scores stabilize) or if the predefined step limit  $H$  is reached. This adaptive simulation mechanism balances computational efficiency and bias reduction, saving resources while maintaining performance.

# 3.4 Process Explanation Model

In this section, we introduce the training procedure of the Process Explanation Model. After training the PRM, the PRM can effectively score the reasoning process; however, the PRM score is an unexplainable scalar and cannot provide natural language critiques. To address this limitation, we designed PEM, a generative model specifically aimed at producing explanations for refinement.

However, directly using off-the-shelf LLMs as PEM often results in explanations misaligned with the PRM's scalar scores, hindering the generator's ability to refine reasoning steps based on external feedback. To address this issue, as shown in Figure 2(c), we propose aligning the PEM with the PRM through Off-policy Preference Learning. This method uses the PRM as a verifier to provide feedback for the PEM-generated explanations, yielding preference data  $\mathcal{D}_{\mathrm{perm}}$ . The PEM is then updated to align its explanations with the PRM's scoring, facilitating the generation of explanations that enhance the policy model's reasoning step through refinement.

Given the state  $s_t$  and reasoning step  $e_t^1$ , the process is as follows:

1. The PRM provides an initial score:

$$
r _ {t} ^ {1} = R \left(s _ {t}, e _ {t} ^ {1}\right),
$$

and the PEM generates an explanation:

$$
c _ {t} = C \left(s _ {t}, e _ {t} ^ {1}, r _ {t} ^ {1}\right).
$$

2. Using external feedback, the policy model  $\pi_{\theta}$  (i.e., RAG) refines the reasoning step:

$$
e _ {t} ^ {2} = \pi_ {\theta} (s _ {t} \mid e _ {t} ^ {1}, c _ {t}, r _ {t} ^ {1}).
$$

3. The PRM re-evaluates the refined step:

$$
r _ {t} ^ {2} = R \left(s _ {t}, e _ {t} ^ {2}\right).
$$

If  $r_t^2 > r_t^1$ , then  $(s_t, e_t^1, c_t)$  is labeled as a positive example with preference label  $p_t = +1$ . Otherwise, it is labeled as a negative example with  $p_t = -1$  (cases where  $r_t^2 = r_t^1$  are discarded). The collected PEM preference training dataset is denoted as:

$$
\mathcal {D} _ {\mathrm {p e m}} = \left\{\left(s _ {t}, e _ {t} ^ {1}, c _ {t}, r _ {t} ^ {1}, r _ {t} ^ {2}, p _ {t}\right) \right\} _ {i = 1} ^ {M _ {e}},
$$

where  $M_e$  is the size of  $\mathcal{D}_{\mathrm{perm}}$ .

During the training phase, since the collected preference data is binary, we employ the KTO Loss [5] to optimize the PEM. This loss is designed for binary preference optimization and is robust to noise in the data. The KTO Loss incorporates a hyperparameter  $\lambda_U > 1$  for negative examples, reflecting loss aversion. In our dataset, the negative examples include the corresponding PRM scores  $r_1$  and  $r_2$ , which can be used to dynamically adjust  $\lambda_U$ , reflecting the degree of loss aversion. Instead of assigning a uniform  $\lambda_U$  for all negative examples as in the original KTO Loss, we introduce a dynamic  $\lambda_U$ :

$$
\lambda_ {U} = \lambda_ {0} \cdot \exp (r _ {1} - r _ {2}),
$$

where  $\lambda_0$  is the base value.

# 3.5 Post-Training Scaling of ReARTeR

In this section, we introduce how ReARTeR enhances the reasoning capabilities of RAG systems through post-training scaling. While test-time scaling can improve the reasoning performance of RAG systems to some extent, for certain weak open-source LLM-based RAG systems  $\pi_{\theta}^{w}$ , their inherent limitations in reasoning capabilities prevent them from solving complex multi-hop questions solely through test-time scaling. Inspired by [37, 43], we propose a step-level offline reinforcement fine-tuning approach to strengthen the reasoning abilities of the RAG system. As illustrated at the bottom of Figure 3 (the retriever is omitted for simplicity), this approach comprises two stages: warm-up stage and step-level offline reinforcement stage.

Warm-Up Stage: In the warm-up stage, we utilize a strong generator-based RAG system  $(\pi_{\theta}^{s})$  to generate a dataset containing reasoning steps:

$$
\mathcal {D} _ {w} = \left\{\left(q _ {i}, e _ {i}, a _ {i}\right) \right\} _ {i = 1} ^ {M _ {w}}
$$

where  $\mathbf{e} = [e_1, e_2, \dots, e_T]$  and  $e_t = [q_t, j_t, d_t, r_t]$ .

During fine-tuning of the weak policy  $\pi_{\theta}^{w}$  using  $\mathcal{D}_w$ , the retriever-generated content  $d_t$  must be masked, as it is not produced by the generator:

$$
\mathcal {L} _ {w} = - \sum_ {i = 1} ^ {M _ {w}} \sum_ {t = 1} ^ {T} \sum_ {k = 1} ^ {| e _ {t} |} \mathbf {1} \left[ o _ {t, k} \notin d _ {t} \right] \log \pi_ {\theta} ^ {w} \left(o _ {t, k} \mid q _ {i}, o _ {<   t}, o _ {t, <   k}\right),
$$

where  $o_{t,k}$  denotes the  $k$ -th token in sequence  $e_t$ ,  $o_{<t}$  represents all tokens generated before step  $t$  (i.e., tokens in  $e_1, \ldots, e_{t-1}$ ), and  $o_{t,<k}$  denotes tokens from the first to the position  $(k-1)$  within  $e_t$ .

Step-Level Offline Reinforcement Stage: In the step-level offline reinforcement stage, the policy model  $\pi_{\theta}^{w}$  iteratively collects step-level preference data using MCTS and leverages this data to improve its reasoning capability via KTO Loss. The updated model is then used to collect new data for further policy updates.

Data Collection: To ensure efficient data collection and balance between positive and negative examples, we adopt the OmegaPRM-based MCTS approach [21] introduced in § 3.2. During rollouts, the generated nodes are scored using the PRM and PCM trained with

the strong generator  $\pi_{\theta}^{s}$ . Reasoning steps with low reward scores are refined to gather higher-quality data.

Iterative Updates: In the first iteration, the policy model is initialized as  $\pi_{\theta,0}^w$ , which is the model fine-tuned during the warm-up stage. Using the collected preference data:

$$
\mathcal {D} _ {0} ^ {u} = \left\{\left(s _ {i}, e _ {i}, M C _ {i}\right) \right\} _ {i = 1} ^ {M _ {u}},
$$

where  $MC_{i} > 0.5$  indicates desirable (positive) reasoning steps and  $MC_{i} \leq 0.5$  indicates undesirable (negative) reasoning steps,  $\mathcal{D}_0^u$  is used to update  $\pi_{\theta,0}^w$  via KTO Loss [5] (with the retrieved document  $d_{i}$  in reasoning steps  $e_{i}$  masked). This yields the updated model  $\pi_{\theta,1}^w$ . In subsequent iterations,  $\pi_{\theta,1}^w$  is used as the policy model to generate preference data  $\mathcal{D}_1^u$ , which is then used to update  $\pi_{\theta,1}^w$  to  $\pi_{\theta,2}^w$ . This process is repeated for  $I$  iterations, progressively improving the reasoning capability of  $\pi_{\theta}^w$ . Compared to the PPO approach used in ReFT [22], our method sacrifices some exploration but achieves more stable updates. Finally,  $\pi_{\theta,I}^w$  represents the policy model after post-training scaling of ReARTeR, which can be combined with test-time scaling to further enhance the reasoning capabilities of RAG systems.

# 4 EXPERIMENTS

In this section, we empirically verify the effectiveness of ReARTeR by addressing the following research questions:

RQ1: How does ReARTeR improve the reasoning capabilities of RAG systems in both closed-source and open-source models?  
RQ2: How do the components of ReARTeR affect test-time scaling?  
RQ3: How does the number of iterations during the post-training process of ReARTeR affect its performance?  
RQ4: How effective is ReARTeR in aligning PEM and PRM?

# 4.1 Experimental Settings

4.1.1 Datasets. In this paper, we focus on leveraging ReARTeR to address complex multi-step question-answering (QA) tasks. To this end, we utilize five benchmark datasets: HotpotQA [45], 2WikiMultiHopQA [8], Musique [35], Bamboogle [27], and StrategyQA [7]. Wikipedia passages serve as the retrieval corpus for all datasets [14]. Following the general experimental setup of RAG [12, 14, 15], we sample 500 examples from the development sets of HotpotQA, 2WikiMultiHopQA, and Musique as test sets. For Bamboogle, which has only 125 examples in its test set, we include all of them as the test set. Since StrategyQA lacks dev or test sets, we sample 500 examples from its training set for testing.

For the training data used in PRM and PCM, and for the posttraining of ReARTeR, we sample 200 examples from the training sets of each dataset. Using the PRM training data construction strategy described in § 3.2, we generate a total of  $M_r = 167$ , 716 training examples. Similarly, using the PCM training data construction strategy described in § 3.4, we generate  $M_e = 769$  training examples. For the post-training phase, the warm-up stage uses  $M_w = 548$  examples, and the preference data collected during each iteration averages  $M_u = 27$ , 822 examples.

4.1.2 Evaluation Metrics. During the evaluation phase, we observed that the outputs of reasoning-optimized RAG systems are typically longer compared to those generated by traditional RAG

systems. Specifically, while the model accurately answers the question, it often includes extensive supplementary information. This renders exact-match metrics such as EM unsuitable for our evaluation tasks. Therefore, we adopt accuracy  $(\mathbf{ACCR}_R)$  as our primary evaluation metric, which determines whether the golden answer is contained within the predicted answer generated by the RAG system. To further refine our evaluation, we employ an LLM-as-Judge approach [17], using GPT4-o [1] as the evaluation model to assess whether the predicted answer is correct. This accuracy metric is referred to as  $\mathbf{ACC}_L$ . The evaluation prompt is as follows:

Given a Question and its Golden Answer, verify whether the Predicted Answer is correct. The prediction is correct if it fully aligns with the meaning and key information of the Golden Answer. Respond with True if the prediction is correct and False otherwise.

Question:

Golden Answer:

Predicted Answer:

During the Process Reward Model Training and Post-Training stages, we use  $\mathrm{ACC}_R$  to determine correctness in Eq. 3, which is more efficient and better suited for collecting large amounts of training data as reward feedback.

4.1.3 Backbone and Baseline Models. To verify the effectiveness of ReARTeR in enhancing the reasoning capabilities of RAG systems, we selected different generators for evaluation. These include the proprietary GPT4o-mini [1] for test-time scaling and the open-source LLaMA3.1-8B [4] (Llama-3.1-8B-Instruct) for both post-training (warm-up from GPT4o-mini) and test-time scaling.

We compared ReARTeR against several baselines: 1. Naive Generation: Directly generating answers using the generator without retrieval. 2. Standard RAG: Traditional retrieval-augmented generation systems. Given that ReARTeR employs multi-path reasoning with CoT processes, which include adaptive retrieval and final answer generation summarized from CoTs, we further compared it with: 3. Branching Methods (Branching): These execute multiple reasoning paths in parallel for a single query, including SuRe [16] and REPLUG [29]. 4. Summarization-based Methods (Summary): LongLLMlingua [13], RECOMP-abstractive [44], and Selective-Context [19]. 5. Adaptive Retrieval Methods (AR): SKR [40] which adaptively retrieve based on generator's knowledge. 6. RAG-CoT Methods (RAG-CoT): These integrate RAG with CoT reasoning, including Self-Ask [26], Iter-RetGen [28], and IRCOT [36]. 7. Test-time Scaling Methods (Test-Time): CR-Planner [18], a recently proposed approach for scaling RAG at test time.

Additionally, we compared ReARTeR with LLaMA3.1-8B as the backbone against recent Open-source Reasoning Models (Reasoning), such as Marco-o1-Qwen-7B [49] and Skywork-o1-Llama3.1-8B [24], which have been extensively optimized for reasoning through large-scale training in general domains and test-time scaling, both integrated into standard RAG configurations.

4.1.4 Implementation Details. The implementation of ReARTeR and the baseline models is based on the open-source RAG framework FlashRAG [14]. The number of samples  $M$  generated per

Table 1: Performance comparisons between ReARTeR and the baselines. The above table shows results with GPT4-o-mini as the generator (Only Test-Time Scaling), while the below table uses LLaMA3.1-8B. The boldface indicates the best performance.  

<table><tr><td rowspan="2">Types</td><td rowspan="2">Models</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">Bamboogle</td><td colspan="2">HotpotQA</td><td colspan="2">Musique</td><td colspan="2">StrategyQA</td></tr><tr><td>ACC_R</td><td>ACC_L</td><td>ACC_R</td><td>ACC_L</td><td>ACC_R</td><td>ACC_L</td><td>ACC_R</td><td>ACC_L</td><td>ACC_R</td><td>ACC_L</td></tr><tr><td rowspan="2">GPT4o-mini</td><td>Naive Generation</td><td>0.348</td><td>0.346</td><td>0.240</td><td>0.280</td><td>0.324</td><td>0.404</td><td>0.134</td><td>0.170</td><td>0.724</td><td>0.724</td></tr><tr><td>Standard RAG</td><td>0.344</td><td>0.292</td><td>0.272</td><td>0.328</td><td>0.342</td><td>0.450</td><td>0.172</td><td>0.188</td><td>0.674</td><td>0.674</td></tr><tr><td rowspan="2">Branching</td><td>SuRe</td><td>0.244</td><td>0.264</td><td>0.168</td><td>0.208</td><td>0.270</td><td>0.380</td><td>0.128</td><td>0.146</td><td>0.550</td><td>0.576</td></tr><tr><td>REPLUG</td><td>0.296</td><td>0.254</td><td>0.224</td><td>0.256</td><td>0.350</td><td>0.428</td><td>0.132</td><td>0.138</td><td>0.654</td><td>0.654</td></tr><tr><td rowspan="3">Summary</td><td>LongLLMLingua</td><td>0.324</td><td>0.316</td><td>0.248</td><td>0.288</td><td>0.358</td><td>0.450</td><td>0.150</td><td>0.172</td><td>0.722</td><td>0.722</td></tr><tr><td>RECOMP-abstractive</td><td>0.298</td><td>0.306</td><td>0.136</td><td>0.176</td><td>0.332</td><td>0.398</td><td>0.118</td><td>0.134</td><td>0.628</td><td>0.628</td></tr><tr><td>Selective-Context</td><td>0.350</td><td>0.290</td><td>0.240</td><td>0.288</td><td>0.366</td><td>0.442</td><td>0.152</td><td>0.172</td><td>0.688</td><td>0.688</td></tr><tr><td rowspan="2">Adaptive</td><td>SKR</td><td>0.364</td><td>0.314</td><td>0.248</td><td>0.288</td><td>0.360</td><td>0.454</td><td>0.162</td><td>0.174</td><td>0.712</td><td>0.712</td></tr><tr><td>Self-Ask</td><td>0.336</td><td>0.478</td><td>0.336</td><td>0.416</td><td>0.392</td><td>0.462</td><td>0.260</td><td>0.270</td><td>0.556</td><td>0.556</td></tr><tr><td rowspan="2">RAG-CoT</td><td>Iter-RetGen</td><td>0.326</td><td>0.270</td><td>0.232</td><td>0.256</td><td>0.374</td><td>0.456</td><td>0.178</td><td>0.188</td><td>0.686</td><td>0.686</td></tr><tr><td>IRCoT</td><td>0.492</td><td>0.114</td><td>0.272</td><td>0.184</td><td>0.434</td><td>0.308</td><td>0.192</td><td>0.214</td><td>0.406</td><td>0.406</td></tr><tr><td>Test-Time</td><td>CR-Planner</td><td>0.520</td><td>0.478</td><td>0.488</td><td>0.524</td><td>0.404</td><td>0.416</td><td>0.272</td><td>0.262</td><td>0.744</td><td>0.744</td></tr><tr><td>Ours</td><td>ReARTeR</td><td>0.554</td><td>0.534</td><td>0.496</td><td>0.544</td><td>0.468</td><td>0.506</td><td>0.296</td><td>0.302</td><td>0.772</td><td>0.772</td></tr><tr><td rowspan="2">LLaMA3.1-8B</td><td>Naive Generation</td><td>0.326</td><td>0.254</td><td>0.144</td><td>0.168</td><td>0.208</td><td>0.268</td><td>0.068</td><td>0.096</td><td>0.672</td><td>0.672</td></tr><tr><td>Standard RAG</td><td>0.336</td><td>0.212</td><td>0.168</td><td>0.216</td><td>0.334</td><td>0.398</td><td>0.104</td><td>0.098</td><td>0.674</td><td>0.674</td></tr><tr><td rowspan="2">Branching</td><td>SuRe</td><td>0.122</td><td>0.262</td><td>0.160</td><td>0.192</td><td>0.266</td><td>0.346</td><td>0.106</td><td>0.144</td><td>0.478</td><td>0.498</td></tr><tr><td>REPLUG</td><td>0.334</td><td>0.204</td><td>0.168</td><td>0.232</td><td>0.290</td><td>0.348</td><td>0.078</td><td>0.090</td><td>0.654</td><td>0.654</td></tr><tr><td rowspan="3">Summary</td><td>LongLLMLingua</td><td>0.304</td><td>0.294</td><td>0.168</td><td>0.216</td><td>0.314</td><td>0.382</td><td>0.088</td><td>0.100</td><td>0.584</td><td>0.584</td></tr><tr><td>RECOMP-abstractive</td><td>0.324</td><td>0.322</td><td>0.104</td><td>0.160</td><td>0.318</td><td>0.380</td><td>0.112</td><td>0.126</td><td>0.628</td><td>0.628</td></tr><tr><td>Selective-Context</td><td>0.266</td><td>0.204</td><td>0.144</td><td>0.200</td><td>0.296</td><td>0.358</td><td>0.092</td><td>0.104</td><td>0.690</td><td>0.690</td></tr><tr><td rowspan="2">Adaptive</td><td>SKR</td><td>0.336</td><td>0.212</td><td>0.176</td><td>0.208</td><td>0.300</td><td>0.372</td><td>0.100</td><td>0.112</td><td>0.662</td><td>0.662</td></tr><tr><td>Self-Ask</td><td>0.306</td><td>0.322</td><td>0.360</td><td>0.432</td><td>0.316</td><td>0.408</td><td>0.222</td><td>0.226</td><td>0.616</td><td>0.616</td></tr><tr><td rowspan="2">RAG-CoT</td><td>Iter-RetGen</td><td>0.310</td><td>0.224</td><td>0.144</td><td>0.176</td><td>0.302</td><td>0.362</td><td>0.084</td><td>0.084</td><td>0.642</td><td>0.642</td></tr><tr><td>IRCoT</td><td>0.338</td><td>0.312</td><td>0.120</td><td>0.104</td><td>0.210</td><td>0.146</td><td>0.060</td><td>0.042</td><td>0.242</td><td>0.242</td></tr><tr><td>Test-Time</td><td>CR-Planer</td><td>0.420</td><td>0.350</td><td>0.304</td><td>0.336</td><td>0.332</td><td>0.350</td><td>0.144</td><td>0.098</td><td>0.664</td><td>0.654</td></tr><tr><td rowspan="2">Reasoning</td><td>Marco-o1</td><td>0.442</td><td>0.184</td><td>0.224</td><td>0.200</td><td>0.352</td><td>0.348</td><td>0.134</td><td>0.104</td><td>0.654</td><td>0.504</td></tr><tr><td>Skywork-o1</td><td>0.344</td><td>0.190</td><td>0.176</td><td>0.160</td><td>0.306</td><td>0.256</td><td>0.092</td><td>0.060</td><td>0.612</td><td>0.326</td></tr><tr><td>Ours</td><td>ReARTeR</td><td>0.470</td><td>0.364</td><td>0.438</td><td>0.484</td><td>0.424</td><td>0.434</td><td>0.244</td><td>0.252</td><td>0.724</td><td>0.724</td></tr></table>

reasoning step for ReARTeR at test-time is set to 3, balancing accuracy and efficiency. The maximum number of reasoning steps  $T$  for Chain-of-Thought (CoT) reasoning is set to 5, where shallow nodes are defined as the first 3 reasoning steps and deep nodes are the remaining steps. The threshold  $\tau$  for initiating the refinement phase is set to 0.5. For the lookahead search, the predefined step limit  $H$  and stopping threshold  $\beta$  are set to 3 and 0.05, respectively. The number  $N$  in PRM training data collection is set to 5. To ensure fairness, we configure the retrieval settings as follows: for iterative retrieval baselines and ReARTeR, the number of external documents retrieved per step is set to Top 1; for single-pass retrieval baselines, the number of retrieved documents is set to 3. The stronger generator used for collecting PRM training data is GPT4-o. The retriever utilized in all experiments is e5-base-v2 [38]. For the PRM, following [18] we fine-tune skywork-reward-llama-3.1-8b-v0.2 [20] with LoRA [9], which is fine-tuned from the general-purpose LLM and excels at scoring in complex scenarios. For the PEM, we fine-tune the Llama-3.2-3B-Instruct [4], which is efficient and effective in generating the explanation for the policy model to refine error reasoning steps. We run all the experiments on machines equipped

with NVIDIA A6000 GPUs and 52-core Intel(R) Xeon(R) Gold 6230R CPUs at  $2.10\mathrm{GHz}$ .

# 4.2 RQ1: Overall Performance

Table 1 presents the experimental results of applying ReARTeR to RAG systems with two different generators: the proprietary GPT4o-mini and the open-source LLaMA3.1-8B, across five multi-step QA datasets. For the RAG system with GPT4o-mini as the generator, where fine-tuning is not feasible, we applied only the Test-Time Scaling component of ReARTeR. Based on the results in Table 1, we observed the following key findings: (1) Compared to baseline models, ReARTeR significantly improves the reasoning capabilities of RAG systems in both closed-source and open-source setups, demonstrating the generalizability of the ReARTeR framework in enhancing RAG systems' reasoning abilities. (2) ReARTeR outperforms Branching methods, indicating that multi-path exploration through Chain-of-Thought (CoT) reasoning is better suited for complex multi-step QA tasks than probability integration in REPLUG or Best-of-K strategies in SuRe. (3) ReARTeR surpasses summarization-based methods, suggesting that conducting CoT reasoning followed

Table 2: Ablation Study of ReARTeR across different generators and datasets.  

<table><tr><td rowspan="2">Model</td><td rowspan="2">Ablation</td><td colspan="2">2WikiMultiHopQA</td><td colspan="2">Bamboogle</td><td colspan="2">HotpotQA</td><td colspan="2">Musique</td></tr><tr><td>ACC_R</td><td>ACC_L</td><td>ACC_R</td><td>ACC_L</td><td>ACC_R</td><td>ACC_L</td><td>ACC_R</td><td>ACC_L</td></tr><tr><td rowspan="6">GPT4o-mini</td><td>w/o Refinement</td><td>0.522</td><td>0.466</td><td>0.474</td><td>0.522</td><td>0.424</td><td>0.456</td><td>0.282</td><td>0.276</td></tr><tr><td>w/o PEM</td><td>0.532</td><td>0.484</td><td>0.486</td><td>0.532</td><td>0.426</td><td>0.462</td><td>0.284</td><td>0.286</td></tr><tr><td>w/o TD-Lookahead</td><td>0.524</td><td>0.490</td><td>0.488</td><td>0.540</td><td>0.458</td><td>0.494</td><td>0.290</td><td>0.294</td></tr><tr><td>w/o Beam Search</td><td>0.526</td><td>0.492</td><td>0.482</td><td>0.522</td><td>0.442</td><td>0.474</td><td>0.278</td><td>0.272</td></tr><tr><td>w/o PRM Data</td><td>0.536</td><td>0.476</td><td>0.474</td><td>0.534</td><td>0.464</td><td>0.504</td><td>0.288</td><td>0.290</td></tr><tr><td>ReARTeR</td><td>0.554</td><td>0.534</td><td>0.496</td><td>0.544</td><td>0.468</td><td>0.506</td><td>0.296</td><td>0.302</td></tr><tr><td rowspan="6">Llama-3.1-8B</td><td>w/o Refinement</td><td>0.444</td><td>0.334</td><td>0.418</td><td>0.440</td><td>0.402</td><td>0.424</td><td>0.230</td><td>0.238</td></tr><tr><td>w/o PEM</td><td>0.450</td><td>0.340</td><td>0.420</td><td>0.446</td><td>0.406</td><td>0.416</td><td>0.234</td><td>0.218</td></tr><tr><td>w/o TD-Lookahead</td><td>0.462</td><td>0.352</td><td>0.428</td><td>0.454</td><td>0.414</td><td>0.438</td><td>0.222</td><td>0.242</td></tr><tr><td>w/o Beam Search</td><td>0.452</td><td>0.346</td><td>0.424</td><td>0.448</td><td>0.416</td><td>0.420</td><td>0.236</td><td>0.246</td></tr><tr><td>w/o PRM Data</td><td>0.466</td><td>0.350</td><td>0.416</td><td>0.458</td><td>0.406</td><td>0.400</td><td>0.238</td><td>0.232</td></tr><tr><td>ReARTeR</td><td>0.470</td><td>0.364</td><td>0.438</td><td>0.484</td><td>0.424</td><td>0.434</td><td>0.244</td><td>0.252</td></tr></table>

by summarization is superior to directly compressing and summarizing external document knowledge for multi-step reasoning tasks. (4) ReARTeR outperforms adaptive retrieval methods, showing that allowing the generator to dynamically decide whether to retrieve in the CoT process can further unlock the model's reasoning potential and improve its ability to answer complex questions. (5) ReARTeR exceeds the performance of RAG-CoT methods, demonstrating that our approach, which leverages Post-Training and Test-Time Scaling, more effectively enhances reasoning capabilities compared to directly combining RAG and CoT reasoning. (6) ReARTeR outperforms CR-Planner, validating that our proposed Trustworthy Process Rewarding mechanism produces superior reasoning paths for RAG systems, thereby improving their ability to handle complex multi-step reasoning problems. (7) ReARTeR surpasses models extensively optimized for reasoning through large-scale training on general domains and test-time scaling, such as Skywork-o1 and Marco-o1. This result indicates that models optimized for general tasks are less effective in RAG-specific reasoning scenarios compared to our framework, further highlighting the effectiveness of ReARTeR in enhancing the reasoning capabilities of RAG systems.

# 4.3 RQ2: Ablation Study of ReARTeR

In this section, we conduct an ablation study to analyze the impact of different components of ReARTeR on the test-time scaling performance of RAG systems. Specifically, we evaluate the following configurations: (1) w/o Refinement: Removing the refinement phase to analyze its effect on the reasoning process of ReARTeR. (2) w/o PEM: Replacing the Process Explanation Model (PEM) with the process reward score directly provided by the PRM during the refinement phase, to evaluate the importance of PEM-generated explanations for refinement. (3) w/o TD-Lookahead: Removing the TD-based lookahead search to validate its role in mitigating early-step bias in the PRM. (4) w/o PRM Data: Training the PRM using data collected with traditional Monte Carlo methods instead of the unbiased data collection strategy proposed in ReARTeR, to



Figure 4: The impact of Post-Training Scaling iterations on ReARTeR using LLaMA-3.1-8B as the generator.


analyze the quality of the PRM trained with our data collection method. (5) w/o Beam Search: Disabling beam search by setting  $M = 1$ , resulting in only a single reasoning path being sampled to generate the CoT.

The experimental results, presented in Table 1, demonstrate that removing any of these components negatively impacts the overall performance of ReARTeR. This highlights the importance of each

component in enhancing the reasoning capabilities of the RAG system. Moreover, these results validate that the unbiased PRM training data collection strategy designed to address the untrustworthy challenges of process reward models enables the training of a more reliable PRM, which provides accurate reward scores for reasoning steps. Additionally, the combination of a more accurate PRM with the TD-based lookahead search enhances the feedback provided during the refinement stage. By leveraging explanations generated by the PEM during the refinement phase, ReARTeR achieves better reasoning step improvements compared to using PRM scores alone.

# 4.4 RQ3: Post-training iterations analysis.

In this section, we analyze the impact of the number of iterations in the Step-Level Offline Reinforcement Stage during the post-training scaling of ReARTeR on the reasoning capabilities of RAG systems. In this experiment, we used LLaMA-3.1-8B as the generator and conducted three iterations of Offline Reinforcement, testing the system on four multi-step reasoning datasets.

The experimental results, presented in Figure 4, demonstrate that the performance of the RAG system on multi-step reasoning datasets improves significantly as the number of Offline Reinforcement iterations increases. Additionally, the results show that our algorithm achieves stable performance improvements across iterations, validating that the proposed Step-Level Offline Reinforcement method provides effective and consistent updates. Due to resource constraints, we did not verify the scalability of our approach on larger datasets or with additional iterations. However, based on the current experimental results, we observe a promising scaling property, suggesting the potential for even greater improvements under resource-abundant conditions.

# 4.5 RQ4: The effective of RARTPR in aligning PEM and PRM.

To evaluate the effectiveness of the alignment strategy for PEM and PRM proposed in RARTPR, we first calculated the improvement rate of process reward scores for reasoning steps with low initial scores after refinement using explanations generated by PEM, both before and after alignment. As shown in Figure 5(a), before aligning PEM with PRM (w/o PEM Align), the improvement rate achieved using explanations from an off-the-shelf LLM-based PEM was only around  $50\%$ . This result indicates that PEM struggles to produce accurate explanations aligned with PRM scores, making it difficult for the RAG system to leverage these explanations to refine reasoning steps and improve PRM scores. In contrast, after aligning PEM and PRM (w PEM Align), we observed a significant increase in the improvement rate, validating the effectiveness of the alignment strategy for enhancing the refinement process and improving reasoning quality. Furthermore, as shown in Figure 5(b), we directly compared the accuracy of RARTPR in solving complex multi-hop queries before and after aligning PEM with PRM. The results demonstrate consistent improvements across multiple datasets after alignment, further confirming the effectiveness of the proposed alignment strategy for PEM and PRM in RARTPR.

We further present a case study of explanations generated by the aligned PEM to demonstrate the effectiveness of PEM under process supervision:

(a) The impact of alignment between PEM and PRM on the improvement rate of process reward scores.

(b) The impact of PEM on ReARTeR's final performance before and after alignment with PRM.  
Figure 5: The impact of aligning PEM and PRM on ReARTeR's overall performance with GPT4-o-mini as generator.

Based on the provided information, I've evaluated the Actor model's current reasoning step and identified the following errors:

# Question Decomposition Errors

- 1.1: The sub-question "Who was the father of John V, Prince of Anhalt-Zerbst?" is a valid sub-question, but the subsequent sub-question "Follow up: Who was John V, Prince Of Anhalt-Zerbst's father?" is not directly related to the original question. This can be improved by rephrasing the sub-question to better align with the original query.

# Retrieval Errors

- 2.1: The retrieved documents contain noise or erroneous information, as the intermediate answer states that the retrieved document does not provide information relevant to the death of John IV, Prince of Anhalt-Zerbst, despite the search result being "Yes".

- 2.2: The retrieved documents are inadequate or irrelevant, as the intermediate answer fails to provide a clear answer to the original question.

# Answer Generation Errors

- 3.1: The answer "John V, Prince of Anhalt-Zerbst died in 1454" is irrelevant to the query, as the original question asks for the death date of John IV, Prince of Anhalt-Zerbst.  
- 3.2: The answer is incomplete, as the original question asks for the death date of John IV, Prince of Anhalt-Zerbst.

# Suggestions for improvement:

1. Refine the sub-question decomposition to ensure that subsequent sub-questions are directly related to the original query. 2. Improve the retrieval process to ensure that relevant and accurate information is retrieved. 3. Verify the accuracy of the answer generation process to provide relevant and correct information. 4. Consider adding additional checks to ensure that the answer is complete and relevant to the original question.

# 5 CONCLUSION

We propose ReARTeR, a framework that enhances the multi-step reasoning capabilities of RAG systems through both post-training and test-time scaling. ReARTeR integrates Trustworthy Process Rewarding, which combines a Process Reward Model for accurate scoring and a Process Explanation Model for explanation-based refinements. During post-training, step-level offline reinforcement fine-tuning with MCTS generates high-quality preference data to optimize the generator. ReARTeR addresses key reasoning challenges, including misalignment between PEM and PRM, bias in PRM training data, and early-step bias in PRM scores, through off-policy preference learning, balanced annotation strategies, and a temporal-difference-based look-ahead search. Experiments on multi-step reasoning benchmarks show that ReARTeR outperforms existing methods, demonstrating its effectiveness in enhancing RAG systems for knowledge-intensive tasks.

# REFERENCES

[1] Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, et al. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774 (2023).  
[2] Cameron B Browne, Edward Powley, Daniel Whitehouse, Simon M Lucas, Peter I Cowling, Philipp Rohlfshagen, Stephen Tavener, Diego Perez, Spyridon Samothrakis, and Simon Colton. 2012. A survey of monte carlo tree search methods. IEEE Transactions on Computational Intelligence and AI in games 4, 1 (2012), 1-43.  
[3] Guanting Dong, Chenghao Zhang, Mengjie Deng, Yutao Zhu, Zhicheng Dou, and Ji-Rong Wen. 2024. Progressive Multimodal Reasoning via Active Retrieval. arXiv preprint arXiv:2412.14835 (2024).  
[4] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al. 2024. The Ilama 3 herd of models. arXiv preprint arXiv:2407.21783 (2024).  
[5] Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, and Douwe Kiela. 2024. Kto: Model alignment as prospect theoretic optimization. arXiv preprint arXiv:2402.01306 (2024).  
[6] Wenqi Fan, Yujuan Ding, Liangbo Ning, Shijie Wang, Hengyun Li, Dawei Yin, Tat-Seng Chua, and Qing Li. 2024. A Survey on RAG Meeting LLMs: Towards Retrieval-Augmented Large Language Models. In Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining (Barcelona, Spain) (KDD '24). Association for Computing Machinery, New York, NY, USA, 6491-6501. https://doi.org/10.1145/3637528.3671470  
[7] Mor Geva, Daniel Khashabi, Elad Segal, Tushar Khot, Dan Roth, and Jonathan Berant. 2021. Did aristole use a laptop? a question answering benchmark with

implicit reasoning strategies. Transactions of the Association for Computational Linguistics 9 (2021), 346-361.  
[8] Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. 2020. Constructing A Multi-hop QA Dataset for Comprehensive Evaluation of Reasoning Steps. In Proceedings of the 28th International Conference on Computational Linguistics. 6609-6625.  
[9] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. [n.d.]. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations.  
[10] Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large Language Models Cannot Self-Correct Reasoning Yet. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=1kmD3fKBPQ  
[11] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong Park. 2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), Kevin Duh, Helena Gomez, and Steven Bethard (Eds.). Association for Computational Linguistics, Mexico City, Mexico, 7036-7050. https://doi.org/10.18653/v1/2024.naacl-long.389  
[12] Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, and Jong C Park. 2024. Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers). 7029-7043.  
[13] Huiqiang Jiang, Qianhui Wu, Xufang Luo, Dongsheng Li, Chin-Yew Lin, Yuqing Yang, and Lili Qiu. 2023. Longllmingua: Accelerating and enhancing llms in long context scenarios via prompt compression. arXiv preprint arXiv:2310.06839 (2023).  
[14] Jiajie Jin, Yutao Zhu, Xinyu Yang, Chenghao Zhang, and Zhicheng Dou. 2024. FlashRAG: A Modular Toolkit for Efficient Retrieval-Augmented Generation Research. arXiv preprint arXiv:2405.13576 (2024).  
[15] Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha, and Jinwoo Shin. [n.d.]. SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs. In The Twelfth International Conference on Learning Representations.  
[16] Jaehyung Kim, Jaehyun Nam, Sangwoo Mo, Jongjin Park, Sang-Woo Lee, Minjoon Seo, Jung-Woo Ha, and Jinwoo Shin. 2024. SuRe: Summarizing Retrievals using Answer Candidates for Open-domain QA of LLMs. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=w4DW6qkRmt  
[17] Haitao Li, Qian Dong, Junjie Chen, Huixue Su, Yujia Zhou, Qingyao Ai, Ziyi Ye, and Yiqun Liu. 2024. LLMs-as-Judges: A Comprehensive Survey on LLM-based Evaluation Methods. arXiv preprint arXiv:2412.05579 (2024).  
[18] Xingxuan Li, Weiwen Xu, Ruochen Zhao, Fangkai Jiao, Shafiq Joty, and Lidong Bing. 2024. Can We Further Elicit Reasoning in LLMs? Critic-Guided Planning with Retrieval-Augmentation for Solving Challenging Tasks. arXiv preprint arXiv:2410.01428 (2024).  
[19] Yucheng Li, Bo Dong, Frank Guerin, and Chenghua Lin. 2023. Compressing Context to Enhance Inference Efficiency of Large Language Models. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, Houda Bouamor, Juan Pino, and Kalika Bali (Eds.). Association for Computational Linguistics, Singapore, 6342-6353. https://doi.org/10.18653/v1/2023.emnlp-main.391  
[20] Chris Yuhao Liu, Liang Zeng, Jiacai Liu, Rui Yan, Jujie He, Chaojie Wang, Shuicheng Yan, Yang Liu, and Yahui Zhou. 2024. Skywork-Reward: Bag of Tricks for Reward Modeling in LLMs. arXiv preprint arXiv:2410.18451 (2024).  
[21] Liangchen Luo, Yinxiao Liu, Rosanne Liu, Samrat Phatale, Harsh Lara, Yunxuan Li, Lei Shu, Yun Zhu, Lei Meng, Jiao Sun, et al. 2024. Improve Mathematical Reasoning in Language Models by Automated Process Supervision. arXiv preprint arXiv:2406.06592 (2024).  
[22] Trung Quoc Luong, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. 2024. Reft: Reasoning with reinforced fine-tuning. arXiv preprint arXiv:2401.08967 (2024).  
[23] Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al. 2024. Self-refine: Iterative refinement with self-feedback. Advances in Neural Information Processing Systems 36 (2024).  
[24] Skywork o1 Team. 2024. Skywork-o1 Open Series. https://huggingface.co/Skywork. https://huggingface.co/Skywork  
[25] Richard Yuanzhe Pang, Weizhe Yuan, Kyunghyun Cho, He He, Sainbayar Sukhbaatar, and Jason Weston. 2024. Iterative reasoning preference optimization. arXiv preprint arXiv:2404.19733 (2024).  
[26] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2022. Measuring and narrowing the compositionality gap in language models. arXiv preprint arXiv:2210.03350 (2022).  
[27] Ofir Press, Muru Zhang, Sewon Min, Ludwig Schmidt, Noah A Smith, and Mike Lewis. 2023. Measuring and Narrowing the Compositionality Gap in Language

Models. In Findings of the Association for Computational Linguistics: EMNLP 2023, 5687-5711.  
[28] Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. 2023. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. arXiv preprint arXiv:2305.15294 (2023).  
[29] Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih. 2023. Replug: Retrieval-augmented black-box language models. arXiv preprint arXiv:2301.12652 (2023).  
[30] Avi Singh, John D Co-Reyes, Rishabh Agarwal, Ankesh Anand, Piyush Patil, Xavier Garcia, Peter J Liu, James Harrison, Jaeoon Lee, Kelvin Xu, et al. 2023. Beyond human data: Scaling self-training for problem-solving with language models. arXiv preprint arXiv:2312.06585 (2023).  
[31] Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. 2024. Scaling lim test-time compute more optimally than scaling model parameters. arXiv preprint arXiv:2408.03314 (2024).  
[32] Richard Sutton. 2019. The Bitter Lesson. http://incompleteideas.net/IncIdeas/BitterLesson.html Incomplete Ideas (blog), 13(1):38.  
[33] Richard S Sutton and Andrew G Barto. 2018. Reinforcement learning: An introduction. MIT press.  
[34] Gerald Tesauro. 1995. Temporal difference learning and TD-Gammon. Commun. ACM 38, 3 (March 1995), 58-68. https://doi.org/10.1145/203330.203343  
[35] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2022. MuSiQue: Multihop Questions via Single-hop Question Composition. Transactions of the Association for Computational Linguistics 10 (2022), 539-554.  
[36] Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. 2023. Interleaving Retrieval with Chain-of-Thought Reasoning for Knowledge-Intensive Multi-Step Questions. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 10014-10037.  
[37] Luong Trung, Xinbo Zhang, Zhanming Jie, Peng Sun, Xiaoran Jin, and Hang Li. 2024. ReFT: Reasoning with Reinforced Fine-Tuning. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), Lun-Wei Ku, Andre Martins, and Vivek Srikumar (Eds.). Association for Computational Linguistics, Bangkok, Thailand, 7601-7614. https://doi.org/10.18653/v1/2024.acl-long.410  
[38] Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. 2022. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533 (2022).  
[39] Peiyi Wang, Lei Li, Zhihong Shao, Runxin Xu, Damai Dai, Yifei Li, Deli Chen, Yu Wu, and Zhifang Sui. 2024. Math-shepherd: Verify and reinforce lms step-by-step

without human annotations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). 9426-9439.  
[40] Yile Wang, Peng Li, Maosong Sun, and Yang Liu. 2023. Self-knowledge guided retrieval augmentation for large language models. arXiv preprint arXiv:2310.05002 (2023).  
[41] Zihan Wang, Yunxuan Li, Yuexin Wu, Liangchen Luo, Le Hou, Hongkun Yu, and Jingbo Shang. 2024. Multi-step problem solving through a verifier: An empirical analysis on model-induced process supervision. arXiv preprint arXiv:2402.02658 (2024).  
[42] Zhenyu Wu, Qingkai Zeng, Zhihan Zhang, Zhaoxuan Tan, Chao Shen, and Meng Jiang. 2024. Enhancing Mathematical Reasoning in LLMs by Stepwise Correction. arXiv preprint arXiv:2410.12934 (2024).  
[43] Yuxi Xie, Anirudh Goyal, Wenyue Zheng, Min-Yen Kan, Timothy P Lillicrap, Kenji Kawaguchi, and Michael Shieh. 2024. Monte Carlo Tree Search Boosts Reasoning via Iterative Preference Learning. arXiv preprint arXiv:2405.00451 (2024).  
[44] Fangyuan Xu, Weijia Shi, and Eunsol Choi. 2024. RECOMP: Improving Retrieval-Augmented LMs with Context Compression and Selective Augmentation. In The Twelfth International Conference on Learning Representations. https://openreview.net/forum?id=mlJLVigNHp  
[45] Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. HotpotQA: A Dataset for Diverse, Explainable Multi-hop Question Answering. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing. 2369-2380.  
[46] Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations (ICLR).  
[47] Zhenrui Yue, Honglei Zhuang, Aijun Bai, Kai Hui, Rolf Jagerman, Hansi Zeng, Zhen Qin, Dong Wang, Xuanhui Wang, and Michael Bendersky. 2024. Inference scaling for long-context retrieval augmented generation. arXiv preprint arXiv:2410.04343 (2024).  
[48] Yuxiang Zhang, Yuqi Yang, Jiangming Shu, Yuhang Wang, Jinlin Xiao, and Jitao Sang. 2024. OpenRFT: Adapting Reasoning Foundation Model for Domain-specific Tasks with Reinforcement Fine-Tuning. arXiv preprint arXiv:2412.16849 (2024).  
[49] Yu Zhao, Huifeng Yin, Bo Zeng, Hao Wang, Tianqi Shi, Chenyang Lyu, Longyue Wang, Weihua Luo, and Kaifu Zhang. 2024. Marco-ol: Towards open reasoning models for open-ended solutions. arXiv preprint arXiv:2411.14405 (2024).

# Footnotes:

Page 0: *Corresponding author. Work partially done at Engineering Research Center of Next-Generation Intelligent Search and Recommendation, Ministry of Education. Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org. Conference acronym 'XX, June 03-05, 2018, Woodstock, NY © 2025 Copyright held by the owner/author(s). Publication rights licensed to ACM. ACM ISBN 978-1-4503-XXXX-X/18/06...$15.00 https://doi.org/XXXXXXXXXXXXXXXXXX 
