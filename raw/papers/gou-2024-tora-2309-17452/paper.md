
# TORA: A TOOL-INTEGRATED REASONING AGENT

# FOR MATHEMATICAL PROBLEM SOLVING

Zhibin Gou $^{1,2*}$ , Zhihong Shao $^{1,2*}$ , Yeyun Gong $^{2\dagger}$ , Yelong Shen $^{3}$

Yujiu Yang $^{1\dagger}$ , Minlie Huang $^{1\dagger}$ , Nan Duan $^{2}$ , Weizhu Chen $^{3}$

$^{1}$ Tsinghua University  $^{2}$ Microsoft Research  $^{3}$ Microsoft Azure AI

{gzb22,szh19}@mails.tsinghua.edu.cn

{yegong, yeshe, nanduan, wzchen}@microsoft.com

# ABSTRACT

Large language models have made significant progress in various language tasks, yet they still struggle with complex mathematics. In this paper, we propose ToRA, a series of Tool-integrated Reasoning Agents designed to solve challenging mathematical problems by seamlessly integrating natural language reasoning with the utilization of external tools (e.g., computation libraries and symbolic solvers), thereby amalgamating the analytical prowess of language and the computational efficiency of tools. To train ToRA, we curate interactive tool-use trajectories on mathematical datasets, apply imitation learning on the annotations, and propose output space shaping to further refine models' reasoning behavior. As a result, ToRA models significantly outperform open-source models on 10 mathematical reasoning datasets across all scales with  $13\% - 19\%$  absolute improvements on average. Notably, ToRA-7B reaches  $44.6\%$  on the competition-level dataset MATH, surpassing the best open-source model WizardMath-70B by  $22\%$  absolute. ToRACODE-34B is also the first open-source model that achieves an accuracy exceeding  $50\%$  on MATH, which significantly outperforms GPT-4's CoT result, and is competitive with GPT-4 solving problems with programs. Additionally, we conduct a comprehensive analysis of the benefits and remaining challenges of tool interaction for mathematical reasoning, providing valuable insights for future research<sup>1</sup>.

Figure 1: Comparing ToRA with baselines on LLaMA-2 base models from 7B to 70B. ToRA models exhibit remarkable improvements over previous state-of-the-art approaches across all scales. In particular, ToRA-70B notably outperforms GPT-4's CoT result on MATH and attains comparable results to GPT-4 solving problems with code.

Figure 2: Examples of three reasoning formats for mathematical reasoning: (a) Rationale-based methods (e.g., CoT prompting) generate step-by-step natural language rationales, (b) Program-based methods (e.g., PAL prompting) solve tasks with program synthesis, and (c) our proposed Tool-integrated Reasoning format interleaves rationales with program-based tool use. For brevity, we present a simple example of single-round tool interaction, where the model creates rationale  $r_1$  for analysis, writes program  $a_1$  to call an external solver, obtains the execution output  $o_1$ , and then generates rationale  $r_2$  to finalize the answer.

# 1 INTRODUCTION

Large language models (LLMs), such as GPT-4 (OpenAI, 2023) and PaLM-2 (Anil et al., 2023), have demonstrated remarkable progress in a wide range of language tasks, particularly in the longstanding challenge of mathematical reasoning (Feigenbaum et al., 1963; Hosseini et al., 2014). However, open-source models, such as LLaMA-2 (Touvron et al., 2023a;b) and Falcon (Penedo et al., 2023), still struggle with advanced mathematical reasoning tasks.

Existing works improve mathematical performance of language models either with step-by-step natural language reasoning (Wei et al., 2022) as illustrated in Fig 2 (a), or by synthesizing and executing programs to obtain the answers (Gao et al., 2022; Chen et al., 2022), as depicted in Fig 2 (b). Both approaches exhibit complementary advantages. Natural language is suitable for semantic analysis, planning, and abstract reasoning (e.g., commonsense reasoning), but struggles with precise computation, symbolic manipulation, and algorithmic processing. Conversely, programs excel in rigorous operations, and can outsource intricate calculations to specialized tools like equation solvers.

To leverage the benefits of both natural language reasoning and program-based tool use, we train open-source models such as LLaMA-2 to reason in a way where natural language reasoning is interleaved with program-based tool use synergistically (as depicted in Fig 2 (c)), thereby largely reducing the gap with closed-source models like GPT-4 in mathematical reasoning. Specifically, we first design the interleaving format of reasoning, curate corresponding interactive tool-use trajectories for mathematical problems from the popular GSM8k (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021) dataset, and then apply imitation learning on the high-quality annotations, leading to a better performance than any existing open-source model. Furthermore, since the curated data is far from exhausting all valid trajectories for a problem, relying solely on imitation learning restricts a model's output space, hindering the flexibility in exploring plausible trajectories during testing. To improve the diversity of plausible reasoning steps and mitigate improper tool-use behavior, we apply output space shaping which additionally trains the models on both self-sampled valid trajectories and invalid ones that have been corrected by a teacher model (e.g., a 34B model can serve as the teacher

$①$  Imitation Learning  
Figure 3: Training TORA contains two steps.  $①$  Imitation Learning: Prompt LLMs like GPT-4 to generate Tool-integrated Reasoning trajectories (TORA-CORPUS) and use this corpus to fine-tune a model  $\mathcal{M}$ ;  $②$  Output Space Shaping: Sample diverse tool-use trajectories with  $\mathcal{M}$ , keep the valid ones, correct the invalid ones with a teacher model  $\mathcal{M}'$ , and retrain  $\mathcal{M}$  on the union of sampled valid trajectories, corrected ones, and the initial TORA-CORPUS to obtain TORA.

for a 7B model). Output space shaping significantly boosts reasoning, allowing open-source models to attain an accuracy exceeding  $50\%$  on the competition-level MATH dataset for the first time.

We evaluate the resulting suite of Tool-integrated Reasoning Agents (TORA) ranging from 7B to 70B on 10 diverse mathematical reasoning datasets. As shown in Fig 1, TORA series significantly outperform open-source models across all scales. Notably, on the competition-level MATH dataset, TORA-7B outperforms the previous SoTA WizardMath-70B (Luo et al., 2023) by  $22\%$  absolute. TORA-CODE-34B beats GPT-4's CoT result (Bubeck et al., 2023) by  $8.3\%$  absolute ( $50.8\%$  vs.  $42.5\%$ ), and is competitive with GPT-4 solving problems with code (GPT-4-Code,  $51.8\%$ ). In addition, we analyze the benefits and remaining challenges of tool interaction for mathematical reasoning, providing valuable insights for future work.

# 2 TORA: TOOL-INTEGRATED AGENTS FOR MATHEMATICAL REASONING

# 2.1 OVERVIEW

ToRA series solve challenging mathematical problems by leveraging both natural language reasoning and program-based tool use. As shown in Fig 2 (c), given a mathematical problem  $q$ , ToRA reasons with natural language, producing  $r_1$ . When reaching a point where program-based tool use is more appropriate for the subsequent task, e.g., equation solving, ToRA generates a program  $a_1$  for tool use following natural language guidance  $r_1$ . The execution output  $o_1$  will be fed to ToRA for subsequent processing including tool use adjustments, sub-tasks solving, or answer finalization. We repeat the process until the model places its answer within "\boxed{}". The resulting trajectory is denoted as  $\tau = r_1a_1o_1\dots r_{n-1}a_{n-1}o_{n-1}r_n$ , where  $r_n$  contains the answer.

Fig 3 presents the training pipeline of ToRA. We first collect interactive tool-use trajectories on popular mathematical datasets. We then apply imitation learning on the resulting annotations, as well as output space shaping to further refine models' reasoning behavior.

# 2.2 COLLECTING INTERACTIVE TOOL-USE TRAJECTORIES

Existing mathematical reasoning datasets primarily contain annotations in either natural language or code, posing a challenge for training tool-integrated agents due to the absence of interactive tool-use annotations. To address this, we utilize GPT-4 to synthesize high-quality trajectories on the GSM8k and MATH training sets. We select GSM8k and MATH as they exhibit diverse reasoning patterns, spanning multiple domains and difficulty levels.

Algorithm 1 Inference of Tool-Integrated Reasoning  
Require: problem  $q$  , model  $\mathcal{G}$  prompt  $p$  , external tools  $\mathcal{E}$  , stop condition  $Stop(\cdot)$  , maximum iteration rounds  $n$  1:  $\tau_0\gets \text{""}$  Trajectory Initialization   
2: for  $i\gets 1$  to n do   
3:  $r_i\sim \mathbb{P}_{\mathcal{G}}(\cdot |p\oplus q\oplus \tau_{i - 1})$  Rationale Generation (Eq. 1)   
4: if  $Stop(r_i)$  then Stopping Criteria   
5: return  $\tau_{i - 1}\oplus r_i$    
6: end if   
7:  $a_i\sim \mathbb{P}_{\mathcal{G}}(\cdot |p\oplus q\oplus \tau_{i - 1}\oplus r_i)$  Program Generation (Eq. 2)   
8:  $o_i\leftarrow \mathcal{E}(a_i)$  Tool Execution   
9:  $\tau_{i}\gets \tau_{i - 1}\oplus r_{i}\oplus a_{i}\oplus o_{i}$  Trajectory Update (Eq. 3)   
10: end for   
11: return  $\tau_{n}$

Prompt Curation We compose instructions along with diverse few-shot examples, utilizing an interleaved format as depicted in Fig 2 (c). These examples showcase interactive tool usage trajectories, incorporating descriptive variable names and combined program outputs. Please refer to Appendix E for the assembled prompts.

Inference Procedure We follow Algorithm 1 and feed GPT-4  $(\mathcal{G})$  with the composed prompt  $p$  to generate a tool-use trajectory  $\tau$  for each question  $q$  from the training set. The trajectory is initialized as an empty string  $\tau_0$ , for each interaction round  $i$ , we first generate a rationale:

$$
r _ {i} \sim \mathbb {P} _ {\mathcal {G}} (\cdot | p \oplus q \oplus \tau_ {i - 1}) \tag {1}
$$

where  $\oplus$  means concatenation. If  $r_i$  includes an answer within "\\boxed{}" (i.e., the stopping condition  $Stop(r_i)$ ), we cease generation, otherwise the model continues to write a program for tool use:

$$
a _ {i} \sim \mathbb {P} _ {\mathcal {G}} (\cdot | p \oplus q \oplus \tau_ {i - 1} \oplus r _ {i}) \tag {2}
$$

In line with Gou et al. (2023), if the model triggers the code execution stop words like "\''output", we supply it with the corresponding execution message and output  $o_i$  by calling tools with  $o_i \gets \mathcal{E}(a_i)$ , facilitating the generation of subsequent steps. Then, we update the trajectory by concatenating it with the newly generated rationale  $r_i$ , program  $a_i$ , and output  $o_i$ :

$$
\tau_ {i} \leftarrow \tau_ {i - 1} \oplus r _ {i} \oplus a _ {i} \oplus o _ {i} \tag {3}
$$

We repeat the above interaction process until we reach the maximum rounds  $n$ .

Trajectory Sampling We set  $n = 3$  and perform inference using GPT-4 with greedy decoding, retaining trajectories that yield correct answers. For questions where GPT-4 fails with greedy decoding, we apply nucleus sampling with a sample size of 10 and keep up to 4 valid trajectories per question. Ultimately, we successfully annotate trajectories for  $98.2\%$  of GSM8k questions and  $83.1\%$  of MATH questions. After filtering out invalid trajectories with tool-use errors or wrong answers, we obtain 16k annotations which constitute our dataset TORA-CORPUS. Table 1 compares TORA-CORPUS with recently proposed mathematical reasoning datasets, while Table 6 in the Appendix displays MATH annotation accuracy details.

# 2.3 TRAINING

Imitation Learning We apply imitation learning on TORA-CORPUS by minimizing negative log-likelihood loss on the trajectory  $\tau$  conditioned on the problem  $q$ :

$$
\mathcal {M} = \arg \min  _ {\mathcal {M}} \sum_ {q, \tau} \sum_ {i = 1} ^ {n - 1} - \log \mathbb {P} _ {\mathcal {M}} \left(r _ {i + 1} a _ {i + 1} | q, r _ {1} \dots o _ {i}\right) \tag {4}
$$

where  $\mathcal{M}$  is the resulting model. After imitation learning, we can simply apply the same procedure in Algorithm 1 by setting prompt to empty  $p = \text{""}$  for inference. Imitation learning leads to state-of-the-art mathematical reasoning performance despite the small scale of TORA-CORPUS.

Table 1: Compared with mathematical reasoning datasets, TORA-CORPUS uniquely combines natural language rationales with program-based tool usage. Note that TORA-CORPUS only employ questions from the original training set of MATH and GSM8k.  

<table><tr><td>Methods</td><td>#Annotation</td><td>Tool</td><td>Interleaving</td><td>LLM Used</td><td>Source</td></tr><tr><td>RFT (Yuan et al., 2023)</td><td>&gt;100k</td><td>X</td><td>X</td><td>LLaMA-2</td><td>GSM8k</td></tr><tr><td>Open-Platypus Lee et al. (2023)</td><td>25k</td><td>X</td><td>X</td><td>GPT-4</td><td>11 datasets with MATH</td></tr><tr><td>WizardMath (Luo et al., 2023)</td><td>&gt;96k</td><td>X</td><td>X</td><td>ChatGPT</td><td>MATH &amp; GSM8k</td></tr><tr><td>Lila (Mishra et al., 2022)</td><td>134k</td><td>✓(PoT)</td><td>X</td><td>-</td><td>20 datasets with MATH &amp; GSM8k</td></tr><tr><td>MathInstruct (Yue et al., 2023)</td><td>260k</td><td>✓(PoT)</td><td>X</td><td>GPT-4</td><td>14 datasets with MATH &amp; GSM8k</td></tr><tr><td>TORA-CORPUS (ours)</td><td>16k</td><td>✓</td><td>✓</td><td>GPT-4</td><td>MATH &amp; GSM8k</td></tr></table>

Output Space Shaping For each question, TORA-CORPUS mostly demonstrates only one valid interactive tool-use trajectory, which may restrict a model's output space, rendering it inflexible in exploring plausible trajectories during testing. We therefore propose output space shaping in order to encourage the diversity of plausible reasoning steps and reduce improper tool-use behavior.

To explore diverse valid trajectories, we apply nucleus sampling to imitation learning models  $\mathcal{M}$  to sample 64 trajectories per training question  $q$ , following the inference procedure in Section 2.2. We retain valid trajectories with correct answers and no tool-use errors. As many samples are duplicates, to further improve diversity and in an attempt to correct models' improper behavior, we seek to leverage invalid trajectories as well. We observe that trajectories with wrong answers are mostly incorrect halfway (Li et al., 2023), and the preceding reasoning is still plausible; in other words, we can obtain valid trajectories by correcting the subsequent portions. Specifically, a wrong trajectory  $\widetilde{\tau}$ , when written in text, can be represented as a sequence of lines separated by line breaks, i.e.,  $\widetilde{\tau} = l_1\dots l_m$ , where  $m$  is the total number of lines in  $\widetilde{\tau}$ . We enumerate possible preceding portions of wrong trajectories, i.e.,  $\widetilde{\tau}[:j] = l_1\dots l_j$ , and leverage a teacher model  $\mathcal{M}'$  to complete the subsequent steps with greedy decoding:  $\tau \gets \mathbb{P}_{\mathcal{M}'}(\cdot | q \oplus \widetilde{\tau}[:j])$  where we abuse the notation  $\mathbb{P}_{\mathcal{M}'}(\cdot)$  to denote the interactive tool use process following Section 2.2. Finally, corrected trajectories as well as valid trajectory samples will be used for model training, thereby shaping the output space.

In our experiments, we always use CodeLLaMA-34B trained on TORA-CORPUS as the teacher model, and apply sampling with the CodeLLaMA series (ranging from 7B to 34B, with imitation learning). We obtain a total of  $233\mathrm{k}$  distinct valid trajectory samples and  $69\mathrm{k}$  corrected ones. From this combined dataset, we randomly select up to 4 trajectories per GSM8k and MATH problem, merge them with TORA-CORPUS, and then train all TORA models on the resulting  $69\mathrm{k}$  annotations.

# 3 EXPERIMENTS

# 3.1 IMPLEMENTATION DETAILS

We fine-tuned LLaMA-2 (Touvron et al., 2023b) and CodeLLaMA (Rozière et al., 2023) series (ranging from 7B to 70B) using TORA-CORPUS with output space shaping, yielding the TORA and TORA-CODE series respectively. We used a learning rate of 2e-5 by default except that we used 1e-5 for the 34B and 70B models. We set the global batch size to 128 and used a linear scheduler with a  $3\%$  warm-up period for 3 epochs. We trained all models with DeepSpeed ZeRO Stage3 (Rajbhandari et al., 2021) and Flash-Attention 2 (Dao, 2023). We used greedy decoding for all results, with the maximum sequence length set to 2,048 and the maximum number of tool executions set to 3.

# 3.2 EVALUATION SETUP

Datasets We evaluated models on GSM8k (Cobbe et al., 2021) and MATH (Hendrycks et al., 2021), along with 8 out-of-distribution datasets, namely GSM-Hard (Gao et al., 2022), SVAMP (Patel et al., 2021), ASDIV (Miao et al., 2020), TabMWP (Lu et al., 2023), SingleEQ, SingleOP, AddSub, and MultiArith (Koncel-Kedziorski et al., 2016), as illustrated in Table 5 in Appendix. The 10 assorted datasets collectively encompass mathematical problems spanning basic arithmetic to competition level, covering middle and high school curricula and various mathematical domains. The problem formats comprise tabular-based, free-form, and multiple-choice questions, ensuring a thorough assessment of the model's mathematical reasoning aptitude.

Table 2: Results on 10 mathematical reasoning tasks. MAWPS results are averaged over four tasks: Singleeq, Singleop, Addsub, and MultArith. Vanilla models are tested with CoT. The best results in each section are in blue, the second-best results are underlined, while the results of our best model are bolded. * ZS: Zero-shot inference without demonstrations.  

<table><tr><td>Model</td><td>Size</td><td>Tools</td><td>ZS*</td><td>GSM8k</td><td>MATH</td><td>GSM-Hard</td><td>SVAMP</td><td>TabMWP</td><td>ASDiv</td><td>MAWPS</td><td>AVG</td></tr><tr><td>Used for training?</td><td></td><td></td><td></td><td>✓</td><td>✓</td><td>X</td><td>X</td><td>X</td><td>X</td><td>X</td><td></td></tr><tr><td colspan="12">Proprietary Models</td></tr><tr><td>GPT-4</td><td>-</td><td>X</td><td>X</td><td>92.0</td><td>42.5</td><td>64.7</td><td>93.1</td><td>67.1</td><td>91.3</td><td>97.6</td><td>78.3</td></tr><tr><td>GPT-4 (PAL)</td><td>-</td><td>✓</td><td>X</td><td>94.2</td><td>51.8</td><td>77.6</td><td>94.8</td><td>95.9</td><td>92.6</td><td>97.7</td><td>86.4</td></tr><tr><td>ChatGPT</td><td>-</td><td>X</td><td>X</td><td>80.8</td><td>35.5</td><td>55.9</td><td>83.0</td><td>69.1</td><td>87.3</td><td>94.6</td><td>72.3</td></tr><tr><td>ChatGPT (PAL)</td><td>-</td><td>✓</td><td>X</td><td>78.6</td><td>38.7</td><td>67.6</td><td>77.8</td><td>79.9</td><td>81.0</td><td>89.4</td><td>73.3</td></tr><tr><td>Claude-2</td><td>-</td><td>X</td><td>X</td><td>85.2</td><td>32.5</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>PaLM-2</td><td>540B</td><td>X</td><td>X</td><td>80.7</td><td>34.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td colspan="12">Open-Source Models</td></tr><tr><td>LLaMA-2</td><td>7B</td><td>X</td><td>X</td><td>13.3</td><td>4.1</td><td>7.8</td><td>38.0</td><td>31.1</td><td>50.7</td><td>60.9</td><td>29.4</td></tr><tr><td>LLaMA-2 SFT</td><td>7B</td><td>X</td><td>✓</td><td>41.3</td><td>7.2</td><td>16.1</td><td>31.9</td><td>27.8</td><td>47.4</td><td>60.0</td><td>33.1</td></tr><tr><td>LLaMA-2 RFT</td><td>7B</td><td>X</td><td>✓</td><td>51.2</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Platypus-2</td><td>7B</td><td>X</td><td>X</td><td>14.4</td><td>5.4</td><td>8.6</td><td>36.7</td><td>26.5</td><td>47.9</td><td>58.4</td><td>28.3</td></tr><tr><td>WizardMath</td><td>7B</td><td>X</td><td>✓</td><td>54.9</td><td>10.7</td><td>20.6</td><td>57.3</td><td>38.1</td><td>59.1</td><td>73.7</td><td>44.9</td></tr><tr><td>CodeLLaMA (PAL)</td><td>7B</td><td>✓</td><td>X</td><td>34.0</td><td>16.6</td><td>33.6</td><td>59.0</td><td>47.3</td><td>61.4</td><td>79.6</td><td>47.4</td></tr><tr><td>Toolformer†</td><td>7B</td><td>✓</td><td>✓</td><td>-</td><td>-</td><td>-</td><td>29.4</td><td>-</td><td>40.4</td><td>44.0</td><td>-</td></tr><tr><td>ToRA</td><td>7B</td><td>✓</td><td>✓</td><td>68.8</td><td>40.1</td><td>54.6</td><td>68.2</td><td>42.4</td><td>73.9</td><td>88.8</td><td>62.4</td></tr><tr><td>ToRA-CODE</td><td>7B</td><td>✓</td><td>✓</td><td>72.6</td><td>44.6</td><td>56.0</td><td>70.4</td><td>51.6</td><td>78.7</td><td>91.3</td><td>66.5 (+19)</td></tr><tr><td>LLaMA-2</td><td>13B</td><td>X</td><td>X</td><td>24.3</td><td>6.3</td><td>13.6</td><td>43.1</td><td>39.5</td><td>56.3</td><td>70.4</td><td>36.2</td></tr><tr><td>LLaMA-2 SFT</td><td>13B</td><td>X</td><td>✓</td><td>51.1</td><td>9.2</td><td>22.3</td><td>46.3</td><td>35.8</td><td>58.6</td><td>75.0</td><td>42.6</td></tr><tr><td>LLaMA-2 RFT</td><td>13B</td><td>X</td><td>✓</td><td>55.3</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Platypus-2</td><td>13B</td><td>X</td><td>X</td><td>23.7</td><td>7.1</td><td>14.3</td><td>50.7</td><td>45.3</td><td>55.1</td><td>69.6</td><td>38.0</td></tr><tr><td>WizardMath</td><td>13B</td><td>X</td><td>✓</td><td>63.9</td><td>14.0</td><td>28.4</td><td>64.3</td><td>46.7</td><td>65.8</td><td>79.7</td><td>51.8</td></tr><tr><td>CodeLLaMA (PAL)</td><td>13B</td><td>✓</td><td>X</td><td>39.9</td><td>19.9</td><td>39.0</td><td>62.4</td><td>59.5</td><td>65.3</td><td>86.0</td><td>53.1</td></tr><tr><td>ToRA</td><td>13B</td><td>✓</td><td>✓</td><td>72.7</td><td>43.0</td><td>57.3</td><td>72.9</td><td>47.2</td><td>77.2</td><td>91.3</td><td>65.9</td></tr><tr><td>ToRA-CODE</td><td>13B</td><td>✓</td><td>✓</td><td>75.8</td><td>48.1</td><td>60.5</td><td>75.7</td><td>65.4</td><td>81.4</td><td>92.5</td><td>71.3 (+18)</td></tr><tr><td>LLaMA-1 RFT</td><td>34B</td><td>X</td><td>✓</td><td>57.9</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>CodeLLaMA (PAL)</td><td>34B</td><td>✓</td><td>X</td><td>53.3</td><td>23.9</td><td>49.4</td><td>71.0</td><td>63.1</td><td>72.4</td><td>91.5</td><td>60.7</td></tr><tr><td>ToRA-CODE</td><td>34B</td><td>✓</td><td>✓</td><td>80.7</td><td>50.8</td><td>63.7</td><td>80.5</td><td>70.5</td><td>84.2</td><td>93.3</td><td>74.8 (+14)</td></tr><tr><td>LLaMA-2</td><td>70B</td><td>X</td><td>X</td><td>57.8</td><td>14.4</td><td>36.0</td><td>73.6</td><td>57.5</td><td>76.0</td><td>92.4</td><td>58.2</td></tr><tr><td>LLaMA-2 SFT</td><td>70B</td><td>X</td><td>✓</td><td>69.3</td><td>14.9</td><td>39.0</td><td>64.0</td><td>53.0</td><td>71.3</td><td>84.8</td><td>56.6</td></tr><tr><td>LLaMA-2 RFT</td><td>70B</td><td>X</td><td>✓</td><td>64.8</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td><td>-</td></tr><tr><td>Platypus-2</td><td>70B</td><td>X</td><td>X</td><td>45.9</td><td>15.0</td><td>24.6</td><td>74.3</td><td>47.3</td><td>72.7</td><td>91.1</td><td>53.0</td></tr><tr><td>WizardMath</td><td>70B</td><td>X</td><td>✓</td><td>81.6</td><td>22.7</td><td>50.3</td><td>80.0</td><td>49.8</td><td>76.2</td><td>86.2</td><td>63.8</td></tr><tr><td>LLaMA-2 (PAL)</td><td>70B</td><td>✓</td><td>X</td><td>55.2</td><td>18.3</td><td>50.0</td><td>74.6</td><td>59.5</td><td>71.9</td><td>92.8</td><td>60.3</td></tr><tr><td>ToRA</td><td>70B</td><td>✓</td><td>✓</td><td>84.3</td><td>49.7</td><td>67.2</td><td>82.7</td><td>74.0</td><td>86.8</td><td>93.8</td><td>76.9 (+13)</td></tr></table>

Metrics We report accuracies of predicted answers. Following Lightman et al. (2023), we round numerical values and use sympy<sup>2</sup> for parsing expressions. Since the SingleEQ, SingleOP, AddSub, and MultiArith datasets focus on different aspects of basic arithmetic, we report their average results under the collective term MAwPS (Koncel-Kedziorski et al., 2016) for all methods.

# 3.3 BASELINES

Proprietary Models We present results from an array of SoTA LLMs, such as OpenAI's GPT-4, ChatGPT (gpt-3.5-turbo), Google's PaLM-2, and Anthropic's Claude-2. By default, we report CoT prompting results, and include PAL (Gao et al., 2022) prompting results for selected models.

Open-Source Models Base models comprise LLaMA-2 and CodeLLaMA with CoT and PAL prompting. Supervised Fine-Tuning (SFT) employs CoT rationales from the original GSM8k and MATH dataset (15k samples) for fine-tuning. Rejection sampling Fine-Tuning (RFT) leverages multiple models to generate diverse reasoning paths for fine-tuning (Yuan et al., 2023). WizardMath augments data using ChatGPT, and conducts SFT and RLHF. Platypus-2, the top model on the LLM Leaderboard  $^{3}$ , is fine-tuned with Open-Platypus reasoning datasets (Lee et al., 2023). We also compare TORA with Toolformer (Schick et al., 2023) which is a model trained to utilize calculators.

Table 3: Results on MATH subtopics.  

<table><tr><td>Model</td><td>Size</td><td>Tool</td><td>Intermediate Algebra</td><td>Precalculus</td><td>Geometry</td><td>Number Theory</td><td>Counting &amp; Probability</td><td>Prealgebra</td><td>Algebra</td><td>Overall</td></tr><tr><td colspan="11">Proprietary Models</td></tr><tr><td>ChatGPT (PAL)</td><td>-</td><td>✓</td><td>18.5</td><td>19.2</td><td>23.2</td><td>48.5</td><td>43.0</td><td>62.7</td><td>45.4</td><td>38.7</td></tr><tr><td>GPT-4 (PAL)</td><td>-</td><td>✓</td><td>32.8</td><td>29.3</td><td>38.0</td><td>58.7</td><td>61.0</td><td>73.9</td><td>59.1</td><td>51.8</td></tr><tr><td colspan="11">Open-Source Models</td></tr><tr><td>WizardMath</td><td>7B</td><td>X</td><td>6.2</td><td>6.0</td><td>6.5</td><td>7.6</td><td>9.5</td><td>18.1</td><td>16.3</td><td>11.2</td></tr><tr><td>ToRA-CODE</td><td>7B</td><td>✓</td><td>35.1 (+28.9)</td><td>31.0 (+25.0)</td><td>24.0 (+17.5)</td><td>50.7 (+43.1)</td><td>30.6 (+21.1)</td><td>55.0 (+36.9)</td><td>61.7 (+45.4)</td><td>44.6 (+33.4)</td></tr><tr><td>w/o Shaping</td><td>7B</td><td>✓</td><td>29.7 (-5.4)</td><td>25.1 (-5.9)</td><td>17.7 (-6.3)</td><td>46.9 (-3.8)</td><td>32.3 (+1.7)</td><td>51.9 (-3.1)</td><td>55.7 (-6.0)</td><td>40.2 (-4.4)</td></tr><tr><td>w/o Rationale</td><td>7B</td><td>✓</td><td>25.5 (-9.6)</td><td>14.7 (-16.3)</td><td>15.4 (-8.6)</td><td>45.9 (-4.8)</td><td>29.7 (-0.9)</td><td>51.0 (-4.0)</td><td>52.4 (-9.3)</td><td>36.8 (-7.8)</td></tr><tr><td>WizardMath</td><td>13B</td><td>X</td><td>6.4</td><td>6.6</td><td>11.5</td><td>9.6</td><td>11.0</td><td>28.5</td><td>21.1</td><td>15.0</td></tr><tr><td>ToRA-CODE</td><td>13B</td><td>✓</td><td>35.7 (+29.3)</td><td>31.1 (+24.5)</td><td>25.7 (+14.2)</td><td>55.6 (+46.0)</td><td>39.5 (+28.5)</td><td>58.7 (+30.2)</td><td>66.7 (+45.6)</td><td>48.1 (+33.1)</td></tr><tr><td>w/o Shaping</td><td>13B</td><td>✓</td><td>32.8 (-2.9)</td><td>26.0 (-5.1)</td><td>24.0 (-1.7)</td><td>52.6 (-3.0)</td><td>38.4 (-1.1)</td><td>55.6 (-3.1)</td><td>61.2 (-5.5)</td><td>44.6 (-3.5)</td></tr><tr><td>w/o Rationale</td><td>13B</td><td>✓</td><td>27.1 (-8.6)</td><td>15.8 (-15.3)</td><td>16.3 (-9.4)</td><td>50.4 (-5.2)</td><td>36.9 (-2.6)</td><td>55.3 (-3.4)</td><td>56.5 (-10.2)</td><td>40.2 (-7.9)</td></tr><tr><td>ToRA-CODE</td><td>34B</td><td>✓</td><td>38.9</td><td>34.6</td><td>27.3</td><td>57.8</td><td>41.4</td><td>63.7</td><td>67.7</td><td>50.8</td></tr><tr><td>w/o Shaping</td><td>34B</td><td>✓</td><td>34.0 (-4.9)</td><td>29.9 (-4.7)</td><td>24.6 (-2.7)</td><td>55.6 (-2.2)</td><td>41.6 (+0.2)</td><td>63.8 (+0.1)</td><td>61.4 (-6.3)</td><td>47.4 (-3.4)</td></tr><tr><td>w/o Rationale</td><td>34B</td><td>✓</td><td>28.3 (-10.6)</td><td>15.8 (-18.8)</td><td>18.0 (-9.3)</td><td>52.4 (-5.4)</td><td>40.7 (-0.7)</td><td>58.6 (-5.1)</td><td>57.5 (-10.2)</td><td>41.9 (-8.9)</td></tr><tr><td>WizardMath</td><td>70B</td><td>X</td><td>9.1</td><td>13.4</td><td>16.9</td><td>16.5</td><td>19.2</td><td>42.7</td><td>35.0</td><td>24.1</td></tr><tr><td>ToRA</td><td>70B</td><td>✓</td><td>37.1 (+28)</td><td>30.4 (+17)</td><td>30.1 (+13.2)</td><td>54.6 (+38.1)</td><td>40.3 (+21.1)</td><td>64.9 (+22.2)</td><td>66.6 (+31.6)</td><td>49.7 (+25.6)</td></tr><tr><td>w/o Shaping</td><td>70B</td><td>✓</td><td>33.8(-3.3)</td><td>28.9(-1.5)</td><td>27.1(-3)</td><td>53.0(-1.6)</td><td>38.0(-2.3)</td><td>62.2(-2.7)</td><td>64.2(-2.4)</td><td>47.3(-2.4)</td></tr><tr><td>w/o Rationale</td><td>70B</td><td>✓</td><td>26.7(-10.4)</td><td>14.7(-15.7)</td><td>20.3(-9.8)</td><td>48.9(-5.7)</td><td>39.2(-1.1)</td><td>59.8(-5.1)</td><td>57.6(-9)</td><td>41.5(-8.2)</td></tr></table>

# 3.4 MAIN RESULTS

Table 2 presents the results of ToRA on 10 mathematical datasets, highlighting the following salient observations: (1) Using interleaved formatting and output space shaping, ToRA consistently surpasses prior state-of-the-art open-source models across all scales, achieving  $13\%$  to  $19\%$  absolute improvements across 10 tasks. (2) ToRA-70B substantially outperforms ChatGPT with both CoT and PAL prompting on GSM8k (84.3% vs. 80.4%) and MATH (49.7% vs. 38.7%), while ToRA-CODE-34B is competitive with GPT-4 solving competition-level MATH dataset with code (50.8% vs. 51.8%). (3) The accuracy of ToRA-CODE is about  $5\%$  higher than ToRA of the same size, demonstrating that continued training on code data significantly benefits program-based tool use. (4) While rationale-based fine-tuning negatively affects out-of-distribution generalization, ToRA displays superior generalization. For instance, WizardMath-70B underperforms the base model on TabMWP (49.8% vs. 57.5%), while ToRA-70B effectively generalizes to this tabular reasoning task (74.0%). (5) ToRA attains fast zero-shot inference speed, averaging 1.02 tool interaction rounds per problem, while effectively addressing problems that require interactive tool utilization.

# 3.5 ABLATION STUDY

Figure 4: Comparison of three formats: (1) Rationale-only: step-by-step natural language reasoning like CoT; (2) Program-only: solving problems with programs like PAL; (3) Tool-integrated Reasoning used by TORA: interweaving rationale and program execution to solve problems. We evaluated GPT-4 with few-shot prompting. We trained LLaMA-2 models to reason in the three types of formats, respectively. For a fair comparison, we do not apply output space shaping for all LLaMA-2 models.

# 3.5.1 COMPARISONS OF FORMATTING

To evaluate the efficacy of the reasoning format adopted by ToRA which interleaves rationales with programs, we compared it with Rationale-only and Program-only formats using GPT-4 and LLaMA-2 trained with the same size of data from MATH. As shown in Fig 4, the ToRA method consistently surpasses Rationale-only and Program-only approaches. Remarkably, using LLaMA-2, the ToRA method achieves substantial improvements of  $29.0\%$  and  $6.7\%$  over Rationale-only and Program-only, respectively. With the closed-source GPT-4, the improvements are  $19.1\%$  and  $9.8\%$ , respectively. This emphasizes the effectiveness of integrating natural language rationales with programs.

# 3.5.2 EFFECTS OF OUTPUT SPACE SHAPING

Figure 5: Ablation on output space shaping strategies using CodeLLaMA: (1)  $\mathrm{TORA}_{-}\mathrm{Correction}$  is trained on TORA-CORPUS without shaping. (2)  $\mathrm{TORA}_{-}\mathrm{Correction}$  employs only the sampling strategy for shaping, trained with up to 4 additional valid trajectory samples per problem. (3) TORA utilizes both the sampling and correction, also trained with up to 4 additional trajectories per problem.

We assess the effectiveness of the output space shaping strategies presented in Section 2.3, specifically sampling and correction. As shown in Fig 5 and Table 3: (1) Output space shaping yields a considerable average improvement of  $3.4\%$  and  $4.0\%$  absolute for GSM8k and MATH, respectively, with greater benefits for smaller models; (2) Applying the sampling strategy results in a  $2.7\%$  absolute improvement on average, while additionally incorporating correction offers a more substantial boost of up to  $4.5\%$ , without using more training data; (3) Output space shaping benefits even the largest model ToRA-70B, with a notable improvement from  $47.3\%$  to  $49.7\%$  on MATH. These findings highlight the effectiveness of our shaping strategies across different model sizes and datasets.

# 3.6 ANALYSIS

We investigate the benefits, detailed patterns, and remaining challenges of tool interaction for mathematical reasoning on the challenging MATH dataset. Performance breakdowns on all subtopics of MATH are reported in Table 3.

Benefits from Tool-Integration for MATH Sub-topics As shown in Table 3, TORA outperforms WizardMath by around  $45\%$  in Algebra and Number Theory, which is attributed to stimulating and shaping tool-use behavior. Problems from the two sub-topics typically need intricate computation and data manipulation. Algebra mainly focuses on solving equations and application problems, while many Number Theory problems can be tackled using brute-force approaches through code.

Patterns of Library Usage for Problem Solving Fig 6 presents the most frequently used libraries for different sub-topics and the corresponding accuracies of their solutions. Tool-use behavior on different mathematical areas demonstrates distinct patterns. sympy and its internal solvers are primarily employed for algebra-related topics. Precalculus exhibits extensive matrix operations via matrices, resulting in a high accuracy. Number Theory depends on algorithms like gcd and lcm. Geometry mainly uses the rational library for fraction-based computations, while the application of other tools is limited, signifying the potential for improvement.

Detailed Impact of Rationale on Different Topics Table 3 shows that using an interleaved format, in contrast to merely writing the program, leads to significant improvements across all subtopics, especially in Precalculus, Algebra, and Geometry, where notable increases range from  $8.6\%$  to  $18.8\%$ .

Figure 6: Library usage frequency and accuracy on each sub-topic of MATH.


Appendix F.1 provides representative examples demonstrating how the rationale aids in planning, multi-round self-correction, and finalizing answers.

Table 4: The failure modes of the ToRA on MATH, and their corresponding percentages in random samples analyzed by humans. We include specific examples of each failure mode in Appendix F.  

<table><tr><td>Error Type</td><td>Definition</td><td>%</td><td>Examples</td></tr><tr><td>Reasoning Error</td><td>Mistakes due to incorrect reasoning steps or missing conditions.</td><td>38%</td><td>Ex. 5</td></tr><tr><td>Hallucination</td><td>Fabrication of numbers or answers.</td><td>5%</td><td>Ex. 6</td></tr><tr><td>Diagram Understanding</td><td>Misinterpretation of the input diagram.</td><td>21%</td><td>Ex. 7</td></tr><tr><td>Inappropriate Tool Use</td><td>Incorrect use of external tools, especially when the problem can’t be solved directly with libraries.</td><td>10%</td><td>Ex. 8</td></tr><tr><td>Syntax Error</td><td>Persistent syntax errors despite multiple correction attempts.</td><td>9%</td><td>Ex. 9</td></tr><tr><td>Runtime Error</td><td>Errors during program execution, unresolved by retrying.</td><td>9%</td><td>Ex. 10</td></tr><tr><td>Rationale-only Error</td><td>Cannot be formalized into a program and the rationale is incorrect.</td><td>3%</td><td>Ex. 11</td></tr><tr><td>False Negative</td><td>Correct answers that don’t fully match the ground truth.</td><td>5%</td><td>Ex. 12</td></tr></table>

Remaining Challenges in Mathematical Reasoning for ToRA To better understand the failure modes and remaining challenges, we manually annotated 100 randomly selected trajectories from the MATH test set, identifying and categorizing their failure modes. The results are shown in Table 4: Primarily, incorrect reasoning steps constitute the primary source of errors for ToRA on complex math reasoning tasks (38%), with some hallucination issues also evident during problem interpretation and answer finalization (5%). Secondly, the misinterpretation of input diagrams contributes significantly to the error rate (21%). This is particularly noticeable in Geometry, Precalculus, and Intermediate Algebra. The diagrams in the MATH dataset are usually detailed in text using the Asymptote language (Hendrycks et al., 2021), thus making it challenging for ToRA to comprehend diagrams purely from textual descriptions. Thirdly, issues with tool usage include Inappropriate Tool Usage (10%), Syntax Error (9%), and Runtime Error (9%). These problems frequently arise when ToRA fails to use tools correctly after several corrections or attempts. There are certain inputs that fail to formalize well as programs (3%), which require abstract reasoning rather than computation. Finally, we also found that there are false negatives when using automatic indicators, i.e., correct predictions that are misjudged as wrong, but the proportion is relatively small (5%).

# 4 CONCLUSION

This paper presents ToRA, a series of novel Tool-integrated Reasoning Agents that synergistically combines natural language rationale with program-based tool-use for mathematical problem solving. Our approach demonstrates the potential of integrating external tools in the reasoning process, enabling language models to effectively tackle complex quantitative tasks. ToRA achieves state-of-the-art performance on 10 diverse mathematical reasoning tasks, substantially outperforming existing rationale-based and program-based approaches. Furthermore, our systematic analysis of the benefits and remaining challenges of tool interaction provides valuable insights for future research, contributing to the development of more advanced and versatile reasoning agents.

# AUTHOR CONTRIBUTIONS

Zhibin Gou proposed the interleaved tool-use format of TORA and curated TORA-CORPUS dataset, implemented the training and evaluation pipeline, conducted experiments and analysis on all datasets, implemented baselines, and was a main contributor to the paper writing. Zhihong Shao proposed the project, conducted preliminary experiments, proposed and implemented the training and evaluation pipelines, proposed and trained all TORA models with output space shaping as well as TORA variants in the ablation study, designed and oversaw experimental analysis, and contributed to many parts of the paper writing. Yeyun Gong, Yelong Shen, Yujiu Yang, Minlie Huang, Nan Duan, and Weizhu Chen provided research mentorship, oversaw project coordination, and advised and contributed to many parts of the writing.

# ACKNOWLEDGMENTS

Zhibin Gou and Yujiu Yang were supported by the National Natural Science Foundation of China (Grant No. U1903213) and the Shenzhen Science and Technology Program (JSGG20220831110203007). Zhihong Shao and Minlie Huang were supported by the NSFC projects (Key project with No. 61936010), and were also supported by the National Science Foundation for Distinguished Young Scholars (with No. 62125604).

# REFERENCES

Rohan Anil, Andrew M Dai, Orhan First, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al. Palm 2 technical report. arXiv preprint arXiv:2305.10403, 2023.  
Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al. Improving language models by retrieving from trillions of tokens. In International conference on machine learning, pp. 2206-2240. PMLR, 2022.  
Sebastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yanzhi Li, Scott M. Lundberg, Harsha Nori, Hamid Palangi, Marco Túlio Ribeiro, and Yi Zhang. Sparks of artificial general intelligence: Early experiments with GPT-4. CoRR, abs/2303.12712, 2023. doi: 10.48550/arXiv.2303.12712. URL https://doi.org/10.48550/arXiv.2303.12712.  
Cristian Bucilua, Rich Caruana, and Alexandru Niculescu-Mizil. Model compression. In Proceedings of the 12th ACM SIGKDD international conference on Knowledge discovery and data mining, pp. 535-541, 2006.  
Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588, 2022.  
Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. Training verifiers to solve math word problems, 2021. URL https://arxiv.org/abs/2110.14168.  
Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. 2023.  
Edward A Feigenbaum, Julian Feldman, et al. Computers and thought, volume 7. New York McGraw-Hill, 1963.  
Yao Fu, Hao Peng, Litu Ou, Ashish Sabharwal, and Tushar Khot. Specializing smaller language models towards multi-step reasoning. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 10421-10430. PMLR, 2023. URL https://proceedings.mlr.press/v202/fu23d.html.  
Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. Pal: Program-aided language models. arXiv preprint arXiv:2211.10435, 2022.  
Zhibin Gou, Zhihong Shao, Yeyun Gong, Yelong Shen, Yujiu Yang, Nan Duan, and Weizhu Chen. Critic: Large language models can self-correct with tool-interactive critiquing. arXiv preprint arXiv:2305.11738, 2023.  
Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. Measuring mathematical problem solving with the math dataset. NeurIPS, 2021.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. Distilling the knowledge in a neural network. arXiv preprint arXiv:1503.02531, 2015.  
Namgyu Ho, Laura Schmid, and Se-Young Yun. Large language models are reasoning teachers. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 14852-14882, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.830. URL https://aclanthology.org/2023.acl-long.830.

Mohammad Javad Hosseini, Hannaneh Hajishirzi, Oren Etzioni, and Nate Kushman. Learning to solve arithmetic word problems with verb categorization. In Proceedings of the 2014 Conference on Empirical Methods in Natural Language Processing (EMNLP), pp. 523-533, 2014.  
Jiaxin Huang, Shixiang Shane Gu, Le Hou, Yuexin Wu, Xuezhi Wang, Hongkun Yu, and Jiawei Han. Large language models can self-improve. CoRR, abs/2210.11610, 2022. doi: 10.48550/arXiv.2210.11610. URL https://doi.org/10.48550/arXiv.2210.11610.  
Weisen Jiang, Han Shi, Longhui Yu, Zhengying Liu, Yu Zhang, Zhenguo Li, and James T Kwok. Backward reasoning in large language models for verification. arXiv preprint arXiv:2308.07758, 2023.  
Rik Koncel-Kedzierski, Subhro Roy, Aida Amini, Nate Kushman, and Hannaneh Hajishirzi. MAWPS: A math word problem repository. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 1152-1157, San Diego, California, June 2016. Association for Computational Linguistics. doi: 10.18653/v1/N16-1136. URL https://aclanthology.org/N16-1136.  
Ariel N Lee, Cole J Hunter, and Nataniel Ruiz. Platypus: Quick, cheap, and powerful refinement of llms. arXiv preprint arXiv:2308.07317, 2023.  
Yifei Li, Zeqi Lin, Shizhuo Zhang, Qiang Fu, Bei Chen, Jian-Guang Lou, and Weizhu Chen. Making language models better reasoners with step-aware verifier. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 5315-5333, 2023.  
Tian Liang, Zhiwei He, Wenxiang Jiao, Xing Wang, Yan Wang, Rui Wang, Yujiu Yang, Zhaopeng Tu, and Shuming Shi. Encouraging divergent thinking in large language models through multi-agent debate. arXiv preprint arXiv:2305.19118, 2023.  
Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let's verify step by step. arXiv preprint arXiv:2305.20050, 2023.  
Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan. Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=DHyHRBwJUTN.  
Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, and Dongmei Zhang. Wizardmath: Empowering mathematical reasoning for large language models via reinforced evol-instruct. arXiv preprint arXiv:2308.09583, 2023.  
Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al. Augmented language models: a survey. arXiv preprint arXiv:2302.07842, 2023.  
Shen-yun Miao, Chao-Chun Liang, and Keh-Yih Su. A diverse corpus for evaluating and developing English math word problem solvers. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, pp. 975-984, Online, July 2020. Association for Computational Linguistics. doi: 10.18653/v1/2020.acl-main.92. URL https://aclanthology.org/2020.acl-main.92.  
Swaroop Mishra, Matthew Finlayson, Pan Lu, Leonard Tang, Sean Welleck, Chitta Baral, Tanmay Rajpurohit, Oyvind Tafjord, Ashish Sabharwal, Peter Clark, and Ashwin Kalyan. Lila: A unified benchmark for mathematical reasoning. In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing (EMNLP), 2022.  
Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al. Webgpt: Browser-assisted question-answering with human feedback. arXiv preprint arXiv:2112.09332, 2021.

OpenAI. Gpt-4 technical report, 2023.  
Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and Marco Tulio Ribeiro. Art: Automatic multi-step reasoning and tool-use for large language models. arXiv preprint arXiv:2303.09014, 2023.  
Aaron Parisi, Yao Zhao, and Noah Fiedel. Talm: Tool augmented language models. arXiv preprint arXiv:2205.12255, 2022.  
Arkil Patel, Satwik Bhattachamishra, and Navin Goyal. Are NLP models really able to solve simple math word problems? In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pp. 2080-2094, Online, June 2021. Association for Computational Linguistics. doi: 10.18653/v1/2021.naacl-main.168. URL https://aclanthology.org/2021.naacl-main.168.  
Guilherme Penedo, Quentin Malartic, Daniel Hesslow, Ruxandra Cojocaru, Alessandro Cappelli, Hamza Alobeidli, Baptiste Pannier, Ebtesam Almazrouei, and Julien Launay. The RefinedWeb dataset for Falcon LLM: outperforming curated corpora with web data, and web data only. arXiv preprint arXiv:2306.01116, 2023. URL https://arxiv.org/abs/2306.01116.  
Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277, 2023.  
Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving. arXiv preprint arXiv:2009.03393, 2020.  
Samyam Rajbhandari, Olatunjri Ruwase, Jeff Rasley, Shaden Smith, and Yuxiong He. Zero-infinity: Breaking thegpu memory wall for extreme scale deep learning. In Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis, pp. 1-14, 2021.  
Baptiste Rozière, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaoqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.  
Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. arXiv preprint arXiv:2302.04761, 2023.  
Zhihong Shao, Fei Huang, and Minlie Huang. Chaining simultaneous thoughts for numerical reasoning. In Yoav Goldberg, Zornitsa Kozareva, and Yue Zhang (eds.), Findings of the Association for Computational Linguistics: EMNLP 2022, Abu Dhabi, United Arab Emirates, December 7-11, 2022, pp. 2533-2547. Association for Computational Linguistics, 2022. doi: 10.18653/v1/2022-findings-emnlp.187. URL https://doi.org/10.18653/v1/2022-findings-emnlp.187.  
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Synthetic prompting: Generating chain-of-thought demonstrations for large language models. In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA, volume 202 of Proceedings of Machine Learning Research, pp. 30706-30775. PMLR, 2023a. URL https://proceedings.mlr.press/v202/shao23a.html.  
Zhihong Shao, Yeyun Gong, Yelong Shen, Minlie Huang, Nan Duan, and Weizhu Chen. Enhancing retrieval-augmented large language models with iterative retrieval-generation synergy. CoRR, abs/2305.15294, 2023b. doi: 10.48550/arXiv.2305.15294. URL https://doi.org/10.48550/arXiv.2305.15294.  
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca, 2023.  
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. Llama 2: Open foundation and fine-tuned chat models. CoRR, abs/2307.09288, 2023b. doi: 10.48550/arXiv.2307.09288. URL https://doi.org/10.48550/arXiv.2307.09288.  
Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, brian richter, Fei Xia, Ed H. Chi, Quoc V Le, and Denny Zhou. Chain of thought prompting elicits reasoning in large language models. In Alice H. Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho (eds.), Advances in Neural Information Processing Systems, 2022. URL https://openreview.net/forum?id=_VjQlMeSB_J.  
Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=WE_vluYUL-X.  
Zheng Yuan, Hongyi Yuan, Chengpeng Li, Guanting Dong, Chuanqi Tan, and Chang Zhou. Scaling relationship on learning mathematical reasoning with large language models. arXiv preprint arXiv:2308.01825, 2023.  
Xiang Yue, Xingwei Qu, Ge Zhang, Yao Fu, Wenhao Huang, Huan Sun, Yu Su, and Wenhu Chen. Mammoth: Building math generalist models through hybrid instruction tuning. arXiv preprint arXiv:2309.05653, 2023.  
Eric Zelikman, Yuhuai Wu, Jesse Mu, and Noah Goodman. Star: Bootstrapping reasoning with reasoning. Advances in Neural Information Processing Systems, 35:15476-15488, 2022.  
Beichen Zhang, Kun Zhou, Xilin Wei, Wayne Xin Zhao, Jing Sha, Shijin Wang, and Ji-Rong Wen. Evaluating and improving tool-augmented computation-intensive math reasoning. arXiv preprint arXiv:2306.02408, 2023.  
Aojun Zhou, Ke Wang, Zimu Lu, Weikang Shi, Sichun Luo, Zipeng Qin, Shaoqing Lu, Anya Jia, Linqi Song, Mingjie Zhan, et al. Solving challenging math word problems using gpt-4 code interpreter with code-based self-verification. arXiv preprint arXiv:2308.07921, 2023a.  
Denny Zhou, Nathanael Scharli, Le Hou, Jason Wei, Nathan Scales, Xuezhi Wang, Dale Schuurmans, Claire Cui, Olivier Bousquet, Quoc V. Le, and Ed H. Chi. Least-to-most prompting enables complex reasoning in large language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023b. URL https://openreview.net/pdf?id=WZH7099tgfM.  
Xinyu Zhu, Junjie Wang, Lin Zhang, Yuxiang Zhang, Yongfeng Huang, Ruyi Gan, Jiaxing Zhang, and Yujiu Yang. Solving math word problems via cooperative reasoning induced language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4471-4485, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.acl-long.245. URL https://aclanthology.org/2023.acl-long.245.

# CONTENTS

A Related Works 15  
B Evaluation Datasets 15  
C Additional Experiments and Analysis 17

C.1 Accuracies of Closed-Source Models on MATH 17  
C.2Effects of # Valid Trajectories for Output Space Shaping 17  
C.3 Impact of Output Space Shaping in Relation to Question Difficulty 17

D Detailed Information of TORA-CORPUS 18  
E Prompts 20  
F Examples 22

F.1 Success Cases 22  
F.2 Failure Cases 24

# A RELATED WORKS

Mathematical Reasoning Recent research has greatly improved reasoning in LLMs with step-by-step natural language reasoning (Polu & Sutskever, 2020; Wei et al., 2022; Zhou et al., 2023b; Zhu et al., 2023; Huang et al., 2022; Liang et al., 2023). However, natural language reasoning struggles with complex computations and symbolic manipulations. To overcome the limitations, recent research has exploited tools like calculators (Cobbe et al., 2021; Shao et al., 2022), code interpreters (Mishra et al., 2022), and symbolic solvers (Zhang et al., 2023). Program-based methods (Gao et al., 2022; Chen et al., 2022; Shao et al., 2023a) transform reasoning tasks into program synthesis tasks, thus offering complementary advantages over natural language reasoning, but they face challenges in nuanced reasoning, planning, and error handling (Gou et al., 2023), where natural language reasoning should be more suitable.

Tool-Augmented Language Models Augmenting LLMs with tools can largely alleviate LLMs' limitations and improve reasoning and generation performance (Parisi et al., 2022; Mialon et al., 2023; Yao et al., 2023). Recent work demonstrates the benefits of integrating retrievers (Borgeaud et al., 2022; Shao et al., 2023b), search engines (Nakano et al., 2021), and multi-tool approaches (Schick et al., 2023; Paranjape et al., 2023; Gou et al., 2023) to improve generation.

Knowledge Distillation Knowledge distillation (KD) transfers knowledge from teacher models to student models (Bucilua et al., 2006; Hinton et al., 2015). Using LLM-generated trajectories for fine-tuning is a form of KD (Fu et al., 2023; Taori et al., 2023; Peng et al., 2023; Ho et al., 2023). Our proposed TORA shows that learning interactive tool-use trajectories is a promising direction to adapt language models to reasoning tasks.

# B EVALUATION DATASETS

We present statistics and examples of the ten evaluation datasets in Table 5.

Table 5: Statistics and examples of the 10 evaluation datasets. In the main result table, we present the average accuracy of SingleEq, SingleOp, AddSub, and MultiArith under the collective name MAWPS.  

<table><tr><td>Dataset</td><td>OOD?</td><td>#Samples</td><td>Example Problem</td><td></td></tr><tr><td>GSM8k (Cobbe et al., 2021)</td><td>IND</td><td>1319</td><td>The ice cream parlor was offering a deal, buy 2 scoops of ice cream, get 1 scoop free. Each scoop cost $1.50. If Erin had $6.00, how many scoops of ice cream should she buy?</td><td></td></tr><tr><td>MATH (Hendrycks et al., 2021)</td><td>IND</td><td>5000</td><td>For a constant c, in cylindrical coordinates (r, θ, z), find the shape described by the equation z = c. 
(A) Line (B) Circle (C) Plane (D) Sphere (E) Cylinder (F) Cone. Enter the letter of the correct option.</td><td></td></tr><tr><td>GSM-Hard (Gao et al., 2022)</td><td>OOD</td><td>1319</td><td>Jean has 30 lollipops. Jean eats 8714250 of the lol-lipops. With the remaining lollipops, Jean wants to pack-age 8714250 lollipops in one bag. How many bags can Jean fill?</td><td></td></tr><tr><td>SVAMP (Patel et al., 2021)</td><td>OOD</td><td>1000</td><td>During summer break 819058 kids from Lawrence county go to camp and the other 668278 kids stay home. How many more kids spent their summer break at the camp compared to those who stayed home?</td><td></td></tr><tr><td>ASDiv (Miao et al., 2020)</td><td>OOD</td><td>2215</td><td>Mrs. Hilt saw an iPod for sale. The price tag said the iPod cost $128, but a sign announced that it was on sale for &quot;35% off.&quot; How much would the iPod cost after the discount?</td><td></td></tr><tr><td>TabMWP (Lu et al., 2023)</td><td>OOD</td><td>1000</td><td>Stem Leaf
2 3, 6, 7, 8, 8
3 0, 7, 9
4 1, 5
5
6 2, 3, 3, 4, 8, 8
7 3, 4, 4, 7, 9
8 5, 5</td><td>Read the table regarding &quot;eight lifting results (lbs)&quot;. Mr. Morrison, a P.E. teacher, wrote down how much weight each of his students could lift. How many people lifted at least 28 pounds?</td></tr><tr><td>SingleEq (Koncel-Kedzierski et al., 2016)</td><td>OOD</td><td>508</td><td>Alyssa&#x27;s dog had puppies. She gave 7 to her friends.She now has 5 puppies left. How many puppies did she have to start with?</td><td></td></tr><tr><td>SingleOp (Koncel-Kedzierski et al., 2016)</td><td>OOD</td><td>562</td><td>Rachel removes 47 bottle caps from a jar. There were originally 87 bottle caps in the jar. How many bottle caps are left in the jar?</td><td></td></tr><tr><td>AddSub (Koncel-Kedzierski et al., 2016)</td><td>OOD</td><td>395</td><td>Sam went to 14 football games this year. He went to 29 games last year. How many football games did Sam go to in all?</td><td></td></tr><tr><td>MultArith (Koncel-Kedzierski et al., 2016)</td><td>OOD</td><td>600</td><td>Paige had 43 math problems and 12 science problems for homework. If she finished 44 of the problems at school, how many problems did she have to do for homework?</td><td></td></tr></table>

Table 6: Accuracies of ChatGPT and GPT-4 on the MATH dataset, with breakdown w.r.t. different mathematical subjects. We apply PAL prompting and the Tool-integrated Reasoning method used by TORA to the two closed-source models.  

<table><tr><td>Model</td><td>Tool</td><td>Intermediate Algebra</td><td>Precalculus</td><td>Geometry</td><td>Number Theory</td><td>Counting &amp; Probability</td><td>Prealgebra</td><td>Algebra</td><td>Overall</td></tr><tr><td></td><td></td><td colspan="8">Test Set</td></tr><tr><td>ChatGPT (PAL)</td><td>✓</td><td>18.5</td><td>19.2</td><td>23.2</td><td>48.5</td><td>43.0</td><td>62.7</td><td>45.4</td><td>38.7</td></tr><tr><td>GPT-4 (PAL)</td><td>✓</td><td>32.8</td><td>29.3</td><td>38.0</td><td>58.7</td><td>61.0</td><td>73.9</td><td>59.1</td><td>51.8</td></tr><tr><td>GPT-4 (Tool-integrated Reasoning)</td><td>✓</td><td>40.0</td><td>37.2</td><td>44.1</td><td>68.9</td><td>67.3</td><td>82.2</td><td>75.8</td><td>61.6</td></tr><tr><td></td><td></td><td colspan="8">Training Set</td></tr><tr><td>GPT-4 (Tool-integrated Reasoning)</td><td>✓</td><td>51.0</td><td>51.5</td><td>42.5</td><td>77.4</td><td>72.2</td><td>89.8</td><td>85.1</td><td>64.3</td></tr><tr><td>w/ best@10</td><td>✓</td><td>72.9</td><td>70.0</td><td>58.9</td><td>91.6</td><td>81.7</td><td>95.5</td><td>96.3</td><td>83.1</td></tr></table>

# C ADDITIONAL EXPERIMENTS AND ANALYSIS

# C.1 ACCURACIES OF CLOSED-SOURCE MODELS ON MATH

Table 6 presents the detailed accuracies of GPT-4 on the MATH dataset. The Tool-integrated Reasoning method used by ToRA significantly outperforms PAL prompting when directly applied to the closed-source GPT-4, further demonstrating the benefits of synergizing natural language reasoning and program-based tool use.

# C.2 EFFECTS OF # VALID TRAJECTORIES FOR OUTPUT SPACE SHAPING

Figure 7: Effects of using different numbers of additional valid trajectories per question for output space shaping.


As shown in Fig 7, it is beneficial to increase the number of additional valid trajectories for output space shaping.

# C.3 IMPACT OF OUTPUT SPACE SHAPING IN RELATION TO QUESTION DIFFICULTY

We compare the effects of output space shaping on MATH problems of different difficulty levels (from level 1 to level 5) in Figure 8, and present the statistics of MATH problems at different levels in Table 7. As can be seen:

- Across these different difficulty levels and model sizes, output space shaping generally brings a significant improvement of  $4.0\%$  on average across different model sizes.  
- Output space shaping brings significant improvements for difficult, long problems. E.g., with TORA-CODE-13B, shaping does not significantly improve level 1 to level 2 problems, but it brings a substantial improvement of  $5.4\%$  to  $5.7\%$  for level 3 to level 5 problems.

Figure 8: Impact of Output Space Shaping in Relation to Question Difficulty.

- After using shaping, TORA-CODE-34B outperforms GPT-4 PAL on problems from Level 1 to Level 4, but there is still a gap at Level 5 (27.3% vs. 30.0%). These problems are usually longer (average about 248.4 characters), require more reasoning steps (>1,000 characters) to solve, and more often include diagram inputs (about 20%). These observations may guide future work to focus more on solving these more difficult problems.

Table 7: Statistics of MATH problems at different levels. Average Answer Length indicates the average length of TORA outputs; Training query coverage indicates the proportion of queries with at least one valid trajectory in TORA-CORPUS relative to the total queries in the original dataset.  

<table><tr><td></td><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Level 4</td><td>Level 5</td></tr><tr><td># Test Samples</td><td>437</td><td>894</td><td>1131</td><td>1214</td><td>1324</td></tr><tr><td>Avg Question Length</td><td>123.8</td><td>150.9</td><td>169.1</td><td>203.0</td><td>248.4</td></tr><tr><td>Avg Answer Length</td><td>503.1</td><td>655.8</td><td>751.2</td><td>881.6</td><td>1083.8</td></tr><tr><td>Training query coverage</td><td>97.7%</td><td>91.6%</td><td>86.5%</td><td>81.3%</td><td>68.0%</td></tr></table>

# D DETAILED INFORMATION OF TORA-CORPUS

We provide a more detailed introduction to the data construction process, quality control, and data statistical information, beyond Sec. 2.2.

Data Format and Quality Control In our preliminary experiments, we found that the tool-integrated reasoning trajectory format generated by zero-shot prompting was somewhat chaotic. Therefore, we designed a few-shot prompting to control the reasoning format, which effectively improved data quality. On the other hand, we increased the annotation success rate by sampling, ensuring more comprehensive coverage of the training query.

Data Filtering Process For the data constructed, we filtered out paths that produced incorrect answers by matching them with standard answers. To prevent the model from learning incorrect

Table 8: Accuracy of TORA-CORPUS on GSM8k and MATH training set. TORA-CORPUS-Greedy uses only the greedy trajectories, while ToRA-Corpus-16k combines sampled trajectories.  

<table><tr><td rowspan="2"></td><td colspan="2">GSM8k</td><td colspan="7">MATH</td></tr><tr><td>All</td><td>All</td><td>Intermediate Algebra</td><td>Precalculus</td><td>Geometry</td><td>Number Theory</td><td>Counting &amp; Probability</td><td>Prealgebra</td><td>Algebra</td></tr><tr><td>TORA-CORPUS-Greedy</td><td>94.4</td><td>64.3</td><td>51.0</td><td>51.5</td><td>70.0</td><td>77.4</td><td>72.2</td><td>89.8</td><td>85.1</td></tr><tr><td>TORA-CORPUS-16k</td><td>98.2</td><td>83.1</td><td>72.9</td><td>70.0</td><td>58.9</td><td>91.6</td><td>81.7</td><td>95.5</td><td>96.3</td></tr></table>

Table 9: Statistics of TORA-CORPUS-16k  

<table><tr><td></td><td>GSM8k</td><td>MATH</td><td>Total</td></tr><tr><td># Train Samples</td><td>7,657</td><td>7,881</td><td>15,538</td></tr><tr><td>Avg Question Length</td><td>236</td><td>189</td><td>211</td></tr><tr><td>Avg Trajectory Length</td><td>678</td><td>704</td><td>691</td></tr><tr><td>Min Trajectory Length</td><td>218</td><td>119</td><td>119</td></tr><tr><td>Max Trajectory Length</td><td>1,713</td><td>2,486</td><td>2,486</td></tr></table>

intermediate reasoning processes, we further filtered out data samples with intermediate program execution errors.

Dataset Statistics In Table 8, we compared the annotation accuracy (i.e., sample coverage) of the training set on GSM8k, MATH, and MATH subtopics of TORA-CORPUS-Greedy using only the greedy trajectories, and TORA-CORPUS-16k combined with sampled trajectories. Furthermore, in Table 9, we reported the statistical data of TORA-CORPUS-16k, such as the number of samples, average question length, average, minimum, and maximum trajectory length, as shown in the following tables.

Rationale as Hints As described in Section 2.2, we annotated interactive tool-use trajectories for the training questions from MATH with GPT-4. GPT-4 achieves a success rate below  $65\%$  using greedy decoding. As MATH was originally annotated with natural language rationales, to improve the annotation success rate, we tried to provide GPT-4 with the human rationales as hints (Zelikman et al., 2022). However, when using this method, GPT-4 tends to replicate the hints and ignore tool-use outputs especially when the outputs are inconsistent with the hints, thus failing to produce high-quality trajectories. Hence, we deferred the utilization of the already-annotated natural language rationales for future investigations. Instead, we employed nucleus sampling to recall valid trajectories for questions that remained unsolved through greedy decoding. This approach significantly boosted annotation accuracy to  $83.1\%$ .

# E PROMPTS

We present instructions and example few-shot prompts of Tool-integrated Reasoning for querying GPT-4.

Listing 1: Prompts of Tool-integrated Reasoning for GPT-4.  
```txt
Integrate step-by-step reasoning and Python code to solve math problems using the following guidelines:
- Analyze the question and write functions to solve the problem; the function should not take any arguments.
- Present the final result in LaTeX using a '\boxed{ }' without any units.
- Utilize the 'pi' symbol and 'Rational' from Sympy for $ \pi $ and fractions, and simplify all fractions and square roots without converting them to decimal values.
Here are some examples you may refer to:
--- Question: Convert the point $(0, -3 \sqrt{3} \{3\}, 3)$ in rectangular coordinates to spherical coordinates. Enter your answer in the form $ (\rho, \theta, \phi) $ where $ \rho > 0, \theta $ 0 \leq \theta < 2 \pi $ and $0 \leq \phi$ \pi
Solution: To convert the point $ (x, y, z) $ in rectangular coordinates to spherical coordinates $ (\rho, \theta, \phi) $, we use the following formulas:
1. $ \rho = \sqrt{3} \{x^2 + y^2 + z^2\}$
2. $ \theta = \arctan \{y\} \{x\}$
3. $ \phi = \arccos \{z\} \{rho\}$
Let's write a function:
``'python
from sympy import sqrt, atan2, acos, pi
def rectangular_to_spherical():
    x, y, z = 0, -3 * sqrt(3), 3
    rho = sqrt(x * z + y * z + z * z)
    theta = atan2(y, x)
    phi = acos(z / rho)
    return rho, theta, phi
spherical_coordinates = rectangular_to_spherical()
print(spherical_coordinates)
...
``'output
(6, -pi/2, pi/3)
The required form of the answer is $(\rho, \theta, \phi) $, where $ \rho > 0$, $0 \leq \theta < 2 \pi$, and $0 \leq \rho$.
Since our initial output has $ \theta = -\pi / 2$, which doesn't meet the condition for $ \theta$ we add $2 \pi$ to adjust its value.
Hence, the spherical coordinates for the point $(0, -3 \sqrt{3} \{3\}, 3)$ are $ \boxed{(6, \frac{\pi}{4} \{3\} \{2\}, \frac{\pi}{4} \{3\})} $
--- Question: Compute $1011_2 + 101_2 - 1100_2 + 1101_2$. Express your answer in base 2.
```

```latex
Solution:
Let's write a single python function for binary computation:
``'python
def binary_sum_diff():
    # Convert binary to decimal
    num1 = int("1011", 2)
    num2 = int("101", 2)
    num3 = int("1100", 2)
    num4 = int("1101", 2)
    # Perform the operations
    result = num1 + num2 - num3 + num4
    # Convert the result back to binary
    result_binary = format(result, "b")
    return result_binary
result = binary_sum_diff()
print(result)
...
``'output
10001
The answer expressed in base 2 is \boxed{10001_2}$
...
Question: Find the matrix that corresponds to projecting onto the vector
\begin{pmatrix}
\end{pmatrix}
``'begin{pmatrix} 2 \ \ -3 \ \ end{pmatrix}.$
Solution:
``'python
from sympy import Matrix, simplify
def projection_matrix():
    v = Matrix([2, -3])
    P = (v * v.T) / (v.T * v) [0]
    # Simplify the elements
    P = P.applyfunc(simplify)
    return P
result = projection_matrix()
print(result)
...
``'output
Matrix([[4/13, -6/13], [-6/13, 9/13]])
The matrix is \$\boxed{\begin{pmatrix}\begin{pmatrix} 4\\ \end{pmatrix}\{13\} & -\frac{\partial}{\partial x}\{6\}\{13\} \end{pmatrix} \begin{pmatrix} -\frac{\partial}{\partial x}\{6\}\{13\} \end{pmatrix} .$
```

```python
from sympy.solve inequalities import solve_univariate_inequality
from sympy.core.relational import StrictLessThan
def solve_inequality():
    x = symbols('x')
    expression = (x**2 - 25) / (x + 5)
    inequality = StrictLessThan(expression, 0)
    solution = solve_univariate_inequality(inequality, x, relational=False)
    simplified_solution = simplify(soliton)
    return simplified解決
result = solve_inequality()
print(result)
...
'output'
Union(Interval.open(-oo, -5), Interval.open(-5, 5))
...
The answer is $boxed{(-\infty, -5)\cup(-5, 5)}$
---
```

# F EXAMPLES

# F.1 SUCCESS CASES

Listing 2: Success case for TORA: Rationale for planning.  
```python
Question: What real values of $x$ are not in the domain of $f(x) = \frac{1}{2} f(x)$?
Solution:
The domain of the function is all real numbers, so we need to find the
values of x for which the denominator of the function is zero.
The denominator of the function is the sum of the absolute values of the
two terms inside the absolute value.
Each of these terms is a quadratic in x, and the roots of a quadratic
equation are real unless the equation is identically zero.
So, we need to find the values of x for which the two quadratic equations
are identically zero.
```
python
from sympy import symbols, Abs, solveset, S
from sympy.core.relational import Ne
def non_domain_values():
    x = symbols('x')
    # These are the values of x for which the function is not defined
    roots1 = solveset(Ne(x**2 - 6*x + 8, 0), x, domain=S.Reals)
    roots2 = solveset(Ne(x**2 + x - 6, 0), x, domain=S.Reals)
    # The roots of a quadratic equation are always in the domain of the
    function, so we need to find the values of x that are not in the
    union of the roots
    non_domain_values = S.Reals - (roots1.union(roots2))
    return non_domain_values
result = non_domain_values()
print(result)
...
output
{2}
```

```txt
The real value of $x$ that is not in the domain of the function is $\boxed{2}$.
```

Listing 3: Success case for ToRA: Self-Correcting Errors with tool feedback.  
```python
Question: The difference between two perfect squares is 133. What is the smallest possible sum of the two perfect squares?
Solution:
We can solve this problem by noting that any perfect square can be written as $(a + b)^2$ or $(a - b)^2$ for some integers $a$ and $b$. So we can rewrite the question as the following problem: Find the minimum possible value for $(a + b)^2 + (a - b)^2$ given that $(a + b)^2 - (a - b)^2 = 133$.***
``'python
from sympy import symbols, Eq, solve
def min_sum_perfect_squares():
    '''Find the smallest possible sum of two perfect squares given that the difference between them is 133.
Returns:
        Integer: The minimum possible sum of the two perfect squares.
    '''a, b = symbols('a b')
# Equation for the difference of the squares
diff_eq = Eq((a + b) **2 - (a - b) **2, 133)
# Solve the equation for b
b_solutions = solve(diff_eq, b)
# Calculate the sum of the squares for each solution of b
sum_squares = [(a + b_sol) **2 + (a - b_sol) **2 for b_sol in b_solts]
# Find the minimum possible sum of the squares
min_sum_squares = min(sum_squares)
return min_sum_squares
result = min_sum_perfect_squares()
print(result)
...
``'output
(a - 133/(4*a)) **2 + (a + 133/(4*a)) **2'
The problem seems to be more complex than anticipated and the solution was incorrect.
Since the difference between two squares can be factored, we can write:
$\mathrm{x}^2 - \mathrm{y}^2 = (\mathrm{x} + \mathrm{y})(\mathrm{x} - \mathrm{y})$
$\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}
= \mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}\mathrm{\Phi}
```

```python
(13, 6) result in the smallest sum of the two perfect squares.  
So, let's modify the previous solution:  
``python  
def min_sum_perfect_squares():  
    min_squares = [13**2, 6**2]  
    min_sum_squares = sum(min_squares)  
    return min_sum_squares  
```  
``output  
205  
``
```

Listing 4: Success case for TORA: Finalizing answer.  
```txt
Question:
Compute \[\sum_{\{k = 1\}}^{\infty}\text{inf}y\frac{\operatorname{frac}\{k + 2\}}{k! + (k + 1)! + (k + 2)!}\] .\]
Solution:
``'python
from sympy import symbols, Sum, factorial, oo, simplify
def infinite_sum():
    expr = (k + 2) / (factorial(k) + factorial(k + 1) + factorial(k + 2))
    sum_expr = Sum(expr, (k, 1, oo))
    result = sum_expr.doit()
    return result
result = infinite_sum()
print(result)
...
'output
Sum((k + 2)/(factorial(k) + factorial(k + 1) + factorial(k + 2)), (k, 1, oo))
...
The sum does not converge to a simple expression, so the answer is $ \
boxed{\sum_{\{k = 1\}}^{\infty}\text{inf}y\frac{\operatorname{frac}\{k + 2\}}{k! + (k + 1)! + (k + 2)!}}$
```

# F.2 FAILURE CASES

Listing 5: Failure case for TORA: Reasoning Error.  
```python
Idx: 4289
Type: Prealgebra
Level: Level 2
Question: 'What is the smallest positive multiple of $13$ that is greater than $500?$'
Solution:
``'python
def smallest_positive_multiple():
    '''What is the smallest positive multiple of $13$ that is greater than $500?$''''
# Start from 500 and keep adding 13 until we get a number that is greater than 500
num = 500
while num <= 500:
```

```txt
num += 13
return num
result = smallest_positive_multiple()
print(result)
...
output
513
...
The smallest positive multiple of $13$ that is greater than $500$ is \
boxed{513}$. Execution: None
Prediction: '513'
Ground Truth: '507'
Score: [False]
Reference CoT: 'Dividing $500$ by $13$ gives a quotient $38$ with a remainder of $6$. In other words, [\text{[500 \cdots div 13=13 \cdots cdot 38+6. \cdots]}]
So, the smallest positive multiple of $13$ that is greater than $500$ is [\text{[13 \cdot 39= \cdots boxed{507}. \cdots]'}]
```

Listing 6: Failure case for TORA: Hallucination.  
```python
Idx: 2775
Type: Intermediate Algebra
Level: Level 3
Question: 'Find the eighth term of the sequence $1440, 1716, 1848, \ldots'
ldots$, whose terms are formed by multiplying the corresponding terms of two arithmetic sequences.' Solution:
``'python
from sympy import symbols, Rational, simplify
def eighth_term():
    '''Find the eighth term of the sequence $1440, 1716, 1848, \ldots$
    whose terms are formed by multiplying the corresponding terms of two arithmetic sequences'''`
a1, d1 = 1440, 36
a2, d2 = 16, 2
# Calculate the eighth term of each sequence
term1 = a1 + 7*d1
term2 = a2 + 7*d2
# Multiply the terms
result = term1 * term2
return result
result = eighth_term()
print(result)
...
output
50760
The eighth term of the sequence is $\boxed{50760}$.
Execution: None
Prediction: '50760'
Ground Truth: '348'
Score: [False]
Reference CoT: 'If $a_n = an + b$ and $b_n = cn + d$ are two arithmetic sequences, then their term-wise product takes the form [an^2 + Bn + C, ] where $A, B, C$ are constants.
```

```latex
Therefore, some sequence $x_n = An^2 + Bn + C$ has $x_0 = 1440,$ $x_1 = 1716,$ and $x_2 = 1848$. Then we have the equations \[\backslash\begin{aligned} & \\ aligned\end{aligned}\] C &= 1440, \[\backslash\backslash\backslash A+B+C&=1716,\backslash\backslash\backslash 4A+2B+C&=1848.\] \end{aligned} \] Subtracting $C=1440$ from the second and third equations, we have $A+B=276$ and $4A+2B=408,$ or $2A+B=204$. Then \[\backslash[A=(2A+B)-(A+B)=204-276=-72,\backslash]\] and so $B=276-A=348$. We conclude that the given sequence has the general formula \[\backslash[x_n=-72n^2+348n+1440.\backslash]\] Then the eighth term is \[\backslash[\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\] Then the eighth term is \[\backslash[\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backslash\backlash 7^2+348\backslash\backslashcdot 7+1440=\backslashboxed{348}. \]
```

Listing 7: Failure case for ToRA: Diagram Understanding. Navigating the complexities of input diagrams is key to solving many problems in Geometry, Precalculus, and Intermediate Algebra. This ideal process encompasses not just visual modalities, but also involves interactions with diagrams, such as adding auxiliary lines. Nonetheless, the utility of computational tools like SymPy is, unfortunately, only marginally beneficial.

```txt
Question: A square has a side length of 10 inches. Congruent isosceles right triangles are cut off each corner so that the resulting octagon has equal side lengths. How many inches are in the length of one side of the octagon? Express your answer as a decimal to the nearest hundredth.
```

```txt
[asy]   
size(150);   
draw((0,0)--(10,0)--(10,10)--(0,10)--(0,0),linewidth(0.7));   
draw((5\*sqrt(2)/(1  $^+$  sqrt(2)),0)--(0,5\*sqrt(2)/(1  $^+$  sqrt(2))),linewidth (0.7));   
draw((10-  $(5\star \mathrm{sqrt}(2) / (1 + \mathrm{sqrt}(2))$  ),0)--(10,5\*sqrt(2)/(1  $^+$  sqrt(2))), linewidth(0.7));   
draw((0,(10-(5\*sqrt(2)/(1+sqrt(2)))))-(5\*sqrt(2)/(1  $^+$  sqrt(2)),10), linewidth(0.7));   
draw((10-  $(5\star \mathrm{sqrt}(2) / (1 + \mathrm{sqrt}(2))$  ),10)-(10,10-5\*sqrt(2)/(1  $^+$  sqrt(2))，linewidth(0.7));   
[/asy]
```

```latex
Solution:  
The side length of the octagon is the hypotenuse of the right triangle formed by the side of the square and the diagonal of the octagon  
The hypotenuse is the sum of the side length of the square and the length of the diagonal of the octagon  
The length of the diagonal of the octagon is  $5 \times \sqrt{3} \times (2) / (1 + \sqrt{3} \times (2))$
```

```python
``python
from sympy import sqrt, Rational
def octagon_side_length():
    square_side_length = 10
        octagon_side_length = square_side_length + 5 * sqrt(2) / (1 + sqrt(2))
```

```txt
return round(octagon_side_length, 2)  
result = octagon_side_length()  
print(result)  
```
``output  
12.93  
``  
The length of one side of the octagon is $\\boxed{12.93}$ inches.
```

```latex
**Ground Truth**: Recall that the ratio of the hypotenuse to the leg of an isosceles right triangle is  $\mathbb{S}\backslash \mathrm{sqrt}\{2\} \mathbb{S}$ . Looking at the removed triangles, we see that the three segments that make up the side of square are  $\mathbb{S}\backslash \mathrm{sqrt}\{2\} \mathbb{S}$ ,  $\mathbb{S}\backslash \mathbb{S}$ , and  $\mathbb{S}\backslash \mathrm{sqrt}\{2\} \mathbb{S}$ , where  $\mathbb{S}\backslash \mathbb{S}$  is the side length of the octagon.
```

```latex
Setting the sum of these three side lengths equal to 10 inches, we find \begin{align*}  
\frac{\operatorname{frac}\{s\}{\sqrt{2}}\} + s + \frac{\operatorname{frac}\{s\}{\sqrt{2}}\}}{s \left( \frac{\operatorname{frac}\{1\} \left( \frac{\sqrt{2}}{2} \right)}{s \left( \frac{\sqrt{2}}{2} \right)} + 1 + \frac{\operatorname{frac}\{1\} \left( \frac{\sqrt{2}}{2} \right)}{s \left( \frac{\sqrt{2}}{2} \right)} \right. } \\ \text{s} = 10 \left( \frac{\operatorname{frac}\{1\} \left( \frac{\sqrt{2}}{2} \right)}{s \left( \frac{\sqrt{2}}{2} \right)} + 1 \right) = 10 (\sqrt{2}) - 1),  
\end{align*}  
\begin{align*}  
\text{s} = 10 \left( \frac{\operatorname{frac}\{1\} \left( \frac{\sqrt{2}}{2} \right)}{s \left( \frac{\sqrt{2}}{2} \right)} + 1 \right) = 10 (\sqrt{2}) - 1),  
\end{align*}  
\begin{align*}  
\text{s} = 10 (\sqrt{2}) - 1),  
\end{align*}  
\begin{align*}  
\text{s} = 10 (\sqrt{2}) - 1),  
\end{align*}  
\begin{align*}  
\text{s} = 10 (\sqrt{2}) - 1),  
\end{align*}  
\begin{align*}  
\text{s} = 150;  
defaultpen(lnewidth(0.7) + fontsize(10));  
real s = 10 / (1 + sqrt(2));  
draw((0,0) -- (10,0) -- (10,10) -- (0,10) -- (0,0));  
draw((5*sqrt(2) / (1 + sqrt(2)), 0) -- (0,5*sqrt(2) / (1 + sqrt(2)))));  
draw((10 - (5*sqrt(2) / (1 + sqrt(2))), 0) -- (10,5*sqrt(2) / (1 + sqrt(2)))));  
draw((0, (10 - (5*sqrt(2) / (1 + sqrt(2)))))) -- (5*sqrt(2) / (1 + sqrt(2)), 10));  
draw((10 - (5*sqrt(2) / (1 + sqrt(2))), 10) -- (10,10 - 5*sqrt(2) / (1 + sqrt(2)))));  
label("\\(s", (10-s/ (2*sqrt(2)), 10-s/ (2*sqrt(2))), SW);  
label("\\displaystyle{\frac{s}{\sqrt{2}}} & \\ label("\\displaystyle{\frac{s}{\sqrt{2}}} & \\ label("\\displaystyle{\frac{s}{\sqrt{2}}} & \\ label("\\displaystyle{\frac{s}{\sqrt{2}}} & \\ label("\\displaystyle{\frac{s}{\sqrt{2}}} & \\
```

Listing 8: Failure case for ToRA: Inappropriate Tool Use.  
```python
Idx: 2489
Type: Intermediate Algebra
Level: Level 5
Question: 'Let $z_1,$ $z_2,$ $z_3$ be complex numbers such that $|z_1| = 2,$ $|z_2| = 3,$ and $|z_3| = 4.$ Find the largest possible value of\ n\ [|z_1 - z_2|^2 + |z_1 - z_3|^2 + |z_2 - z_3|^2 \]'
Solution:
``python
from sympy import symbols, simplify, Max
from sympy/functions import Abs
def max_expression_value():
    '''Let $z_1,$ $z_2,$ $z_3$ be complex numbers such that $|z_1| = 2,$ $|z_2| = 3,$ and $|z_3| = 4.$ Find the largest possible value of \[\left[ \left|z_1 - z_2\right|^2 + \left|z_1 - z_3\right|^2 + \left|z_2 - z_3\right|^2 \right]\]''' # Define the variables
    z1, z2, z3 = symbols('z1 z2 z3')
    # Define the expression
    expr = Abs(z1 - z2) ** 2 + Abs(z1 - z3) ** 2 + Abs(z2 - z3) ** 2
    # Define the conditions
    conditions = [Abs(z1) - 2, Abs(z2) - 3, Abs(z3) - 4]
    # Find the maximum value of the expression under the conditions
    max_value = Max(expr, *conditions)
    # Simplify the maximum value
    simplified_max_value = simplify(max_value)
    return simplified_max_value
```

```txt
result = max_expression_value()
print(result)
...
...
```
```
```
output
Max(Abs(z1) - 2, Abs(z2) - 3, Abs(z3) - 4, Abs(z1 - z2) **2 + Abs(z1 - z3) **2 + Abs(z2 - z3) **2)
...
The largest possible value of $|z_1 - z_2|^2 + |z_1 - z_3|^2 + |z_2 - z_3| ^2$ is $\boxed{25}$
```
Execution: None
Prediction: '25'
Ground Truth: '87'
Score: [False]
Reference CoT: 'We have that\n\begin{align*}\n{z_1} \end{align*}
z_1|^2, \{\n{z_2} \} overline{z_{1}}^2 & = |z_2|^2, \{\n{z_3} \} overline{z_{1}}^2
z_3 & = |z_3|^2. \n\end{align*}
z_1 - z_2
|z_1 - z_3|^2 + |z_2 - z_3|^2
|z_1 - z_2)
+ (z_1 - z_3)
(\\overline{z_{1} - z_{3}})
(\\overline{z_{1} - z_{3}})
(\\overline{z_{1} - z_{3}})
(\\overline{z_{1} - z_{3}})
(\\overline{z_{1} - z_{3}})
(\\overline{z_{1} - z_{3}})
(\\overline{z_{1} - z_{3}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{1}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{3}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{4}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{5}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{6}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{7}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{8}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{9}})
(\\overline{z_{10}}
| \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\underline{n} | \n\uberm
Z = {0, Z; nztwo
= (3/4, 3*sqrt[15]/4); \nzthree = (-11/4, -3*sqrt[15]/4); \nndraw (Circle((0, 0), 2), red); ndraw(Circle((0, 0), 3), green); ndraw(Circle((0, 0), 4), blue); ndraw(zone--ztwo--ztthree--cycle); \nndot("$z_1$", zone, E);
\ndot("$z_2$", ztwo, N); ndot("$z_3$", zthree, SW); \n[asy] \n
nAlternative: For equality to occur, we must have $z_1 + z_2 + z_3 = 0.
$. Without loss of generality, we can assume that $z_1 = 2.
$. Then
$z_2 + z_3 = -2.
$. Let $z_2 = x + iy$
so that $z_3 = -x - 2 - iy,
where $x$ and $y$
are real numbers. We need\n\begin{align*}
| z_2
| ^2 = x^2 + y^2 & = 9 \n\end{align*}
| z_3 | ^2 = (x + 2)^2 + y^2 & = 16. \n\end{align*}
Subtracting the first equation from the second, we get $4x + 4
= 7.
$, or $x = \dfrac34.
One solution is $z_2 = \dfrac34 + i\cdfrak{x}
dfrac\{3\backslash sqrt\{15\}\} \{4\}$
and $z_3 = -\dfrac\{11\}4 + i\cdfrak{x}
dfrac\{3\backslash sqrt\{15\}\} \{4\}.
$. This example shows that equality is possible, so the
maximum value is $\boxed{\textbf{xbox}\textbf{\textit{xbox}}}$.
```

Listing 9: Failure case for ToRA: Syntax Error.  
```javascript
Idx: 106
Type: Algebra
Level: Level 5
Question: 'Below is a portion of the graph of a function, $y=h(x)$:\n\nasy]\nimport graph; size(8cm); real lsf=0.5; pen dps=linewidth(0.7)+fontsize(10); defaultpen(dps); pen ds=black; real xmin=-0.75,xmax=8.25,ymin=-1.25,ymax=10.25;\n\npen cqcqcq=rgb(0.75,0.75,0.75);\n\ngrid\*/ pen gs=linewidth(0.7)+cqcqcq+linetype("2 2"); real gx=1,gy=1;\n
```

```txt
nfor(real i=ceil(xmin/gx) *gx; i<=floor(xmax/gx) *gx; i+=gx) draw((i, ymin) -- (i, ymax), gs); for(real i=ceil(ymin/gy) *gy; i<=floor(ymax/gy) *gy; i+=gy) draw((xmin, i) -- (xmin, i), gs); \n\nLabel laxis; laxis.p=fontsize(10); \n\nxaxis("", xmin, xmax, Ticks (laxis, Step=1.0, Size=2, NoZero), Arrows(6), above=true); yaxis "", ymin, ymax, Ticks (laxis, Step=1.0, Size=2, NoZero), Arrows(6), above=true); \nreal f1(real x){return (x-0.5) * (x-2.5) * (x-6.5) * (x-7.5)/16+x; }\ndraw(graph(f1,-0.25,8.25), linewidth(1)); \nclip((xmin, ymin) -- (xmin, ymax) -- (xmax, ymax) -- (ymax, ymin) -- cycle); \nlabel "$y=h(x)$", (8.5, 8), E); \n[/asy]\n\nWhat is the sum of all integers $x$ in the interval shown ($0\leq x\leq 8$) such that $h(x)>x$ ?'
```

```python
Solution:
``python
from sympy import symbols, solve
def sum_of_x():
    '''Below is a portion of the graph of a function, $y=h(x)$:
[asy]
import graph; size(8cm); real lsf=0.5; pen dps=linewidth(0.7)+fontsize (10); defaultpen(dps); pen ds=black; real xmin=-0.75,xmax=8.25,ymin=-1.25,ymax=10.25;
```

```txt
pen cqcqcq=rgb(0.75,0.75,0.75);   
/\*grid\*/ pen gs  $\equiv$  linewidth(0.7)+cqcqc+linetype("2 2"); real  $\mathrm{gx} = 1,\mathrm{gy} = 1;$  for(real i  $\equiv$  ceil(xmin/gx)*gx;  $i <   =$  floor(xmax/gx)*gx;i+=gx) draw((i,ymin)--(i, ymax),gs); for(real i  $\equiv$  ceil(ymin/gy)*gy;i  $\coloneqq$  floor(ymax/gy)*gy;i+=gy) draw((xmin,i)-(xmax,i),gs);
```

```txt
Label laxis; laxis.p=fontsize(10);  
xaxis("",xmin, xmax, Ticks(laxis, Step=1.0, Size=2, NoZero), Arrows(6), above=true); yaxis("",ymin, ymax, Ticks(laxis, Step=1.0, Size=2, NoZero), Arrows(6), above=true);  
real f1(real x) {return (x-0.5) * (x-2.5) * (x-6.5) * (x-7.5) / 16+x;}  
draw(graph(f1, -0.25, 8.25), linewidth(1));  
clip((xmin, ymin) -- (xmin, ymax) -- (xmax, ymax) -- (xmax, ymin) -- cycle);  
label("\\\$y=h(x)\)", (8.5, 8), E);  
[/asy]
```

```python
What is the sum of all integers $x$ in the interval shown ($0\le x\le 8$) such that $h(x)>x$ ?"""
x = symbols('x')
h_x = (x - 0.5) * (x - 2.5) * (x - 6.5) * (x - 7.5) / 16 + x
# Find the x-values where h(x) > x
x_values = [x for x in range(9) if h_x.subs(x, x) > x]
# Calculate the sum of these x-values
sum_x = sum(x_values)
return sum_x
```

```txt
result = sum_of_x()
print(result)
```
```
```
output
```

```txt
TypeError: cannot determine truth value of Relational
```

```txt
The code above has an error. The `if` statement can only be used with boolean values, but the `h_x.subs(x, x) > x` comparison is not a boolean value. We need to find a different way to solve this problem.
```

```txt
Let's analyze the problem. We are asked to find the sum of all integers $x$ in the interval shown ($0\leq x\leq 8$) such that $h(x)>x$.
```

We can see that $h(x)$ is a parabola that opens upwards. For a parabola that opens upwards, the maximum value of the parabola will occur at the midpoint of the interval, which is $x = \frac{\partial}{\partial x} \{0 + 8\} \{2\} = 4$.

For $x < 4$, $h(x)$ is below the line $y = x$, so $h(x) \leq x$. For $x > 4$, $h(x)$ is above the line $y = x$, so $h(x) > x$.

Therefore, we only need to find the sum of the integers $x$ in the interval $0 \leq x \leq 4$ such that $h(x) > x$.

Here's the corrected code:

``python
``python
from sympy import symbols, solve

```python
def sum_of_x:
    ""
Find the sum of all integers x in the interval shown ($0\leq x\leq 8$)
    such that h(x)>x.
Returns:
    int: The sum of all integers x in the interval shown ($0\leq x\leq 8$)
    ) such that h(x)>x.
    ""
x = symbols('x')
h_x = (x - 0.5) * (x - 2.5) * (x - 6.5) * (x - 7.5) / 16 + x
# Find the x-values where h(x) > x and 0 <= x <= 4
x_values = [x for x in range(5) if h_x.subs(x, x) > x]
# Calculate the sum of these x-values
sum_x = sum(x_values)
return sum_x
```

result = sum_of_x()
print(result)

```
``'output
SyntaxError: invalid decimal literal

The code above still has an error. The `h_x.subs(x, x) > x'` comparison is not a boolean value. We need to find a different way to solve this problem.

Let's analyze the problem. We are asked to find the sum of all integers $x$ in the interval shown ($0\leq x\leq 8$) such that $h(x)>x$.

We can see that $h(x)$ is a parabola that opens upwards. For a parabola that opens upwards, the maximum value of the parabola will occur at the midpoint of the interval, which is $x = \frac{\partial}{\partial x} \{0 + 8\} \{2\} = 4$.

For $x < 4$, $h(x)$ is below the line $y = x$, so $h(x) \leq x$. For $x > 4$, $h(x)$ is above the line $y = x$, so $h(x) > x$.

Therefore, we only need to find the sum of the integers $x$ in the interval $0 \leq x \leq 4$ such that $h(x) > x$.

Here's the corrected code:

``python
``python
def sum_of_x():

```markdown
```
Find the sum of all integers x in the interval shown ($0\leq x\leq 8$)
such that h(x)>x.
Returns:
    int: The sum of all integers x in the interval shown ($0\leq x\leq 8$)
) such that h(x)>x.
```
# Define the interval
interval = range(5)
# Find the x-values where h(x) > x
x_values = [x for x in interval if x * (x - 4) * (x - 6) * (x - 7) / 16 > x]
# Calculate the sum of these x-values
sum_x = sum(x_values)
return sum_x
result = sum_of_x()
print(result)
```
```
输出
SyntaxError: invalid decimal literal
```
Reach max function call limit.
Execution: None
Prediction: 'SyntaxError:invaliddecimalliteral'
Ground Truth: '26'
Score: [False]
Reference CoT: 'We could check $h(x)$ separately for each integer $x$
from $0$ to $8$: for example, $h(0)\approx approx 3.8$, so $h(0)>0$, but $h
(1)\approx approx -0.7$, so $h(1)\not>1$, and so on.\nHowever, it is
easier to see at a glance which $x$ satisfy $h(x)>x$ by superimposing
a graph of $y=x$ on the graph of $y=h(x)$:\n\ndraw
((-0.75,-0.75)--(8.25,8.25),red+1);\nimport graph; size(8cm); real
1sf=0.5; pen dps=linewidth(0.7)+fontsize(10); defaultpen(dps); pen ds
=black; real xmin=-0.75,xmax=8.25,ymin=-1.25,ymax=10.25;\n\npen
cqcqcq=rgb(0.75,0.75,0.75);\n\nreal xmin=-0.75,0.75);\n\ngrid* / pen gs=linewidth(0.7)+cqqcq+
linetype("2 2"); real gx=1,gy=1;\nfor(real i=ceil(xmin/gx)*gx;i<= floor(xmax/gx)*gx;i+=gx) draw((i,ymin)--(i,ymax),gs); for(real i=ceil
(ymin/gy)*gy;i<=floor(ymax/gy)*gy;i+=gy) draw((xmin,i)--(xmin,i),gs);\n\nlLabel laxis; laxis.p=fontsize(10);\n\nxaxis,Step=1.0,Size=2,NoZero),Arrows(6),above=true); yaxis("",ymin,
ymax,Ticks(laxis,Step=1.0,Size=2,NoZero),Arrows(6),above=true);\nreal
f1(real x){return (x-0.5)* (x-2.5)* (x-6.5)* (x-7.5)/16+x};\ndraw(graph
(f1,-0.25,8.25),linewidth(1));\nclip((xmin,ymin)--(xmin,ymax)--(xmin,
ymax)--(xmin,ymin)--cycle);\nlabel "$y=h(x)$", (8.5,8),E);\ndot((0,0),
blue); dot((3,3),blue); dot((4,4),blue); dot((5,5),blue); dot((6,6),
blue); dot((8,8),blue);\nThe six blue dots above mark the
integer points $(x,x)$ which lie below the graph of $y=h(x)$,
indicating that $h(x)>x$. Their $x$ coordinates are $0,3,4,5,6,8,$
which add up to $\\boxed{26}$.'
```

Listing 10: Failure case for ToRA: Runtime Error. ToRA failed to fall back from program timeout to a rationale-based solution. For Intermediate Algebra and Precalculus problems like this example, direct brute-force solutions are often infeasible, resulting in exceptions like timeout. Addressing these challenges requires complex reasoning over math expressions, and may involve sophisticated problem-solving and proof techniques like forward and backward reasoning (Jiang et al., 2023), as well as result verification (Gou et al., 2023; Zhou et al., 2023a).

Idx: 2261

```txt
Type: Intermediate Algebra
Level: Level 3
Question: 'For some positive integers $m$ and $n,$\n[\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][\sqrt[4][
```

Listing 11: Failure case for TORA: Rationale-only Error.

```txt
Idx: 2638
Type: Intermediate Algebra
Level: Level 1
Question: 'The graph of $y = f(x)$ is shown below.\n\n[asy]\nunitsize(0.5 cm);\n\nreal func(real x) \{n real y; n if (x >= -3 && x <= 0) \{y = -2 - x; \}n if (x >= 0 && x <= 2) \{y = sqrt(4 - (x - 2)^2) - 2; \}n if (x >= 2 && x <= 3) \{y = 2*(x - 2); \}n return(y); \}n\nint i, n; n\nfor (i = -5; i <= 5; ++i) \{n draw((i, -5) -- (i, 5), gray(0.7)); \}n draw((-5, i) -- (5, i), gray(0.7)); \}n\nndraw(-5, 0) -- (5, 0), Arrows(6)); \}ndraw((0, -5) -- (0, 5), Arrows(6)); \}n\nlabel("\\(x\$", (5, 0), E); \)\\(y\$", (0, 5), N); \)\\(x\$", (3, 5), N"; \)\\(y\$", (3, -2), UnFill); \)\\(x\$", (3, -2), UnFill); \)\\(y\$", (3, -2), UnFill); \)\\(x\$", (3, -2), UnFill); \)\\(y\$", (3, -2), UnFill); \)\\(x\$", (3, -2), UnFill); \)\\(y\$", (3, -2), UnFill); \)\\(x\$", (3, -2), UnFill); \)\\\(y\$", (3, -2), UnFill); \)\\(x\$", (3, -2), UnFill); \)\\(y\$", (3, -2), UnFill); \)\\(x\$", (3, -2), UnFill); \)\\(y\$", (3, -2), UnFill); \)\\(x\$", (3, -2), UnFill); \)\\(y\$", (\)0.5\) )\}n\nndraw(real x) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{nreturn(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x)) \{n return(-f(x) \()\) [graph(n)], (i, -5) -- (i, 5), gray(0.7)); \}n draw(graf[n], (-5, i) -- (5, i), gray(0.7)); \}n\ndraw(graf[n], (-5, 0) -- (5, 0), Arrows(6)); \}n draw(graf[n], (0, -5) -- (0, 5), Arrows(6)); \}n\nlabel(graf[n], "\\(x\$", (5, 0), E); \)\\(x\$", (0, 5), N); \)\\(x\$", (0, 5), N"; \)\\(x\$", (0, 5), N"; \)\\(x\$", (0, 5), N"; \)\\(x\$", (0, 5), N"; \)\\(x\$", (0, -5), red); \)\\(x\$", (0, -5), red; \)\\(x\$", (0, -5), red; \)\\(x\$", (0, -5), red; \)\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5),red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (0, -5), red; \)\\\(x\$", (12, 0)) * (\text{graf}[2]);\nadd(\text{shift}((24, 0)) * (\text{graf}[3]);\nadd(\text{shift}((6, -12)) * (\text{graf}[4]))\nadd(\text{shift}((18, -12)) * (\text{graf}[5]))\n[\)/asym] \\nEnter the letter of the graph of $y = -f(x).$' Solution: The correct graph is the one that reflects the original graph y = f(x) about the y-axis. This is graph D.
```

```txt
So, the answer is $ \boxed{\text{D}}$. Execution: None Prediction: 'D' Ground Truth: 'B' Score: [False] Reference CoT: 'The graph of $y = -f(x) $ is the reflection of the graph of $y = f(x) $ in the $x$-axis. The correct graph is $ \boxed{\text{B}}$.
```

Listing 12: Failure case for TORA: False Negative.  
```python
Idx: 2228  
Type: Intermediate Algebra  
Level: Level 3  
Question: 'Find all the real roots of \n[ [(12x - 1)(6x - 1)(4x - 1)(3x - 1) = 5. ] Enter all the real roots, separated by commas.'  
Solution:  
``python  
from sympy import symbols, solve, simplify  
def find_real Roots():  
    '''Find all the real roots of  
[ [(12x - 1)(6x - 1)(4x - 1)(3x - 1) = 5. ] Enter all the real roots,  
    separated by commas.'  
    x = symbols('x')  
    equation = (12*x - 1) * (6*x - 1) * (4*x - 1) * (3*x - 1) - 5
```

```txt
roots = solve(equation, x)
real_roots = [root for root in roots if root.is_real]
return real_roots
result = find_real_roots()
print(result)
...
output
[-1/12, 1/2]
The real roots of the equation are $\\boxed{-\\frac{1}{12}, \\\frac{1}{2}\} \\{2}\}.
Execution: None
Prediction: '- $\frac{1}{12}, \frac{1}{2}$ 'Ground Truth: ' $\frac{1}{2}, -\frac{1}{12}$ 'Score: [False]
Reference CoT: 'We can expand using the pairs $(3x-1)(12x-1)=36x^2-15x+1$ and $(6x-1)(4x-1)=24x^2-10x+1,$ so\n\[(36x^2-15x+1)(24x^2-10x+1)=5.\]Let $y=12x^2-5x.$ Then\n\[(3y+1)(2y+1)=5.\]This simplifies to $6y^2+5y-4=0,$ which factors as $(2y-1)(3y+4)=0.$ Hence, $y=\\frac{1}{2}\$ or $y=-\{\frac{4}{3}\}$ and factors as\n\{\{3\}.\\n\nIf $12x^2-5x=\\frac{1}{2}\}$ then $24x^2-10x-1=0,$ which factors as\n\[(2x-1)(12x+1)=0.\]Hence, $x=\\frac{1}{2}\$ or $x=-\{\frac{4}{3}\}$ and factors as\n\{\{12\}.\\n\nIf $12x^2-5x=-\{\frac{4}{3}\}$ then\n\{\{3\},\\nThen\n\{\{36x^2-15x+4=0,\}which has no real solutions.\n\nTherefore, the real roots are \$\\boxed{\{\frac{1}{2}\}, -\{\frac{1}{2}\}}.
\]
```

# Footnotes:

Page 0: *Equal contribution. See Contributions section for details. Work done during an internship at Microsoft. †Corresponding authors. $^{1}$ Code and models are available at https://github.com/microsoft/ToRA. 
Page 5: $^{2}$ https://www.sympy.org <sup>3</sup>https://huggingface.co/spaces/HuggingFaceH4/open_11m_leaderboard 
