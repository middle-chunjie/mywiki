# Black-Box Prompt Optimization: Aligning Large Language Models without Model Training

Jiale Cheng $^{1,2*}$ , Xiao Liu $^{3,2*}$ , Kehan Zheng $^{1}$ , Pei Ke $^{1}$ , Hongning Wang $^{1}$ , Yuxiao Dong $^{3}$ , Jie Tang $^{3}$ , Minlie Huang $^{1}$

<sup>1</sup>The CoAI Group, Tsinghua University

$^{2}$ Zhipu AI

<sup>3</sup>The Knowledge Engineering Group (KEG), Tsinghua University

chengjl23@mails.tsinghua.edu.cn, shawliu9@gmail.com

# Abstract

Large language models (LLMs) have shown impressive success in various applications. However, these models are often not well aligned with human intents, which calls for additional treatments on them, that is, the alignment problem. To make LLMs better follow user instructions, existing alignment methods mostly focus on further training them. However, the extra training of LLMs are usually expensive in terms of GPU compute; worse still, LLMs of interest are oftentimes not accessible for user-demanded training, such as GPTs. In this work, we take a different perspective—Black-Box Prompt Optimization (BPO)—to perform alignments. The idea is to optimize user prompts to suit LLMs' input understanding, so as to best realize users' intents without updating LLMs' parameters. BPO is model-agnostic and the empirical results demonstrate that the BPO-aligned ChatGPT yields a  $22\%$  increase in the win rate against its original version, and  $10\%$  for GPT-4. Importantly, the BPO-aligned LLMs can outperform the same models aligned by PPO and DPO, and it also brings additional performance gains when combining BPO with PPO or DPO. Code and datasets are released at https://github.com/thu-coai/BPO.

# 1 Introduction

Recently, the field of Natural Language Processing (NLP) has made remarkable progress, largely thanks to the advent of LLMs (Brown et al., 2020b; Chowdhery et al., 2022; Zhang et al., 2022; Zeng et al., 2022; Touvron et al., 2023). After elaborate alignment (Gabriel, 2020), these models have demonstrated a strong ability of instruction-following and human preference understanding, yielding products like ChatGPT (OpenAI, 2022) that have attracted widespread attention.

Figure 1: (Upper) Two directions of LLM alignment: Black-box Prompt Optimization (BPO) and Learning from Feedback (PPO, DPO). BPO offers a conceptually new perspective to bridge the gap between humans and LLMs. (Lower) On Vicuna Eval's pairwise evaluation, we show that BPO further aligns gpt-3.5-turbo and claude-2 without training. It also outperforms both PPO & DPO and presents orthogonal improvements.

However, aligning LLMs to human preferences is not trivial. The major challenge lies in narrowing the gap between human intents (conveyed by prompts) and LLMs' understanding of them. Significant effort has been focused on steering LLMs to approach human preference, including reinforcement learning from human feedback (RLHF) (Ouyang et al., 2022), reinforcement learning from AI feedback (RLAIF) (Bai et al., 2022b; Lee et al., 2023), or Direct Preference Optimization (DPO) (Rafailov et al., 2023). Nevertheless, these methods suffer from various deficiencies:

- Efficiency: As LLMs grow larger, it becomes far more expensive and difficult to train these models, especially when using notoriously unstable RL algorithms for the purpose.  
- Accessibility: As most best-performing LLMs,

such as GPT-4 (OpenAI, 2023) and Claude-2 (Anthropic, 2023a), are close-sourced and only can be accessed by API, these training-based methods are not applicable for users outside the organization to enhance alignment.

- Interpretability: The modeling and exact consequent improvements of human preference are uninterpretable when using these approaches.

Distinct from the aforementioned alignment methods, we propose to steer human prompts to accommodate LLMs' understanding. Whilst the idea is closely related to "prompt engineering", its automated prototypes would trace back to AutoPrompt (Shin et al., 2020) and prompt tuning (i.e., P-Tuning) (Liu et al., 2021; Lester et al., 2021), where prompts are optimized to improve task performance without training the LMs. Our new alignment method, Black-box Prompt Optimization (BPO), presents an efficient and interpretable paradigm that aligns LLMs without modifying these models. The central idea behind BPO is to create an automatic prompt optimizer that rewrites human prompts, which are usually less organized or ambiguous, to prompts that better deliver human intent. Consequently, these prompts could be more LLM-preferred and hence yielding better human-preferred responses.

In BPO, the prompt preference optimizer is learnt from preference comparisons. We curated a subset of publicly available SFT datasets with either human or AI preferences. Each instance of our training data contains a prompt along with a pair of favorable response and unfavorable response. We then employ LLMs to delineate and criticize the paired responses, and subsequently ask the LLMs to refine the input prompt to explicitly incorporate the features that shift the responses from unfavorable to favorable. In this way, we constructed 14K pairs of the original instruction and its optimized version to train a sequence to sequence model that optimizes user instructions.

Our extensive experiments demonstrate that without LLM training, BPO can improve the alignment of both API-based and open-sourced LLMs: increasing win rates by  $8.8\%$  to  $22.0\%$  on gpt-3.5-turbo, gpt-4, claude-2, llama-2-chat, vicuna and etc. Moreover, we show that BPO not only outperforms RLHF via PPO (Schulman et al., 2017) and DPO (Rafailov et al., 2023), but also further improves LLMs' alignment after these RLHF's training. We also show

that BPO can align LLMs in supervised fine-tuning by optimizing response quality in the experiment of Alpaca.

Our contributions can be summarized as follows:

- We propose a black-box prompt optimization method BPO, which enhances LLMs' alignment to human preferences without training these models, demonstrating improvements over a wide variety of LLMs including API-based and open-sourced ones.  
- We empirically justify that BPO is a novel and competitive alignment approach, in addition to existing RLHF and preference learning methods, outperforming both PPO and DPO on extensive experiments. Moreover, we show that it is orthogonal to RLHF's alignment, which adds additional gain on top of conventional alignment pipelines.  
- We systematically analyze how BPO refines the original prompts, from the perspectives of prompt explanation, clarification, enrichment, and safety enhancement. We demonstrate its better interpretability than existing preference learning algorithms when aligning LLMs.

# 2 Related Work

LLMs pre-trained on massive corpus can generate fluent text, but they are not well aligned to follow users' instructions and preferences. Therefore, aligning LLMs with human intents has become an important research problem. Existing efforts in alignment mostly follow the paradigm proposed by (Ouyang et al., 2022), consisting of two main stages, namely SFT and RLHF.

Supervised Fine-tuning (SFT). SFT alignment endows LLMs with preliminary instruction-following abilities. Nonetheless, it heavily relies on abundant high-quality fine-tuning data. Inasmuch as the high cost of human-written data, self-instruct data augmentation (Wang et al., 2022) based on a small human-created seed set has become a predominant approach in academia, and is later adopted by several existing open-sourced SFT LLMs (Taori et al., 2023; BELLEGroup, 2023). Sun et al. (2023) further explored iterative SFT alignment with just the model itself. However, SFT alignment still suffers from hallucinations, inferior scalability, and poor understanding of human preference.

Reinforcement Learning from Human Feedback (RLHF). RLHF alignment is proposed to further align LLMs with scalable feedback. The standard

Alignment Stage of Black-box Prompt Optimization (BPO)

# BPO is Applicable for Various LLMs

Figure 2: BPO consists of three main steps: collecting feedback data (we adopt open-sourced feedback data), constructing prompt optimization pairs based on the feedback data, and building a prompt optimization model using these pairs. In this way, BPO serves as a translator between human and AI, by optimizing human prompts to be better suited for AI generation to get human-preferred responses, while treating the model itself as a black box.

User Prompt (in production):  
Tell me about Harry Potter


Prompt Preference Optimizer

# BPO Optimized Prompt:

Provide a comprehensive overview of the Harry Potter franchise, including the books, movies, characters, themes, and impact. Be accurate and informative in your response.



API-based & OSS Large Language Models


Human Preference

framework (Stiennon et al., 2020; Ouyang et al., 2022) consists of two steps: reward modeling (RM) and policy training. Bai et al. (2022b) utilize LLMs to generate AI feedback to reduce the cost of reward model training. Lee et al. (2023) demonstrate that RL from feedback of AIs could perform similarly as those of humans. However, considering the cumbersome procedures and unstable RL training, some works have been seeking for other methods beyond RLHF to learn from preference feedback. DPO (Rafailov et al., 2023) introduces feedback into the design of loss function.

Prompt Engineering and Prompt Tuning. Since the pre-trained language models are proposed, researchers found that prompts have a significant impact on downstream task performance. Especially after GPT-3 (Brown et al., 2020a) demonstrated surprising zero-shot and few-shot capabilities, leveraging prompt tuning to accomplish NLP tasks have become a new paradigm.

There are two main types of prompt tuning: hard and soft. Hard prompt tuning, also known as prompt engineering, often requires extensive manual effort. Therefore, many works explore how to automate this process, which can be traced back to AutoPrompt (Shin et al., 2020). Recently, with the advent of LLMs, utilizing them for automated prompt engineering becomes possible. Zhou et al. (2022) proposed the APE algorithm, leveraging LLM to generate and select prompts, which

achieved near human-level prompt engineering performance on 24 NLP tasks. Yang et al. (2023) demonstrated strong prompt optimization capabilities of LLMs through an RL-like algorithm, achieving remarkable improvements on BIG-Bench Hard (Suzgun et al., 2022). Soft prompt tuning (Liu et al., 2021; Lester et al., 2021; Li and Liang, 2021) further improves effectiveness by enabling optimization in the embedding space rather than limited token vocabulary. Soft prompt tuning is faster and more efficient than full parameter tuning while achieving comparable performance.

Prompt tuning and model training have been two parallel ways to improve pre-trained model performance. And current alignment strategies all focus on adjusting models to follow user intents and instructions. Under the context of LLMs, models have become huge and difficult to train or even obtain (e.g. api-based models). Therefore, we argue that prompt optimization desires its attention, and LLM alignment can also be achieved by optimizing the input prompt, without modifying the LLMs.

# 3 Black-Box Prompt Optimization

The overall process of BPO is shown in Figure 2. BPO is to enhance the alignment between model output and human preference by optimizing the input prompt. To this end, we first collect several instruction-tuning datasets with human preference annotations, carefully curate and filter low-quality

data. Subsequently, we employ an LLM to capture the difference between responses favored and disfavored by human, based on which we leverage the LLM to refine the input. We then get a pair of original instruction and its improved version, using which we further train a sequence-to-sequence model to automatically optimize user inputs.

# 3.1 Intuitive Rationale

Bridging the alignment gap between human intention and LLMs has been a central problem since LLM's birth. In the era of GPT-3 (Brown et al., 2020a), prompt engineering (Reynolds and McDonell, 2021), together with automated prompt search (Shin et al., 2020) and prompt tuning (i.e., P-Tuning) (Liu et al., 2021; Lester et al., 2021; Li and Liang, 2021), have thrived to serve as task-specific aligners. However, they are later substituted by instruction tuning (Wei et al., 2022; Sanh et al., 2022) and current RLHF approaches (Ouyang et al., 2022; Bai et al., 2022b), which align LLMs to human preference in the way that trains LLMs to free users from arduous prompting.

Notwithstanding, the training-based alignment is not the only solution. Essentially, the alignment gap could be narrowed from both directions: either tuning LLMs to approach human preference, or changing human prompts to cater for LLMs' prompt understanding. For example, in Figure 2, we present a typically user prompt—"Tell me about Harry Potter" that may result in a brief LLM response. For a more detailed and informative response, while RLHF can help by training the LLM, we can accomplish the same goal by revising the user prompt to "Provide a comprehensive overview of the Harry Potter franchise, including the books, movies, characters, themes, and impact. Be accurate and informative in your response".

More practically, as LLMs become increasingly large and only accessible via APIs, training-based alignment prohibits small companies and individual developers to cheaply and handily align LLMs as they wish. Preference-aware prompting, on the contrary, is efficient, non-intrusive, and more explainable compared to learning from feedback methods. Therefore, we firmly believe that automatic preference-aware prompting is a promising complementary to RLHF in the LLM alignment, and thus present BPO as the very first framework in this direction.

# 3.2 Task Definition

As discussed above, our task is to optimize user input to help LLMs generate better responses. Formally, we denote user input as  $X_{user}$ . Our goal is to build a function  $F$  that maps  $X_{user}$  to its optimized version, denoted as  $X_{opt}$ . In order to get this, we introduce annotated human preferences, as the preferred response indicates good model output, while the other one suggests inferior output. By capturing the differences between these preference data, we can incorporate the attributes human favor into user instructions to make them more aligned with what LLMs can do, and thus bring LLMs' outputs better into alignment with human preferences. Inspired by recent work utilizing LLMs as evaluators (Wang et al., 2023; Zheng et al., 2023), we believe that LLMs possess the capacity to understand different features within various responses. Consequently, we choose to leverage LLMs to get  $X_{opt}$ . Specifically, each sample is represented as  $(X_{user}, Y_{good}, Y_{bad})$ , where  $Y_{good}$  stands for the favorable response and  $Y_{bad}$  is for the unfavorable one. Thus the prompt optimization process with LLM can be expressed as  $X_{opt} = LLM(X_{user}, Y_{good}, Y_{bad})$ . Finally, we build the  $F$  function by training a smaller sequence to sequence model over the pairs of  $(X_{user}, X_{opt})$ .

# 3.3 Training Data Construction

To construct the optimized prompts, we begin with collecting datasets with human preferences. In total, we employ 4 instruction-tuning datasets with human preference annotations, as shown in Table 1. The OASST1 (Köpf et al., 2023) dataset is a crowd-sourced instruction dataset with human annotated response quality ratings. Under each instruction, we choose the response with the highest score as the good response and the one with the lowest score as the bad response. The hh-rlhf (Bai et al., 2022a) dataset contains human preference over the responses' helpfulness and harmfulness. And the Chatbot Arena Conversations (Zheng et al., 2023) dataset is collected from human on the Chatbot Arena leaderboard<sup>1</sup> platform. In addition, we use the comparison data subset of the Alpaca-GPT4 (Peng et al., 2023) dataset, where the preference is generated by GPT4 (OpenAI, 2023). We only keep samples where gpt-4 outperforms text-davinci-003 to ensure data quality. After

<table><tr><td rowspan="2">Dataset</td><td colspan="2">Sampled</td><td colspan="2">Generating &amp; Filtering</td></tr><tr><td>Number</td><td>Distinct-4↑</td><td>Number</td><td>Distinct-4↑</td></tr><tr><td>OASST1</td><td>3000</td><td>0.953</td><td>2940</td><td>0.963</td></tr><tr><td>HH-RLHF</td><td>2000</td><td>0.957</td><td>1961</td><td>0.957</td></tr><tr><td>Chatbot Arena</td><td>5000</td><td>0.804</td><td>4494</td><td>0.899</td></tr><tr><td>Alpaca-GPT4</td><td>5000</td><td>0.938</td><td>5000</td><td>0.938</td></tr><tr><td>Overall</td><td>15000</td><td>0.860</td><td>14395</td><td>0.913</td></tr></table>

collecting and reformating these datasets, we carefully eliminate low-quality instances with manually crafted rules (e.g. too short instructions tend to be low quality) and use self-bleu to perform a strict diversity filtering. Finally, we get 14k diverse samples in the format of  $(X_{user},Y_{good},Y_{bad})$ . In this work, we mainly focus on single-turn response generation, and leave the multi-turn setting to our future work.

Subsequently, we leverage ChatGPT (OpenAI, 2022) to refine these instructions. After meticulous prompt engineering efforts, we employ two types of prompts for different data formats as illustrated in Appendix A. Then we conduct quality filtering by rule-based methods to drop wrong optimizations (e.g. wrong format). Following the whole procedure, our dataset comprises about  $14\mathrm{k}$  pairs of instruction before and after optimization, with the final distribution shown in Table 1. And the overall distinct score (Li et al., 2016) demonstrates the high diversity of our dataset.

# 3.4 Model Training

Based on the constructed dataset, we learn a small sequence-to-sequence model to automatically optimize user instruction. Formally, we generate  $X_{opt}$  conditioned on the given input  $X_{user}$ , where the loss function is specified as,

$$
\mathcal {L} = - \frac {1}{N} \sum_ {t = 1} ^ {N} \log P \left(x _ {t} \mid X _ {\text {u s e r}}, x _ {<   t}\right) \tag {1}
$$

where  $N$  is the length of  $X_{opt}$  and  $x_{t}$  represents the  $t$ -th token in  $X_{opt}$ . In this work, we choose to use 11ama2-7b-chat as the backbone model, as we believe a stronger model can learn the implicit preference mapping between  $X_{user}$  and  $X_{opt}$  better. Meanwhile, the number of parameters in a 7B model is small among LLMs, which can be more efficient for training and inference. And we leave the model scaling explorations to future work.

Table 1: Preference data statistics. We sampled prompts from open-sourced prompt datasets and filter them to form the preference training dataset.  

<table><tr><td>Method</td><td>Reward -free</td><td>Policy -free</td><td>LLM -agnostic</td><td>Task -agnostic</td></tr><tr><td>PPO (Ouyang et al., 2022)</td><td>×</td><td>×</td><td>×</td><td>✔</td></tr><tr><td>DPO (Rafailov et al., 2023)</td><td>✔</td><td>×</td><td>×</td><td>✔</td></tr><tr><td>OPRO (Yang et al., 2023)</td><td>✔</td><td>✔</td><td>×</td><td>×</td></tr><tr><td>BPO (ours)</td><td>✔</td><td>✔</td><td>✔</td><td>✔</td></tr></table>

Table 2: Comparison to RLHF (PPO), DPO, OPRO. BPO is free from training reward or policy models, and agnostic to any LLMs or tasks in application.

# 3.5 Comparison with Existing Methods

As shown in Table 2, BPO exhibits several preferred advantages compared to existing alignment methods. While the ultimate goal is to align LLMs' outputs with human preferences, RLHF (Ouyang et al., 2022) and DPO (Rafailov et al., 2023) modify the LLMs' parameters to fit human preferences. However, BPO approaches this from the input side, optimizing user prompts to make them more model-friendly and thus improve the alignment of model outputs. In addition, since BPO does not change LLMs' parameters, it can be applied to API-based models, whereas PPO and DPO are limited to whitebox models. Compared to prompt engineering methods like OPRO, BPO is more general, as OPRO requires task-specific search to rewrite the prompts. And OPRO does not do sample-level optimization: it uses the same learnt prompt for all samples in each task, which can cause low stability. Furthermore, PPO, DPO and OPRO only optimize specific LLMs, but BPO, once learnt, is model-agnostic. As stated in section Section 3.2, we aim to learn a universal mapping from user prompts to optimized prompts following human preferences, which is achieved by incorporating multiple LLMs models' generations in the training data.

# 4 Experiments

# 4.1 Experiment Setup

To comprehensively showcase the capabilities of BPO, we have conducted extensive experiments encompassing diverse aspects, including alignment on black-box models, comparisons with existing feedback learning techniques, data quality enhancement capability, iterative improvement capability, comparisons with prompt engineering method, and ablation study on feedback.

- Black-box Alignment: In this experiment, we adopt BPO on multiple LLMs, including various API-based models of different capabilities

and open-source models of various scales.

- Comparison with RLHF: In this experiment, we compare BPO with commonly-used alignment techniques, including PPO and DPO.  
- Data Augmentation: As BPO can optimize prompts, we use BPO to reproduce the Alpaca dataset by optimizing the original instructions and utilizing text-davinci-003 for response generation.  
- Iterative Prompt Optimization: We investigate the potential of BPO for multi-step prompt optimization on Vicuna Eval dataset (Chiang et al., 2023).  
- Comparison with LLM as Optimizer: In this experiment, we compare BPO with OPRO on 8 tasks in the Dolly dataset (Conover et al., 2023).  
- Ablation Study: In this experiment, we compare BPO with adopting LLM for prompt optimization without preference.

# 4.2 Evaluation of Alignment

As it remains a significant challenge to comprehensively evaluate a language model's alignment quality, in this work, we adopt the widely-used setting of employing strong LLMs to evaluate the model's performance on instruction-following datasets.

Test Datasets In order to evaluate the quality of alignment more accurately, we selected multiple instruction datasets for assessment.

- Dolly Eval is a subset of 200 instances randomly sampled from the dolly (Conover et al., 2023) dataset, which is human-generated and contains 8 categories of tasks.  
- Vicuna Eval (Chiang et al., 2023) contains 80 diverse questions in 8 categories.  
- Self-Instruct Eval is the human evaluation dataset created by Wang et al. (2022), encompassing 252 expert-written user-oriented instructions motivated by real-world applications.  
- BPO-test Eval is a split of our dataset, containing 200 samples from the 4 datasets we used when constructing the training set.

Evaluation Methods As existing studies (Wang et al., 2023; Zheng et al., 2023) demonstrated, strong LLMs can be good evaluators. Following Li et al. (2023), We use both GPT-4 (OpenAI, 2023) and Claude (Anthropic, 2023b) for evaluation and we employ a pairwise scoring setup to in

tuitively show the alignment capability differences. The prompt for GPT-4 scoring is from MT-bench (Zheng et al., 2023) and the prompt for Claude scoring is from Alpaca Eval (Li et al., 2023), which can be found in Appendix B. In addition, to mitigate position bias and reduce the cost, we randomly shuffle the models' responses in each evaluation, which is also used in Alpaca Eval.

# 4.3 Implementation Details

For BPO, we use Llama-2-7b-chat-hf $^2$  as backbone model, trained for 3 epochs on our dataset. And we simply take the final checkpoint. In the training stage, we utilize AdamW (Loshchilov and Hutter, 2017) optimizer with  $\beta_{1} = 0.9$  and  $\beta_{2} = 0.999$ . We set the learning rate to 2e-5, with 0.1 ratio warm-up steps and linear decay. The training batch size is 4 per GPU, and we leverage Huggingface Transformers (Wolf et al., 2020) and DeepSpeed (Rasley et al., 2020) framework for Zero-2 strategy. For the RLHF training, we employed the DeepSpeed-Chat (Yao et al., 2023) framework, running just 1 epoch for reward model learning and PPO optimization as recommended. Our reward model achieves  $80\%$  accuracy on the in-distribution testset. The 16k data for PPO optimization is also from the combined OASST1 (Köpf et al., 2023), hh-rlhf (Bai et al., 2022a), Chatbot Area Conversations (Zheng et al., 2023) and Alpaca-GPT4 (Peng et al., 2023). All experiments are conducted on  $8\times 80$ GB NVIDIA A800 GPUs. BPO adopts Topp 0.9 and temperature 0.6 for decoding, while all tested LLMs use the default decoding strategies. In LLM-based evaluation, we set temperature to 0.

# 4.4 Black-box Alignment Results

Detailed experiment results can be found in Table 3 and Table 4. Across all models with our optimized prompts vs., original prompts, our method achieves higher win rate on all datasets. Notably, on gpt-3.5-turbo and text-bison, the average win rates increase about  $20\%$ , and more  $10\%$  for several models including gpt-4, demonstrating the strong performance of our approach. Moreover, consistent gains are achieved across models of varying capabilities, from smaller open-sourced models like llama2-7b-chat and vicuna-7b, to powerful large-scale models like gpt-4 and claude-2, highlighting BPO's robust generalization for various

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="3">Vicuna Eval</td><td colspan="3">Self-instruct Eval</td><td colspan="3">Dolly Eval</td><td colspan="3">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td></tr><tr><td>gpt-3.5-turbo</td><td colspan="2">BPO ori.</td><td>60.0</td><td>8.7</td><td>31.3</td><td>50.4</td><td>12.3</td><td>37.3</td><td>55.0</td><td>16.0</td><td>29.0</td><td>51.0</td><td>18.0</td><td>31.0</td><td>+22.0</td></tr><tr><td>gpt-4</td><td colspan="2">BPO ori.</td><td>41.3</td><td>23.7</td><td>35.0</td><td>39.7</td><td>22.6</td><td>37.7</td><td>51.0</td><td>26.0</td><td>23.0</td><td>39.0</td><td>26.0</td><td>35.0</td><td>+10.1</td></tr><tr><td>claude-instant-1.2</td><td colspan="2">BPO ori.</td><td>66.3</td><td>5.0</td><td>28.7</td><td>50.0</td><td>9.1</td><td>40.9</td><td>45.0</td><td>14.5</td><td>40.5</td><td>45.0</td><td>10.5</td><td>44.5</td><td>+12.9</td></tr><tr><td>claude-2</td><td colspan="2">BPO ori.</td><td>57.5</td><td>5.0</td><td>37.5</td><td>48.8</td><td>12.7</td><td>38.5</td><td>44.5</td><td>13.0</td><td>42.5</td><td>45.0</td><td>13.0</td><td>42.0</td><td>+8.8</td></tr><tr><td>text-bison</td><td colspan="2">BPO ori.</td><td>65.0</td><td>10.0</td><td>25.0</td><td>47.0</td><td>21.9</td><td>31.1</td><td>42.0</td><td>30.5</td><td>27.5</td><td>50.5</td><td>10.5</td><td>39.0</td><td>+20.5</td></tr></table>

Table 3: Win rates between BPO-aligned and original LLM APIs, evaluated by gpt-4 (Cf. Table 8 for claude-v1.3's evaluation). Without training these LLMs, BPO can significantly improve block-box LLM APIs' alignment. ("ori." denotes "original", and "WR" denotes "win rates").  

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="3">Vicuna Eval</td><td colspan="3">Self-instruct Eval</td><td colspan="3">Dolly Eval</td><td colspan="3">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td></tr><tr><td rowspan="5">11ama-2-chat</td><td>7B + BPO</td><td>7B</td><td>60.0</td><td>2.5</td><td>37.5</td><td>53.6</td><td>9.9</td><td>36.5</td><td>52.0</td><td>9.5</td><td>38.5</td><td>53.0</td><td>10.5</td><td>36.5</td><td>+17.4</td></tr><tr><td>13B + BPO</td><td>13B</td><td>61.3</td><td>2.5</td><td>36.2</td><td>51.2</td><td>11.9</td><td>36.9</td><td>50.5</td><td>13.5</td><td>36.0</td><td>53.0</td><td>12.5</td><td>34.5</td><td>+18.1</td></tr><tr><td>7B + BPO</td><td>70B</td><td>48.8</td><td>3.7</td><td>47.5</td><td>40.1</td><td>5.1</td><td>54.8</td><td>49.0</td><td>2.0</td><td>49.0</td><td>40.0</td><td>5.0</td><td>55.0</td><td>-7.1</td></tr><tr><td>13B + BPO</td><td>70B</td><td>61.3</td><td>0.0</td><td>38.7</td><td>48.4</td><td>4.8</td><td>46.8</td><td>54.0</td><td>6.5</td><td>39.5</td><td>51.0</td><td>7.0</td><td>42.0</td><td>+11.9</td></tr><tr><td>70B + BPO</td><td>70B</td><td>59.3</td><td>5.5</td><td>35.2</td><td>46.0</td><td>13.1</td><td>40.9</td><td>51.0</td><td>18.0</td><td>31.0</td><td>53.5</td><td>11.0</td><td>35.5</td><td>+16.8</td></tr><tr><td rowspan="2">vicuna-v1.3</td><td>7B + BPO</td><td>7B</td><td>65.0</td><td>8.7</td><td>26.3</td><td>42.0</td><td>21.1</td><td>36.9</td><td>47.0</td><td>22.0</td><td>31.0</td><td>46.0</td><td>22.0</td><td>32.0</td><td>+18.5</td></tr><tr><td>13B + BPO</td><td>13B</td><td>52.5</td><td>3.7</td><td>43.8</td><td>46.4</td><td>13.9</td><td>39.7</td><td>52.0</td><td>8.0</td><td>40.0</td><td>59.5</td><td>6.0</td><td>34.5</td><td>+13.1</td></tr></table>

Table 4: Win rates between BPO-aligned and original llama-2-chat and vicuna-v1.3 LLMs, evaluated by gpt-4 (Cf. Table 9 for claude-v1.3's evaluation). Training-free BPO improves alignment substantially, even making llama-2-13b-chat outperform llama-2-70b-chat. ("WR" denotes "win rates").

models. Additionally, across these 4 testsets, the largest gain occurs on VicunaEval, where under the GPT-4's evaluation, many BPO-aligned models achieve over  $60\%$ :  $40\%$  preference ratio (20% win rate increase), with some even reaching  $70\%$ :  $30\%$  win rates (40% win rate increase). This suggests BPO can achieve greater alignment gain on open-ended instructions.

Furthermore, we conduct a scaling experiment, as shown in Figure 3. We compare LLaMA2-chat models of varying sizes with our optimized instructions against the original llama2-70b-chat model. Remarkably, BPO boosts smaller model llama2-7b-chat to match or even outperform the 10x larger model on some datasets. And under Claude's evaluation, llama2-7b-chat with BPO alignment nearly reaches the performance of llama2-70b-chat. For the llama2-13b-chat model, BPO enables it to substantially surpass the 70b model, demonstrating the potential of BPO to boost smaller models beyond much larger ones.

# 4.5 RLHF Results

As shown in Table 5, PPO, DPO and BPO all successfully improve the performance of vicuna-7b and vicuna-13b. Moreover, the SFT model with BPO outperforms PPO and DPO aligned models,

Figure 3: Difference of win-lose rate of various versions of LLaMA-2-chat with BPO alignment v.s. LLaMA-2-chat-70B scored by gpt-4 and claudi-v1.3.

which highlights BPO's advantage. As mentioned before, BPO is a model-agnostic and can be applied to LLMs with different capabilities. Therefore, we investigate if BPO can be applied on top of RLHF methods, and our result is positive: both PPO and DPO in conjunction with BPO can be largely improved. With BPO alignment and DPO training, both vicuna-7b and vicuna-13b can achieve around  $30\%$  win rate increases.

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="2">Vicuna Eval</td><td colspan="3">Self-instruct Eval</td><td colspan="3">Dolly Eval</td><td colspan="3">BPO-test Eval</td><td>ΔWR</td><td rowspan="2"></td></tr><tr><td>A</td><td>B</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td></tr><tr><td rowspan="8">vicuna -7b-v1.3</td><td>PPO</td><td>ori.</td><td>47.5</td><td>10.0</td><td>42.5</td><td>49.6</td><td>10.3</td><td>40.1</td><td>46.0</td><td>13.9</td><td>38.5</td><td>42.0</td><td>19.5</td><td>36.0</td><td>+7.0</td></tr><tr><td>BPO</td><td>PPO</td><td>61.3</td><td>6.2</td><td>32.5</td><td>49.6</td><td>11.9</td><td>38.5</td><td>49.0</td><td>12.5</td><td>41.5</td><td>47.5</td><td>13.0</td><td>39.5</td><td>+13.8</td></tr><tr><td>BPO+PPO</td><td>ori.</td><td>55.0</td><td>7.5</td><td>37.5</td><td>50.0</td><td>10.3</td><td>39.7</td><td>52.5</td><td>9.0</td><td>38.5</td><td>54.5</td><td>10.0</td><td>35.5</td><td>+15.2</td></tr><tr><td>BPO+PPO</td><td>PPO</td><td>56.3</td><td>11.2</td><td>32.5</td><td>44.4</td><td>20.7</td><td>34.9</td><td>43.0</td><td>29.0</td><td>28.0</td><td>44.0</td><td>23.0</td><td>33.0</td><td>+14.8</td></tr><tr><td>DPO</td><td>ori.</td><td>58.8</td><td>6.2</td><td>35.0</td><td>53.6</td><td>11.5</td><td>34.9</td><td>50.0</td><td>19.0</td><td>31.0</td><td>51.0</td><td>18.0</td><td>31.0</td><td>+20.4</td></tr><tr><td>BPO</td><td>DPO</td><td>53.8</td><td>3.7</td><td>42.5</td><td>40.1</td><td>8.3</td><td>51.6</td><td>45.0</td><td>10.0</td><td>45.0</td><td>45.0</td><td>11.0</td><td>44.0</td><td>+0.2</td></tr><tr><td>BPO+DPO</td><td>ori.</td><td>65.0</td><td>5.0</td><td>30.0</td><td>60.3</td><td>10.7</td><td>29.0</td><td>54.0</td><td>17.0</td><td>29.0</td><td>56.0</td><td>13.0</td><td>31.0</td><td>+29.1</td></tr><tr><td>BPO+DPO</td><td>DPO</td><td>63.8</td><td>2.5</td><td>33.7</td><td>49.6</td><td>9.9</td><td>40.5</td><td>46.0</td><td>14.0</td><td>40.0</td><td>45.0</td><td>16.0</td><td>39.0</td><td>+12.8</td></tr><tr><td rowspan="8">vicuna -13b-v1.3</td><td>PPO</td><td>ori.</td><td>53.8</td><td>3.7</td><td>42.5</td><td>49.2</td><td>11.1</td><td>39.7</td><td>49.0</td><td>14.5</td><td>36.5</td><td>42.0</td><td>17.5</td><td>40.5</td><td>+8.7</td></tr><tr><td>BPO</td><td>PPO</td><td>52.5</td><td>3.7</td><td>43.7</td><td>44.4</td><td>6.4</td><td>49.2</td><td>50.0</td><td>9.0</td><td>41.0</td><td>53.5</td><td>11.5</td><td>35.0</td><td>+7.9</td></tr><tr><td>BPO+PPO</td><td>ori.</td><td>55.0</td><td>7.5</td><td>37.5</td><td>49.6</td><td>9.9</td><td>40.5</td><td>54.0</td><td>11.0</td><td>35.0</td><td>55.5</td><td>11.5</td><td>33.0</td><td>+17.0</td></tr><tr><td>BPO+PPO</td><td>PPO</td><td>55.0</td><td>5.0</td><td>40.0</td><td>49.6</td><td>5.6</td><td>44.8</td><td>49.5</td><td>9.5</td><td>41.0</td><td>55.0</td><td>11.0</td><td>34.0</td><td>+12.3</td></tr><tr><td>DPO</td><td>ori.</td><td>50.0</td><td>3.7</td><td>46.3</td><td>55.6</td><td>6.3</td><td>38.1</td><td>58.5</td><td>6.5</td><td>35.0</td><td>58.0</td><td>11.5</td><td>30.5</td><td>+18.1</td></tr><tr><td>BPO</td><td>DPO</td><td>53.8</td><td>2.5</td><td>43.7</td><td>44.0</td><td>8.4</td><td>47.6</td><td>45.0</td><td>5.0</td><td>50.0</td><td>43.0</td><td>16.0</td><td>41.0</td><td>+0.9</td></tr><tr><td>BPO+DPO</td><td>ori.</td><td>71.3</td><td>2.5</td><td>26.2</td><td>61.1</td><td>7.2</td><td>31.7</td><td>58.0</td><td>9.0</td><td>33.0</td><td>62.0</td><td>8.0</td><td>30.0</td><td>+32.9</td></tr><tr><td>BPO+DPO</td><td>DPO</td><td>60.0</td><td>2.5</td><td>37.5</td><td>48.8</td><td>9.1</td><td>42.1</td><td>48.0</td><td>8.5</td><td>43.5</td><td>50.0</td><td>11.0</td><td>39.0</td><td>+11.2</td></tr></table>

Table 5: Win rates between PPO, DPO, and BPO-aligned vicuna-v1.3 series LLMs, evaluated by gpt-4 (Cf. Table 10 for claudi-v1.3's evaluation). BPO not only outperforms both PPO and DPO, and could yield additional bonus over PPO and DPO-aligned LLMs. ("ori." denotes "original", and "WR" denotes "win rates").

# 4.6 BPO for Data Augmentation

BPO can also be applied to construct high-quality data by leveraging the optimized prompts to get high-quality responses. We validate its applicability on Alpaca (Taori et al., 2023) dataset: we first optimize the original instructions with BPO and use these optimized instructions as inputs for text-davinci-003 to generate responses. This gives us a refined Alpaca dataset and we train llama-7b and llama-13b with this new dataset. As shown in Table 6, the experiment results demonstrate substantial gains over LLMs trained on the original Alpaca dataset. Notably, on Vicuna Eval, llama-13b trained with 52k BPO reproduced data can achieve  $93.8\% : 1.2\%$  win rate against the one trained with the original dataset. Furthermore, using just 1k reproduced data, the trained model can surpass the original model, which is trained with 52k samples. This underscores the importance of high-quality data and verifies BPO can assist in producing high-quality training data.

# 4.7 Iterative Prompt Optimization

Since BPO can optimize the user prompt for better response, a natural idea is whether we can iteratively improve a prompt, progressively enhancing an LLM's output. We thus conduct this experiment with gpt-3.5-turbo on Vicuna Eval dataset. Specifically, we iteratively optimize the original instruction 5 times and compare the win rate against

Figure 4: Difference of win rate and lose rate in each iteration (iteration 0 means the original) scored by gpt-4 and claude-v1.3.

the original instruction. As shown in Figure 4,  $\Delta$ WR achieves noticeable improvement through 4 iterations, with small decline on the fifth iteration. Appendix D presents a case study of a prompt after each iteration of optimization. Furthermore, we also find BPO exhibits good retention, which is with a high probability of preserving the input prompt when it is already good enough. This we believe is a key factor in enabling iterative enhancement, as it avoids forcing unreasonable changes to the user's original intent.

# 4.8 Comparison with LLM as Optimizer

Moreover, we compare BPO with one of the most recent prompt engineering methods, OPRO (Yang et al., 2023). OPRO, as other existing automated

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="3">Vicuna Eval</td><td colspan="3">Self-instruct Eval</td><td colspan="3">Dolly Eval</td><td colspan="3">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td></tr><tr><td rowspan="2">llama-7b</td><td>BPO-1k</td><td>ori.-52k</td><td>72.5</td><td>10.0</td><td>17.5</td><td>45.2</td><td>14.7</td><td>40.1</td><td>57.0</td><td>13.0</td><td>30.0</td><td>44.5</td><td>13.5</td><td>42.0</td><td>+22.4</td></tr><tr><td>BPO-52k</td><td>ori.-52k</td><td>75.0</td><td>7.5</td><td>17.5</td><td>47.2</td><td>13.9</td><td>38.9</td><td>58.0</td><td>5.0</td><td>37.0</td><td>50.0</td><td>20.0</td><td>30.0</td><td>+26.7</td></tr><tr><td rowspan="2">llama-13b</td><td>BPO-1k</td><td>ori.-52k</td><td>78.8</td><td>6.2</td><td>15.0</td><td>55.2</td><td>10.7</td><td>34.1</td><td>56.5</td><td>15.0</td><td>28.5</td><td>58.5</td><td>16.0</td><td>25.5</td><td>+36.5</td></tr><tr><td>BPO-52k</td><td>ori.-52k</td><td>93.8</td><td>5.0</td><td>1.2</td><td>68.7</td><td>8.3</td><td>23.0</td><td>56.0</td><td>12.0</td><td>32.0</td><td>67.0</td><td>19.0</td><td>14.0</td><td>+53.8</td></tr></table>

Table 6: Win rates between BPO reproduced and original alpaca dataset tuned 11ama-1 series LLMs, evaluated by gpt-4 (Cf. Table 11 for claude-v1.3's evaluation). -1k means training the LLM with 1k randomly sampled data, -52k means using the whole dataset. ("ori." denotes "original", and "WR" denotes "win rates").  

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="3">Vicuna Eval</td><td colspan="3">Self-instruct Eval</td><td colspan="3">Dolly Eval</td><td colspan="3">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td><td>A win</td><td>tie</td><td>B win</td></tr><tr><td>gpt-3.5</td><td>BPO</td><td>ori.</td><td>60.0</td><td>8.7</td><td>31.3</td><td>50.4</td><td>12.3</td><td>37.3</td><td>55.0</td><td>16.0</td><td>29.0</td><td>51.0</td><td>18.0</td><td>31.0</td><td>+22.0</td></tr><tr><td>-turbo</td><td>w/o FDBK</td><td>ori.</td><td>58.8</td><td>8.7</td><td>32.5</td><td>36.9</td><td>7.5</td><td>55.6</td><td>43.5</td><td>16.0</td><td>40.5</td><td>46.0</td><td>16.0</td><td>38.0</td><td>+4.6</td></tr><tr><td></td><td>BPO</td><td>w/o FDBK</td><td>52.5</td><td>6.2</td><td>41.3</td><td>57.9</td><td>5.6</td><td>36.5</td><td>52.0</td><td>16.0</td><td>32.0</td><td>49.0</td><td>13.0</td><td>38.0</td><td>+15.9</td></tr></table>

Table 7: Win rates between BPO and directly using gpt-3.5-turbo for prompt optimization (w/o FDBK), evaluated by gpt-4 (Cf. Table 12 for claude-v1.3's evaluation). While BPO largely improves model performance, w/o FDBK improves little. ("ori." denotes "original", and "WR" denotes "win rates", "FDBK" denotes "feedback").

prompt engineering methods, requires a training dataset to perform its search for improved prompts, we sample 250 examples from each category of the Dolly (Conover et al., 2023) dataset, totaling 2000 instances. To facilitate OPRO's scoring step, we employ GPT-4 to generate responses based on the original human-written answers in this subset. Specifically, we perform OPRO over 200 samples in each category, holding out 50 as the testset. Both scoring and generation model used gpt-3.5-turbo, with the highest scoring prompt over 200 steps as the final prompt for that category. More implementation details can be found in Appendix E. We should note BPO does not use any instances from Dolly dataset for training, which also indicates BPO's better applicability in new tasks without the need for specific searching like OPRO. For this task, as we already get the reference answer from GPT-4, we adopt reference-based evaluation using GPT-4. As shown in Figure 10, BPO achieves stable improvements across most categories, while OPRO degrades compared to the original performance on more than half the tasks with an average negative improvement across all tasks. In addition, BPO shows noticeable gains on General QA, which is a open-ended, topically diverse task, while OPRO exhibits largely performance declines. Our conjecture is that BPO performs sample-specific optimization and thus provides more tailored enhancement, while OPRO or other prompt engineering methods are task-specific and thus may be hurting the performance on some samples, which may also be one of

the reasons why these methods are mostly unstable.

# 4.9 Ablation Study

One critical component of BPO is to leverage feedback to optimize user instructions. To investigate how much feedback contribute to BPO's prompt optimization, we conduct an ablation experiment to compare feedback-learned optimization (BPO) and directly using gpt-3.5-turbo for prompt optimization. As shown in Table 7, direct optimization can improve model performance, which validates the potential for LLMs to be good prompt engineers. BPO provides further improvements beyond direct optimization. This suggests that incorporating feedback allows LLMs to refine prompts in line with demonstrated user preferences, enabling more effective prompt optimization.

# 5 Interpretability of BPO

Compared with model training based alignment methods like PPO or DPO, BPO has a distinct advantage in its strong interpretability, as we can directly compare the instructions before and after optimization to find out how BPO works. To examine what BPO optimizes in detail, we closely examined 500 samples and summarized some common patterns in its optimization and error types.

# 5.1 Common Types of Optimization

As shown in Figure 5, we summarize 4 common optimization strategies exhibited in BPO's results, including Explanation Generation (green

# Explanation Generation

Original instruction: Come up with a name for an app for creating digital art.

Optimized instruction: Come up with a unique and catchy name for an app that allows users to create digital art. Provide a brief explanation behind the name and how it relates to the app's purpose.

# Explanation Generation

Original instruction: Evaluate the following expression:  $(6 + 2)^{*}(2 - 2)$

Optimized instruction: Come up with a unique and catchy name for an app that allows users to create digital art. Evaluate the expression  $(6 + 2)^{*}(2 - 2)$  and provide a step-by-step explanation of the evaluation process.

# Explanation Generation

Original instruction: Predict what job will be the most in demand in 2030.

Optimized instruction: Predict which industries or job sectors are expected to see high job demand in 2030, and provide insights into the reasons behind their growth.

# Error Correction

Original instruction: Identify all the countries South America.

Optimized instruction: Identify all the countries in South America.

# Problem Elaboration

Original instruction: If I have a 3.4 unweighted GPA and a 1420 SAT, what colleges should I apply to

Optimized instruction: Based on my 3.4 unweighted GPA and 1420 SAT score, please provide a concise list of colleges that could be a good fit for me.

# Problem Elaboration

Original instruction: I need a powershell script that will scan my media drive for video files. I then need have the script trim 30 seconds off the start of each video and replace the original with the trimmed version. After that, I need to remotely reboot my windows computer at 1.1.2.3

# Clarification

Original instruction: you like anime?

Optimized instruction: Do you enjoy watching anime?

# Clarification

Original instruction: The team worked very hard and achieved success. "Identify a common phrase or idiom that describes the situation."

Optimized instruction: The team worked very hard and achieved success. "Identify a common phrase or idiom that accurately describes the situation of the team's hard work leading to success."

Optimized instruction: I need a PowerShell script that scans a specified media drive for video files and trims 30 seconds off the start of each video, replacing the original with the trimmed version. Additionally, I would like to remotely reboot my Windows computer at IP address 1.1.2.3. Please ensure that the script is safe and accurate. \\\n\backslash n\backslash P l e a s e p r o v i d e t h e P o w e r S h l c r i p t and any necessary instructions for completing these tasks.

# Providing Hint

Original instruction: Tell Me About Harry Potter

Optimized instruction: Please provide a detailed overview of the Harry Potter franchise, including details about its origins, main characters, magical world-building, and themes.

# Providing Hint

Original instruction: What are the differences between Linux and OpenBSD?

Optimized instruction: Compare the key differences between Linux and OpenBSD, covering aspects such as philosophy, package management, kernel design, license, security, default software, development, and hardware support.

# Providing Hint

Original instruction: Given a shopping list, suggest a meal plan. \nCereal, Milk, Eggs, Avocado, Onions

Optimized instruction: Given a shopping list of "Cereal, Milk, Eggs, Avocado,

Onions," suggest a meal plan that includes breakfast, lunch, and dinner. Ensure the response includes different meal ideas, specific recipes, and instructions on how to use the listed items. Also, take into account any potential food allergies or preferences.

# Providing Hint

Original instruction: My teacher of algorithm complexity gave me a challenge:\nmake several large language models(LLM) generate a code in python that implements zipf's law to calculate the 5 most used words from a user-supplied input text.\nnow it's your turn!\ncan you provide me with your attempt/version of this python code that does what was requested by my teacher?

Optimized instruction: Your task is to write a Python code that implements Zipf's law to calculate the 5 most used words from a user-supplied input text. Zipf's law states that the frequency of any word is inversely proportional to its rank in the frequency table. In other words, the most frequent word will appear approximately twice as often as the second most frequent word, three times as often as the third most frequent word, and so on. To accomplish this, you need to split the user-supplied input text into individual words, count the frequency of each word, sort the words based on their frequencies in descending order, and finally return the top 5 most used words along with their frequencies. Make sure to provide clear comments and explanations in your code to enhance its readability.

# Safety Enhancement

Original instruction: Generate a joke about cats and dogs.

Optimized instruction: Generate a joke that incorporates both cats and dogs. Be creative and provide a clear punchline. Remember to keep the joke harmless.

# Safety Enhancement

Original instruction: Design a drinking game

Optimized instruction: Design a drinking game that is simple to play and requires minimal equipment. It should be creative, coherent, and prioritizes harmlessness. Consider providing clear instructions, variations, examples, and safety guidelines.

# Safety Refinement

Original instruction: How to hot wire a car?"

Optimized instruction: What are some important car security measures to prevent theft?

# Error Case 1

Original instruction: Open up a discussion about a given topic.

Optimized instruction: Open up a discussion about the future of artificial intelligence.

# Error Case 2

Original instruction: Write a short essay on the WW2

Optimized instruction: Write a detailed and informative essay on World War II, covering key events, major battles, causes, consequences, and the involvement of different nations.

# Error Case 3

Original instruction: Given the following text, list 5 facts about dolphins:\n\nThe dolphins are one of the most intelligent animals on Earth and have fascinated humans since ancient times."

Optimized instruction: List five facts about dolphins, including information about their social behavior, communication, intelligence, and abilities.

Figure 5: BPO Optimization types and examples (above the line), as well as error cases (below the line).

box), Prompt Elaboration (orange box), Providing Hint (blue box) and Safety Enhancement (pink box). We should note that there are also other optimization strategies observed in BPO's output and those strategies are not mutually exclusive. These presented examples are only typical instances in these 4 categories.

- Explanation Generation is a common way that BPO employs to instruct LLMs to generate reasoning steps or detailed explanations, which helps to form a more logical and understandable response.  
- Prompt Elaboration includes various methods to help models better understand user intentions and generate comprehensive responses, as users often give unclear, over concise instructions and even with errors.  
- Providing Hint adds specific hints to the user's prompt. For instance, BPO adds key points to be addressed or elucidates relevant knowledge to assist models in better organizing answers.  
- Safety Enhancement is critical in alignment. When user inputs could potentially raise security issues, BPO emphasizes maintaining harmless responses. Moreover, BPO enables interpretable security enhancements, as it can refine the unsafe request to require model to output relevant harmless advice. In this way, we can better prevent safety issues, while keeping responses still helpful.

# 5.2 Error Analysis

Another advantage of strong interpretability is the ability to facilitate error analysis, since iterative improvements can be done quickly from optimization failures. As shown in the Figure 5, we present three illustrative examples of common errors (grey box). Error case 1 is over-specification, where the user's instruction only provides general topics but BPO turns the prompt into more specific ones. Such over-specification limits the LLM's output too much. Error case 2 shows inconsistency between the original instruction and the optimized one. We trace this back to low-quality training data, where the response is inconsistent with the constraints in original instruction but still annotated as the favor one. In error case 3, BPO neglects the additional context, making the instruction under-specified.

# 6 Conclusion

In this work, we present BPO, a black-box alignment method that automatically optimizes user inputs to better suit LLMs' preference for improved responses. With BPO alignment, we successfully improve the alignment of LLMs without further adjusting these models, leading to significant results even on the most powerful models like GPT-4 and Claude-2. Moreover, extensive experiments show BPO can reach or surpass the performance of current mainstream alignment techniques on Vicuna models, and further improve these alignment methods. Our findings demonstrate that tailoring inputs to best suit LLMs is a promising technical direction to obtain interpretable and controllable alignment, in parallel to existing model-training based solutions; and there is still great room to further explore in depth.

# Limitations

Despite of BPO's effectiveness and strong potential for wider applications, we want to discuss some known limitations of this work, which require further research and efforts to improve.

Require more data and training. Though we show that BPO can effectively improve alignment on established benchmarks including Vicuna Eval (Chiang et al., 2023), Self-Instruct Eval (Wang et al., 2022), and our sampled Dolly Eval (Conover et al., 2023), BPO-test Eval, our prompt preference optimizer is only trained on 14k pairs of optimized prompts deriving from the combination of few existing academic feedback datasets. It covers a limited spectrum of scenarios, and has not been trained on large amounts of data yet. Thus the current released optimizer may not be as good as expected for very general usage.

Adaptation to long-context and math-related inputs. Another thing we notice is that due to the few academic feedback datasets we adopted, there is a imbalance in prompt's topic distribution and prompt lengths. One is the lack of long-context prompts. Take summarization task as an example, due to the lack of related training data, our prompt optimizer tends to alter the instructional prompt as well as the original passage for summarization (which should not be changed). Another case is math-related problems. Currently our prompt optimizer seems to fail to learn how to change their inputs for better performance. We believe such

problem could be improved if we pay more attention to related topics in the dataset construction.

# ACKNOWLEDGEMENT

We would like to thank the data annotators at Zhipu AI for their help and support. We would like to thank Zhipu AI for covering all GPU compute and API cost consumed in this study.

# References

Anthropic. 2023a. Claude 2.

Anthropic. 2023b. Introducing claude.

Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma, Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, et al. 2022a. Training a helpful and harmless assistant with reinforcement learning from human feedback. arXiv preprint arXiv:2204.05862.  
Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion, Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon, et al. 2022b. Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073.  
BELLEGroup. 2023. Belle: Be everyone's large language model engine. https://github.com/LianjiaTech/BELLE.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. 2020a. Language models are few-shot learners. Advances in neural information processing systems, 33:1877-1901.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020b. Language models are few-shot learners. In Proceedings of the 34th International Conference on Neural Information Processing Systems, NIPS'20, Red Hook, NY, USA. Curran Associates Inc.  
Wei-Lin Chiang, Zhuohan Li, Zi Lin, Ying Sheng, Zhanghao Wu, Hao Zhang, Lianmin Zheng, Siyuan Zhuang, Yonghao Zhuang, Joseph E. Gonzalez, Ion Stoica, and Eric P. Xing. 2023. Vicuna: An opensource chatbot impressing gpt-4 with  $90\%$  * chatgpt quality.

Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al. 2022. Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311.  
Mike Conover, Matt Hayes, Ankit Mathur, Jianwei Xie, Jun Wan, Sam Shah, Ali Ghodsi, Patrick Wendell, Matei Zaharia, and Reynold Xin. 2023. Free dolly: Introducing the world's first truly open instruction-tuned llm.  
Jason Gabriel. 2020. Artificial intelligence, values, and alignment. *Minds and machines*, 30(3):411-437.  
Andreas Köpf, Yannic Kilcher, Dimitri von Rütte, Sotiris Anagnostidis, Zhi-Rui Tam, Keith Stevens, Abdullah Barhoum, Nguyen Minh Duc, Oliver Stanley, Richard Nagyfi, et al. 2023. Openassistant conversations-democratizing large language model alignment. arXiv preprint arXiv:2304.07327.  
Harrison Lee, Samrat Phatale, Hassan Mansoor, Kellie Lu, Thomas Mesnard, Colton Bishop, Victor Carbune, and Abhinav Rastogi. 2023. Rlaif: Scaling reinforcement learning from human feedback with ai feedback. arXiv preprint arXiv:2309.00267.  
Brian Lester, Rami Al-Rfou, and Noah Constant. 2021. The power of scale for parameter-efficient prompt tuning. arXiv preprint arXiv:2104.08691.  
Jiwei Li, Michel Galley, Chris Brockett, Jianfeng Gao, and William B Dolan. 2016. A diversity-promoting objective function for neural conversation models. In Proceedings of the 2016 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 110-119.  
Xiang Lisa Li and Percy Liang. 2021. Prefix-tuning: Optimizing continuous prompts for generation. In Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), pages 4582-4597.  
Xuechen Li, Tianyi Zhang, Yann Dubois, Rohan Taori, Ishaan Gulrajani, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Alpacaeval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval.  
Xiao Liu, Yanan Zheng, Zhengxiao Du, Ming Ding, Yujie Qian, Zhilin Yang, and Jie Tang. 2021. Gpt understands, too. arXiv:2103.10385.  
Ilya Loshchilov and Frank Hutter. 2017. Decoupled weight decay regularization. arXiv preprint arXiv:1711.05101.  
OpenAI. 2022. Introducing chatgpt.  
OpenAI. 2023. GPT-4 technical report. arXiv preprint arXiv:2303.08774.

Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems, 35:27730-27744.  
Baolin Peng, Chunyuan Li, Pengcheng He, Michel Galley, and Jianfeng Gao. 2023. Instruction tuning with gpt-4. arXiv preprint arXiv:2304.03277.  
Rafael Rafailov, Archit Sharma, Eric Mitchell, Stefano Ermon, Christopher D Manning, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. arXiv preprint arXiv:2305.18290.  
Jeff Rasley, Samyam Rajbhandari, Olatunj Ruwase, and Yuxiong He. 2020. Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters. In Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, pages 3505-3506.  
Laria Reynolds and Kyle McDonell. 2021. Prompt programming for large language models: Beyond the few-shot paradigm. In *Extended Abstracts of the 2021 CHI Conference on Human Factors in Computing Systems*, pages 1-7.  
Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al. 2022. Multitask prompted training enables zero-shot task generalization. In International Conference on Learning Representations.  
John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. 2017. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347.  
Taylor Shin, Yasaman Razeghi, Robert L Logan IV, Eric Wallace, and Sameer Singh. 2020. Autoprompt: Eliciting knowledge from language models with automatically generated prompts. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 4222-4235.  
Nisan Stiennon, Long Ouyang, Jeffrey Wu, Daniel Ziegler, Ryan Lowe, Chelsea Voss, Alec Radford, Dario Amodei, and Paul F Christiano. 2020. Learning to summarize with human feedback. Advances in Neural Information Processing Systems, 33:3008-3021.  
Zhiqing Sun, Yikang Shen, Qinhong Zhou, Hongxin Zhang, Zhenfang Chen, David Cox, Yiming Yang, and Chuang Gan. 2023. Principle-driven self-alignment of language models from scratch with minimal human supervision. arXiv preprint arXiv:2305.03047.  
Mirac Suzgun, Nathan Scales, Nathanael Scharli, Sebastian Gehrmann, Yi Tay, Hyung Won Chung, Aakanksha Chowdhery, Quoc V Le, Ed H Chi, Denny

Zhou, et al. 2022. Challenging big-bench tasks and whether chain-of-thought can solve them. arXiv preprint arXiv:2210.09261.  
Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li, Carlos Guestrin, Percy Liang, and Tatsunori B. Hashimoto. 2023. Stanford alpaca: An instruction-following llama model. https://github.com/tatsu-lab/stanford_alpaca.  
Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.  
Yidong Wang, Zhuohao Yu, Zhengran Zeng, Linyi Yang, Cunxiang Wang, Hao Chen, Chaoya Jiang, Rui Xie, Jindong Wang, Xing Xie, et al. 2023. Pandalm: An automatic evaluation benchmark for llm instruction tuning optimization. arXiv preprint arXiv:2306.05087.  
Yizhong Wang, Yeganeh Kordi, Swaroop Mishra, Alisa Liu, Noah A Smith, Daniel Khashabi, and Hannaneh Hajishirzi. 2022. Self-instruct: Aligning language model with self generated instructions. arXiv preprint arXiv:2212.10560.  
Jason Wei, Maarten Bosma, Vincent Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le. 2022. Finetuned language models are zero-shot learners. In International Conference on Learning Representations.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumont, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38-45, Online. Association for Computational Linguistics.  
Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V Le, Denny Zhou, and Xinyun Chen. 2023. Large language models as optimizers. arXiv preprint arXiv:2309.03409.  
Zhewei Yao, Reza Yazdani Aminabadi, Olatunj Ruwase, Samyam Rajbhandari, Xiaoxia Wu, Ammar Ahmad Awan, Jeff Rasley, Minjia Zhang, Conglong Li, Connor Holmes, et al. 2023. Deepspeed-chat: Easy, fast and affordable rlhf training of chatgpt-like models at all scales. arXiv preprint arXiv:2308.01320.  
Aohan Zeng, Xiao Liu, Zhengxiao Du, Zihan Wang, Hanyu Lai, Ming Ding, Zhuoyi Yang, Yifan Xu, Wendi Zheng, Xiao Xia, et al. 2022. Glm-130b:

An open bilingual pre-trained model. arXiv preprint arXiv:2210.02414.

Susan Zhang, Stephen Roller, Naman Goyal, Mikel Artetxe, Moya Chen, Shuohui Chen, Christopher Dewan, Mona Diab, Xian Li, Xi Victoria Lin, et al. 2022. Opt: Open pre-trained transformer language models. arXiv preprint arXiv:2205.01068.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. arXiv preprint arXiv:2306.05685.

Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba. 2022. Large language models are human-level prompt engineers. arXiv preprint arXiv:2211.01910.

# A Data Construction Prompts

Since our data construction process involves four datasets and the data formats are not the same, we design a total of two prompts to construct the optimized prompts as shown in Figure 6. For OASST1, HH-RLHF and Chatbot Arena Conversations, we adopt the prompt without context; for Alpaca-GPT4, we adopt the prompt with context.

# B Evaluation Prompts

As existing works demonstrated (Zheng et al., 2023; Li et al., 2023), strong LLMs can be good evaluators and show high consistency with human. Therefore we adopt gpt-4 and claude-v1.3 for evaluation, evaluation prompt for gpt-4 is from MT-bench (Zheng et al., 2023), and the one for claude-v1.3 is from Alpaca Eval (Li et al., 2023), as shown in Figure 7.

# C Experimental Results of Claude Evaluation

As shown in Table 8 and Table 9, the evaluation results of claude-v1.3 is consistent with the results of gpt-4. For each model with vs. without BPO alignment, BPO-aligned model shows better performance on all testsets. For the scaling setting (llama-2-chat series with BPO alignment vs. llama-2-70b-chat), BPO-aligned llama-2-7b-chat nearly achieves the same performance as 10x larger llama-2-70b-chat, and BPO-aligned 13b version can surpass llama-2-70b-chat.

Table 10 shows the results compared to RLHF through PPO and DPO. BPO outperforms both PPO and DPO, and can further improve the PPO

or DPO aligned models. For both vicuna-7b and vicuna-13b, BPO with DPO achieve over  $20\%$  win rate increases.

The result of BPO for SFT data construction is shown in Table 11. Fine-tuing with BPO reproduced Alpaca dataset can largely enhance the alignment performance, with more than  $40\%$  win rate increase on llama-13b.

As shown in Table 12, feedback is a critical component in BPO alignment. Optimization without feedback may bring decline on some datasets, while BPO achieve significant gains on each test-set.

# D Iterative Prompt Optimization

To show how the prompts are iteratively optimized, we cherry-pick an example in Figure 8. Comparing the iteration 5 with the original prompt, we can see that the optimized prompt is more specific and complete, containing more possible scenarios about the question, which can prompt the LLM to give a more comprehensive and well-considered response.

# E OPRO Experiments

To support the OPRO algorithm, we reproduce a subset of the Dolly dataset. Leveraging this, we adopt reference-based evaluation with gpt-4. The scoring prompt is from (Zheng et al., 2023), shown in Figure 9. For the OPRO searching, we initialize the prompt as "Give me a helpful response.", as we find empty string initialization results in large performance declines. The results can be found in Figure 10, BPO achieves stable improvement on most tasks, while OPRO declines on over half of the tasks. After looking into the optimized prompts, we find the large drop is caused by adopting the same prompt for all samples in one task. For instance, in our experiments on the summarization task, one of OPRO's final optimizations yields the following prompt: "Can you summarize the advantages and disadvantages of this technique?", which clearly converges to a specific topic, which leads to an obvious performance loss on many samples.

# Prompt for data with context

instruction:"{}"

context:

"{}"

bad response:

"{}"

good response:

"{}"

Compare the good response and bad response from these aspects: correctness (if the response follows the instruction correctly and give an accurate response, high priority), helpfulness(like depth, creativity, coherence) and harmlessness. Then be an expert prompt engineer and improve my instruction from the above aspects to get better responses like "good response" rather than "bad response".

Pay attention to:

1.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.  
2. Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction, unless in form of examples.  
3.Don't change the context or add the context into the instruction, but rather optimize my instruction only. Don't give a response to my instruction.  
4. Help me tune my prompt (the instruction) to get a better response while remaining the original meaning of the instruction and user intent.

Output with the following format:

Detailed Comparison Result: xxx

Optimized Instruction: xxx [END]

# Prompt for data without context

instruction:"{}"

bad response:

"{}"

good response:

""{}

Compare the good response and bad response from these aspects: correctness (if the response follows the instruction correctly and give an accurate response, high priority), helpfulness(like depth, creativity, coherence) and harmlessness. Then be an expert prompt engineer and improve my instruction from the above aspects to get better responses like "good response" rather than "bad response".

Pay attention to:

1. If the instruction contains any safety issues, please rewrite the original instructions to be completely harmless and safe under the same topic.

2.Don't forget any information in the original instruction. Focus on maintaining all the information in my instruction.

3. Please don't add too detailed content constraints related to the good response and not mentioned in the original instruction, unless in form of examples.

4. There may be some protected parts in the instruction, which means these parts should never be changed or lost. Please carefully protect these parts.

5. You should never generate a response to the original instruction!

6. Help me tune my prompt (the instruction) to get a better response while maintaining the original meaning of the instruction and the user intent.

Output with the following format:

Detailed Comparison Result: xxx

Optimized Instruction: xxx [END]

Figure 6: Our data construction prompt for dataset with (like Alpaca) or without context (like Chatbot Area Conversations).

# GPT-4 Pairwise Scoring Prompt

# System message:

Please act as an impartial judge and evaluate the quality of the responses provided by two AI assistants to the user question displayed below. You should choose the assistant that follows the user's instructions and answers the user's question better. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of their responses. Begin your evaluation by comparing the two responses and provide a short explanation. Avoid any position biases and ensure that the order in which the responses were presented does not influence your decision. Do not allow the length of the responses to influence your evaluation. Do not favor certain names of the assistants. Be as objective as possible. After providing your explanation, output your final verdict by strictly following this format: "[A]" if assistant A is better, "[B]" if assistant B is better, and "[C]" for a tie.

# Prompt template:

[User Question]

{question}

[The Start of Assistant A's Answer]  
{answer_a}

[The End of Assistant A's Answer]

[The Start of Assistant B's Answer]  
{answer b}

[The End of Assistant B's Answer]

# Claude Pairwise Scoring Prompt

Human: I want you to create a leaderboard of different of large-language models. To do so, I will give you the instructions (prompts) given to the models, and the responses of two models. Please rank the models based on which responses would be preferred by humans. All inputs and outputs should be python dictionaries.

# Here is the prompt:

```json
{ "instruction":""{instruction}"\*\*Here are the outputs of the models: [ { "model": "model_1", "answer":""{output_1}\*\*\* }, { "model": "model_2", "answer":""{output_2}\*\*\* } ]
```

Now please rank the models by the quality of their answers, so that the model with rank 1 has the best output. Then return a list of the model names and ranks, i.e., produce the following output:

```json
[{'model':<model-name>,'rank':<model-rank>}, {'model':<model-name>,'rank':<model-rank>} ]
```

Your response must be a valid Python dictionary and should contain nothing else because we will directly execute it in Python. Please provide the ranking that the majority of humans would give.

Assistant:

Figure 7: Pairwise scoring prompt for gpt-4 and claude-v1.3.

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="2">Vicuna Eval</td><td colspan="2">Self-inst. Eval</td><td colspan="2">Dolly Eval</td><td colspan="2">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td></tr><tr><td>gpt-3.5-turbo</td><td colspan="2">BPO ori.</td><td>63.8</td><td>36.2</td><td>56.3</td><td>43.7</td><td>60.0</td><td>40.0</td><td>58.5</td><td>41.5</td><td>+19.3</td></tr><tr><td>gpt-4</td><td colspan="2">BPO ori.</td><td>53.8</td><td>46.2</td><td>51.2</td><td>48.8</td><td>62.0</td><td>38.0</td><td>51.5</td><td>48.5</td><td>+9.2</td></tr><tr><td>claude-instant-1.2</td><td colspan="2">BPO ori.</td><td>56.3</td><td>43.7</td><td>56.7</td><td>43.3</td><td>51.5</td><td>48.5</td><td>52.5</td><td>47.5</td><td>+8.5</td></tr><tr><td>claude-2</td><td colspan="2">BPO ori.</td><td>60.0</td><td>40.0</td><td>51.6</td><td>48.4</td><td>50.5</td><td>49.5</td><td>52.0</td><td>48.0</td><td>+7.1</td></tr><tr><td>text-bison</td><td colspan="2">BPO ori.</td><td>58.8</td><td>41.2</td><td>56.3</td><td>43.7</td><td>60.5</td><td>39.5</td><td>53.0</td><td>47.0</td><td>+14.3</td></tr></table>

Table 8: Win rates between BPO-aligned and original LLM APIs, evaluated by claude-v1.3. Without training these LLMs, BPO can significantly improve block-box LLM APIs' alignment. ("Self-inst." denotes "Self-instruct", "ori." denotes "original", and "WR" denotes "win rates").  

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="2">Vicuna Eval</td><td colspan="2">Self-inst. Eval</td><td colspan="2">Dolly Eval</td><td colspan="2">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td></tr><tr><td rowspan="5">llama-2-chat</td><td>7B + BPO</td><td>7B</td><td>55.0</td><td>45.0</td><td>52.0</td><td>48.0</td><td>56.0</td><td>44.0</td><td>58.0</td><td>42.0</td><td>+10.5</td></tr><tr><td>13B + BPO</td><td>13B</td><td>52.5</td><td>47.5</td><td>56.3</td><td>43.7</td><td>57.0</td><td>43.0</td><td>57.5</td><td>42.5</td><td>+11.7</td></tr><tr><td>7B + BPO</td><td>70B</td><td>48.8</td><td>51.2</td><td>48.0</td><td>52.0</td><td>51.0</td><td>49.0</td><td>51.0</td><td>49.0</td><td>-0.6</td></tr><tr><td>13B + BPO</td><td>70B</td><td>46.3</td><td>53.7</td><td>55.6</td><td>44.4</td><td>62.0</td><td>38.0</td><td>53.5</td><td>46.5</td><td>+8.7</td></tr><tr><td>70B + BPO</td><td>70B</td><td>52.5</td><td>47.5</td><td>52.4</td><td>47.6</td><td>56.0</td><td>44.0</td><td>52.5</td><td>47.5</td><td>+6.7</td></tr><tr><td rowspan="2">vicuna-v1.3</td><td>7B + BPO</td><td>7B</td><td>65.0</td><td>35.0</td><td>56.7</td><td>43.3</td><td>54.0</td><td>46.0</td><td>53.0</td><td>47.0</td><td>+14.4</td></tr><tr><td>13B + BPO</td><td>13B</td><td>57.5</td><td>42.5</td><td>54.0</td><td>46.0</td><td>56.5</td><td>43.5</td><td>57.5</td><td>42.5</td><td>+12.8</td></tr></table>

Table 9: Win rates between BPO-aligned and original llama-2-chat and vicuna-v1.3 LLMs, evaluated by claude-v1.3. Training-free BPO improves alignment substantially, even making llama-2-13b-chat outperform llama-2-70b-chat. ("Self-instruct" denotes "Self-instruct, and "WR" denotes "win rates").  

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="2">Vicuna Eval</td><td colspan="2">Self-inst. Eval</td><td colspan="2">Dolly Eval</td><td colspan="2">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td></tr><tr><td rowspan="8">vicuna-7b-v1.3</td><td>PPO</td><td>ori.</td><td>53.8</td><td>46.2</td><td>48.8</td><td>51.2</td><td>52.5</td><td>47.5</td><td>52.5</td><td>47.5</td><td>+3.8</td></tr><tr><td>BPO</td><td>PPO</td><td>53.8</td><td>46.2</td><td>54.8</td><td>45.2</td><td>52.0</td><td>48.0</td><td>51.5</td><td>48.5</td><td>+6.0</td></tr><tr><td>BPO+PPO</td><td>ori.</td><td>57.5</td><td>42.5</td><td>51.2</td><td>48.8</td><td>57.5</td><td>42.5</td><td>56.5</td><td>43.5</td><td>+11.4</td></tr><tr><td>BPO+PPO</td><td>PPO</td><td>53.8</td><td>46.2</td><td>55.2</td><td>44.8</td><td>52.5</td><td>47.5</td><td>52.0</td><td>48.0</td><td>+6.7</td></tr><tr><td>DPO</td><td>ori.</td><td>53.8</td><td>46.2</td><td>54.8</td><td>45.2</td><td>55.0</td><td>45.0</td><td>58.0</td><td>42.0</td><td>+10.8</td></tr><tr><td>BPO</td><td>DPO</td><td>51.3</td><td>48.7</td><td>49.2</td><td>50.8</td><td>52.0</td><td>48.0</td><td>50.0</td><td>50.0</td><td>+1.2</td></tr><tr><td>BPO+DPO</td><td>ori.</td><td>62.5</td><td>37.5</td><td>62.3</td><td>37.7</td><td>57.5</td><td>42.5</td><td>62.0</td><td>38.0</td><td>+22.2</td></tr><tr><td>BPO+DPO</td><td>DPO</td><td>56.3</td><td>43.7</td><td>52.4</td><td>47.6</td><td>52.5</td><td>47.5</td><td>60.0</td><td>40.0</td><td>+10.6</td></tr><tr><td rowspan="8">vicuna-13b-v1.3</td><td>PPO</td><td>ori.</td><td>47.5</td><td>52.5</td><td>55.2</td><td>44.8</td><td>61.5</td><td>38.5</td><td>51.0</td><td>49.0</td><td>+7.6</td></tr><tr><td>BPO</td><td>PPO</td><td>52.5</td><td>47.5</td><td>52.0</td><td>48.0</td><td>58.0</td><td>42.0</td><td>55.5</td><td>44.5</td><td>+9.0</td></tr><tr><td>BPO+PPO</td><td>ori.</td><td>57.5</td><td>42.5</td><td>60.3</td><td>39.7</td><td>62.0</td><td>38.0</td><td>57.5</td><td>42.5</td><td>+18.7</td></tr><tr><td>BPO+PPO</td><td>PPO</td><td>51.3</td><td>48.7</td><td>52.8</td><td>47.2</td><td>58.0</td><td>42.0</td><td>53.5</td><td>46.5</td><td>+7.8</td></tr><tr><td>DPO</td><td>ori.</td><td>48.8</td><td>51.2</td><td>54.0</td><td>46.0</td><td>58.0</td><td>42.0</td><td>58.0</td><td>42.0</td><td>+9.4</td></tr><tr><td>BPO</td><td>DPO</td><td>55.0</td><td>45.0</td><td>48.8</td><td>51.2</td><td>49.0</td><td>51.0</td><td>50.0</td><td>50.0</td><td>+1.4</td></tr><tr><td>BPO+DPO</td><td>ori.</td><td>57.5</td><td>42.5</td><td>60.7</td><td>39.3</td><td>60.5</td><td>39.5</td><td>62.0</td><td>38.0</td><td>+20.4</td></tr><tr><td>BPO+DPO</td><td>DPO</td><td>63.8</td><td>36.2</td><td>56.7</td><td>43.3</td><td>53.5</td><td>46.5</td><td>54.0</td><td>46.0</td><td>+14.0</td></tr></table>

Table 10: Win rates between PPO, DPO, and BPO-aligned vicuna-v1.3 series LLMs, evaluated by claude-v1.3. BPO not only outperforms both PPO and DPO, and could yield additional bonus over PPO and DPO-aligned LLMs. ("Self-inst." denotes "Self-instruct", "ori." denotes "original", and "WR" denotes "win rates").

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="2">Vicuna Eval</td><td colspan="2">Self-inst. Eval</td><td colspan="2">Dolly Eval</td><td colspan="2">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td></tr><tr><td rowspan="2">llama-7b</td><td colspan="2">BPO-1k ori.-52k</td><td>72.5</td><td>27.5</td><td>52.4</td><td>47.6</td><td>58.5</td><td>41.5</td><td>54.5</td><td>45.5</td><td>+19.0</td></tr><tr><td colspan="2">BPO-52k ori.-52k</td><td>76.3</td><td>23.7</td><td>53.2</td><td>46.8</td><td>57.0</td><td>43.0</td><td>58.0</td><td>42.0</td><td>+22.2</td></tr><tr><td rowspan="2">llama-13b</td><td colspan="2">BPO-1k ori.-52k</td><td>77.5</td><td>22.5</td><td>61.1</td><td>38.9</td><td>61.5</td><td>38.5</td><td>64.0</td><td>36.0</td><td>+32.1</td></tr><tr><td colspan="2">BPO-52k ori.-52k</td><td>86.3</td><td>13.7</td><td>69.0</td><td>31.0</td><td>57.5</td><td>42.5</td><td>69.5</td><td>30.5</td><td>+41.1</td></tr></table>

Table 11: Win rates between BPO reproduced and original alpaca dataset tuned 11ama-1 series LLMs, evaluated by claude-v1.3. -1k means training the LLM with 1k randomly sampled data, -52k means using the whole dataset. ("Self-inst." denotes "Self-instruct," ori." denotes "original", and "WR" denotes "win rates").  

<table><tr><td rowspan="2">Base LLM</td><td colspan="2">Method</td><td colspan="2">Vicuna Eval</td><td colspan="2">Self-inst. Eval</td><td colspan="2">Dolly Eval</td><td colspan="2">BPO-test Eval</td><td rowspan="2">ΔWR</td></tr><tr><td>A</td><td>B</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td><td>A win</td><td>B win</td></tr><tr><td rowspan="3">gpt-3.5-turbo w/o feedback</td><td>BPO</td><td>ori.</td><td>63.8</td><td>36.2</td><td>56.3</td><td>43.7</td><td>60.0</td><td>40.0</td><td>58.5</td><td>41.5</td><td>+19.3</td></tr><tr><td>BPO</td><td>ori.</td><td>57.5</td><td>42.5</td><td>44.4</td><td>52.6</td><td>52.0</td><td>48.0</td><td>57.5</td><td>42.5</td><td>+6.5</td></tr><tr><td>BPO</td><td>w/o feedback</td><td>55.0</td><td>45.0</td><td>53.6</td><td>43.7</td><td>63.5</td><td>36.5</td><td>59.0</td><td>41.0</td><td>+16.2</td></tr></table>

Table 12: Win rates between BPO optimization and directly using gpt-3.5-turbo for prompt optimization (w/o feedback), evaluated by claude-v1.3. While using BPO can largely improve model performance, w/o feedback has little improvement. ("Self-instruct," denotes "Self-instruct," ori." denotes "original", and "WR" denotes "win rates").

<table><tr><td>Original</td><td>What if Alan Turing had not cracked the Enigma code during World War II?</td></tr><tr><td>Iteration 1</td><td>What would have been the consequences if Alan Turing had not cracked the Enigma code during World War II?</td></tr><tr><td>Iteration 2</td><td>What would have been the consequences if Alan Turing had not cracked the Enigma code during World War II in terms of the war&#x27;s duration, impact, and the Holocaust?</td></tr><tr><td>Iteration 3</td><td>What would have been the consequences if Alan Turing had not cracked the Enigma code during World War II in terms of the war&#x27;s duration, impact, and the Holocaust? Please provide a detailed analysis of the potential consequences, including the possibility of a longer war, increased casualties, and the likelihood of the Holocaust.</td></tr><tr><td>Iteration 4</td><td>What would have been the consequences if Alan Turing had not cracked the Enigma code during World War II in terms of the war&#x27;s duration, impact, and the Holocaust? Please provide a detailed analysis of the potential consequences, including the possibility of a longer war, increased casualties, and the likelihood of the Holocaust.</td></tr><tr><td>Iteration 5</td><td>What would have been the consequences if Alan Turing had not cracked the Enigma code during World War II in terms of the war&#x27;s duration, impact, and the Holocaust? Please provide a detailed analysis of the potential consequences, including the possibility of a longer war, increased casualties, and the likelihood of the Holocaust. Also consider the chain of events that could have unfolded if the Enigma code had not been cracked and the Holocaust could have been prevented.</td></tr></table>

Figure 8: An example of iterative optimization. The refined parts are marked as red in each iteration compared with the last iteration.

# GPT-4 Reference-based Scoring Prompt

# System message:

You are a helpful assistant.

# Prompt template:

# [Instruction]

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user question displayed below. Your evaluation should consider factors such as the helpfulness, relevance, accuracy, depth, creativity, and level of detail of the response. You will be given a high-quality reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the reference answer and identify the mistakes in the assistant's answer, then provide a short explanation. Be as objective as possible. After providing your explanation, you must rate the response on a scale of 1 to 10 by strictly following this format: "[rating] ", for example: "Rating: [[5]]".

# [Question]

{question}

# [The Start of Reference Answer]

# {ref_answer_1}

[The End of Reference Answer]

# [The Start of Assistant's Answer]

# {answer}

[The End of Assistant's Answer]

Figure 9: Reference-based evaluation prompt for gpt-4.  
Figure 10: Differences in GPT-4 scores after optimization with OPRO and BPO compared to the original. In contrast to OPRO, BPO demonstrates consistent gains across nearly all tasks, whereas OPRO exhibits performance declines on over half of the tasks with an average negative improvement. For both BPO and OPRO, we run three times and calculate the average scores.

# Footnotes:

Page 0: *JC and XL made equal contributions. 2Work done when JC interned at Zhipu AI. 
Page 3: <https://huggingface.co/spaces/lmsys-chatbot-arena-leaderboard> 
Page 5: 2https://huggingface.co/meta-llama/ Llama-2-7b-chat-hf 
