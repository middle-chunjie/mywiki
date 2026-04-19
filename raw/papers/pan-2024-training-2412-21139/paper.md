# Training Software Engineering Agents and Verifiers with SWE-Gym

Jiayi Pan  $^{*1}$  Xingyao Wang  $^{*2}$  Graham Neubig  $^{3}$  Navdeep Jaitly  $^{4}$  Heng Ji  $^{2}$  Alane Suhr  $^{†1}$  Yizhe Zhang  $^{†4}$

# Abstract

We present SWE-Gym, the first environment for training software engineering (SWE) agents. SWE-Gym contains 2,438 real-world task instances, each comprising a Python codebase with an executable runtime environment, unit tests, and a task specified in natural language. We use SWE-Gym to train language model based SWE agents, and achieve up to  $19\%$  absolute gains in resolution rate on the popular SWE-Bench Verified and Lite test sets. We also experiment with inference-time scaling through verifiers trained on agent trajectories sampled from SWE-Gym. When combined with our fine-tuned SWE agents, we achieve  $32.0\%$  and  $26.0\%$  on SWE-Bench Verified and Lite, respectively, reflecting a new state-of-the-art for open-weight SWE agents. To facilitate further research, we publicly release SWE-Gym, models, and agent trajectories.

# 1. Introduction

Language models (LMs) have remarkable promise in automating software engineering (SWE) tasks, as most clearly measured by recent progress on benchmarks like SWE-Bench (Jimenez et al., 2024) and Commit0 (Zhao et al., 2024). While LM-based SWE agents have shown significant performance gains through improving agent-computer interfaces (Yang et al., 2024) and prompting strategies (Wang et al., 2024c), advances in SWE agents have been limited by a reliance on proprietary models, with limited research to improve the underlying LM itself.

Unlike other domains where supervised fine-tuning and reinforcement learning have significantly improved LM capabilities, such as chat (Ouyang et al., 2022), math reason-

* Equal contribution. † Equal supervision. ¹UC Berkeley ²UIUC ³CMU ⁴Apple. Correspondence to: Jiayi Pan <jiayipan@berkeley.edu>, Xingyao Wang <xingyao6@illinois.edu>, Alane Suhr <suhr@berkeley.edu>, Yizhe Zhang <yiz-zhang@apple.com>.

Proceedings of the  $42^{nd}$  International Conference on Machine Learning, Vancouver, Canada. PMLR 267, 2025. Copyright 2025 by the author(s).


Figure 1: SWE-Gym enables scalable improvements for software engineering agents. Top: Scaling the amount of training data shows consistent performance improvements as we obtain more training trajectories, with no signs of saturation at 491 trajectories. We use temperature  $t = 0$  for evaluation. Bottom: For inference time scaling, we generate a number of candidate trajectories per task and select the best using a verifier trained on SWE-Gym. This approach demonstrates roughly log-linear gains with the number of sampled solutions.  $t = 0$  (excluded from regression) is used as the first hypothesis to be consistent with the top figure; later rollouts use  $t = 0.5$ .

ing (Shao et al., 2024; Yuan et al., 2024), and web navigation (Pan et al., 2024), software engineering currently lacks suitable training environments, and creating environments is uniquely challenging. Real-world software engineering requires interaction with an executable runtime that has been prepared with the appropriate software dependencies and reproducible test suites, among other requirements. These challenges are reflected in the existing resources (Tab. 1). For example, the SWE-Bench (Jimenez et al., 2024) training split contains only solutions (git patches that solve the task), missing the step-by-step actions taken by the developer to create each solution, and executable environments and re

ward signals. R2E (Jain et al., 2024) uses synthetic tasks that are very far from real-world problems, while datasets such as APPS (Hendrycks et al., 2021a) focus only on isolated tasks rather than realistic repository-level coding problems.

To bridge this gap, we present SWE-Gym, the first training environment combining real-world software engineering tasks from GitHub issues with pre-installed dependencies and executable test verification. SWE-Gym contains 2,438 Python tasks sourced from 11 popular open-source repositories (Tab. 2), providing useful environments for training LMs as agents and verifiers.

SWE-Gym supports training state-of-the-art openweight SWE agents. Based on the OpenHands (Wang et al., 2024c) agent scaffold for general-purpose software development (§2), we fine-tune a 32B Qwen-2.5 coder model (Hui et al., 2024b) using only 491 agent-environment interaction trajectories sampled using SWE-Gym, and achieve substantial absolute improvements of +12.3% (to 15.3%) and +13.6% (to 20.6%) in resolution rate on SWE-Bench Lite and SWE-Bench Verified respectively (§4.2).

SWE-Gym is effective across agent scaffolds. In another agent scaffold based on a specialized workflow (MoatlessTools; Orwall 2024; §2), we experiment with self-improvement, where the LM interacts with SWE-Gym, receives reward from it, and learns to improve itself through rejection sampling fine-tuning. This self-improvement boosts performance up to  $19.7\%$  on SWE-Bench Lite.

SWE-Gym supports training verifier models to enable inference-time scaling. We use test suites included in SWE-Gym to determine whether sampled agent trajectories are successful or not. Given these samples, we train a verifier model (i.e., an outcome-supervised reward model; Cobbe et al., 2021) that estimates a trajectory's probability of success. This enables inference-time scaling, where we sample multiple agent trajectories and select the one with the highest estimated reward according to the verifier. This further improves the resolution rate to  $32.0\%$  ( $+11.4\%$  absolute improvement) on SWE-Bench Verified (§5.1.1; Fig. 1 bottom) and  $26.0\%$  on SWE-Bench Lite (§5.1.2), establishing a new state-of-the-art among systems with publicly accessible weights (Tab. 9).

Our baseline training and inference-time scaling methods on SWE-Gym yield continuously improved results with increasing compute (Fig. 1). In the training phase, performance scales with the number of sampled trajectories up to our current limit of 491 trajectories, suggesting that performance is currently limited by the compute budget for sampling rather than the number of tasks in SWE-Gym. Similarly, using the agent and verifier trained by SWE-Gym, the bottom panel shows that using more compute during inference time steadily improves the performance.

# 2. Related Work

Agents that solve GitHub issues. We focus on software engineering agents designed to automatically resolve GitHub issues within the SWE-Bench framework (Jimenez et al., 2024). These agents take a GitHub issue and its associated code repository as input and generate a valid code modification (i.e., a git diff patch) to address the issue. The correctness of these modifications is verified using a human-written test suite. Existing agent designs are categorized by the extent of human priors integrated into their workflows: Specialized workflows (Xia et al., 2024; Orwell, 2024; Zhang et al., 2024b; Chen et al., 2024) involve human-defined stages (e.g., localization, code editing, patch re-ranking), where a LM is iteratively prompted for each stage to produce the final result. This approach reduces the task horizon and minimizes the need for long-term planning. However, specialized workflows require significant human engineering, may not generalize to novel issue types, and can fail if intermediate steps encounter problems. In contrast, general-purpose prompting ((Yang et al., 2024; Wang et al., 2024c)) rely on LM's ability to plan over long horizons and generate actions based on a history of interactions without heavily pre-defined workflows. While more flexible, general approaches demand higher capabilities from the underlying LM and can be computationally expensive due to multiple interaction rounds. The most successful existing SWE agents are built on proprietary language models like GPT-4 or Claude and utilize specialized workflows to overcome these models' limitations. This contrasts with other sequential decision-making domains (Silver et al., 2017; Akkaya et al., 2019), where learning-based approaches, such as reinforcement learning, drive success by enabling systems to learn from interactions and rewards to develop task competence. A key barrier in the SWE agent domain is the lack of appropriate training environments. Our experiments show that SWE-Gym can be used to build strong learning-based agents, accelerating research in this area.

**Environments for training software agents.** There is no existing dataset suitable for training software engineering agents. SWE-Bench (Jimenez et al., 2024) is widely used for evaluating software engineering performance, but its training split lacks executable environments and success signals present in the evaluation split, making it useful only for imitation learning approaches. HumanEval (Chen et al., 2021) is designed for standalone code generation tasks, akin to coding competitions. Therefore, it falls short of addressing the complex challenges inherent in real-world, repository-level software engineering tasks, which involve thousands of files, millions of lines of code, and tasks such as bug fixing, feature development, and system optimization. Similarly, R2E (Jain et al., 2024) is a small evaluation dataset

Table 1: SWE-Gym is the first publicly available training environment combining real-world SWE tasks from GitHub issues with pre-installed dependencies and executable test verification. Repository-level: whether each task is situated in a sophisticated repository; Executable Environment: whether each task instance comes with an executable environment with all relevant dependencies pre-installed; Real task: whether task instruction is collected from human developers.  

<table><tr><td>Dataset (split)</td><td>Repository-Level</td><td>Executable Environment</td><td>Real task</td><td>#Instances (total)</td><td>#Instances (train)</td></tr><tr><td>CodeFeedback (Zheng et al., 2024b)</td><td>X</td><td>X</td><td>✓</td><td>66,383</td><td>66,383</td></tr><tr><td>APPS (Hendrycks et al., 2021a)</td><td>X</td><td>✓</td><td>✓</td><td>10,000</td><td>5,000</td></tr><tr><td>HumanEval (Chen et al., 2021)</td><td>X</td><td>✓</td><td>✓</td><td>164</td><td>0</td></tr><tr><td>MBPP (Tao et al., 2024)</td><td>X</td><td>✓</td><td>✓</td><td>974</td><td>374</td></tr><tr><td>R2E (Jain et al., 2024)</td><td>✓</td><td>✓</td><td>X</td><td>246</td><td>0</td></tr><tr><td>SWE-Bench (train) (Jimenez et al., 2024)</td><td>✓</td><td>X</td><td>✓</td><td>19,008</td><td>19,008</td></tr><tr><td>SWE-Gym Raw</td><td>✓</td><td>X</td><td>✓</td><td>64,689</td><td>64,689</td></tr><tr><td>SWE-Bench (test) (Jimenez et al., 2024)</td><td>✓</td><td>✓</td><td>✓</td><td>2,294</td><td>0</td></tr><tr><td>SWE-Gym</td><td>✓</td><td>✓</td><td>✓</td><td>2,438</td><td>2,438</td></tr></table>

with 246 instances and, due to its synthetic nature, lacks the realism and complexity in real-world software engineering scenario. Our proposed SWE-Gym instead uses real-world GitHub issues as task, and associated executable unit tests for evaluation. This results in realistic and complex task formulations, aligning closely with real-world challenges.

Post-training: From chatbots and reasoners to agents. Post-training, which fine-tunes pre-trained language models using supervised or reinforcement learning, significantly improves model performance across various domains. Techniques like RLHF (Ouyang et al., 2022) have become standard for adapting language models into chatbots, improving both performance and alignment (Qwen Team, 2024). In math reasoning, datasets such as MATH (Hendrycks et al., 2021b) and GSM-8K (Cobbe et al., 2021) facilitate the training and evaluation of policy and verifier models (Cobbe et al., 2021; Wang et al., 2024a). Earlier works (Wang et al., 2024b; Chen et al., 2023; Zeng et al., 2023; Wu et al., 2024) demonstrate that distilling agent trajectories from stronger models improve weaker models. Recent studies (Xi et al., 2024; Zhai et al., 2024; Bai et al., 2024) explore self-improving methods, showing that reinforcement learning or rejection sampling fine-tuning guided by reward enables LMs to enhance themselves without more capable teachers.

However, post-training typically depends on expert demonstration data or training environments with reliable reward signals, which are largely absent in the software engineering domain. This has led to a reliance on prompting-based methods with proprietary language models. Our work addresses this gap with SWE-Gym, a training environment based on real-world software engineering tasks that uses expert-written tests as reward signals. Our experiments demonstrate that SWE-Gym can build strong SWE agents without prompt engineering.

# 3. SWE-Gym Environment

SWE-Gym comprises 2,438 real-world software engineering tasks sourced from pull requests in 11 popular Python repositories, with pre-configured executable environments and expert-validated test cases, constructed in close alignment with SWE-Bench (Jimenez et al., 2024). These repositories are separate from those used in SWE-Bench to avoid contamination. These tasks require SWE agents to develop test-passing solutions for real-world GitHub issues using provided codebases and executable environments. Such agents must map from natural language descriptions of the issue, as well as the initial state of the repository, to a pull request represented as a git patch.

We also identify a subset of 230 tasks, SWE-Gym Lite, which contains generally easier and more self-contained tasks that are suitable for rapid prototyping, in alignment with SWE-Bench Lite (Jimenez et al., 2024). To support future research in SWE agent development and automatic dataset synthesis, we also release SWE-Gym Raw, a large set of Python GitHub issues without executable environments (64,689 instances spanning 358 Python repositories).

# 3.1. Dataset Construction

Identify Repositories. We first use SEART GitHub search<sup>1</sup> to filter a list of initial repositories. Unlike SWE-Bench, which focuses on the top 5k most downloaded PyPI libraries (Jimenez et al., 2024), we select Python repositories that were created before July 1, 2022 and have more than 500 stars, with at least 300 lines of code, more than 500 pull requests (PRs) and 100 contributors. This results in 358 repositories.

Extracting Training Instances from Repositories. We use SWE-Bench's instance extraction script to convert these repositories into task instances, each corresponding to a

<table><tr><td>Category</td><td>Metric</td><td>SWE-Gym</td><td>SWE-Gym Lite</td></tr><tr><td rowspan="2">Size</td><td># Instances</td><td>2,438 (2,294)</td><td>230 (300)</td></tr><tr><td># Repos</td><td>11 (12)</td><td>11 (12)</td></tr><tr><td>Issue Text</td><td>Length by Words</td><td>239.8 (195.1)</td><td>186.2 (175.9)</td></tr><tr><td rowspan="2">Codebase</td><td># Non-test Files</td><td>971.2 (2944.2)</td><td>818.8 (2988.5)</td></tr><tr><td># Non-test Lines</td><td>340675.0 (363728.4)</td><td>340626.2 (377562.4)</td></tr><tr><td rowspan="3">Gold Patch</td><td># Lines edited</td><td>69.8 (32.8)</td><td>10.6 (10.1)</td></tr><tr><td># Files edited</td><td>2.5 (1.7)</td><td>1.0 (1.0)</td></tr><tr><td># Func. edited</td><td>4.1 (3.0)</td><td>1.4 (1.34)</td></tr><tr><td rowspan="2">Tests</td><td># Fail to Pass</td><td>10.0 (9.0)</td><td>2.04 (3.5)</td></tr><tr><td># Total</td><td>760.8 (132.5)</td><td>99.9 (85.2)</td></tr></table>

Table 2: Statistics comparing SWE-Gym with the SWE-Bench test split (in parenthesis). Except for size metrics, we report the average value across instances.


Figure 2: Repository distribution of SWE-Gym instances.

GitHub issue including the natural language description of the issue, a snapshot of the repository in which the issue was created, and a set of unit tests. Over the 358 repositories, we extract 64,689 task instances. We refer to this dataset as SWE-Gym Raw, which is over three times larger than the 19k instances gathered in previous work (Jimenez et al., 2024) and includes nearly ten times as many repositories.

While SWE-Gym Raw instances contain code, issue descriptions, and the solution, they do not contain executable environments or a guarantee that its unit tests are effective in evaluating the correctness of a solution. Thus, we focus on 11 repositories with numerous instances and semi-manually create executable environments for them.

Version Training Instances. Associating instances with their respective version numbers (e.g. 1.2.3) and setting up environments version-by-version makes the environment collection process more practical by avoiding redundant setup work. We generalize SWE-Bench's versioning script to support versioning via script execution, and semi-automatically collect versions for each instance based on information available in the repository (e.g., pyproject.toml, git tag, etc).

Setup Executable Environments and Verify Instances. Creating executable environments with pre-installed dependencies is crucial for developing software engineering agents, as it mirrors deployment settings and allows for incremental unit test feedback. Configuring dependencies for specific codebase versions is challenging due to the lack of a universal Python package installation method and backward compatibility issues, especially for older GitHub issues. Ignoring these environments could introduce distribution bias, diminishing SWE-Gym's utility. To address this, we manually configure dependencies for each task instance using relevant configuration files (e.g., requirements.txt), CI scripts, or documentation from the repository snapshot at the time of issue creation. We then use SWE-Bench's

execution-based validation script to ensure that the gold patch (the human-submitted code diff) passes more unit tests than the original code. This process required approximately 200 human annotation hours $^{2}$  and 10,000 CPU core hours. After validation and filtering out failed instances, we obtained 2,438 unit-test-validated instances from 11 repositories. For full reproducibility, we publicly release pre-built Docker images for each instance, totaling 6 TB.

# 3.2. SWE-Gym Lite

Solving software engineering tasks is computationally intensive, costing usually $1 or more per task with frontier models (Wang et al., 2024c). To improve research efficiency via faster agent evaluation, Jimenez et al. (2024) introduce SWE-Bench Lite, a canonical subset of 300 instances from SWE-Bench. Following the SWE-Bench Lite filtering pipeline,<sup>3</sup> we delineate the SWE-Gym Lite split, comprising 230 instances. Similar to SWE-Bench Lite, this subset excludes tasks that require editing more than one file, tasks with poorly described problem statements, those with excessively complex ground-truth code diffs, and tests focused on error message validation.

# 3.3. Dataset Statistics

Fig. 2 illustrates that the task distribution across repositories exhibits a long-tail pattern. Notably, tasks associated with pandas comprise nearly one-third of the total, whereas tasks related to bokeh represent a mere one percent.

Our analysis suggests that tasks in SWE-Gym are on average harder than those included in SWE-Bench. Tab. 2 shows that SWE-Gym has statistics similar to SWE-Bench, with several key differences. Codebases in SWE-Gym, on average, have relatively fewer files than SWE-Bench, but a

similar number of total lines of code. However, gold patches in SWE-Gym have significantly more lines and files edited when compared to SWE-Bench's gold patches. Additionally, we find models have consistently lower performance on SWE-Gym compared to SWE-Bench.<sup>4</sup> Beyond models and scaffolds overfitting to SWE-Bench, the decreased performance on SWE-Gym may also be due to our inclusion of sophisticated repositories like pandas and MONAI.

# 4. Training LMs as Agents with SWE-Gym

We experiment with training language model agents using SWE-Gym. We use two agent scaffolds (OpenHands, Wang et al. 2024c, §4.2; Moatless Tools, Orwell 2024, §4.3).

# 4.1. Setting

Agent Scaffolds. Recent LM-based SWE agents comprise a base language model, and a set of tools and prompts this base model has access to. This set of tools and prompting strategies is referred to as an agent scaffold, and recent work has developed numerous scaffolds for different purposes (refer to §2 for examples). We experiment with two types of agent scaffolds: one for general-purpose prompting (OpenHands CodeAct; Wang et al. 2024c) and one for specialized workflows (MoatlessTools; Orwell 2024), which allows us to measure the efficacy of SWE-Gym across diverse deployment settings.

Policy Improvement Algorithm. We use SWE-Gym to improve the underlying LM for a given SWE agent. As a baseline, we employ a simple policy improvement algorithm: rejection sampling fine-tuning (a.k.a. filtered behavior cloning), where we fine-tune the base LM on success trajectories sampled from SWE-Gym.

Evaluation Metrics. We use the standard SWE agent benchmarks SWE-Bench Lite and Verified (Jimenez et al., 2024) for evaluation. We report (1) resolution rate (\%), the proportion of resolved task instances, and (2) Empty Patch (\%), the proportion of trajectories where none of the code in the repository is edited. We use OpenHands remote runtime (Neubig & Wang, 2024) to parallelize evaluation (e.g., execute unit tests).

Technical Details. For base LMs, we use Qwen-2.5-Coder-Instruct (Hui et al., 2024a) 7B, 14B, and 32B. §B.2 contains training run details.

# 4.2. Training General-Purpose Prompting Agents

In this section, we use OpenHands (version CodeActAgent 2.1, Wang et al. 2024b;c) as our agent scaffold, which is based on general-purpose ReAct-style prompting (Yao et al.,

2023). In contrast to specialized-workflows-agents (§2), it relies on the LM to generate actions and do planning. It equips the base LM with a bash terminal and a file editor. We disable the browser feature of OpenHands in this work.

Trajectory Collection. By rejection sampling, we obtain 491 successful trajectories from SWE-Gym. These trajectories are sampled from gpt-4o-2024-08-06 and claude-3-5-sonnet-20241022 with different temperature settings. Each successful trajectory, on average, has roughly 19 turns and approximately 19,000tokens. Although SWE-Gym offers many more tasks and allows repeated sampling, our 491 trajectories are limited primarily by computational budget.

Training on SWE-Gym trajectories turns LM into effective agents to fix issues. As shown in Tab. 3, the pretrained base model achieves resolution rates of  $3.0\%$  and  $7.0\%$  on SWE-Bench Lite and Verified, respectively. After fine-tuning on 491 trajectories<sup>6</sup>, it improves by up to  $12.3\%$ $(3.0\% \rightarrow 15.3\%)$  and  $13.6\%$ $(7.0\% \rightarrow 20.6\%)$ .

Training reduces stuck-in-loop behavior. For agent tasks, open-weight LMs often get stuck in loops, where the model perpetually generates the same action for multiple turns, especially when prompted with general-purpose prompts (§2). Thus, we report Stuck in Loop (%), the percentage of trajectories where the agent repeats the same action three times consecutively. As shown in Tab. 3, zero-shot pretrained models often get stuck in loops; even the largest 32B model is trapped in  $29.4\%$  of SWE-Bench Verified tasks. Fine-tuning on trajectories from SWE-Gym consistently reduces the stuck-in-loop rate by  $4.6 - 18.6\%$  across both SWE-Bench Lite and Verified tasks, except for the 32B model on SWE-Bench Lite, which increases by  $1.5\%$  due to its already low loop rate. This coincides with a decrease in the empty patch rate, likely enabling the agent to perform more code edits.

Performance scales with model size. Rather unsurprisingly, larger base models consistently improve the resolution rate, empty patch rate, and stuck-in-loop rate (Tab. 3).

Self-improvement remains ineffective. In addition to fine-tuning on trajectories sampled from strong teacher models, we also experiment with fine-tuning on trajectories sampled directly from the policy being updated. We use the fine-tuned 32B model to sample 6 trajectories per SWE-Gym instance (using temperature  $t = 0.5$ ), obtaining 868 successful trajectories (i.e., on-policy trajectories). We further fine-tune the base 32B model on a mixture of 868 on-policy trajectories and the previously collected 491 off-policy trajectories. When evaluating this fine-tuned model on SWE-Bench Lite, we observe the resolution rate drop from 15.3

Table 3: Model performance (fine-tuned on 491 SWE-Gym-sampled trajectories) on SWE-Bench (Jimenez et al., 2024) using OpenHands (Wang et al., 2024c) as agent scaffold. We use Qwen-2.5-Coder-Instruct as the base model.  

<table><tr><td rowspan="2">Model Size</td><td colspan="3">Empty Patch (%, ↓)</td><td colspan="3">Stuck in Loop (%, ↓)</td><td colspan="3">Avg. Turn(s)</td><td colspan="3">Resolve Rate (%, ↑)</td></tr><tr><td>zero-shot</td><td>fine-tuned</td><td>Δ</td><td>zero-shot</td><td>fine-tuned</td><td>Δ</td><td>zero-shot</td><td>fine-tuned</td><td>Δ</td><td>zero-shot</td><td>fine-tuned</td><td>Δ</td></tr><tr><td colspan="13">SWE-Bench Lite (300 instances)</td></tr><tr><td>7B</td><td>40.3</td><td>29.7</td><td>-10.7</td><td>47.0</td><td>31.0</td><td>-16.0</td><td>20.3</td><td>22.2</td><td>+1.9</td><td>1.0 (± 1.0)</td><td>10.0 (± 2.4)</td><td>+9.0</td></tr><tr><td>14B</td><td>49.7</td><td>18.1</td><td>-31.6</td><td>31.7</td><td>27.1</td><td>-4.6</td><td>23.2</td><td>21.4</td><td>-1.8</td><td>2.7 (± 1.9)</td><td>12.7 (± 2.3)</td><td>+10.0</td></tr><tr><td>32B</td><td>27.0</td><td>18.1</td><td>-8.9</td><td>16.7</td><td>18.1</td><td>+1.5</td><td>15.5</td><td>29.3</td><td>+13.9</td><td>3.0 (± 1.4)</td><td>15.3 (± 2.5)</td><td>+12.3</td></tr><tr><td colspan="13">SWE-Bench Verified (500 instances)</td></tr><tr><td>7B</td><td>45.8</td><td>33.8</td><td>-12.0</td><td>39.6</td><td>21.0</td><td>-18.6</td><td>21.9</td><td>35.3</td><td>+13.4</td><td>1.8 (± 1.1)</td><td>10.6 (± 2.1)</td><td>+8.8</td></tr><tr><td>14B</td><td>44.9</td><td>14.5</td><td>-30.4</td><td>32.1</td><td>21.3</td><td>-10.7</td><td>25.5</td><td>30.1</td><td>+4.6</td><td>4.0 (± 1.6)</td><td>16.4 (± 2.0)</td><td>+12.4</td></tr><tr><td>32B</td><td>9.5</td><td>13.8</td><td>+4.3</td><td>29.4</td><td>23.8</td><td>-5.6</td><td>24.6</td><td>31.6</td><td>+7.0</td><td>7.0 (± 1.3)</td><td>20.6 (± 2.1)</td><td>+13.6</td></tr></table>

to  $8.7\%$ , suggesting that self-improvement is not yet working. We hypothesize that we could achieve improved results using more advanced policy optimization methods, such as proximal policy optimization (PPO) (Schulman et al., 2017), or with a stronger base model. These directions remain promising avenues for future investigation.

# 4.3. Self-Improvement with Specialized Workflow

Unlike OpenHands, which offers freedom in long-horizon planning, MoatlessTools constrains the language model's action space to pre-defined specialized workflows, reducing task horizons. Specialized workflows outperform general-purpose prompting for open-weight LMs. In Tab. 3 and Tab. 4, the 7B and 32B LM achieve zero-shot resolution rates of  $7\%$  and  $19\%$  with MoatlessTools, compared to  $1.0\%$  and  $3.0\%$  with OpenHands on SWE-Bench Lite.

Given MoatlessTools' improved zero-shot performance and shorter task horizon, we hypothesize that self-improvement without a strong teacher is achievable using this scaffold and training on SWE-Gym. With a limited compute budget, we conduct this experiment with only 7B and 32B models, using LoRA (Hu et al., 2022) for the 32B model for improved efficiency. We use the 7B model for ablation experiments.

We use iterative rejection sampling fine-tuning for policy improvement. Each iteration involves (a) performing 30 high-temperature (1.0) rollouts per task on SWE-Gym-Lite and adding successful trajectories to the fine-tuning dataset, and (b) fine-tuning the policy on these filtered trajectories. After two iterations, further improvements are negligible.

Data Bias Impacts Performance. Repeated sampling, as in Brown et al. (2024), shows that task success probability follows a long-tail distribution (Fig. 6), where more samples increase solved instances. While broader task coverage benefits training, it introduces a bias toward easier tasks, making it suboptimal to train on all successful trajectories, as first observed in math reasoning Tong et al. (2024).

Mitigating Bias with Per-Instance Capping. We introduce per-instance capping—a method that limits the maximum number of selected samples per task. As illustrated in Fig. 6,

Table 4: resolution rate (RR) and Empty patch rate (EP) on SWE-Bench Lite with the MoatlessTools Scaffold after online rejection sampling fine-tuning (temperature  $t = 0$ ).  

<table><tr><td rowspan="2">Setting</td><td colspan="2">7B Model</td><td colspan="2">32B Model</td></tr><tr><td>EP(%,↓)</td><td>RR(%,↑)</td><td>EP(%,↓)</td><td>RR(%,↑)</td></tr><tr><td>Zero-Shot</td><td>56.3%</td><td>7.0%</td><td>24.3%</td><td>19.0%</td></tr><tr><td>Iteration 1</td><td>29.0%</td><td>9.0%</td><td>18.3%</td><td>19.7%</td></tr><tr><td>Iteration 2</td><td>23.3%</td><td>10.0%</td><td>9.7%</td><td>19.7%</td></tr></table>

this balances dataset bias and size. A low cap reduces dataset size and performance (§5.2), while a high cap skews the distribution toward easier tasks. Empirically, a threshold of 2 achieves a good balance, slightly outperforming the full dataset and improving training speed (Tab. 6). We rank trajectories by the number of model response rounds required, preferring fewer.

Results. Results. After two policy improvement iterations (Tab. 4), the 7B model's resolution rate increased from  $7.0\%$  to  $9.0\%$  after the first iteration and to  $10.0\%$  after the second. In contrast, the 32B model improved from  $19.0\%$  to  $19.7\%$  after the first iteration with no further gains. We attribute the limited gains in the 32B model to the scaffold's restricted action space and the rejection sampling fine-tuning method.

# 5. Scaling Agent Performance with SWE-Gym

We explore two scaling directions enabled by SWE-Gym to enhance agent performance: inference-time scaling (§5.1) and training-time data scaling (§5.2).

# 5.1. Inference-Time Scaling with Verifiers

Trajectories sampled from SWE-Gym can be used not only for training a policy, but also for training a verifier (i.e., reward) model. We train an outcome-supervised reward model (ORM) (Cobbe et al., 2021) that, given the relevant context of the task execution (including the problem statement, agent trajectory, and current git diff), generates a score that estimates the probability that the agent has solved the problem. We experiment with using this model to rerank

candidate trajectories sampled from a SWE agent policy, and show that such learned verifiers enable effective inference-time scaling for further performance improvement.

# 5.1.1. VERIFIER FOR GENERAL-PURPOSE PROMPTING

For OpenHands agents (Wang et al., 2024b;c) with general-purpose prompting (§2), we train a verifier (ORM) that takes as input the trajectory  $\tau = [o_1, a_1, o_2, a_2, \ldots, o_n, a_n]$ , represented as an interleaved sequence of observations and actions, and generates a scalar reward  $r \in [0,1]$ . Observations  $o_k$  include the task problem statement, command execution output, error messages, etc; action  $a_k$  can be bash command or file operations (e.g., edit, view) from the agent.

Training and Inference. We fine-tune 32B Qwen2.5-Coder-Instruct to label trajectories as successful or unsuccessful using output tokens <YES> and <NO> respectively. For training data, we re-use two sets of trajectories we sampled on SWE-Gym for agent training in §4.2: (1) off-policy trajectories which contain 443 successful trajectories; (2) on-policy trajectories which contain 875 successful trajectories sampled from the fine-tuned Qwen2.5-Coder-Instruct-32B. We combine both on-policy and off-policy trajectories, randomly sample the same amount of unsuccessful trajectories from each subset (1,318 each), and combine them as our dataset for verifier training (total 2,636 trajectories). We fine-tune the model to predict <YES> for successful trajectories and <NO> for unsuccessful ones.

At inference time, conditioned on the prompt and the agent trajectory  $\tau$ , we use SGLang (Zheng et al., 2024a) to obtain the log probability of the next token being  $<\mathrm{YES}>$  ( $l_y$ ) or  $<\mathrm{NO}>$  ( $l_n$ ). We then calculate the reward as the probability of success by normalizing the log probability:  $r = \exp(l_y) / (\exp(l_y) + \exp(l_n))$ .

Metrics. We report two metrics: (1)  $\mathrm{Pass}@\mathbf{k}$ , the proportion of tasks with at least one successful solution among  $k$  samples, and (2)  $\mathrm{Best}@\mathbf{k}$ , the success rate of the highest-reward trajectories selected by the verifier from  $k$  samples per task.  $\mathrm{Pass}@\mathbf{k}$  measures solution discovery (upper bound for  $\mathrm{Best}@\mathbf{k}$ );  $\mathrm{Best}@\mathbf{k}$  evaluates verifier accuracy. Mean and variance calculation are detailed in §B.1, following Lightman et al. (2023).

Results. Fig. 3 shows how Pass@k and Best@K scale with the number of sampled agent trajectories using the finetuned 32B model as the agent model. Pass@k demonstrates strong improvement, rising from 20.6 to  $37.8\%$  resolution rate as  $k$  increases from 1 to 8, and up to  $42.8@k = 16$ . The Best@k metric, which relies on our verifier's ability to se

Figure 3: Increasing inference-time compute improves performance on SWE-Bench Verified with a learnt verifier. Both the agent and the verifier are a Qwen2.5-Coder-Instruct-32B model fine-tuned on the corresponding dataset (§5.1.1). OpenHands is used as the agent scaffold.

lect the best trajectory, demonstrates more modest but steady progress, improving from a resolution rate of  $20.6@1$  to  $29.8@8$ , and up to  $32.0@16$ . The gap between Pass@k and Best@k, due to the imperfect performance of our trained verifier, indicates there is room for improvements in reward modeling for coding agents. Surprisingly, we found that fine-tuning the verifier model using LoRA (Hu et al., 2022)  $(29.8@8)$  with Unsloth (Unsloth Team, 2024) performs better than full-parameter fine-tuning  $(27.2@8)$ , potentially due regularization. Furthermore, as shown in Fig. 1 (bottom), the Best@k curve exhibits strong linearity on a logarithmic scale, indicating a promising scaling behavior.

Training data matters for verifier. We experiment with variations on the choice of training data for our verifier model. Using full-parameter fine-tuning on Qwen-2.5-Coder-Instruct-32B, we use different mixtures of on- and off-policy trajectories, as well as different distributions of successful and unsuccessful trajectories. As shown in Fig. 8, our ablation study demonstrates that the choice of training data can significantly impact verifier performance. Training with a mixture of off-policy and on-policy data yields the best results (our default setting), reaching a resolution rate of  $27@8$ . In contrast, using only on-policy data from the fine-tuned model shows moderate but limited improvement, while training exclusively on off-policy data from Claude and GPT leads to early performance plateaus around  $22\%$  resolution rate. Our findings indicate that verifier training benefits most from a diverse dataset combining both off-policy and on-policy examples.

# 5.1.2. VERIFIER FOR SPECIALIZED WORKFLOW

For MoatlessTools agents with specialized workflows, given that it doesn't have a turn-taking action-observation trajectory like OpenHands CodeActAgent, we prepare verifier

Figure 4: Scaling inference-time compute for MoatlessTools Agents (32B) with learned verifiers on SWE-Bench Lite. Temperature  $t = 0.5$ .

inputs through a parsing process adopted from Zhang et al. (2024a), which combines task descriptions, relevant agent context, and generated patches.<sup>9</sup> We train the verifier to map from this input to a single token indicating task success.

Following the training procedure described in §5.1.1, we train 7B and 32B verifiers using on-policy trajectories from the last (2nd round of sampling, applying LoRA (Hu et al., 2022). To address the easy-data bias in the training dataset, we cap the number of successful trajectories per instance at two and balance the data by subsampling failure cases to match the same number of successful ones.

Results. We evaluate the verifiers by sampling from an agent policy with  $k = 8$  at temperature 0.5. As shown in Fig. 4 and Fig. 7, these verifiers enable effective scaling across verifier and policy sizes: the 7B verifier improves from 10 to  $13.3\%$  resolution rate on SWE-Bench Lite when paired with a 7B policy, while the 32B verifier improves from 19.7 to  $26.3\%$  when paired with a 32B policy. The 7B verifier plateaus after  $k = 4$  samples when ranking trajectories from both 7B and 32B agents. In contrast, the 32B verifier continues improving even at  $k = 8$ , suggesting that verifier size significantly affects scaling behavior.

# 5.2. Training-Time Scaling with Data

We then examine how scaling the amount of training data affects agent performance using 491 sampled trajectories from §4.2. We simulate three scaling methods through subsampling: (1) Scaling trajectories, where trajectories are randomly dropped (Fig. 5); (2) Scaling unique task instances, where only one successful trajectory per task instance is selected (Fig. 9); and (3) Scaling repositories, which sequentially includes all instances from each repository to assess repository-level diversity.

Figure 5: Scaling effects of increasing the number of randomly sampled trajectories for training.

Setup. Using OpenHands (Wang et al., 2024c) and the fine-tuning approach described in §4.2, we evaluate these scaling approaches on SWE-Bench Verified: scaling the number of trajectories, by subsampling from the full trajectory dataset from §4.2 (at most 491 trajectories); unique instance scaling on these trajectories deduplicated by instance ID (at most 294 trajectories), and repository-based scaling where we sort repositories alphabetically and include all trajectories from each repository in order (e.g., first  $25\%$  contains complete trajectories from the first N repositories). We compare models trained on  $25\%$ ,  $50\%$ , and  $100\%$  of the full dataset for each approach, sampling training subsets using the methods described above for each scaling approach. $^{10}$

Scaling trends suggest instance and repository diversity is not yet a bottleneck. Fig. 5 demonstrates substantial scaling behavior, with consistent improvements in resolution rate as the number of training trajectories randomly increases, particularly for the 32B model. These results suggest that SWE-Gym's current size and repository diversity are likely not performance bottlenecks - further improvements could likely be achieved by allocating more computing resources to sampling more training trajectories.

Fig. 9 reveals comparable overall performance between different scaling approaches up to where dedduplication takes effect. While Random Scaling (No Dedup.) achieves higher final performance, this is likely due to having more trajectories (491 vs 294) rather than better scaling efficiency. Among deduplicated approaches, Repository Scaling shows stronger initial performance at  $25\%$  data, suggesting that complete repository coverage may provide more coherent learning signals early in training. These results suggest that the repository and instance diversity of SWE-Gym is not yet a bottleneck - further improvements could likely be achieved by simply sampling more agent trajectory data for training, regardless of duplication or repository distribution.

# 6. Conclusions, Limitations, and Future Work

In this paper, we introduce SWE-Gym, the first training environment that addresses critical gaps in enabling scalable learning for software engineering agents. By combining real-world Python tasks with repository-level context, pre-configured execution environments, and test verifications, SWE-Gym will be a foundation for advancing LM agent training research. Through extensive experiments, we demonstrate that SWE-Gym enables both agent and verifier models to achieve significant improvements in resolving complex software tasks. Our findings highlight the scalability of these approaches, revealing potential for continuous performance gains with increased compute.

We see many research directions that we are excited to explore in the future:

1. Automatic Environment Synthesis SWE-Gym, while effective, is limited by its environment diversity, including the number of repositories, types of tasks, and programming languages. We view environment synthesis—via automated environment creation, test-case generation, or task generation—as a critical next step.  
2. Self-Improvement with Reinforcement Learning Despite notable progress, our self-improvement results are modest. Training language model agents with large-scale online reinforcement learning is a promising direction for further improvements.  
3. Human-Agent Interaction Current SWE settings focus solely on task completion, neglecting human-in-the-loop collaboration, which is essential for real-world software engineering. Methods like user simulation or learning from offline human-agent interaction data might offer ways for developing collaborative agents that align with human.

# Impact Statement

This work presents SWE-Gym, an environment for training software engineering agents, with strong empirical results on its effectiveness. We discuss a few important societal implications to consider. First, improving automated software engineering capabilities could increase developer's productivity and accessibility across industries. Although current models are primarily research artifacts and not yet production-ready, they can support critical open-source infrastructure and potentially make software development more accessible. Secondly, as these agents become more capable, they may impact software engineering jobs and require careful consideration around code ownership, licensing, and attribution. Additionally, while we focus on legitimate software engineering tasks, similar techniques

could potentially be misused to automate the creation of malicious code. We encourage future work to further explore frameworks for responsible deployment of software engineering agents, including considerations around security, safety, and economic impacts.

# Acknowledgments

We thank John Yang and Ofir Press for helpful discussions, and John Yang for assistance in reproducing data analysis results from SWE-Bench. We thank Modal Labs<sup>11</sup> for the GPU compute support through its Academic Credits Program. XW and HJ are partially supported by U.S. DARPA ITM Program No. FA8650-23-C-7316. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of DARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.

# References

Akkaya, I., Andrychowicz, M., Chociej, M., Litwin, M., McGrew, B., Petron, A., Paino, A., Plappert, M., Powell, G., Ribas, R., et al. Solving rubik's cube with a robot hand. arXiv preprint arXiv:1910.07113, 2019.  
Badertdinov, I., Trofimova, M., Anapolskiy, Y., Abramov, S., Zainullina, K., Golubev, A., Polezhaev, S., Litvintseva, D., Karasik, S., Fisin, F., Skvortsov, S., Nekrashevich, M., Shevtsov, A., and Yangel, B. Scaling data collection for training software engineering agents. Nebius blog, 2024.  
Bai, H., Zhou, Y., Cemri, M., Pan, J., Suhr, A., Levine, S., and Kumar, A. Digirl: Training in-the-wild device-control agents with autonomous reinforcement learning. ArXiv, abs/2406.11896, 2024. URL https://api-semanticscholar.org/CorpusID:270562229.  
Brown, B., Juravsky, J., Ehrlich, R., Clark, R., Le, Q. V., R'e, C., and Mirhoseini, A. Large language monkeys: Scaling inference compute with repeated sampling. ArXiv, abs/2407.21787, 2024. URL https://api-semanticscholar.org/CorpusID:271571035.  
Chen, B., Shu, C., Shareghi, E., Collier, N., Narasimhan, K., and Yao, S. Fireact: Toward language agent fine-tuning. ArXiv, abs/2310.05915, 2023. URL https://api-semanticscholar.org/CorpusID:263829338.

Chen, D., Lin, S., Zeng, M., Zan, D., Wang, J.-G., Cheshkov, A., Sun, J., Yu, H., Dong, G., Aliev, A., Wang, J., Cheng, X., Liang, G., Ma, Y., Bian, P., Xie, T., and Wang, Q. Coder: Issue resolving with multi-agent and task graphs. CoRR in ArXiv, abs/2406.01304, 2024.  
Chen, M., Tworek, J., Jun, H., Yuan, Q., Ponde, H., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D. W., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Babuschkin, I., Balaji, S., Jain, S., Carr, A., Leike, J., Achiam, J., Misra, V., Morikawa, E., Radford, A., Knight, M. M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code. ArXiv, abs/2107.03374, 2021. URL https://api_semanticscholar.org/CorpusID:235755472.  
Cobbe, K., Kosaraju, V., Bavarian, M., Chen, M., Jun, H., Kaiser, L., Plappert, M., Tworek, J., Hilton, J., Nakano, R., Hesse, C., and Schulman, J. Training verifiers to solve math word problems. ArXiv, abs/2110.14168, 2021. URL https://api.sementicscholar.org/CorpusID:239998651.  
Golubev, A., Polezhaev, S., Zainullina, K., Trofimova, M., Badertdinov, I., Anapolskiy, Y., Litvintseva, D., Karasik, S., Fisin, F., Skvortsov, S., Nekrashevich, M., Shevtsov, A., Abramov, S., and Yangel, B. Leveraging training and search for better software engineering agents. Nebius blog, 2024. https://nebius.com/blog/posts/training-and-search-for-software-engineing-agents.  
Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., and Steinhardt, J. Measuring coding challenge competence with APPS. In Vanschoren, J. and Yeung, S. (eds.), Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual, 2021a.  
Hendrycks, D., Burns, C., Kadavath, S., Arora, A., Basart, S., Tang, E., Song, D. X., and Steinhardt, J. Measuring mathematical problem solving with the math dataset. ArXiv, abs/2103.03874, 2021b. URL https://api.sementicscholar.org/CorpusID:232134851.  
Hu, E. J., Shen, Y., Wallis, P., Allen-Zhu, Z., Li, Y., Wang, S., Wang, L., and Chen, W. Lora: Low-rank adaptation of large language models. In The Tenth International Conference on Learning Representations, ICLR

2022, Virtual Event, April 25-29, 2022. OpenReview.net, 2022. URL https://openreview.net/forum? id=nZeVKeeFYf9.  
Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Dang, K., et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024a.  
Hui, B., Yang, J., Cui, Z., Yang, J., Liu, D., Zhang, L., Liu, T., Zhang, J., Yu, B., Dang, K., et al. Qwen2. 5-coder technical report. arXiv preprint arXiv:2409.12186, 2024b.  
Jain, N., Shetty, M., Zhang, T., Han, K., Sen, K., and Stoica, I. R2E: turning any github repository into a programming agent environment. In *Forty-first International Conference on Machine Learning*, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=kXHgEYFyf3.  
Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. R. Swe-bench: Can language models resolve real-world github issues? In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net, 2024. URL https://openreview.net/forum?id=VTF8yNQM66.  
Lightman, H., Kosaraju, V., Burda, Y., Edwards, H., Baker, B., Lee, T., Leike, J., Schulman, J., Sutskever, I., and Cobbe, K. Let's verify step by step. ArXiv, abs/2305.20050, 2023. URL https://api-semanticscholar.org/CorpusID:258987659.  
Ma, Y., Cao, R., Cao, Y., Zhang, Y., Chen, J., Liu, Y., Liu, Y., Li, B., Huang, F., and Li, Y. Lingma swe-gpt: An open development-process-centric language model for automated software improvement. arXiv preprint arXiv:2411.00622, 2024.  
Modal. Modal: High-performance AI infrastructure. https://modal.com/, 2024. Accessed: 2024-12-18.  
Neubig, G. and Wang, X. Evaluation of LLMs as Coding Agents on SWE-Bench (at 30x Speed!). All Hands AI blog, 2024.  
Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al. Training language models to follow instructions with human feedback. Advances in neural information processing systems, 35:27730-27744, 2022.

Pan, J., Zhang, Y., Tomlin, N., Zhou, Y., Levine, S., and Suhr, A. Autonomous evaluation and refinement of digital agents. ArXiv, abs/2404.06474, 2024. URL https://api.sementicscholar.org/CorpusID:269009430.  
PyTorch Team. torch tune: PyTorch native posttraining library. https://github.com/pytorch/torch tune, 2024.  
Qwen Team. Qwen2.5: A party of foundation models, September 2024. URL https://qwenlm.github.io/blog/qwen2.5/.  
Schulman, J., Wolski, F., Dhariwal, P., Radford, A., and Klimov, O. Proximal policy optimization algorithms. ArXiv, abs/1707.06347, 2017. URL https://api.sementicscholar.org/CorpusID:28695052.  
Shao, Z., Wang, P., Zhu, Q., Xu, R., Song, J., Bi, X., Zhang, H., Zhang, M., Li, Y., Wu, Y., et al. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. arXiv preprint arXiv:2402.03300, 2024.  
Silver, D., Hubert, T., Schrittwieser, J., Antonoglou, I., Lai, M., Guez, A., Lanctot, M., Sifre, L., Kumaran, D., Graepel, T., Lillicrap, T. P., Simonyan, K., and Hassabis, D. Mastering chess and shogi by self-play with a general reinforcement learning algorithm. ArXiv, abs/1712.01815, 2017. URL https://api_semanticscholar.org/CorpusID:33081038.  
Tao, N., Ventresque, A., Nallur, V., and Saber, T. Enhancing program synthesis with large language models using many-objective grammar-guided genetic programming. Algorithms, 17(7):287, 2024. doi: 10.3390/A17070287. URL https://doi.org/10.3390/a17070287.  
Tong, Y., Zhang, X., Wang, R., Wu, R. M., and He, J. Dart-math: Difficulty-aware rejection tuning for mathematical problem-solving. ArXiv, abs/2407.13690, 2024. URL https://api.sementicscholar.org/CorpusID:271270574.  
Unsloth Team. Easily finetune and train LLMs. Get faster with unsloth. https://unsloth.ai/, 2024.  
Wang, P., Li, L., Shao, Z., Xu, R., Dai, D., Li, Y., Chen, D., Wu, Y., and Sui, Z. Math-shepherd: Verify and reinforce LLMs step-by-step without human annotations. In Ku, L.-W., Martins, A., and Srikumar, V. (eds.), Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 9426-9439, Bangkok, Thailand, August 2024a. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.510. URL https://aclanthology.org/2024.acl-long.510.

Wang, X., Chen, Y., Yuan, L., Zhang, Y., Li, Y., Peng, H., and Ji, H. Executable code actions elicit better LLM agents. In *Forty-first International Conference on Machine Learning*, ICML 2024, Vienna, Austria, July 21-27, 2024. OpenReview.net, 2024b. URL https://openreview.net/forum?id=jJ9BoXAfFa.  
Wang, X., Li, B., Song, Y., Xu, F. F., Tang, X., Zhuge, M., Pan, J., Song, Y., Li, B., Singh, J., Tran, H. H., Li, F., Ma, R., Zheng, M., Qian, B., Shao, Y., Muennighoff, N., Zhang, Y., Hui, B., Lin, J., Brennan, R., Peng, H., Ji, H., and Neubig, G. OpenHands: An Open Platform for AI Software Developers as Generalist Agents. CoRR in ArXiv, abs/2407.16741, 2024c.  
Wu, Z., Bai, H., Zhang, A., Gu, J., Vinod Vydiswaran, V., Jaitly, N., and Zhang, Y. Divide-or-conquer? which part should you distill your llm? ArXiv, 2024.  
Xi, Z., Ding, Y., Chen, W., Hong, B., Guo, H., Wang, J., Yang, D., Liao, C., Guo, X., He, W., Gao, S., Chen, L., Zheng, R., Zou, Y., Gui, T., Zhang, Q., Qiu, X., Huang, X., Wu, Z., and Jiang, Y.-G. Agentgym: Evolving large language model-based agents across diverse environments. ArXiv, abs/2406.04151, 2024. URL https://api-semanticscholar.org/CorpusID:270285866.  
Xia, C. S., Deng, Y., Dunn, S., and Zhang, L. Agentless: Demystifying llm-based software engineering agents. CoRR, abs/2407.01489, 2024. doi: 10.48550/ARXIV.2407.01489. URL https://doi.org/10.48550/arXiv.2407.01489.  
Yang, J., Jimenez, C. E., Wettig, A., Lieret, K., Yao, S., Narasimhan, K., and Press, O. Swe-agent: Agent-computer interfaces enable automated software engineering. CoRR, abs/2405.15793, 2024. doi: 10.48550/ARXIV.2405.15793. URL https://doi.org/10.48550/arXiv.2405.15793.  
Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K. R., and Cao, Y. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net, 2023. URL https://openreview.net/forum?id=WE_vluYUL-X.  
Yuan, L., Cui, G., Wang, H., Ding, N., Wang, X., Deng, J., Shan, B., Chen, H., Xie, R., Lin, Y., Liu, Z., Zhou, B., Peng, H., Liu, Z., and Sun, M. Advancing LLM reasoning generalists with preference trees. CoRR, abs/2404.02078, 2024. doi: 10.48550/ARXIV.2404.02078. URL https://doi.org/10.48550/arXiv.2404.02078.

Zeng, A., Liu, M., Lu, R., Wang, B., Liu, X., Dong, Y., and Tang, J. Agenttuning: Enabling generalized agent abilities for llms. In Annual Meeting of the Association for Computational Linguistics, 2023. URL https://apisemantic scholar.org/CorpusID:264306101.  
Zhai, Y., Bai, H., Lin, Z., Pan, J., Tong, S., Zhou, Y., Suhr, A., Xie, S., LeCun, Y., Ma, Y., and Levine, S. Fine-tuning large vision-language models as decision-making agents via reinforcement learning. ArXiv, abs/2405.10292, 2024. URL https://api_semanticscholar.org/CorpusID:269790773.  
Zhang, K., Yao, W., Liu, Z., Feng, Y., Liu, Z., Murthy, R., Lan, T., Li, L., Lou, R., Xu, J., Pang, B., Zhou, Y., Heinecke, S., Savarese, S., Wang, H., and Xiong, C. Diversity empowers intelligence: Integrating expertise of software engineering agents. ArXiv, abs/2408.07060, 2024a. URL https://apisemantic scholar.org/CorpusID:271860093.  
Zhang, Y., Ruan, H., Fan, Z., and Roychoudhury, A. Autocoderover: Autonomous program improvement. In ISSTA, 2024b.  
Zhao, W., Jiang, N., Lee, C., Chiu, J. T., Cardie, C., Galle, M., and Rush, A. M. Commit0: Library generation from scratch, 2024. URL https://arxiv.org/abs/2412.01769.  
Zheng, L., Yin, L., Xie, Z., Sun, C., Huang, J., Yu, C. H., Cao, S., Kozyrakis, C., Stoica, I., Gonzalez, J. E., Barrett, C., and Sheng, Y. Sglang: Efficient execution of structured language model programs, 2024a. URL https://arxiv.org/abs/2312.07104.  
Zheng, T., Zhang, G., Shen, T., Liu, X., Lin, B. Y., Fu, J., Chen, W., and Yue, X. Opencode interpreter: Integrating code generation with execution and refinement. ArXiv, abs/2402.14658, 2024b. URL https://api_semanticscholar.org/CorpusID:267782452.  
Örwall, A. Moatless Tool. https://github.com/aorwall/moatless-tools, 2024. Accessed: 2024-10-22.

# A. Comparison with Concurrent Works

Ma et al. (2024) trains an LM agent, Lingma SWE-GPT, using a method similar to our rejection sampling fine-tuning baseline, with a dataset comparable to our SWE-Gym Raw splits. Without executable unit test feedback, they rely on manually defined heuristics to filter out low-quality trajectories, such as comparing similarity between submitted patches and edit locations with gold patches. The model weights are publicly accessible but not the training pipeline or the dataset.

Most relevant to our work are two consecutive blog posts by Golubev et al. (2024) and Badertdinov et al. (2024), who also construct an executable training environment with real-world tasks from GitHub. Instead of manual configuration, they employ a general environment setup script and simply discard instances that fail the setup process. This approach leads to key differences in dataset size and distribution: while it biases the environment away from tasks with complex dependencies, they successfully collect 6,415 instances, about 1.5 times larger than our dataset. In Golubev et al. (2024), they also study training agents and verifiers with the environment. Additionally, they explore a lookahead setting where a trained verifier ranks and selects the best next action. With a substantially large collection of agent trajectories (80,036 compared to thousands in our experiments) and model size (72B compared to 32B), Their best system achieves  $40\%$  accuracy on SWE-Bench Verified. While their dataset and agent trajectories are publicly accessible, the model is not.

In comparison, with a comparable dataset size, our SWE-Gym has executable feedback, avoids potential dataset bias through manual configuration of environments, while providing comprehensive analysis of agent and verifier training, their scaling behaviors, and positive results on agent self-improvement. Our system achieves competitive results with significantly lower compute and a smaller model size (32B vs 72B). Lastly, we open source all artifacts of the project, including dataset, model weights, agent trajectory data and the training pipeline.

<table><tr><td rowspan="2">Model
Name, Model Size</td><td colspan="2">SWE-Bench</td><td colspan="2">Openness</td></tr><tr><td>Lite</td><td>Verified</td><td>Model</td><td>Environment</td></tr><tr><td>Ma et al. (2024), 72B</td><td>22.0</td><td>30.2</td><td>✓</td><td>✗</td></tr><tr><td>Golubev et al. (2024) Agent and Verifier, 72B</td><td>-</td><td>40.6</td><td>✗</td><td>✓</td></tr><tr><td>Our SWE-Gym Agent and Verifier, 32B</td><td>26.0</td><td>32.0</td><td>✓</td><td>✓</td></tr></table>

Table 5: Comparison of model performance on SWE-Bench benchmark and if the model weights and environments are publicly accessible (openness).  

<table><tr><td>Cap</td><td># Traj</td><td>Empty Patch (%,↓)</td><td>resolution rate (%,↑)</td></tr><tr><td>0 (Zero-shot)</td><td>0</td><td>56.3</td><td>7.0</td></tr><tr><td>1</td><td>36</td><td>37.3</td><td>9.0</td></tr><tr><td>2</td><td>62</td><td>29.0</td><td>9.7</td></tr><tr><td>3</td><td>82</td><td>43.7</td><td>7.7</td></tr><tr><td>No Cap (All)</td><td>172</td><td>30.7</td><td>9.3</td></tr></table>

Table 6: resolution rate and empty patch rate on SWE-Bench Lite with a 7B model trained using different instance capping strategies (Cap).

# B. Experiment Details

# B.1. Mean and Variance for Pass@N and Best@N.

We mostly follow (Lightman et al., 2023) for obtaining the mean and variance for the Pass@N and Best@N curve. Given a total of M rounds of rollouts, for  $N < M$ , we calculate the mean and variance across 100 randomly selected sub-samples of size  $N$  from the  $M$  rollouts. For the OpenHands CodeActAgent inference-time scaling curve at §3, we exclude this calculation for  $N = 1$ , as we use a temperature of 0 for the first attempt.

# B.2. OpenHands Agent Experiments

During training, we use OpenHands's remote runtime (Neubig & Wang, 2024) feature to execute agent trajectories in parallel on SWE-Gym. We use torch tune (PyTorch Team, 2024) for full parameter fine-tuning with a learning rate of  $1\mathrm{e} - 4$  maximum 5 epochs, global batch size of 8, max context length of 32768. We fine-tuned both 7B, 14B, and 32B variant of the model, and experiments were performed with 2-8x NVIDIA H100 80G GPU on modal (Modal, 2024). The

<table><tr><td rowspan="2"></td><td rowspan="2">Original</td><td rowspan="2">Dedup.</td><td colspan="2">Sorted by Random (Dedup.)</td><td colspan="2">Sorted by Repo (Dedup.)</td></tr><tr><td>First 25%</td><td>First 50%</td><td>First 25%</td><td>First 50%</td></tr><tr><td>getmoto/moto</td><td>155</td><td>72</td><td>12</td><td>33</td><td>0</td><td>46</td></tr><tr><td>Project-MONAI/MONAI</td><td>95</td><td>53</td><td>17</td><td>25</td><td>53</td><td>53</td></tr><tr><td>pandas-dev/pandas</td><td>70</td><td>61</td><td>14</td><td>30</td><td>0</td><td>0</td></tr><tr><td>python/mypy</td><td>46</td><td>27</td><td>7</td><td>12</td><td>0</td><td>0</td></tr><tr><td>dask/dask</td><td>45</td><td>29</td><td>8</td><td>17</td><td>6</td><td>29</td></tr><tr><td>iterative/dvc</td><td>36</td><td>24</td><td>8</td><td>12</td><td>0</td><td>0</td></tr><tr><td>conan-io/conan</td><td>20</td><td>12</td><td>1</td><td>7</td><td>12</td><td>12</td></tr><tr><td>pydantic/pydantic</td><td>11</td><td>7</td><td>2</td><td>4</td><td>0</td><td>0</td></tr><tr><td>facebookresearch/hydra</td><td>7</td><td>5</td><td>2</td><td>5</td><td>0</td><td>5</td></tr><tr><td>bokeh/bokeh</td><td>3</td><td>2</td><td>1</td><td>1</td><td>2</td><td>2</td></tr><tr><td>modin-project/modin</td><td>3</td><td>2</td><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Total</td><td>491</td><td>294</td><td>73</td><td>147</td><td>73</td><td>147</td></tr></table>

Table 7: Distribution of success trajectories used in training-time scaling experiments (§5.2). Dedup. denotes that the trajectories are deduplicated by randomly select ONE success trajectory per instance ID; Sorted by random (repo) X% (Dedup.) denotes a subset of trajectories taken from the first X% from deduct. instances that are sorted randomly (by repository name).  

<table><tr><td rowspan="2"></td><td rowspan="2">Resolved</td><td rowspan="2">Count</td><td rowspan="2">Mean</td><td rowspan="2">Std</td><td rowspan="2">Min</td><td rowspan="2">Max</td><td colspan="7">Percentiles</td></tr><tr><td>5%</td><td>10%</td><td>25%</td><td>50%</td><td>75%</td><td>90%</td><td>95%</td></tr><tr><td rowspan="2">Num. of Messages</td><td>X</td><td>5,557.0</td><td>39.2</td><td>31.9</td><td>7.0</td><td>101.0</td><td>9.0</td><td>9.0</td><td>9.0</td><td>29.0</td><td>61.0</td><td>100.0</td><td>101.0</td></tr><tr><td>✓</td><td>491.0</td><td>39.9</td><td>19.9</td><td>13.0</td><td>101.0</td><td>19.0</td><td>21.0</td><td>25.0</td><td>33.0</td><td>47.5</td><td>65.0</td><td>87.0</td></tr><tr><td rowspan="2">Num. of Tokens</td><td>X</td><td>5,557.0</td><td>17,218.3</td><td>17,761.6</td><td>1,615.0</td><td>167,834.0</td><td>1,833.0</td><td>1,907.0</td><td>2,268.0</td><td>12,305.0</td><td>26,434.0</td><td>41,182.2</td><td>51,780.6</td></tr><tr><td>✓</td><td>491.0</td><td>18,578.5</td><td>11,361.4</td><td>2,560.0</td><td>81,245.0</td><td>5,813.0</td><td>8,357.0</td><td>11,559.5</td><td>15,999.0</td><td>22,040.5</td><td>31,632.0</td><td>39,512.5</td></tr></table>

Table 8: Statistics of SWE-Gym-sampled trajectories. We use the tokenizer from Qwen-2.5-Coder-Instruct-7B to estimate the number of tokens.

Figure 6: Success distribution over 30 rounds on SWE-Gym Lite with 7B model in zero-shot. The distribution is naturally biased toward easy tasks. Per instance capping reduces this bias but lowers the total trajectory count for training. We set temperature  $t = 1$  during sampling.

only exception is in the main experiment of §5.1.1, where we use LoRA (Hu et al., 2022)  $(29.8\% @8)$  via Unsloth library (Unsloth Team, 2024) to train the verifier for max 2 epochs, while other hyper-parameter stays the same.

Inference during evaluation is bounded by either 100 interaction turns or the base LM's  $32\mathrm{k}$  context window length, whichever is reached first.

# B.3. MoatlessTools Agent Experiments

All MoatlessTools models are trained with a context window of 10240. For experiments with the 7B model, we use torch tune to train the policy model with full-finetuning using 4 H100 GPUs. We set batch size to 8, learning rate to  $2 \times 10^{-5}$ , and train for 5 epochs.

For the 32B model, we use Unsloth (Unsloth Team, 2024) with a single H100 GPU for LoRA fine-tuning. We set the number of epochs to 5, batch size to 8, LoRA rank to 64, and learning rate to  $5 \times 10^{-4}$ . We use the same configuration for verifier training.

For MoatlessAgent experiments, we serve the agent with FP8 quantization for improved throughput, which we found to have minimal effects on model performance.

# B.4. Details of OpenHands Trajectory Sampling

As detailed in Tab. 10, we collect a few sets of trajectories for fine-tuning experiments. We collect dataset  $D_0$  by sample gpt-4o-2024-08-06 on SWE-Gym Lite with temperature 0 and collected 19 trajectories that eventually solve the task (evaluated by unit test in SWE-Gym). We then varied the temperatures (setting  $\mathsf{t} = \{0.2, 0.3, 0.4, 0.5, 0.8\}$ ) and sample on SWE-Gym Lite. Combining these instances with  $D_0$ , we get 106 trajectories that solve the given problem  $(D_1)$ . We set the maximum number of turns to be 30 for both  $D_0$  and  $D_1$ . To experiment on the effect of max turn, we set max number of turns to 50 and sample gpt-4o-2024-08-06 (19 resolved out of 230) and craude-3-5-sonnet-20241022 (67 resolved out of 230) with temperature 0 on SWE-Gym Lite, and sample gpt-4o-2024-08-06 (temperature  $\mathsf{t} = \{0, 1\}$ ) on SWE-Gym full set (in total 299 resolved out of 4876 instances).

Training Software Engineering Agents and Verifiers with SWE-Gym  

<table><tr><td>Agent</td><td>Model</td><td>Model Size</td><td>Training Data</td><td>Resolved (%)</td></tr><tr><td colspan="5">SWE-Bench Verified (500 instances)</td></tr><tr><td>RAG</td><td>SWE-Llama (Jimenez et al., 2024)</td><td>7B</td><td>10K instances</td><td>1.4</td></tr><tr><td>RAG</td><td>SWE-Llama (Jimenez et al., 2024)</td><td>13B</td><td>10K instances</td><td>1.2</td></tr><tr><td>Lingma Agent (Ma et al., 2024)</td><td>Lingma SWE-GPT (v0925)</td><td>7B</td><td>90K PRs from 4K repos</td><td>18.2</td></tr><tr><td>Lingma Agent (Ma et al., 2024)</td><td>Lingma SWE-GPT (v0925)</td><td>72B</td><td>90K PRs from 4K repos</td><td>28.8</td></tr><tr><td>OpenHands (Wang et al., 2024c) (Ours)</td><td>fine-tuned Qwen2.5-Coder-Instruct</td><td>32B</td><td>491 agent trajectories from 11 repos</td><td>20.6</td></tr><tr><td>OpenHands w/ Verifier (Wang et al., 2024c) (Ours)</td><td>fine-tuned Qwen2.5-Coder-Instruct</td><td>32B (Agent &amp; Verifier)</td><td>491 agent trajectories from 11 repos for agent + 1318 × 2 success/failure agent trajectories for verifier</td><td>32.0</td></tr></table>

Table 9: Performance comparison with SWE-Bench (Jimenez et al., 2024) baselines with publicly accessible weights. Data source: https://www.swebench.com/, Accessed on Dec 21, 2024.  

<table><tr><td>Trajectory Set</td><td>Sampled from Model</td><td>Sampled on Dataset</td><td>Temperature</td><td>Max Turns</td><td>Success trajectories</td></tr><tr><td>D0</td><td>gpt-4o-2024-08-06</td><td>SWE-Gym Lite</td><td>0</td><td>30</td><td>19 (8.26%)</td></tr><tr><td></td><td></td><td></td><td colspan="2">(Cumulative) Total D0</td><td>19</td></tr><tr><td rowspan="5">D1\D0</td><td>gpt-4o-2024-08-06</td><td>SWE-Gym Lite</td><td>0.2</td><td>30</td><td>11 (4.78%)</td></tr><tr><td>gpt-4o-2024-08-06</td><td>SWE-Gym Lite</td><td>0.3</td><td>30</td><td>17 (7.39%)</td></tr><tr><td>gpt-4o-2024-08-06</td><td>SWE-Gym Lite</td><td>0.4</td><td>30</td><td>21 (9.13%)</td></tr><tr><td>gpt-4o-2024-08-06</td><td>SWE-Gym Lite</td><td>0.5</td><td>30</td><td>18 (7.83%)</td></tr><tr><td>gpt-4o-2024-08-06</td><td>SWE-Gym Lite</td><td>0.8</td><td>30</td><td>20 (8.70%)</td></tr><tr><td></td><td></td><td></td><td colspan="2">(Cumulative) Total D1</td><td>106</td></tr><tr><td rowspan="4">D2\D1</td><td>gpt-4o-2024-08-06</td><td>SWE-Gym Lite</td><td>0</td><td>50</td><td>19 (8.26%)</td></tr><tr><td>claude-3-5-sonnet-20241022</td><td>SWE-Gym Lite</td><td>0</td><td>50</td><td>67 (29.1%)</td></tr><tr><td>gpt-4o-2024-08-06</td><td>SWE-Gym Full</td><td>0</td><td>50</td><td>*111 (4.55%)</td></tr><tr><td>gpt-4o-2024-08-06</td><td>SWE-Gym Full</td><td>1</td><td>50</td><td>188 (7.71%)</td></tr><tr><td></td><td></td><td></td><td colspan="2">(Cumulative) Total D2</td><td>491</td></tr></table>

* Run into infrastructure-related error where some instances failed to complete, this number might be under estimate of actual number of success trajectories.  
Table 10: Summary of trajectories sampled from SWE-Gym.

This gives us in total  $106 + 19 + 67 + 299 = 491$  success trajectories, which forms our final training trajectories  $D_{2}$ .

# B.5. MoatlessTools ORM Prompt

The following is a pseudo-code that generates a prompt for MoatlessTools Verifier (ORM), which is modified from (Zhang et al., 2024a). Unlike (Zhang et al., 2024a), which relies on proprietary models like Claude-3.5-Sonnet for context extraction, we obtain context directly from the agent's trajectory being evaluated.

SYSTEMMESSAGE  $=$  ""You are an expert in python for software engineering and code

$\rightarrow$  review. Your responsibility is to review the patches generated by language  
→ models to fix some issues and provide feedback on the quality of their  
code."

USER_MESSAGE  $=$  ""I want you to evaluate an LLM-generated candidate patch that

$\leftrightarrow$  tries to resolve an issue in a codebase.

To assist you in this task, you are provided with the following information:

- You are given an issue text on a github repository (wrapped with  
$\leftrightarrow$  <issue_description></issue_description>).  
- You are also given some identified code spans that are relevant to the issue.

Each code span is wrapped with <code spans file_path=FILE_PATH

$\leftrightarrow$  span_id=SPAN_ID></codeSpan> tags, where FILE_PATH is the path to the  
$\hookrightarrow$  file containing the code span, and SPAN_ID is the unique identifier for  
the code span.

```latex
Each code span also comes with the line numbers for you to better understand  $\leftrightarrow$  the context. It's possible that the code span are not sufficient to fix  $\leftrightarrow$  the issue, adjust your score accordingly.   
- You are given the candidate patch that tries to resolve the target issue. For your convenience, you are given the hunks of original code and the code  $\leftrightarrow$  after applying the patch. The code before the patch is wrapped with  $<  _{\mathrm{before\_patch}}>$  /beforePatch> and  $\leftrightarrow$  the code after the patch is wrapped with  $<  _{\mathrm{after\_patch}}>$  /afterPatch>. Note that the file names in beforePatch starts with 'a/' and the file names  $\leftrightarrow$  in afterPatch starts with 'b/'.
```

```txt
<issue_description>
{issue_text}
</issue_description>
<before_batch>
{before_batch}
</before_batch>
<after_batch>
{after_batch}
</after_batch>
{code_spans}
```

Response in "True" or "False" for whether the patch has resolved the issue."""

# B.6. OpenHands ORM Prompt

The following is a pseudo-code that generates a prompt for OpenHands Verifier (ORM).

```txt
SYSTEMMESSAGE  $=$  "'You are an expert judge evaluating AI assistant interactions.  $\nrightarrow$  Your task is to determine if the assistant successfully resolved the user's  $\nrightarrow$  request.
```

Key evaluation criteria:

1. Did the assistant complete the main task requested by the user?  
2. Did the assistant handle all edge cases and requirements specified?  
3. Were there any errors or issues in the final solution?  
4. Did the assistant verify the solution works as intended?

```txt
Respond only with "<judgement>YES</judgement>" or "<judgement>NO</judgement)".
```

```python
USER_MESSAGE = '''Please evaluate the following interaction between an AI  $\rightarrow$  assistant and a user:
```

```txt
$= = =$  INTERACTIONLOG  $= = =$  \*\* + traj_str +\*\*   
 $= = =$  END INTERACTION  $= = =$
```

```txt
Based on the above interaction, did the assistant successfully resolve the user's  $\leftrightarrow$  initial request? Respond with YES or NO.''
```

```txt
messages = [
    {'role': 'system', 'content': SYSTEMMESSAGE},
    {'role': 'user', 'content': USERMESSAGE},
    {'role': 'assistant', 'content': <judgement>' + ("YES" if resolved else  $\rightarrow$  "NO") + </judgement>}
```

The last assistant messages that contains judgement is only provided during training time. At inference time, the trained verifier is responsible predicting the probability of 'Yes' and 'No'.


Figure 7: Scaling inference-time compute for MoatlessTools Agents (7B and 32B) with their corresponding learned verifiers. Temperature  $t = 0.5$ .

Figure 8: Ablation study for verifier training (§5.1.1). Performances are evaluated on SWE-Bench Verified. Both the agent and the verifier are Qwen2.5-Coder-Instruct-32B model fine-tuned on the corresponding dataset. OpenHands (Wang et al., 2024c) is used as the agent scaffold.

Figure 9: Comparison of three data sampling approaches using 32B LM: scaling trajectories (dedup.), scaling unique task instances, and scaling repositories ( $\S 5.2$ ).

# Footnotes:

Page 2: 1https://seart-ghs.si.usi.ch/ 
Page 3: 2Annotations are done by a subset of the authors. 3For details on its construction process, see https://www. swebench.com/lite.html. 
Page 4: $^{4}\S \mathrm{B}.4$  contains details of these experiments. $^{5}$ Tab. 8 contains more statistics of the sampled trajectories.  
 $^{6}$ We use a sampling temperature of 0 unless otherwise specified. 
Page 6: $^{7}$  §B.6 includes the verifier prompt template. <sup>8</sup>We keep only trajectories within 32k-token length for training, which may reduce their number compared to Section 4.2. 
Page 7: <sup>9</sup>We provide the prompt template in §B.5. 10Tab. 7 contains detailed statistics of these datasets. 
Page 8: 11https://modal.com/ 
