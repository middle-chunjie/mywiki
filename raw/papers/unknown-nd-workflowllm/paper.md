WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models
=================================================================================

Shengda Fan1, Xin Cong2∗, Yuepeng Fu2, Zhong Zhang2, Shuyan Zhang3, Yuanwei Liu4,  
 Yesai Wu2, Yankai Lin1†, Zhiyuan Liu2, Maosong Sun2  
1Renmin University of China2Tsinghua University3The University of Manchester  
4Wuhan University  
fanshengda@ruc.edu.cn, xin.cong@outlook.com  
Indicates equal contribution. Corresponding author.

###### Abstract

Recent advancements in large language models (LLMs) have driven a revolutionary paradigm shift in process automation from Robotic Process Automation to Agentic Process Automation by automating the workflow orchestration procedure based on LLMs.
However, existing LLMs (even the advanced OpenAI GPT-4o) are confined to achieving satisfactory capability in workflow orchestration.
To address this limitation, we present WorkflowLLM, a data-centric framework elaborately designed to enhance the capability of LLMs in workflow orchestration.
It first constructs a large-scale fine-tuning dataset WorkflowBench with $106,763$ samples, covering $1,503$ APIs from $83$ applications across $28$ categories.
Specifically, the construction process can be divided into three phases:
(1) Data Collection: we collect real-world workflow data from Apple Shortcuts and RoutineHub, transcribing them into Python-style code.
We further equip them with generated hierarchical thought via ChatGPT.
(2) Query Expansion: we prompt ChatGPT to generate more task queries to enrich the diversity and complexity of workflows.
(3) Workflow Generation: we leverage an annotator model trained on collected data to generate workflows for synthesized queries. Finally, we merge the synthetic samples that pass quality confirmation with the collected samples to obtain the WorkflowBench.
Based on WorkflowBench, we fine-tune Llama-3.1-8B to obtain WorkflowLlama.
Our experiments show that WorkflowLlama demonstrates a strong capacity to orchestrate complex workflows, while also achieving notable generalization performance on previously unseen APIs. Additionally, WorkflowBench exhibits robust zero-shot generalization capabilities on an out-of-distribution task planning dataset, T-Eval.
Our data and code are available at <https://github.com/OpenBMB/WorkflowLLM>.

1 Introduction
--------------

Process Automation (PA)*(Cichocki et al., [1997])*, as a long-standing pursuit of the human race, aims to automate repetitive tasks to minimize human labor and improve efficiency.
Tracing back to the agricultural era, humanity has employed waterwheels and oxen to automate farming practices.
Robotic Process Automation (RPA), the current predominant PA technique, abstracts the repetitive task into a workflow (i.e., a program that can execute automatically) by orchestrating various actions (e.g., functions or APIs)*(Ivančić et al., [2019]; Hofmann et al., [2020]; Wewerka \& Reichert, [2020]; Agostinelli et al., [2020]; Ferreira et al., [2020])*.
While RPA successfully reduces the human labor via automated workflow execution, the process of orchestrating workflows still requires substantial manual effort.
Recently, large language models (LLMs)*(OpenAI, [2022]; [2023]; Touvron et al., [2023a]; [b]; Dubey et al., [2024])* have achieved remarkable performance beyond natural language processing*(Ahn et al., [2022]; Cheng et al., [2023]; Qian et al., [2024])*.
The emergence of LLMs has unveiled a paradigm shift trend, moving from Robotic Process Automation to Agentic Process Automation (APA)*(Ye et al., [2023]; Zeng et al., [2023]; Huang et al., [2024]; Wornow et al., [2024]; Li et al., [2024])* which automates the workflow orchestration process by utilizing LLMs to build the workflow.

However, such a paradigm shift trend is constrained by the limited ability of LLMs to orchestrate complex workflows, which in turn leads to two crucial limitations in current APA methods:
(1) Constrained Action Scale: Current LLMs can only orchestrate small-scale workflows with a limited number of actions. The most advanced OpenAI GPT-4 is capable of managing workflows with an average of only 6.1 actions, even when equipped with advanced decision-making mechanisms *(Ye et al., [2023])*. This falls short of the complexity required to meet real-world demands.
For instance, as a widely-used representative, Apple Shortcuts*(Apple, [2024])* involves an average of $70.4$ actions.
(2) Simple Logical Structure: Currently, most existing work mainly focuses on generating sequential actions*(Yao et al., [2022]; Qin et al., [2024]; Chen et al., [2024])* while workflows of the real-world applications usually involve intricate logical structures such as branches and loops.
For example, Apple Shortcuts averages $2.6$ nested branch/loop logical structures.
As a result, there is an urgent need to unlock the workflow orchestration capability of LLMs to expedite the paradigm shift in process automation.

<img src='x1.png' alt='Refer to caption' title='' width='747' height='248' />

*Figure 1: Overview of WorkflowLLM. It first constructs WorkflowBench through a three-phase pipeline and fine-tunes WorkflowLlama, which can generate workflows based on the user’s query (appointment management in this case).*

To address these challenges, we propose WorkflowLLM, a data-centric framework including dataset construction, model training, and evaluation to enhance LLMs’ workflow orchestration capabilities (shown in [Figure 1]).
Specifically, we first construct WorkflowBench, which consists of $106,763$ supervised fine-tuning instances, encompassing $1,503$ APIs across $83$ applications, structured through three primary phases:

* •

    Data Collection:
    We select shortcuts from RoutineHub as high-quality data sources because they represent a robust RPA application with numerous expert-developed workflows available.
    We curate $14,771$ human-annotated, high-quality shortcuts spanning $28$ diverse categories (e.g., Business, Health \& Fitness, Productivity), alongside associated metadata including titles, functionality descriptions, and API documentations.
    As the raw workflow data is not directly suitable for LLMs to process, and considering that Python allows more convenient parameter passing and control logic*(Ye et al., [2023]; Wang et al., [2024b])*, we transcribe the shortcut source code into Python-like code.
    Subsequently, we prompt ChatGPT to generate comments, task plans, and task queries at varying levels of granularity—from fine-grained to coarse-grained—to enrich the data with detailed thought processes and enhance the learning efficacy of LLMs*(Wei et al., [2023])*.

* •

    Query Expansion:
    To enrich the diversity and complexity of workflows, we utilize ChatGPT to generate additional task queries.
    Specifically, we first sample applications with diverse functionalities and select their APIs, along with built-in APIs, to prompt ChatGPT to generate task queries that leverage these sampled APIs to accomplish specific tasks.
    To further ensure workflow complexity, we also sample real-world workflow examples as demonstrations to guide ChatGPT in generating similar workflows.

* •

    Workflow Generation:
    As existing LLMs even GPT-4o still struggle in workflow generation, we first train a workflow annotator model based on the collected real-world shortcuts.
    Then we utilize the trained annotator to generate workflows for the expanded task queries.
    To prevent low-quality workflows generated by the annotator model from affecting subsequent training, we perform quality confirmation to ensure dataset integrity.
    We first utilize ChatGPT to refine the generated workflows to fix existing minor bugs in them and then use rule-based filtering to remove workflows with logical errors.

To evaluate the capability of LLMs in workflow orchestration, we employ two metrics: the reference-code-based metric CodeBLEU and the model-based metric Pass Rate.
Experimental results demonstrate that WorkflowLlama consistently and significantly outperforms all baselines, including GPT-4o even with the in-context learning technique, across both metrics under unseen instructions and unseen APIs settings.
Furthermore, WorkflowBench demonstrates strong generalization capabilities in out-of-distribution (OOD) scenarios, particularly on the T-Eval benchmark*(Chen et al., [2024])*, where it achieves an F1 plan score of 77.5%.

2 Related Work
--------------

#### Process Automation

RPA has gained considerable attention for automating repetitive tasks in various productivity scenarios*(Ivančić et al., [2019]; Hofmann et al., [2020]; Wewerka \& Reichert, [2020]; Agostinelli et al., [2020]; Ferreira et al., [2020])*.
RPA predominantly relies on handcrafted workflows (e.g., programming, recording human behavior), making them highly suitable for automating well-structured, routine processes*(Herm et al., [2020])*.
However, such approaches require substantial efforts and in-depth domain expertise, resulting in high setup costs and limited adaptability.
Recent advancements in LLMs have spurred interest in integrating these models into RPA to enhance flexibility and reduce dependency on manual workflow creation. *Ye et al. ([2023])* introduced the concept of APA, which utilizes LLMs to autonomously orchestrate workflows based on human instructions.
Subsequently, several studies have sought to apply APA in various domains, including travel planning*(Xie et al., [2024])*, smartphone applications*(Huang et al., [2024])*, enterprise automation*(Wornow et al., [2024])*, financial question answering*(Zeng et al., [2023])*, and data analysis*(Li et al., [2024])*.
Despite relying on advanced LLMs (e.g., GPT-4), these approaches have often exhibited suboptimal performance, highlighting challenges faced by existing LLMs in workflow orchestration.
While *Li et al. ([2024])* made an effort to fine-tune Mixtral-8$\times$7B*(Jiang et al., [2024])*, it could only orchestrate sequential workflows with an average of $15.6$ actions, remaining insufficient for real-world requirements.
This work addresses a critical gap by proposing WorkflowLLM framework to enhance the workflow orchestration capabilities of LLMs to meet real-world demands.

#### Tool Learning

Workflow orchestration driven by LLMs frequently depends on external tools, such as APIs, to extend their operational capabilities.
Recent studies have demonstrated that LLMs can effectively acquire and utilize external tools by learning from their documentation, thereby solving complex tasks that would otherwise be beyond the model’s native capabilities*(Wu et al., [2023]; Schick et al., [2024]; Qin et al., [2023b]; [2024])*.
This integration enables LLMs to access real-time knowledge and perform specialized operations, particularly for executing intricate processes*(Yang et al., [2023]; Nakano et al., [2021]; Qin et al., [2023a]; Wang et al., [2024c]; Gao et al., [2023])*.
To further enhance this capability, several efforts have introduced datasets specifically designed to fine-tune LLMs for tool interactions*(Zhuang et al., [2024]; Qin et al., [2024]; Wang et al., [2024a])*.
However, these datasets are often constrained to limited actions scale, thus limiting their effectiveness for managing complex, real-world workflows.
Compared to tool learning scenarios, orchestrating workflows demands more sophisticated planning and reasoning that current LLMs have yet to fully realize.
In response to these limitations, we present WorkflowLLM to significantly improve LLMs’ capabilities in workflow orchestration.
Besides, *Shen et al. ([2024])* also used Apple’s Shortcuts but aimed to assess LLMs’ tool utilization ability.
In contrast, we emphasize a different scenario, workflow orchestration and aim to enhance the workflow orchestration ability rather than evaluation alone.

<img src='x2.png' alt='Refer to caption' title='' width='747' height='273' />

*Figure 2: Illustration of our WorkflowLLM which contains three phases to construct WorkflowBench, followed by the supervised fine-tuning phase to derive WorkflowLlama.*

3 WorkflowLLM
-------------

As [Figure 2] shows, WorkflowLLM introduces a data-centric framework to enhance the capability of LLMs in workflow orchestration by constructing a high-quality supervised fine-tuning dataset WorkflowBench.
In this section, we outline the dataset construction process, which is carried out in three distinct phases: Data Collection, Query Expansion, and Workflow Generation.

### 3.1 Data Collection

We first give the introduction to Apple Shortcuts and RoutineHub, and describe how we crawl and filter to get high-quality data.
We then convert the shortcuts into Python-style workflow code.
Inspired by Chain-of-Thought*(Wei et al., [2022]; Chen et al., [2023])*, we prompt ChatGPT to generate hierarchical thoughts, including comments, task plans, and task queries, progressing from fine-grained to coarse-grained details for each shortcut.

#### Apple Shortcuts and RoutineHub

Apple Shortcuts, as a representative application of RPA, is developed by Apple Inc.
This tool facilitates the automation of a series of actions, enabling users to efficiently perform a diverse range of tasks.
The actions within Shortcuts are APIs provided by both built-in Apple applications, such as Safari, and third-party applications like OpenAI. Each application may provide multiple actions.
For instance, OpenAI provides APIs that facilitate voice conversations and text interactions with ChatGPT.
Through a simple drag-and-drop interface, users can construct complex workflows, such as navigating to the nearest coffee shop or downloading watermark-free images from TikTok.

RoutineHub111https://routinehub.co/ is a prominent community for sharing shortcuts, with a collection of thousands of shortcuts across both iOS and macOS platforms.
All shortcuts on RoutineHub are categorized into $28$ workflow categories (e.g., Business, Health \& Fitness, Productivity, etc).
RoutineHub records the metadata of each shortcut (e.g., title, description, iCloud URL), providing valuable information.

#### Crawling and Filtering

For each shortcut, we crawl the title, developer-provided description, and iCloud URL linked to Apple.
As RoutineHub does not provide the source code for these shortcuts, we further crawl it from their iCloud URLs.
Besides, we merge shortcuts collected by ShortcutsBench*(Shen et al., [2024])*, sourced from platforms like ShareShortcuts222https://shareshortcuts.com and MacStories333https://www.macstories.net/shortcuts, to further expand the scale of our dataset.
However, the source code of these shortcuts lacks detailed information about the involved actions, such as API metadata.
Inspired by ShortcutsBench*(Shen et al., [2024])*, we extract action information from macOS’s built-in definition files and third-party application interface definition files.
For each API, we record its name, description, parameter names, parameter types, default values, return value types, and return value name, which provides a valuable resource for LLMs to efficiently interpret and utilize these APIs, even in zero-shot scenarios.

To ensure compatibility between the crawled shortcuts and the action interfaces, we implement a stringent filtering mechanism to verify that all API calls are executed correctly.
During this process, we identify that some shortcuts contain non-interpretable binary sequences as API parameters, potentially disrupting the training process of language models.
To maintain data quality, we remove these samples from the dataset. As a result, we curate a final set of $14,771$ high-quality shortcuts, ensuring the reliability of the dataset for subsequent data expansion and model training.

#### Shortcuts Transcription

The original shortcut source codes are written in property lists format*(Hummert \& Humphries, [2022])*, which sequentially encodes logical constructs like branches and loops. This encoding is notably different from the types of data commonly used in the pre-training of LLMs.
To address this gap, we convert the shortcuts into abstract syntax trees (ASTs), apply pre-order traversal to transform them into Python code, with further algorithmic details provided in Appendix [A] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models").
Furthermore, the original shortcuts use hexadecimal strings as variable names, leading to reduced semantic clarity.
To improve interpretability, we use ChatGPT to automatically reassign these variables with more contextually meaningful names, thereby enhancing the overall readability and utility of the code for further language model training.
A typical comparison between property lists and Python code can be found in Appendix [C] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models").

#### Thought Generation

To provide informative guidance for LLMs in orchestrating workflows, we design a three-level thought hierarchy from fine-grained to coarse-grained:
(1) Low-level comments are intended to clarify the purpose of each action within the workflow.
(2) Median-level plans represent an abstraction over a sequence of actions, outlining the collective goal of these steps.
(3) High-level queries reflect the user’s requirements, specifying the intended outcome without prescribing specific methods to achieve it.
These three levels of thought are generated through a bottom-up approach.
Specifically, given the transcribed workflow $w$, let the set of actions in the workflow $w$ be denoted as $\mathcal{A}$, where each action $a_{i}\in\mathcal{A}$ corresponds to a function calling in the Python code.
For each action $a_{i}$, we generate a corresponding comment $c_{i}$ by prompting ChatGPT.
Subsequently, given the action set $\mathcal{A}\={a_{i}}$ and comments $\mathcal{C}\={c_{i}}$ of workflow $w$, we prompt ChatGPT to generate the corresponding task plan $\mathcal{P}$.
We combine the task plan $\mathcal{P}$, the comments $\mathcal{C}$, and the action set $\mathcal{A}$ of the workflow $w$ to generate the high-level task query $\mathcal{Q}$.
This bottom-up manner is analogous to the summarization task, effectively ensuring content reliability and minimizing the risk of hallucination.

<img src='x3.png' alt='Refer to caption' title='' width='747' height='277' />

*Figure 3: Illustration of data field composition in WorkflowBench comprising Task Query, API documentations, Task Plan, and Workflow code with Comments.*

Finally, as [Figure 3] shows, each workflow $w$ is represented as: $w\={\mathcal{Q},\mathcal{D},\mathcal{P},\mathcal{A}}$,
where the workflow $w$ consists of the task query $\mathcal{Q}$, action documentation $\mathcal{D}$ for all involved actions, the task plan $\mathcal{P}$, and all actions represented as annotated Python code $\mathcal{A}$.
A detailed example can be found in Appendix [D] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models").

### 3.2 Query Expansion

After performing a comprehensive statistical analysis on the collected data, we find that the data exhibits significant complexity, with an average of 70.4 actions and 12 branches, surpassing the complexity of existing workflow-related benchmarks. However, the diversity of the data is relatively low. Specifically, 40.3% of the workflows fall under the Utilities category, and over 99% of the APIs used are Apple’s built-in APIs (i.e., those classified as is_workflow_actions APP).

Therefore, we intend to expand the dataset by focusing on two key aspects:
(1) Diversity: making up for the lack of diversity in real data and covering a broad range of APIs and workflow categories to enhance the model’s utility and robustness; (2) Complexity: matching the action scale and logical complexity of the real-world data to ensure that they can effectively represent real-world problems and orchestrate nodes accordingly.
To this end, we sample APIs from diverse applications and multiple workflows with representative logical structures (e.g., whether they contain branches or loops) to synthesize additional queries.

To ensure that the number of APIs in the synthesized dataset aligns with real-world usage, we sample $n$ APIs based on real-world distributions. Approximately $\lfloor n/2\rfloor$ are drawn from Apple’s built-in API set (e.g., openurl or sendemail), with the remainder from third-party applications (e.g., OpenAI). The total number of built-in and external APIs is thus $n$.

To ensure that the sampled APIs can interact coherently, we do not sample directly from the entire API set.
Instead, we first randomly select 1-5 applications and then choose all APIs from these selected applications.
This method ensures that the selected APIs are functionally compatible and capable of representing real-world workflows.

The prompt used for ChatGPT to synthesize queries consists of four components: (1) a general prompt to describe the task query generation task, (2) documentations for the sampled APIs, (3) in-context examples from the collected data for reference, and (4) the workflow category to which the query belongs to.
By controlling the workflow category and in-context examples, we can ensure the diversity and complexity of the generated data. As seen from Figure [4], the synthesized query has a more balanced category distribution and uses more third-party APIs. Although most of the APIs used are still built-in APIs, this is reasonable considering that they carry necessary operations.

### 3.3 Workflow Generation

<img src='x4.png' alt='Refer to caption' title='' width='830' height='412' />

*Figure 4:  Comparison of the distributions across workflow categories, APPs, and action counts in the collected data and the final dataset. The upper section shows the original data collected from Apple Shortcuts and RoutineHub, while the lower section presents the expanded dataset distributions.*

*Table 1: Detailed statistics of WorkflowBench. Seed. refers to the collected data from Shortcuts. Train. and Test. refers to the training set and the test set of WorkflowBench respectively.*

| Statistics | Seed. | Train. | Test. |
| --- | --- | --- | --- |
| Num. of Instances | 14,771 | 105,573 | 1,190 |
| Num. of APPs | 71 | 83 | 31 |
| Num. of APIs | 584 | 1,503 | 324 |
| Num. of Categories | 28 | 28 | 28 |
| Avg. Action | 70.4 | 78.5 | 41.7 |
| Avg. IF | 12.0 | 7.4 | 7.9 |
| Avg. LOOP | 0.7 | 0.5 | 0.5 |
| Avg. Nested Depth | 2.6 | 2.7 | 2.1 |

To annotate the corresponding workflows of the synthesized queries effectively, we train an annotator model based on the collected shortcuts data to support more diverse applications and categories, while ensuring consistency with the real-world data as much as possible.

#### Annotator Training

First, we construct the supervised fine-tuning (SFT) dataset based on the collected human-labeled shortcuts.
Specifically, each workflow data point comprises a query $\mathcal{Q}$, the corresponding action documentation $\mathcal{D}$, the task plan $\mathcal{P}$, and the workflow represented as annotated Python code $\mathcal{A}_{\text{commented}}$.
During the SFT process, as shown in Figure [3], we take the query $\mathcal{Q}$, the corresponding action documentation $\mathcal{D}$ as the input to guide the model to generate a task plan $\mathcal{P}$, followed by the step-by-step generation of the current thought (i.e., the comment $c_{i}$) and the corresponding action $a_{i}$, which includes the action name and its associated parameters.
We use the trained annotator to generate workflows $\mathcal{A^{\prime}}$ from synthesized queries.

#### Quality Confirmation

Due to the limited accuracy of the annotator model, the generated workflows may contain errors to some extent.
For example, we identify issues in $\mathcal{A^{\prime}}$ (e.g., extraneous branches not relevant to the query and incorrect function call formats).
To enhance the overall quality, we prompt ChatGPT with in-context samples to refine both $\mathcal{A^{\prime}}_{\text{commented}}$ and $\mathcal{P^{\prime}}$, ensuring that the workflow accurately addresses the query.
Then, we use rule-based filtering to remove workflows with fundamental errors.
Specifically, we remove samples that don’t incorporate code, don’t utilize the given APIs, or violate parameter constraints associated with those APIs.

Finally, we derive a synthesized dataset of $91,992$ instances, which is combined with the initially collected data to form the final WorkflowBench.
It contains $106,763$ instances with $1,503$ APIs across $83$ applications, which are used to train WorkflowLlama.
The statistics of WorkflowBench are listed in [Table 1] and the distribution comparisons of workflow categories, APPs, and the number of actions between the collected data and final data are demonstrated in [Figure 4].
From the statistical results, we can see that the synthetic data maintains complexity while expanding diversity.

4 Experiments
-------------

### 4.1 Experimental Setup

Training Details We fine-tune the annotator and WorkflowLlama on LLaMA-3.1-8B*(Dubey et al., [2024])* for 3 epochs using the AdamW optimizer*(Loshchilov \& Hutter, [2019])*. A linear learning rate scheduler is used with a peak learning rate of $2\times 10^{-5}$ and a warm-up ratio of $0.1$. Each mini-batch contains $32$ examples, and the maximum sequence length is set as $8,192$ tokens.

Baselines To provide a comprehensive comparison, we select several representative LLMs as baselines for our experiments. These baselines include proprietary models such as GPT-4o-mini and GPT-4o, as well as open-source models like Qwen2-7B*(qwe, [2024])*, Llama-3.1-8B, and Llama-3.1-70B*(Dubey et al., [2024])*.
Additionally, we apply in-context learning (ICL)*(Dong et al., [2022])* with one random-sampled instance to these baselines to better adapt them for workflow orchestration.

Metrics In the main experiments, we use both reference-code-based metrics and a model-based evaluation to comprehensively evaluate the quality of the generated workflows.
For reference-based metrics, we apply CodeBLEU *(Ren et al., [2020])* with four components:

* •

    BLEU measures N-gram overlap for token-level similarity.

* •

    Weighted N-Gram Match assigns higher weights to critical code tokens like keywords.

* •

    Syntactic AST Match compares the Abstract Syntax Trees (ASTs) to assess syntactic accuracy.

* •

    Semantic Data-Flow Match evaluates logical correctness by comparing data-flow relationships between variables.

Together, these components provide a comprehensive evaluation of both syntactic and semantic aspects of the workflows.
We follow *Ren et al. ([2020])*, setting the four components to 0.1, 0.1, 0.4, and 0.4, respectively, and calculate a weighted sum to obtain the CodeBLEU score.
For model-based evaluation, we elaborately prompt ChatGPT as the automatic evaluator to evaluate the Pass Rate of the generated workflows.

### 4.2 Effectiveness of Evaluator

To validate the reliability of the ChatGPT evaluator in terms of Pass Rate, we sample 30 instruction-response pairs (i.e., task queries and their corresponding workflow codes) for each model in Table[4.2], forming a human-evaluated dataset of $330$ instances ($30\times 11\=330$). First, we use GPT-4o-mini to label whether each instance could complete the given tasks only using the provided APIs. Then, human evaluators re-label the sampled data according to the same criteria. Ultimately, $268$ instances are labeled consistently by both the ChatGPT evaluator and human evaluators, achieving an agreement rate of $\textbf{81.2}\%$, demonstrating the reliability and effectiveness of the evaluator.

*Table 2: Performance comparison of various models on the test set of WorkflowBench under the unseen instructions (ID) and unseen APIs (OOD) settings (%).*

{widetabular}

lcccccccccccc Model CodeBLEU Pass Rate  
BLEU Weighted N-Gram AST Data-Flow Overall  
 ID OOD ID OOD ID OOD ID OOD ID OOD ID OOD  
 Proprietary Models  
GPT-4o-mini 0.4 0.4 1.5 1.6 29.5 29.5 37.0 36.3 26.8 26.5 54.8 47.5  
 w/ ICL 0.5 0.5 1.7 1.8 35.3 34.4 35.1 34.2 28.3 27.7 66.0 57.7

GPT-4o 0.5 0.4 1.8 1.7 33.5 31.8 37.3 36.9 28.5 27.7 56.6 47.5  
 w/ ICL 0.5 0.5 1.8 1.8 37.1 35.3 38.0 36.6 30.2 30.0 67.5 57.6  
 Open-Source Models  
Qwen2-7B 0.4 0.4 1.2 1.3 27.2 27.7 33.2 33.1 24.4 24.5 25.6 22.6  
 w/ ICL 0.5 0.5 1.2 1.3 30.2 29.8 32.4 32.9 25.2 25.3 28.2 26.4  
Llama-3.1-8B 0.6 0.7 1.2 1.4 31.0 29.6 30.0 30.8 24.6 24.3 33.0 24.5  
 w/ ICL 0.7 0.7 1.3 1.4 34.0 32.4 32.6 32.4 25.3 25.2 40.2 32.7  
Llama-3.1-70B 0.4 0.4 1.4 1.5 29.9 30.0 37.8 37.6 27.3 27.2 55.4 42.3  
 w/ ICL 0.4 0.4 1.6 1.5 34.1 32.9 39.138.4 29.5 28.7 67.6 61.4  
WorkflowLlama (8B) 9.47.011.098.355.148.8 38.0 35.3 39.335.176.970.4

### 4.3 Main Experiments

Settings The main experiments are conducted using the test set of WorkflowBench. Ideally, by scaling both the quantity and diversity of instructions and unique tools within the training data, WorkflowLlama is expected to generalize to novel instructions and APIs that are not seen during training.
This is particularly important because it enables users to define custom APIs and allows WorkflowLlama to adapt based solely on the provided documentation.
To evaluate this capability, we assess WorkflowLlama’s generalization performance at two levels: (1) Unseen Instructions, considers an In-Distribution (ID) setting, which involves using the same set of APIs as those in the training data, and (2) Unseen APIs, considers an Out-Of-Distribution (OOD) setting, involving only 50 common APIs required to construct workflows and APIs that are absent from the training data.
Since WorkflowBench contains a comprehensive set of APIs, which poses a substantial challenge for LLMs in terms of API comprehension and selection, we provide the correct APIs directly as input. It allows us to focus on the workflow orchestration, bypassing the issue of API selection.

Main Results The results are placed in Table[4.2], from which we derive that:

1. 1.

    Although multiple workflows can successfully complete a query, there is a positive correlation between the reference-free Pass Rate metric and the reference-based CodeBLEU metric. Given that the Pass Rate metric derived from ChatGPT aligns with human evaluations over 80% of the time, CodeBLEU serves as a reliable proxy for evaluating workflow orchestration capabilities.

2. 2.

    All models demonstrate a certain capacity for workflow orchestration. This may stem from their inherent instruction-following and code-generation capabilities.
    We find that models like GPT-4o and Llama-3.1-70B, which perform better on generic tasks, also excel in workflow orchestration.
    In addition, prompting with in-context samples significantly enhances the models’ performance.

3. 3.

    We find that scores on text overlap metrics such as BLEU and weighted N-gram are low for all models. Even the fine-tuned WorkflowLlama only achieves $8.2\%$ and $9.7\%$ on these two metrics.
    This is because the reference codes consist mainly of workflows with function names and arguments, and contain few Python-related keywords, making exact matching challenging.
    In contrast, models achieve better scores on syntactic AST match and semantic data-flow match.

4. 4.

    After fine-tuning, WorkflowLlama shows a significant improvement in its ability to orchestrate actions. The performance of WorkflowLlama even outperforms powerful closed-source models GPT-4o with ICL by a large margin. Specifically, WorkflowLlama achieves a $\textbf{39.3}\%$ score on CodeBLEU and a $\textbf{76.9}\%$ Pass Rate under ID settings, demonstrating the validity of our proposed WorkflowLLM framework and WorkflowBench dataset.

5. 5.

    WorkflowLlama demonstrates strong generalization capabilities. Even though it has not been trained on the same instructions or APIs, it still significantly outperforms the vanilla Llama-3.1 on all metrics, ahead of or close to the more powerful foundation models.
    Notably, our method achieves $\textbf{35.1}\%$ in CodeBLEU and $\textbf{70.4}\%$ in Pass Rate, outperforming all strong baselines.

### 4.4 Analysis of Workflow Complexity

<img src='x5.png' alt='Refer to caption' title='' width='761' height='445' />

<img src='x6.png' alt='Refer to caption' title='' width='761' height='449' />

<img src='x7.png' alt='Refer to caption' title='' width='761' height='449' />

*Figure 5: Performance comparisons based on the number of actions, the number of Branch \& Loop, and the nested depth of the reference code.*

To evaluate the models’ ability to generate workflows of varying complexity, we break down the performance of CodeBLEU according to the total number of actions, the number of branches and loops, and the nested depth of the reference code. As shown in Figure[5], the performance of all models deteriorates as the number of actions or the logical complexity increases, indicating the challenge of orchestrating complex workflows.
However, across all levels of complexity, WorkflowLlama significantly outperforms all other models. Moreover, the relative performance of WorkflowLlama improves as the complexity of the workflow increases, which demonstrates fine-tuning with WorkflowBench significantly enhances the model’s ability to handle more complex workflows.

### 4.5 Out-of-distribution Generalization to T-Eval (Chen et al., [2024])

*Table 3: Comparisons of F1 scores on the PLAN task of T-Eval. (Bold denotes the best score among models of the same category.)*

| Model | F1 |
| --- | --- |
| Proprietary Models | |
| Claude2 | 84.9 |
| GPT-3.5 | 86.6 |
| GPT-4 | 86.7 |
| Open-Source Models | |
| Qwen-7B | 63.1 |
| Mistral-7B | 64.9 |
| Llama-3.1-8B | 68.2 |
| Qwen-14B | 69.7 |
| Llama-2-13B | 65.1 |
| Vicuna-13B | 54.0 |
| Baichuan2-13B | 52.1 |
| WizardLM-70B | 42.7 |
| Llama-2-70B | 63.1 |
| Qwen-72B | 73.4 |
| WorkflowLlama (8B) | 77.5 |

Settings To further evaluate the generalization capability of WorkflowLlama, we conduct experiments using an OOD benchmark, T-Eval, a widely-used benchmark to evaluate the multi-step decision-making capability of LLMs to utilize APIs.
The original data format in T-Eval is based on JSON or strings, which differ significantly from the Python-based format employed in WorkflowBench. To ensure the evaluation metrics’ consistency between ours and the original paper, we convert WorkflowBench into JSON format while preserving the metadata of workflows and the specifics of queries. Subsequently, we retrain WorkflowLlama on the transformed dataset.
We employ the F1 Score proposed in the original paper to measure the alignment with the reference API sequences.

Results The results are shown in Table[3] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models"). As observed, WorkflowLlama demonstrates strong OOD generalization performance on the T-Eval benchmark, despite being trained on different domains and tasks using different APIs. Notably, WorkflowLlama significantly outperforms the vanilla Llama3.1-8B as well as larger open-source models like Llama-2-70B and Qwen-72B, highlighting that fine-tuning with WorkflowBench enhances the model’s out-of-distribution planning ability.

### 4.6 Ablation Study

*Table 4: Ablation study results of Natural Language Thoughts on Workflow Orchestration (%).*

{widetabular}

lccccc Model CodeBLEU  
BLEU Weighted N-Gram AST Data-Flow Overall  
WorkflowLlama 9.4 11.1 55.1 38.0 39.3 
 w/o Task Plan 9.1 10.7 53.9 36.6 38.2 
 w/o Comment 9.1 10.8 54.9 35.3 38.1 
 w/o Task Plan \& Comment 8.8 10.2 53.7 35.1 37.4 
 w/o Synthetic Data 7.8 9.4 53.5 35.4 37.3 
Settings To assess the efficacy of WorflowBench’s components, we conduct an ablation study under the settings of unseen instructions (i.e., the ID setting).

Results Table [4.6] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models") presents the performance results when the model is trained under different conditions: without synthetic data, without the task plan $\mathcal{P}$, without action-level comments $\mathcal{C}$, and without both $\mathcal{C}$ and $\mathcal{P}$. The experimental results reveal two key findings. First, the two types of natural language thoughts enhance the reasoning capabilities of the model. Removing either type of thought leads to a decline in CodeBLEU performance. Second, training on large-scale synthetic data further improves performance, highlighting the effectiveness of the WorkflowBench expansion process.

### 4.7 Case Study

<img src='x8.png' alt='Refer to caption' title='' width='830' height='314' />

*Figure 6: Case study of generated code between vanilla Llama-3.1-8B and WorkflowLlama.*

To further illustrate the effect of fine-tuning on WorkflowBench, we present a typical example in Figure [6] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models"). In this case, the vanilla Llama-3.1 model exhibits two types of errors. First, the model does not adhere to the given instructions for workflow orchestration, using APIs outside the provided list, i.e., hallucination APIs. Specifically, it uses the time.sleep() function instead of is_workflow_actions_delay() to set a timer. Second, due to its relatively weak workflow orchestration capabilities, the model fails to complete all user instructions. Specifically, it does not activate airplane mode using the is_workflow_actions_airplanemode_set() function.
Fine-tuning on WorkflowBench effectively alleviates these two issues. However, we observe that fine-tuning also introduces redundant actions. For instance, WorkflowLlama repeats the parsing check of the clipboard’s content. We will address this redundancy problem in future work.

5 Conclusion
------------

In this paper, we present WorkflowLLM to enhance the capability of large language models in workflow orchestration.
In WorkflowLLM, WorkflowBench is constructed covering $106,763$ workflows with $1,503$ APIs across $83$ applications through a three-phase pipeline.
By fine-tuning Llama-3.1-8B on WorkflowBench, we derive WorkflowLlama which can achieve superior performance on the workflow orchestration task exceeding all comparable baselines including the most advanced OpenAI GPT-4o.
Moreover, we adapt our WorkflowLlama on the T-Eval dataset and the experimental results reveal the generalization ability of our constructed WorkflowBench.
We believe that our constructed dataset has the potential to contribute to advancements in APA.

References
----------

* qwe (2024)Qwen2 technical report.2024.
* Agostinelli et al. (2020)Simone Agostinelli, Andrea Marrella, and Massimo Mecella.Towards intelligent robotic process automation for bpmers.*arXiv preprint arXiv:2001.00804*, 2020.
* Ahn et al. (2022)Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Chuyuan Fu, Keerthana Gopalakrishnan, Karol Hausman, et al.Do as i can, not as i say: Grounding language in robotic affordances.*arXiv preprint arXiv:2204.01691*, 2022.
* Apple (2024)Apple.Shortcuts app, 2024.URL <https://apps.apple.com/us/app/shortcuts/id915249334>.Accessed: 2024-05-09.
* Chen et al. (2023)Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W. Cohen.Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks.*Transactions on Machine Learning Research*, 2023.ISSN 2835-8856.URL [https://openreview.net/forum?id\=YfZ4ZPt8zd](https://openreview.net/forum?id=YfZ4ZPt8zd "").
* Chen et al. (2024)Zehui Chen, Weihua Du, Wenwei Zhang, Kuikun Liu, Jiangning Liu, Miao Zheng, Jingming Zhuo, Songyang Zhang, Dahua Lin, Kai Chen, et al.T-eval: Evaluating the tool utilization capability of large language models step by step.In *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)*, pp. 9510–9529, 2024.
* Cheng et al. (2023)Liying Cheng, Xingxuan Li, and Lidong Bing.Is gpt-4 a good data analyst?*arXiv preprint arXiv:2305.15038*, 2023.
* Cichocki et al. (1997)Andrzej Cichocki, Helal A Ansari, Marek Rusinkiewicz, and Darrell Woelk.*Workflow and process automation: concepts and technology*, volume 432.Springer Science \& Business Media, 1997.
* Dong et al. (2022)Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui.A survey on in-context learning.*arXiv preprint arXiv:2301.00234*, 2022.
* Dubey et al. (2024)Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, et al.The llama 3 herd of models.*arXiv preprint arXiv:2407.21783*, 2024.
* Ferreira et al. (2020)Deborah Ferreira, Julia Rozanova, Krishna Dubba, Dell Zhang, and Andre Freitas.On the evaluation of intelligent process automation.*arXiv preprint arXiv:2001.02639*, 2020.
* Gao et al. (2023)Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig.PAL: program-aided language models.In Andreas Krause, Emma Brunskill, Kyunghyun Cho, Barbara Engelhardt, Sivan Sabato, and Jonathan Scarlett (eds.), *International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA*, volume 202 of *Proceedings of Machine Learning Research*, pp. 10764–10799. PMLR, 2023.URL <https://proceedings.mlr.press/v202/gao23f.html>.
* Herm et al. (2020)Lukas-Valentin Herm, Christian Janiesch, Alexander Helm, Florian Imgrund, Kevin Fuchs, Adrian Hofmann, and Axel Winkelmann.A consolidated framework for implementing robotic process automation projects.In *Business Process Management: 18th International Conference, BPM 2020, Seville, Spain, September 13–18, 2020, Proceedings 18*, pp. 471–488. Springer, 2020.
* Hofmann et al. (2020)Peter Hofmann, Caroline Samp, and Nils Urbach.Robotic process automation.*Electronic markets*, 30(1):99–106, 2020.
* Huang et al. (2024)Tian Huang, Chun Yu, Weinan Shi, Zijian Peng, David Yang, Weiqi Sun, and Yuanchun Shi.Promptrpa: Generating robotic process automation on smartphones from textual prompts.*arXiv preprint arXiv:2404.02475*, 2024.
* Hummert \& Humphries (2022)Christian Hummert and Georgina Louise Humphries.*Property Lists*, pp. 157–165.Springer International Publishing, Cham, 2022.ISBN 978-3-030-98467-0.doi: 10.1007/978-3-030-98467-0˙6.URL [https://doi.org/10.1007/978-3-030-98467-0_6](https://doi.org/10.1007/978-3-030-98467-0_6 "").
* Ivančić et al. (2019)Lucija Ivančić, Dalia Suša Vugec, and Vesna Bosilj Vukšić.Robotic process automation: systematic literature review.In *Business Process Management: Blockchain and Central and Eastern Europe Forum: BPM 2019 Blockchain and CEE Forum, Vienna, Austria, September 1–6, 2019, Proceedings 17*, pp. 280–295. Springer, 2019.
* Jiang et al. (2024)Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, et al.Mixtral of experts.*arXiv preprint arXiv:2401.04088*, 2024.
* Li et al. (2024)Zelong Li, Shuyuan Xu, Kai Mei, Wenyue Hua, Balaji Rama, Om Raheja, Hao Wang, He Zhu, and Yongfeng Zhang.Autoflow: Automated workflow generation for large language model agents.*arXiv preprint arXiv:2407.12821*, 2024.
* Loshchilov \& Hutter (2019)Ilya Loshchilov and Frank Hutter.Decoupled weight decay regularization.In *7th International Conference on Learning Representations, ICLR 2019, New Orleans, LA, USA, May 6-9, 2019*. OpenReview.net, 2019.URL [https://openreview.net/forum?id\=Bkg6RiCqY7](https://openreview.net/forum?id=Bkg6RiCqY7 "").
* Nakano et al. (2021)Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al.Webgpt: Browser-assisted question-answering with human feedback.*ArXiv preprint*, abs/2112.09332, 2021.
* OpenAI (2022)OpenAI.OpenAI: Introducing ChatGPT, 2022.URL <https://openai.com/blog/chatgpt>.
* OpenAI (2023)OpenAI.Gpt-4 technical report, 2023.
* Qian et al. (2024)Chen Qian, Wei Liu, Hongzhang Liu, Nuo Chen, Yufan Dang, Jiahao Li, Cheng Yang, Weize Chen, Yusheng Su, Xin Cong, Juyuan Xu, Dahai Li, Zhiyuan Liu, and Maosong Sun.Chatdev: Communicative agents for software development.In Lun-Wei Ku, Andre Martins, and Vivek Srikumar (eds.), *Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2024, Bangkok, Thailand, August 11-16, 2024*, pp. 15174–15186. Association for Computational Linguistics, 2024.URL [https://aclanthology.org/2024.acl-long.810](https://aclanthology.org/2024.acl-long.810 "").
* Qin et al. (2023a)Yujia Qin, Zihan Cai, Dian Jin, Lan Yan, Shihao Liang, Kunlun Zhu, Yankai Lin, Xu Han, Ning Ding, Huadong Wang, et al.Webcpm: Interactive web search for chinese long-form question answering.*arXiv preprint arXiv:2305.06849*, 2023a.
* Qin et al. (2023b)Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, et al.Tool learning with foundation models.*arXiv preprint arXiv:2304.08354*, 2023b.
* Qin et al. (2024)Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun.Toolllm: Facilitating large language models to master 16000+ real-world apis.In *The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024*. OpenReview.net, 2024.URL [https://openreview.net/forum?id\=dHng2O0Jjr](https://openreview.net/forum?id=dHng2O0Jjr "").
* Ren et al. (2020)Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma.Codebleu: a method for automatic evaluation of code synthesis.*arXiv preprint arXiv:2009.10297*, 2020.
* Schick et al. (2024)Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom.Toolformer: Language models can teach themselves to use tools.*Advances in Neural Information Processing Systems*, 36, 2024.
* Shen et al. (2024)Haiyang Shen, Yue Li, Desong Meng, Dongqi Cai, Sheng Qi, Li Zhang, Mengwei Xu, and Yun Ma.Shortcutsbench: A large-scale real-world benchmark for api-based agents.*arXiv preprint arXiv:2407.00132*, 2024.
* Touvron et al. (2023a)Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurelien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample.Llama: Open and efficient foundation language models.*arXiv preprint arXiv:2302.13971*, 2023a.
* Touvron et al. (2023b)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al.Llama 2: Open foundation and fine-tuned chat models.*arXiv preprint arXiv:2307.09288*, 2023b.
* Wang et al. (2024a)Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji.Executable code actions elicit better LLM agents.*CoRR*, abs/2402.01030, 2024a.doi: 10.48550/ARXIV.2402.01030.URL [https://doi.org/10.48550/arXiv.2402.01030](https://doi.org/10.48550/arXiv.2402.01030 "").
* Wang et al. (2024b)Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji.Executable code actions elicit better llm agents.In *Forty-first International Conference on Machine Learning*, 2024b.
* Wang et al. (2024c)Zilong Wang, Yuedong Cui, Li Zhong, Zimin Zhang, Da Yin, Bill Yuchen Lin, and Jingbo Shang.Officebench: Benchmarking language agents across multiple applications for office automation.*CoRR*, abs/2407.19056, 2024c.doi: 10.48550/ARXIV.2407.19056.URL [https://doi.org/10.48550/arXiv.2407.19056](https://doi.org/10.48550/arXiv.2407.19056 "").
* Wei et al. (2022)Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.Chain-of-thought prompting elicits reasoning in large language models.*Advances in neural information processing systems*, 35:24824–24837, 2022.
* Wei et al. (2023)Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc Le, and Denny Zhou.Chain-of-thought prompting elicits reasoning in large language models, 2023.
* Wewerka \& Reichert (2020)Judith Wewerka and Manfred Reichert.Robotic process automation–a systematic literature review and assessment framework.*arXiv preprint arXiv:2012.11951*, 2020.
* Wornow et al. (2024)Michael Wornow, Avanika Narayan, Krista Opsahl-Ong, Quinn McIntyre, Nigam H Shah, and Christopher Re.Automating the enterprise with foundation models.*arXiv preprint arXiv:2405.03710*, 2024.
* Wu et al. (2023)Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan.Visual chatgpt: Talking, drawing and editing with visual foundation models.*arXiv preprint arXiv:2303.04671*, 2023.
* Xie et al. (2024)Jian Xie, Kai Zhang, Jiangjie Chen, Tinghui Zhu, Renze Lou, Yuandong Tian, Yanghua Xiao, and Yu Su.Travelplanner: A benchmark for real-world planning with language agents.*arXiv preprint arXiv:2402.01622*, 2024.
* Yang et al. (2023)Linyao Yang, Hongyang Chen, Zhao Li, Xiao Ding, and Xindong Wu.Chatgpt is not enough: Enhancing large language models with knowledge graphs for fact-aware language modeling.*arXiv preprint arXiv:2306.11489*, 2023.
* Yao et al. (2022)Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao.React: Synergizing reasoning and acting in language models.*ArXiv preprint*, abs/2210.03629, 2022.
* Ye et al. (2023)Yining Ye, Xin Cong, Shizuo Tian, Jiannan Cao, Hao Wang, Yujia Qin, Yaxi Lu, Heyang Yu, Huadong Wang, Yankai Lin, et al.Proagent: From robotic process automation to agentic process automation.*arXiv preprint arXiv:2311.10751*, 2023.
* Zeng et al. (2023)Zhen Zeng, William Watson, Nicole Cho, Saba Rahimi, Shayleen Reynolds, Tucker Balch, and Manuela Veloso.Flowmind: automatic workflow generation with llms.In *Proceedings of the Fourth ACM International Conference on AI in Finance*, pp. 73–81, 2023.
* Zhuang et al. (2024)Yuchen Zhuang, Yue Yu, Kuan Wang, Haotian Sun, and Chao Zhang.Toolqa: A dataset for llm question answering with external tools.*Advances in Neural Information Processing Systems*, 36, 2024.

Appendix A Algorithm of Transcribing Shortcuts
----------------------------------------------

Data: Shortcut file to be transcribed

Result: Abstract syntax tree of the actions

Initialize an empty tree with a root node and set current_node to root

foreach *action in action list* do

Determine action_type and mode from action

if *action_type is Conditional* then

HandleConditional(mode, action)

else if *action_type is RepeatEach* then

HandleLoop(mode, action)

else if *action_type is RepeatCount* then

HandleLoop(mode, action)

else if *action_type is ChooseFromMenu* then

HandleMatchCase(mode, action)

else

HandleDefault(action)

Function *HandleConditional(mode, action)*:

if *mode \=\= 0 (start if)* then

AddNode(action)

Set current_node to new node

else if *mode \=\= 1 (else)* then

Move current_node to parent node

AddNode(action)

Set current_node to new node

else if *mode \=\= 2 (end if)* then

Move current_node to parent node

Function *HandleLoop(mode, action)*:

if *mode \=\= 0 (start loop)* then

AddNode(action)

Set current_node to new node

else if *mode \=\= 2 (end loop)* then

Move current_node to parent node

Function *HandleMatchCase(mode, action)*:

if *mode \=\= 0 (start match)* then

AddNode(action)

Set current_node to new node

else if *mode \=\= 1 (start case)* then

if *current_node is match node* then

AddNode(action)

Set current_node to new node

else

Move current_node to parent match node

AddNode(action)

Set current_node to new node

else if *mode \=\= 2 (end match)* then

Move current_node to parent node

Function *HandleDefault(action)*:

AddNode(action)

Function *AddNode(action)*:

Create new node with action

Append new node to current_node.children

Set parent of new node to current_node

*Algorithm 1 Recursive Parsing of Property List to Construct Abstract Syntax Tree*

Appendix B Prompt Design
------------------------

### B.1 Workflow Orchestration Prompt

[⬇](data:text/plain;base64,WW91IGFyZSBhIHZlcnkgaGVscGZ1bCBBSSBhc3Npc3RhbnQgd2hvIGNhbiB3cml0ZSBjb3JyZXNwb25kaW5nIFB5dGhvbiBtYWluIGNvZGUgYmFzZWQgb24gdXNlcidzIHF1ZXJ5IGFuZCB1c2FibGUgUHl0aG9uIGZ1bmN0aW9uIGludGVyZmFjZS4KClBsZWFzZSBnZW5lcmF0ZSBweXRob24gbWFpbiBjb2RlIGJhc2VkIG9uIHRoZSBmb2xsb3dpbmcgcXVlcnkgOgoge3F1ZXJ5fQpZb3UgY2FuIHN0YXJ0IGJ5IHVzaW5nIG5hdHVyYWwgbGFuZ3VhZ2UgdG8gcGxhbiB5b3VyIHRvb2wgY2FsbCBzdHJhdGVneSwgYW5kIHRoZW4gZ2VuZXJhdGUgdGhlIGNvbXBsZXRlIGNvZGUuIEZvciBleGFtcGxlLCBgVGhvdWdodDoKPHRvb2wgY2FsbCBzdHJhdGVneT4KCkNvZGU6CmBgYHB5dGhvbgo8bWFpbiBjb2RlPgpgYGBgLgpOb3RlIHRoYXQgeW91ciBvdXRwdXQgc2hvdWxkIGFsd2F5cyBpbmNsdWRlIGBDb2RlOgpgYGBweXRob24KPG1haW4gY29kZT4KYGBgYCwgZm9ybWF0dGVkIGFjY29yZGluZ2x5LgpIZXJlIGFyZSBzb21lIHVzZWZ1bCBmdW5jdGlvbiBpbnRlcmZhY2UgeW91IG1heSB1c2U6CiB7YXBpc19kb2NzfQ==)

YouareaveryhelpfulAIassistantwhocanwritecorrespondingPythonmaincodebasedonuser’squeryandusablePythonfunctioninterface.

Pleasegeneratepythonmaincodebasedonthefollowingquery:

{query}

Youcanstartbyusingnaturallanguagetoplanyourtoolcallstrategy,andthengeneratethecompletecode.Forexample,‘Thought:

<toolcallstrategy>

Code:

‘‘‘python

<maincode>

‘‘‘‘.

Notethatyouroutputshouldalwaysinclude‘Code:

‘‘‘python

<maincode>

‘‘‘‘,formattedaccordingly.

Herearesomeusefulfunctioninterfaceyoumayuse:

{apis_docs}

### B.2 Evaluator Prompt

[⬇](data:text/plain;base64,WW91IGFyZSBhIGtpbmRseSBjb2RlIHJldmlld2VyLCBJIHdpbGwgcHJvdmlkZSB5b3Ugd2l0aCBhIHF1ZXJ5LCBhIGxpc3Qgb2YgYWxsb3dlZCBhcGlzIGFuZCBhIHBpZWNlIG9mIGNvZGUgdG8gYmUgcmV2aWV3ZWQsIHlvdSBoZWxwIG1lIHRvIGNoZWNrIGlmIHRoZSBjb2RlIHRvIGJlIHJldmlld2VkIGlzIGNvbXBsaWFudCB3aXRoIG91ciBzcGVjaWZpY2F0aW9ucy4KVGhlIHJlcXVpcmVtZW50cyBhcmUgYXMgZm9sbG93czoKMS4gWW91ICoqc2hvdWxkIHJldHVybiBUcnVlIGV2ZW4gaWYgdGhlIGNvZGUgaW1wbGVtZW50cyBhZGRpdGlvbmFsIGZ1bmN0aW9uYWxpdHkgbm90IHJlcXVpcmVkIGluIHRoZSBxdWVyeSoqLCBhcyBsb25nIGFzIGl0IHJvdWdobHkgaW1wbGVtZW50cyB0aGUgcmVxdWlyZW1lbnRzIGluIHRoZSBxdWVyeS4KMi4gV2UgZG9uJ3QgaW1wb3NlIGFueSByZXF1aXJlbWVudHMgb24gY29kZSByZWFkYWJpbGl0eSBvciBuYW1pbmcgY29udmVudGlvbnMuIFlvdSAqKnNob3VsZCByZXR1cm4gVHJ1ZSBhcyBsb25nIGFzIHRoZSByZXZpZXdlZCBjb2RlIGRvZXNuJ3QgdXNlIGRpc2FsbG93ZWQgZnVuY3Rpb25zIGFuZCByZWFzb25hYmx5IGFjY29tcGxpc2hlcyB3aGF0IGlzIGFza2VkIGluIHRoZSBxdWVyeSBpbiBnZW5lcmFsIHRlcm1zKiouIFRoZXJlJ3Mgbm8gbmVlZCB0byBnZXQgc3RyaWN0bHkgaHVuZyB1cCBvbiB0aGUgZGV0YWlscy4KMy4gUmV0dXJuIEZhbHNlIGlmIHRoZSBjb2RlIGZhaWxzIHRvIGZ1bGZpbGwgdGhlIHJlcXVpcmVtZW50IGluIHRoZSBxdWVyeS4gZS5nLiBpZiBpdCBpcyBwcm9wb3NlZCBpbiB0aGUgcXVlcnkgdG8gdHVybiBkb3duIHRoZSBiYXR0ZXJ5IGxldmVsIG9mIHRoZSBwaG9uZSBhbmQgdGhlIGJyaWdodG5lc3Mgb2YgdGhlIHNjcmVlbiwgaXQgaXMgYSBmYWlsdXJlIHRvIGZ1bGZpbGwgb25seSBhbnkgb25lIG9mIHRoZSBmdW5jdGlvbnMuCjQuIEJ1aWx0LWluIHB5dGhvbiBzeW50YXggc3VjaCBhcyBgaWZgLCBgbG9vcGAsIGBpbnB1dCgpYCwgYW5kIGBwcmludCgpYCBhcmUgYWxsb3dlZC4gIFJldHVybiBGYWxzZSBpZiB0aGUgY29kZSB1c2VzICoqYW55IGV4dGVybmFsIGZ1bmN0aW9ucyBvciBhcGlzKiogbm90IGluIGFsbG93ZWQgYXBpcyBsaXN0IGFuZCBub3QgYSBidWlsdC1pbiBmdW5jdGlvbiBzdWNoIGFzIGlucHV0KCksIHByaW50KCkuIEZvciBleGFtcGxlLCBpZiBJIHByb3ZpZGUgdGhlIGlzX3dvcmtmbG93X29wZW51cmwgZnVuY3Rpb24sIHRoaXMgc2hvdWxkIGJlIHVzZWQuIEFueSB1c2Ugb2YgYW55IG90aGVyIGxpYnJhcnkgbGlrZSByZXF1ZXN0cyBldGMuIGlzIGEgRmFsc2UuCnF1ZXJ5OntxdWVyeX0KbGlzdCBvZiBhbGxvd2VkIGFwaXM6IHthcGlzfQpjb2RlIHRvIHJldmlldzoge2NvZGV9CgpZb3VyIGFuc3dlcjogW1RydWUgb3IgRmFsc2Ugd2l0aCBpbnRlcnByZXRhdGlvbl0=)

Youareakindlycodereviewer,Iwillprovideyouwithaquery,alistofallowedapisandapieceofcodetobereviewed,youhelpmetocheckifthecodetobereviewediscompliantwithourspecifications.

Therequirementsareasfollows:

1.You**shouldreturnTrueevenifthecodeimplementsadditionalfunctionalitynotrequiredinthequery**,aslongasitroughlyimplementstherequirementsinthequery.

2.Wedon’timposeanyrequirementsoncodereadabilityornamingconventions.You**shouldreturnTrueaslongasthereviewedcodedoesn’tusedisallowedfunctionsandreasonablyaccomplisheswhatisaskedinthequeryingeneralterms**.There’snoneedtogetstrictlyhunguponthedetails.

3.ReturnFalseifthecodefailstofulfilltherequirementinthequery.e.g.ifitisproposedinthequerytoturndownthebatterylevelofthephoneandthebrightnessofthescreen,itisafailuretofulfillonlyanyoneofthefunctions.

4.Built-inpythonsyntaxsuchas‘if‘,‘loop‘,‘input()‘,and‘print()‘areallowed.ReturnFalseifthecodeuses**anyexternalfunctionsorapis**notinallowedapislistandnotabuilt-infunctionsuchasinput(),print().Forexample,ifIprovidetheis_workflow_openurlfunction,thisshouldbeused.Anyuseofanyotherlibrarylikerequestsetc.isaFalse.

query:{query}

listofallowedapis:{apis}

codetoreview:{code}

Youranswer:[TrueorFalsewithinterpretation]

### B.3 Comment Generation Prompt

[⬇](data:text/plain;base64,QSBTaG9ydGN1dCBpcyBhIHNlcXVlbmNlIG9mIGFjdG9ucywgd2hlcmUgZWFjaCBhY3Rpb24gaXMgYW4gQVBJIGNhbGwsIHRvIGV4ZWN1dGUgdXNlci1wcm92aWRlZCBxdWVyaWVzLgpBcyBhIHVzZXItZnJpZW5kbHkgYW5kIHBhdGllbnQgYXNzaXN0YW50LCB5b3VyIHRhc2sgaXMgdG8gcHJvdmlkZSBhIHNldCBvZiBkZXNjcmlwdGlvbiBvZiBlYWNoIGxpbmUgb2YgdGhlIGNvZGUgc2NyaXBwZXQuIFRvIHNhdmUgdGltZSwgSSBoYXZlIHJldHJpZXZlZCBhbGwgdGhlIGxpbmVzIGV4Y2x1c2l2ZSBvZiBibGFuayBsaW5lcyBvZiB0aGUgY29kZSBzbmlwcGV0IGFuZCBsaXN0ZWQgYXMgYSBkaWN0aW9uYXJ5IGJlbG93IHRoZSBjb2RlLgoKWW91ciBhbnN3ZXIgc2hvdWxkIGJlIGluIHRoZSBqc29uIGZvcm1hdCBhcyBmb2xsb3dzOgpgYGBqc29uCnsKICAgICJsaW5lIHgiOiAiPGRlc2NyaXB0aW9uLW9mLWxpbmUteD4iLAogICAgImxpbmUgeCsxIjogIjxkZXNjcmlwdGlvbi1vZi1saW5lLXgrMT4iLAogICAgIi4uLiI6ICIuLi4iLAogICAgImxpbmUgeCtuIjogIjxkZXNjcmlwdGlvbi1vZi1saW5lLXgrbj4iCn1gYGAKClRoZSBjb2RlIGlzIDoKe2NvZGV9ClRoZSBsaW5lcyBhcmUge2xpbmVzfQ==)

AShortcutisasequenceofactons,whereeachactionisanAPIcall,toexecuteuser-providedqueries.

Asauser-friendlyandpatientassistant,yourtaskistoprovideasetofdescriptionofeachlineofthecodescrippet.Tosavetime,Ihaveretrievedallthelinesexclusiveofblanklinesofthecodesnippetandlistedasadictionarybelowthecode.

Youranswershouldbeinthejsonformatasfollows:

‘‘‘json

{

"linex":"<description-of-line-x>",

"linex+1":"<description-of-line-x+1>",

"...":"...",

"linex+n":"<description-of-line-x+n>"

}‘‘‘

Thecodeis:

{code}

Thelinesare{lines}

### B.4 Task Plan Generation Prompt

[⬇](data:text/plain;base64,QmFzZWQgb24gdGhpcyBsaW5lIGJ5IGxpbmUgZGVzY3JpcHRpb24gb2YgdGhlIGNvZGUsIGdlbmVyYXRlIGEgZmxvd2NoYXJ0IG9mIGEgd29ya2Zsb3cgYnkgbmF0dXJhbCBsYW5ndWFnZS4KVGhpcyBpcyB0aGUgY29kZToKe2NvZGV9)

Basedonthislinebylinedescriptionofthecode,generateaflowchartofaworkflowbynaturallanguage.

Thisisthecode:

{code}

### B.5 Task Query Generation Prompt

[⬇](data:text/plain;base64,QXMgYSBoZWxwZnVsIGFzc2lzdGFudCwgcGxlYXNlIGhlbHAgbWUgY3JhZnQgYSBxdWVyeS4gVGhpcyBxdWVyeSwgZm9ybWF0dGVkIGFzIGEgcXVlc3Rpb24sIHNob3VsZCBkZXNjcmliZSB0aGUgdGFzayBhIHVzZXIgd2FudHMgdG8gY29tcGxldGUgYW5kIGFkaGVyZSB0byB0aGUgZm9sbG93aW5nIGNyaXRlcmlhOgoxLiBPbmUgb2YgdGhlIHNvbHV0aW9uIHRvIHRoZSB0YXNrIGRlc2NyaWJlZCBpbiB0aGUgcXVlcnkgY291bGQgYmUgdGhlIHB5dGhvbiBjb2RlIGJlbG93LgoyLiBJdCBzaG91bGQgYmUgY2xvc2UgdG8gcmVhbC13b3JsZCBwcm9ibGVtcyBvciByZXF1ZXN0cy4KMy4gSXQgc2hvdWxkIGluY2x1ZGUgbWFqb3IgcGFydHMgb2YgdGhlIGNvZGUuCjQuIFRoZSBxdWVyeSBzaG91bGQgbm90IHNwZWNpZnkgcHl0aG9uLgoKRm9yIGV4YW1wbGUsIHRoZSBjb2RlIGlzOgp7SUNMX2NvZGV9CkFuZCB0aGUgZXhwZWN0ZWQgb3V0cHV0IHF1ZXJ5IHNob3VsZCBiZSBzaW1pbGFyIHRvOgp7SUNMX3F1ZXJ5fQoKUGxlYXNlIGNyYWZ0IGEgcXVlcnkgYmFzZWQgb24gdGhlIGV4YW1wbGVzIGFuZCB0aGUgZm9sbG93aW5nIGNvZGU6Cntjb2RlfQ==)

Asahelpfulassistant,pleasehelpmecraftaquery.Thisquery,formattedasaquestion,shoulddescribethetaskauserwantstocompleteandadheretothefollowingcriteria:

1.Oneofthesolutiontothetaskdescribedinthequerycouldbethepythoncodebelow.

2.Itshouldbeclosetoreal-worldproblemsorrequests.

3.Itshouldincludemajorpartsofthecode.

4.Thequeryshouldnotspecifypython.

Forexample,thecodeis:

{ICL_code}

Andtheexpectedoutputqueryshouldbesimilarto:

{ICL_query}

Pleasecraftaquerybasedontheexamplesandthefollowingcode:

{code}

### B.6 Query Expansion Prompt

[⬇](data:text/plain;base64,WW91IGFyZSBleGNlcHRpb25hbGx5IHNraWxsZWQgYXQgY3JhZnRpbmcgcmVhbC13b3JsZCB1c2VyIHF1ZXJpZXMgZ2l2ZW4gc29tZSBhcGlzLiBIZXJlIGFyZSBleGFtcGxlczp7ZXhhbXBsZXN9LiBQbGVhc2UgZ2FpbiBpbnNwaXJhdGlvbiBmcm9tIHRoZSBmb2xsb3dpbmcgYXBpIGRvY3MgdG8gY3JlYXRlIGEgaGlnaC1xdWFsaXR5IHJlYWx3b3JsZCBxdWVyeS4KQXBpIGRvY3MgZm9yIGluc3BpcmF0aW9uOgpgYGBweXRob24Ke2FwaXNfc3RyaW5nfQpgYGAKUGxlYXNlIHJlZmVyIHRvIHRoZSBhYm92ZSBleGFtcGxlcyBhbmQgY3JhZnQgYSBuZXcgb25lIQpSZXF1aXJlbWVudHM6IEFQSSBuYW1lIGlzIHN0cmljdGx5IHByb2hpYml0ZWQgZnJvbSBhcHBlYXJpbmcgaW4gdGhlIGdlbmVyYXRlZCBxdWVyeS4gRWFjaCBxdWVyeSBzaG91bGQgYmUgY29tcGxpY2F0ZWQgZW5vdWdoIGFuZCBjYW4gYmUgc29sdmVkIHVzaW5nIGFsbCBhcGlzIGFib3ZlLiBUaGUgcXVlcnkgKipzaG91bGQgYmUgY2VudGVyZWQgYXJvdW5kIHtjYXRlZ29yeX0gdGhlbWUqKiBhbmQgc2hvdWxkIG5vdCBiZSBzcHJlYWQgb3V0IGludG8gdW5yZWxhdGVkIHBpZWNlcy4=)

Youareexceptionallyskilledatcraftingreal-worlduserqueriesgivensomeapis.Hereareexamples:{examples}.Pleasegaininspirationfromthefollowingapidocstocreateahigh-qualityrealworldquery.

Apidocsforinspiration:

‘‘‘python

{apis_string}

‘‘‘

Pleaserefertotheaboveexamplesandcraftanewone!

Requirements:APInameisstrictlyprohibitedfromappearinginthegeneratedquery.Eachqueryshouldbecomplicatedenoughandcanbesolvedusingallapisabove.Thequery**shouldbecenteredaround{category}theme**andshouldnotbespreadoutintounrelatedpieces.

### B.7 Quality Confirmation Prompt

[⬇](data:text/plain;base64,WW91IGFyZSBleGNlcHRpb25hbGx5IHNraWxsZWQgYXQgcG9saXNoaW5nIHRvb2wgY2FsbGluZyBwbGFuIChpLmUuLCB0aG91Z2h0KSBhbmQgcHl0aG9uIGNvZGUgZ2l2ZW4gYSB0YXNrLgoKR2l2ZW4gdGFzazoKe3F1ZXJ5fQoKCk9sZCB0b29sIGNhbGxpbmcgcGxhbjoKe3Rob3VnaHR9CgogT2xkIGNvZGU6Cntjb2RlfQoKIFVzZWQgQVBJIGRvYzoKe2FwaXN9CgpIZXJlIGFyZSBleGFtcGxlcyBmb3IgeW91IHRvIHJlZmVyOntJQ0xfY29udGV4dH0uClBsZWFzZSBtYWtlIHN1cmUgdGhlIGNvZGUgaXMgbG9naWNhbGx5IGNvcnJlY3QgYW5kIG9wZXJhdGlvbmFsLgoKUmVxdWlyZW1lbnRzOgpbMV0gRW5zdXJlIHRoYXQgYm90aCBwbGFuIGFuZCBjb2RlIHJlc3BvbmQgY29ycmVjdGx5IHRvIHRoZSB0YXNrIGFuZCB0aGF0IGNvZGUgY2FsbHMgbWF0Y2ggdGhlIHBsYW4sIHdoaWNoIHlvdSBjYW4gZG8gYnkgdHdlYWtpbmcsIGVtYmVsbGlzaGluZywgYW5kIG1vZGlmeWluZyBib3RoIHBsYW4gYW5kIGNvZGUuClBsYW4gZG9lcyBub3QgaGF2ZSB0byBiZSBvbmUtdG8tb25lIGNvcnJlc3BvbmRlbmNlIG9mIGNvZGU7IHBsYW4gY2FuIGJlIGFiYnJldmlhdGVkLgpbMl0gUGxlYXNlIGVuc3VyZSB0aGF0IHRoZSBjb2RlIGNvbmZvcm1zIHRvIHB5dGhvbiBzeW50YXguIEVuc3VyZSB0aGF0IGFsbCBweXRob24gY29kZSBpcyBjb21wbGV0ZSBhbmQgcnVubmFibGUuIFlvdSBjYW4gYWRkIGNvZGUgd2hlbiBuZWNlc3NhcnkuClszXSBFdmVyeSBsaW5lIG9mIGNvZGUgc2hvdWxkIGJlIHByZWNlZGVkIGJ5IGEgY29tbWVudCBtYXJrZWQgd2l0aCBhICIjIi4gV2hlbiBtb2RpZnlpbmcgdGhlIGNvZGUsIHBsZWFzZSBtb2RpZnkgdGhlIGluLWxpbmUgY29tbWVudHMgYWNjb3JkaW5nbHkuCls0XSBFbnN1cmUgdGhhdCBhbGwgZnVuY3Rpb24gcGFyYW1ldGVyIGNhbGxzIGFyZSBjb3JyZWN0IGFuZCB5b3UgY2FuIGNoYW5nZSB0aGUgY29kZSBpbiBjYXNlIG9mIGVycm9ycy4KWzVdIFRob3VnaHQgYW5kIGNvZGUgc2hvdWxkIGJlIGFzIGNvbmNpc2Ugd2hpbGUga2VlcGluZyB0aGUgbWVhbmluZyBpbnRhY3QuCls2XSBJZiB0aGVyZSBhcmUgY2FzZXMgaW5jbHVkaW5nIGludmFsaWQgYmluYXJ5IGNvZGUsIHJlcGxhY2UgdGhlbSB3aXRoIHJlYXNvbmFibGUgdGV4dCwgZGVsZXRlIHRoZW0sIG9yIHJlcGxhY2UgdGhlbSB3aXRoIGEgcmVhZGluZyBvcGVyYXRpb24gb24gYSBmaWxlIChlc3BlY2lhbGx5IHdoZW4gdGhlIGJpbmFyeSBjb2RlIGlzIGFuIGVuY29kZWQgaW1hZ2UpLgpSZXNwb25kIHN0cmljdGx5IHdpdGggSlNPTi4K)

Youareexceptionallyskilledatpolishingtoolcallingplan(i.e.,thought)andpythoncodegivenatask.

Giventask:

{query}

Oldtoolcallingplan:

{thought}

Oldcode:

{code}

UsedAPIdoc:

{apis}

Hereareexamplesforyoutorefer:{ICL_context}.

Pleasemakesurethecodeislogicallycorrectandoperational.

Requirements:

[1]Ensurethatbothplanandcoderespondcorrectlytothetaskandthatcodecallsmatchtheplan,whichyoucandobytweaking,embellishing,andmodifyingbothplanandcode.

Plandoesnothavetobeone-to-onecorrespondenceofcode;plancanbeabbreviated.

[2]Pleaseensurethatthecodeconformstopythonsyntax.Ensurethatallpythoncodeiscompleteandrunnable.Youcanaddcodewhennecessary.

[3]Everylineofcodeshouldbeprecededbyacommentmarkedwitha"#".Whenmodifyingthecode,pleasemodifythein-linecommentsaccordingly.

[4]Ensurethatallfunctionparametercallsarecorrectandyoucanchangethecodeincaseoferrors.

[5]Thoughtandcodeshouldbeasconcisewhilekeepingthemeaningintact.

[6]Iftherearecasesincludinginvalidbinarycode,replacethemwithreasonabletext,deletethem,orreplacethemwithareadingoperationonafile(especiallywhenthebinarycodeisanencodedimage).

RespondstrictlywithJSON.

### B.8 Variable Rename Prompt

[⬇](data:text/plain;base64,WW91IGFyZSBhIGhlbHBmdWwgYXNzaXN0YW50IGZvciByZW5hbWluZyB2YXJpYWJsZSBuYW1lcyBpbiBhIGNvZGUgc25pcHBldC4KVGhlIGZvbGxvd2luZyBjb2RlIHNuaXBwZXQgaXMgYSBwYXJ0IG9mIGEgcHJvZ3JhbSwgYW5kIHZhcmlhYmxlcyBhcmUgbmFtZWQgaW4gZm9ybWF0ICd2YXJpYWJsZXhfJy4KWW91ciB0YXNrIGlzIHRvIHJlbmFtZSB0aGVzZSB2YXJpYWJsZXMgc28gdGhhdCB0aGV5IGNvbmZvcm0gdG8gdGhlIHByb2dyYW1taW5nIHNwZWNpZmljYXRpb24gYW5kIGhhdmUgc29tZSBzZW1hbnRpYyBtZWFuaW5nLCB3aGljaCBjYW4gYmUgaW5mZXJlZCBieSByZWxhdGl2ZSBmdW5jdGlvbiBjYWxscwpBbmQgeW91ciBvdXRwdXQgc2hvdWxkIG9ubHkgYmUgYSBkaWN0aW9uYXJ5IGNvbnRhaW5pbmcgdGhlIG9sZCBuYW1lLW5ldyBuYW1lIGtleSB2YWx1ZSBwYWlyClRoZSBkZWZpbml0aW9uIG9mIHNvbWUgZnVuY3Rpb25zIGFyZSBub3QgaW5jbHVkZWQsIGFuZCB5b3Ugc2hvdWxkbid0IG1vZGlmeSB0aGVtLgpGb2xsb3dpbmcgdGhlIGNvZGUsIHRoZXJlJ3MgYSBkaWN0aW9uYXJ5IHRoYXQgY29udGFpbnMgc2hvcnQgZGVzY3JpcHRpb24gb2YgdGhlIHV1aWQtbmFtZWQgdmFyaWFibGUuIEFuZCB5b3UgY2FuIHRha2UgaXQgYXMgcmVmZXJlbmNlLgpOb3RlIHRoYXQgd2hpbGUgdGhlIGRlc2NyaXB0aW9uIG1pZ2h0IGJlIHRoZSBzYW1lLCBidXQgdGhlIGFjdHVhbCBtZWFuaW5nIGlzIGRpZmZlcmVudCBhY3Jvc3MgZGlmZmVyZW50IHZhcmlhYmxlcy4gU28geW91IHNob3VsZCBub3QganVzdCBjb3B5IHRoZSBzaG9ydCBkZXNjcmlwdGlvbi4gSW5zdGVhZCB5b3UnZCBiZXR0ZXIgY29ucHJlaGVuc2l2ZWx5IGNvbnNpZGVyIHRoZSBkZXNjcmlwdGlvbiwgbmFtZXMgb2YgY2FsbGVkIGZ1bmN0aW9ucywgYW5kIHRoZSBnZW5lcmFsIGxvZ2ljLgpUaGUgY29kZSBpcyBhcyBmb2xsb3dzOgp7Y29kZX0KVGhlIGRpY3Rpb25hcnkgaXMgYXMgZm9sbG93czoKe2Rlc2NyaXB0aW9ufQpUbyBzYXZlIHRpbWUsIEkgaGF2ZSByZXRyaWV2ZWQgYWxsIHRoZSB2YXJpYWJsZXMgdGhhdCByZXF1aXJlcyB0byBiZSByZW5hbWVkOgp7dmFyaWFibGVzfQ==)

Youareahelpfulassistantforrenamingvariablenamesinacodesnippet.

Thefollowingcodesnippetisapartofaprogram,andvariablesarenamedinformat’variablex_’.

Yourtaskistorenamethesevariablessothattheyconformtotheprogrammingspecificationandhavesomesemanticmeaning,whichcanbeinferedbyrelativefunctioncalls

Andyouroutputshouldonlybeadictionarycontainingtheoldname-newnamekeyvaluepair

Thedefinitionofsomefunctionsarenotincluded,andyoushouldn’tmodifythem.

Followingthecode,there’sadictionarythatcontainsshortdescriptionoftheuuid-namedvariable.Andyoucantakeitasreference.

Notethatwhilethedescriptionmightbethesame,buttheactualmeaningisdifferentacrossdifferentvariables.Soyoushouldnotjustcopytheshortdescription.Insteadyou’dbetterconprehensivelyconsiderthedescription,namesofcalledfunctions,andthegenerallogic.

Thecodeisasfollows:

{code}

Thedictionaryisasfollows:

{description}

Tosavetime,Ihaveretrievedallthevariablesthatrequirestoberenamed:

{variables}

Appendix C Case Study of Shortcuts
----------------------------------

We provide a real-world shortcut example, which includes the following three presentation forms: the rwa property list configuration file, the Python code after transcription and variable renaming, and the visual interface on MacOS.

The raw property list configuration file is presented below. For the sake of brevity, we have omitted the middle portion containing the actions.

```

{
  "WFWorkflowClientVersion": "754",
  "WFWorkflowClientRelease": "2.1.2",
  "WFWorkflowMinimumClientVersion": 411,
  "WFWorkflowIcon": {
    "WFWorkflowIconStartColor": 4274264319,
    "WFWorkflowIconImageData": "b’’",
    "WFWorkflowIconGlyphNumber": 59672
  },
  "WFWorkflowImportQuestions": [],
  "WFWorkflowTypes": ["WatchKit", "ActionExtension"],
  "WFWorkflowInputContentItemClasses": ["WFURLContentItem"],
  "WFWorkflowActions": [
    {
      "WFWorkflowActionIdentifier": "is.workflow.actions.count",
      "WFWorkflowActionParameters": {
        "WFCountType": "Items",
        "UUID": "F292DD85-A8D2-4EBF-93E8-AC45F1C38310"
      }
    },
    {
      "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
      "WFWorkflowActionParameters": {
        "WFControlFlowMode": 0,
        "WFConditionalActionString": "0",
        "GroupingIdentifier": "51B09BBE-EF2D-4635-B820-412BADC6D64C",
        "WFCondition": "Equals"
      }
    },
    ...
    {
      "WFWorkflowActionIdentifier": "is.workflow.actions.conditional",
      "WFWorkflowActionParameters": {
        "GroupingIdentifier": "05DA8CFC-73E5-47EC-BBF6-7A23BD4D6C27",
        "WFControlFlowMode": 2
      }
    }
  ]
}

```

It can be observed that this configuration file employs non-semantic hexadecimal strings to represent variables and uses keywords such as is.workflow.actions.conditional and GroupingIdentifier to implement logic controls like conditions, making it inherently difficult to read and comprehend. Consequently, we have converted it into a Python-like code format. The Python code, after transcription, variable renaming, and commenting, is shown as follows:

[⬇](data:text/plain;base64,IyBUaGlzIGxpbmUgY2FsbHMgdGhlIGZ1bmN0aW9uIGlzX3dvcmtmbG93X2FjdGlvbnNfY291bnQgd2l0aCBhIHBhcmFtZXRlciBvZiBXRkNvdW50VHlwZSBzZXQgdG8gJ0l0ZW1zJywgd2hpY2ggY2hlY2tzIHRoZSBjb3VudCBvZiB3b3JrZmxvdyBhY3Rpb25zIHJlbGF0ZWQgdG8gaXRlbXMgYW5kIGFzc2lnbnMgdGhlIHJlc3VsdCB0byB3b3JrZmxvd19hY3Rpb25fY291bnQuCndvcmtmbG93X2FjdGlvbl9jb3VudCA9IGlzX3dvcmtmbG93X2FjdGlvbnNfY291bnQoIFdGQ291bnRUeXBlPSdJdGVtcycpCiMgVGhpcyBsaW5lIGNoZWNrcyBpZiB0aGUgd29ya2Zsb3dfYWN0aW9uX2NvdW50IGlzIGVxdWFsIHRvICcwJywgd2hpY2ggbWVhbnMgdGhlcmUgYXJlIG5vIGF2YWlsYWJsZSBhY3Rpb25zIGZvciBpdGVtcy4KaWYgd29ya2Zsb3dfYWN0aW9uX2NvdW50ID09ICcwJzoKICAgICMgSWYgdGhlcmUgYXJlIG5vIGFjdGlvbnMsIHRoaXMgbGluZSBjYWxscyB0aGUgZnVuY3Rpb24gaXNfd29ya2Zsb3dfYWN0aW9uc191cmwgd2l0aCBhIHBhcmFtZXRlciBvZiBXRlVSTEFjdGlvblVSTCBzZXQgdG8gYSBzcGVjaWZpYyBBbWF6b24gVVJMIHRvIGdldCB0aGUgVVJMIGZvciB0aGUgd29ya2Zsb3cgYWN0aW9ucyBhbmQgYXNzaWducyBpdCB0byB3b3JrZmxvd19hY3Rpb25fdXJsLgogICAgd29ya2Zsb3dfYWN0aW9uX3VybCA9IGlzX3dvcmtmbG93X2FjdGlvbnNfdXJsKCBXRlVSTEFjdGlvblVSTD0naHR0cHM6Ly93d3cuYW1hem9uLmNvbS9ncC9oaXN0b3J5JykKICAgICMgVGhpcyBsaW5lIGRpc3BsYXlzIHRoZSB3ZWJwYWdlIGRlZmluZWQgYnkgd29ya2Zsb3dfYWN0aW9uX3VybCBieSBjYWxsaW5nIHRoZSBpc193b3JrZmxvd19hY3Rpb25zX3Nob3d3ZWJwYWdlIGZ1bmN0aW9uLgogICAgaXNfd29ya2Zsb3dfYWN0aW9uc19zaG93d2VicGFnZSggV0ZVUkw9d29ya2Zsb3dfYWN0aW9uX3VybCkKIyBUaGlzIGxpbmUgc3RhcnRzIHRoZSBlbHNlIGNsYXVzZSB0aGF0IGV4ZWN1dGVzIGlmICdVcGRhdGVLaXQnIGlzIG5vdCBmb3VuZCBpbiBteV93b3JrZmxvd3MuCmVsc2U6CiAgICAjIEluIHRoaXMgbGluZSwgdGhlIGNvZGUgcHJvbXB0cyB0aGUgdXNlciBmb3IgaW5wdXQgd2l0aCAnUGxlYXNlIGVudGVyIHRoZSB2YWx1ZTonLCBjYXB0dXJlcyBpdCwgYW5kIGNhbGxzIHRoZSBmdW5jdGlvbiBpc193b3JrZmxvd19hY3Rpb25zX2dldHZhcmlhYmxlIHRvIGdldCBhIGNvcnJlc3BvbmRpbmcgdmFyaWFibGUgYW5kIGFzc2lnbnMgdGhlIHJlc3VsdCB0byB1c2VyX2lucHV0X3ZhbHVlLgogICAgdXNlcl9pbnB1dF92YWx1ZSA9IGlzX3dvcmtmbG93X2FjdGlvbnNfZ2V0dmFyaWFibGUoIFdGVmFyaWFibGU9Zid7aW5wdXQoIlBsZWFzZSBlbnRlciB0aGUgdmFsdWU6Iil9JykKICAgICMgVGhpcyBsaW5lIHByb2Nlc3NlcyB0aGUgdXNlcl9pbnB1dF92YWx1ZSBieSBjYWxsaW5nIHRoZSBmdW5jdGlvbiBpc193b3JrZmxvd19hY3Rpb25zX2RldGVjdF9saW5rLCB3aGljaCBleHRyYWN0cyBhIGxpbmsgZnJvbSB0aGUgdXNlcidzIGlucHV0LCBhbmQgYXNzaWducyB0aGUgZGV0ZWN0ZWQgbGluayB0byBkZXRlY3RlZF9saW5rLgogICAgZGV0ZWN0ZWRfbGluayA9IGlzX3dvcmtmbG93X2FjdGlvbnNfZGV0ZWN0X2xpbmsoIFdGSW5wdXQ9dXNlcl9pbnB1dF92YWx1ZSkKICAgICMgSGVyZSwgdGhlIGRldGVjdGVkX2xpbmsgaXMgdXNlZCBhcyBpbnB1dCBmb3IgdGhlIGZ1bmN0aW9uIGlzX3dvcmtmbG93X2FjdGlvbnNfZ2V0aXRlbWZyb21saXN0IHRvIHJldHJpZXZlIGFuIGl0ZW0gZnJvbSBhIGxpc3QgYW5kIGFzc2lnbnMgdGhlIHJlc3VsdCB0byBpdGVtX2Zyb21fbGlzdC4KICAgIGl0ZW1fZnJvbV9saXN0ID0gaXNfd29ya2Zsb3dfYWN0aW9uc19nZXRpdGVtZnJvbWxpc3QoIFdGSW5wdXQ9ZGV0ZWN0ZWRfbGluaykKICAgICMgRmluYWxseSwgdGhpcyBsaW5lIGRpc3BsYXlzIHRoZSB3ZWJwYWdlIGFzc29jaWF0ZWQgd2l0aCB0aGUgcmV0cmlldmVkIGl0ZW0gZnJvbSBpdGVtX2Zyb21fbGlzdCBieSBjYWxsaW5nIGlzX3dvcmtmbG93X2FjdGlvbnNfc2hvd3dlYnBhZ2UuCiAgICBpc193b3JrZmxvd19hY3Rpb25zX3Nob3d3ZWJwYWdlKCBXRlVSTD1pdGVtX2Zyb21fbGlzdCkKIyBUaGlzIGxpbmUgcmV0cmlldmVzIHRoZSB1c2VyJ3Mgd29ya2Zsb3dzIGJ5IGNhbGxpbmcgdGhlIGZ1bmN0aW9uIGlzX3dvcmtmbG93X2FjdGlvbnNfZ2V0bXl3b3JrZmxvd3MgYW5kIGFzc2lnbnMgdGhlIHJlc3VsdCB0byBteV93b3JrZmxvd3MuCm15X3dvcmtmbG93cyA9IGlzX3dvcmtmbG93X2FjdGlvbnNfZ2V0bXl3b3JrZmxvd3MoKQojIFRoaXMgbGluZSBjaGVja3MgaWYgJ1VwZGF0ZUtpdCcgZXhpc3RzIGluIHRoZSB1c2VyJ3Mgd29ya2Zsb3dzLgppZiAnVXBkYXRlS2l0JyBpbiBteV93b3JrZmxvd3M6CiAgICAjIElmICdVcGRhdGVLaXQnIGlzIGZvdW5kLCB0aGlzIGxpbmUgY3JlYXRlcyBhIGRpY3Rpb25hcnkgbmFtZWQgdXBkYXRla2l0X2RldGFpbHMgdGhhdCBjb250YWlucyB0aGUgZGV0YWlscyBmb3IgdGhlIHVwZGF0ZSBraXQsIGluY2x1ZGluZyBpdHMgbmFtZSwgdmVyc2lvbiwgYW5kIFJvdXRpbmVIdWIgSUQuCiAgICB1cGRhdGVraXRfZGV0YWlscyA9IHsnU2hvcnRjdXQgTmFtZSc6ICdCdXkgS2luZGxlIEJvb2snLCAnQ3VycmVudCBWZXJzaW9uJzogJzEuMCcsICdSb3V0aW5lSHViIElEJzogJzEzNjAnfQogICAgIyBUaGlzIGxpbmUgY2FsbHMgdGhlIGZ1bmN0aW9uIGlzX3dvcmtmbG93X2FjdGlvbnNfcnVud29ya2Zsb3cgdG8gZXhlY3V0ZSB0aGUgd29ya2Zsb3cgbmFtZWQgJ1VwZGF0ZUtpdCcgd2l0aCB0aGUgcGFyYW1ldGVycyBXRlNob3dXb3JrZmxvdyBzZXQgdG8gRmFsc2UgYW5kIFdGSW5wdXQgc2V0IHRvIHRoZSBkZXRhaWxzIGZyb20gdXBkYXRla2l0X2RldGFpbHMuCiAgICBpc193b3JrZmxvd19hY3Rpb25zX3J1bndvcmtmbG93KCBXRldvcmtmbG93TmFtZT0nVXBkYXRlS2l0JywgV0ZTaG93V29ya2Zsb3c9RmFsc2UsIFdGSW5wdXQ9dXBkYXRla2l0X2RldGFpbHMpCiAgICAjIFRoaXMgbGluZSBjb250YWlucyB0aGUgcGFzcyBzdGF0ZW1lbnQsIGluZGljYXRpbmcgdGhhdCBpZiAnVXBkYXRlS2l0JyBpcyBub3QgZm91bmQsIHRoZSBwcm9ncmFtIHdpbGwgZG8gbm90aGluZy4KICAgIHBhc3M=)

#Thislinecallsthefunctionis_workflow_actions_countwithaparameterofWFCountTypesetto’Items’,whichchecksthecountofworkflowactionsrelatedtoitemsandassignstheresulttoworkflow_action_count.

workflow_action_count\=is_workflow_actions_count(WFCountType\=’Items’)

#Thislinechecksiftheworkflow_action_countisequalto’0’,whichmeanstherearenoavailableactionsforitems.

ifworkflow_action_count\=\=’0’:

#Iftherearenoactions,thislinecallsthefunctionis_workflow_actions_urlwithaparameterofWFURLActionURLsettoaspecificAmazonURLtogettheURLfortheworkflowactionsandassignsittoworkflow_action_url.

workflow_action_url\=is_workflow_actions_url(WFURLActionURL\=’https://www.amazon.com/gp/history’)

#Thislinedisplaysthewebpagedefinedbyworkflow_action_urlbycallingtheis_workflow_actions_showwebpagefunction.

is_workflow_actions_showwebpage(WFURL\=workflow_action_url)

#Thislinestartstheelseclausethatexecutesif’UpdateKit’isnotfoundinmy_workflows.

else:

#Inthisline,thecodepromptstheuserforinputwith’Pleaseenterthevalue:’,capturesit,andcallsthefunctionis_workflow_actions_getvariabletogetacorrespondingvariableandassignstheresulttouser_input_value.

user_input_value\=is_workflow_actions_getvariable(WFVariable\=f’{input("Please␣enter␣the␣value:")}’)

#Thislineprocessestheuser_input_valuebycallingthefunctionis_workflow_actions_detect_link,whichextractsalinkfromtheuser’sinput,andassignsthedetectedlinktodetected_link.

detected_link\=is_workflow_actions_detect_link(WFInput\=user_input_value)

#Here,thedetected_linkisusedasinputforthefunctionis_workflow_actions_getitemfromlisttoretrieveanitemfromalistandassignstheresulttoitem_from_list.

item_from_list\=is_workflow_actions_getitemfromlist(WFInput\=detected_link)

#Finally,thislinedisplaysthewebpageassociatedwiththeretrieveditemfromitem_from_listbycallingis_workflow_actions_showwebpage.

is_workflow_actions_showwebpage(WFURL\=item_from_list)

#Thislineretrievestheuser’sworkflowsbycallingthefunctionis_workflow_actions_getmyworkflowsandassignstheresulttomy_workflows.

my_workflows\=is_workflow_actions_getmyworkflows()

#Thislinechecksif’UpdateKit’existsintheuser’sworkflows.

if’UpdateKit’inmy_workflows:

#If’UpdateKit’isfound,thislinecreatesadictionarynamedupdatekit_detailsthatcontainsthedetailsfortheupdatekit,includingitsname,version,andRoutineHubID.

updatekit_details\={’Shortcut␣Name’:’Buy␣Kindle␣Book’,’Current␣Version’:’1.0’,’RoutineHub␣ID’:’1360’}

#Thislinecallsthefunctionis_workflow_actions_runworkflowtoexecutetheworkflownamed’UpdateKit’withtheparametersWFShowWorkflowsettoFalseandWFInputsettothedetailsfromupdatekit_details.

is_workflow_actions_runworkflow(WFWorkflowName\=’UpdateKit’,WFShowWorkflow\=False,WFInput\=updatekit_details)

#Thislinecontainsthepassstatement,indicatingthatif’UpdateKit’isnotfound,theprogramwilldonothing.

pass

<img src='x9.png' alt='Refer to caption' title='' width='656' height='1183' />

*Figure 7: The visual interface of the shortcut RoutineHub · Buy Kindle Book.*

We can clearly see that the transcribed Python code is of high quality, with strong readability, making it well-suited for training LLMs that have been pre-trained on extensive Python code.

For easy understanding, we also present the visual interface of this shortcut in Figure [7] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models"), which corresponds to the transcribed Python code on a line-by-line basis.

Appendix D Case Study of WorkflowBench
--------------------------------------

In this section, we provide a workflow example of WorkflowBench based on the shortcut listed in Appendix [C] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models"). We omit the annotated workflow code that has been shown in Appendix [C] ‣ 4.4 Analysis of Workflow Complexity ‣ 4.3 Main Experiments ‣ 4.2 Effectiveness of Evaluator ‣ 4 Experiments ‣ WorkflowLLM: Enhancing Workflow Orchestration Capability of Large Language Models").

The query $\mathcal{Q}$ is:

[⬇](data:text/plain;base64,Q291bGQgeW91IGd1aWRlIG1lIG9uIGhvdyB0byBkZXNpZ24gYSBzb2x1dGlvbiB0aGF0IG1hbmFnZXMgd29ya2Zsb3cgYWN0aW9ucz8gSSBuZWVkIGl0IHRvIGNoZWNrIGZvciBleGlzdGluZyBpdGVtcywgcmV0cmlldmUgcHVyY2hhc2UgaGlzdG9yeSBmcm9tIGEgc3BlY2lmaWMgc291cmNlIGlmIG5vIGl0ZW1zIGFyZSBwcmVzZW50LCBhbmQgYWxzbyBwcm92aWRlIHRoZSBvcHRpb24gdG8gaW5wdXQgYSBzcGVjaWZpYyBpdGVtIGZvciBkZXRhaWxlZCByZXRyaWV2YWwgaWYgaXRlbXMgYXJlIGF2YWlsYWJsZS4gQWRkaXRpb25hbGx5LCBJIHdhbnQgdGhlIHNvbHV0aW9uIHRvIHRyaWdnZXIgYSBzcGVjaWZpYyB3b3JrZmxvdyBpZiBpdCBpcyBhdmFpbGFibGUgd2l0aGluIG15IGV4aXN0aW5nIHByb2Nlc3Nlcy4=)

Couldyouguidemeonhowtodesignasolutionthatmanagesworkflowactions?Ineedittocheckforexistingitems,retrievepurchasehistoryfromaspecificsourceifnoitemsarepresent,andalsoprovidetheoptiontoinputaspecificitemfordetailedretrievalifitemsareavailable.Additionally,Iwantthesolutiontotriggeraspecificworkflowifitisavailablewithinmyexistingprocesses.

The list of provided APIs is:

[⬇](data:text/plain;base64,WwogICAgJ2lzLndvcmtmbG93LmFjdGlvbnMuc2hvd3dlYnBhZ2UnLAogICAgJ2lzLndvcmtmbG93LmFjdGlvbnMuZ2V0aXRlbWZyb21saXN0JywKICAgICdpcy53b3JrZmxvdy5hY3Rpb25zLmdldHZhcmlhYmxlJywKICAgICdpcy53b3JrZmxvdy5hY3Rpb25zLnVybCcsCiAgICAnaXMud29ya2Zsb3cuYWN0aW9ucy5nZXRteXdvcmtmbG93cycsCiAgICAnaXMud29ya2Zsb3cuYWN0aW9ucy5jb3VudCcsCiAgICAnaXMud29ya2Zsb3cuYWN0aW9ucy5ydW53b3JrZmxvdycsCiAgICAnaXMud29ya2Zsb3cuYWN0aW9ucy5kZXRlY3QubGluaycKXQ==)

[

’is.workflow.actions.showwebpage’,

’is.workflow.actions.getitemfromlist’,

’is.workflow.actions.getvariable’,

’is.workflow.actions.url’,

’is.workflow.actions.getmyworkflows’,

’is.workflow.actions.count’,

’is.workflow.actions.runworkflow’,

’is.workflow.actions.detect.link’

]

The task plan $\mathcal{P}$ is:

[⬇](data:text/plain;base64,MS4gKipTdGFydCoqCiAgIC0gVGhlIHByb2Nlc3MgYmVnaW5zLgoyLiAqKlJldHJpZXZlIFdvcmtmbG93IEFjdGlvbiBDb3VudCoqCiAgIC0gQ2FsbCB0aGUgZnVuY3Rpb24gYGlzX3dvcmtmbG93X2FjdGlvbnNfY291bnRgIHdpdGggcGFyYW1ldGVyIGBXRkNvdW50VHlwZWAgc2V0IHRvICdJdGVtcycuCiAgIC0gU3RvcmUgdGhlIHJlc3VsdCBpbiBgd29ya2Zsb3dfYWN0aW9uX2NvdW50YC4KMy4gKipDaGVjayBXb3JrZmxvdyBBY3Rpb24gQ291bnQqKgogICAtICoqRGVjaXNpb24qKjogSXMgYHdvcmtmbG93X2FjdGlvbl9jb3VudGAgZXF1YWwgdG8gJzAnPwogICAgIC0gKipZZXMqKjogUHJvY2VlZCB0byBzdGVwIDQuCiAgICAgLSAqKk5vKio6IFByb2NlZWQgdG8gc3RlcCAxMC4KNC4gKipHZXQgV29ya2Zsb3cgQWN0aW9uIFVSTCoqCiAgIC0gQ2FsbCB0aGUgZnVuY3Rpb24gYGlzX3dvcmtmbG93X2FjdGlvbnNfdXJsYCB3aXRoIHBhcmFtZXRlciBgV0ZVUkxBY3Rpb25VUkxgIHNldCB0byAnaHR0cHM6Ly93d3cuYW1hem9uLmNvbS9ncC9oaXN0b3J5Jy4KICAgLSBTdG9yZSB0aGUgcmVzdWx0IGluIGB3b3JrZmxvd19hY3Rpb25fdXJsYC4KNS4gKipEaXNwbGF5IFdlYnBhZ2UqKgogICAtIENhbGwgdGhlIGZ1bmN0aW9uIGBpc193b3JrZmxvd19hY3Rpb25zX3Nob3d3ZWJwYWdlYCB3aXRoIGBXRlVSTGAgc2V0IHRvIGB3b3JrZmxvd19hY3Rpb25fdXJsYC4KNi4gKipFbmQgV29ya2Zsb3cgQWN0aW9uIENoZWNrKioKICAgLSBFbmQgdGhlIHByb2Nlc3Mgb2YgY2hlY2tpbmcgd29ya2Zsb3cgYWN0aW9ucyBpZiB0aGUgY291bnQgaXMgJzAnLgo3LiAqKlJldHJpZXZlIE15IFdvcmtmbG93cyoqCiAgIC0gQ2FsbCB0aGUgZnVuY3Rpb24gYGlzX3dvcmtmbG93X2FjdGlvbnNfZ2V0bXl3b3JrZmxvd3NgLgogICAtIFN0b3JlIHRoZSByZXN1bHQgaW4gYG15X3dvcmtmbG93c2AuCjguICoqQ2hlY2sgZm9yIFVwZGF0ZUtpdCoqCiAgIC0gKipEZWNpc2lvbioqOiBJcyAnVXBkYXRlS2l0JyBpbiBgbXlfd29ya2Zsb3dzYD8KICAgICAtICoqWWVzKio6IFByb2NlZWQgdG8gc3RlcCA5LgogICAgIC0gKipObyoqOiBQcm9jZWVkIHRvIHN0ZXAgMTEuCjkuICoqSGFuZGxlIFVwZGF0ZUtpdCoqCiAgIC0gQ3JlYXRlIGEgZGljdGlvbmFyeSBgdXBkYXRla2l0X2RldGFpbHNgIHdpdGg6CiAgICAgLSAnU2hvcnRjdXQgTmFtZSc6ICdCdXkgS2luZGxlIEJvb2snCiAgICAgLSAnQ3VycmVudCBWZXJzaW9uJzogJzEuMCcKICAgICAtICdSb3V0aW5lSHViIElEJzogJzEzNjAnCiAgIC0gQ2FsbCB0aGUgZnVuY3Rpb24gYGlzX3dvcmtmbG93X2FjdGlvbnNfcnVud29ya2Zsb3dgIHdpdGg6CiAgICAgLSBgV0ZXb3JrZmxvd05hbWVgIHNldCB0byAnVXBkYXRlS2l0JwogICAgIC0gYFdGU2hvd1dvcmtmbG93YCBzZXQgdG8gRmFsc2UKICAgICAtIGBXRklucHV0YCBzZXQgdG8gYHVwZGF0ZWtpdF9kZXRhaWxzYC4KMTAuICoqUmVjZWl2ZSBVc2VyIElucHV0IChpZiBubyB1cGRhdGUga2l0KSoqCiAgIC0gUHJvbXB0IHVzZXI6ICJQbGVhc2UgZW50ZXIgdGhlIHZhbHVlOiIuCiAgIC0gQ2FwdHVyZSBpbnB1dCBhbmQgY2FsbCB0aGUgZnVuY3Rpb24gYGlzX3dvcmtmbG93X2FjdGlvbnNfZ2V0dmFyaWFibGVgLgogICAtIFN0b3JlIHRoZSByZXN1bHQgaW4gYHVzZXJfaW5wdXRfdmFsdWVgLgoxMS4gKipEZXRlY3QgTGluayBmcm9tIElucHV0KioKICAgLSBDYWxsIHRoZSBmdW5jdGlvbiBgaXNfd29ya2Zsb3dfYWN0aW9uc19kZXRlY3RfbGlua2Agd2l0aCBgV0ZJbnB1dGAgc2V0IHRvIGB1c2VyX2lucHV0X3ZhbHVlYC4KICAgLSBTdG9yZSB0aGUgcmVzdWx0IGluIGBkZXRlY3RlZF9saW5rYC4KMTIuICoqR2V0IEl0ZW0gZnJvbSBMaXN0KioKICAgLSBDYWxsIHRoZSBmdW5jdGlvbiBgaXNfd29ya2Zsb3dfYWN0aW9uc19nZXRpdGVtZnJvbWxpc3RgIHdpdGggYFdGSW5wdXRgIHNldCB0byBgZGV0ZWN0ZWRfbGlua2AuCiAgIC0gU3RvcmUgdGhlIHJlc3VsdCBpbiBgaXRlbV9mcm9tX2xpc3RgLgoxMy4gKipTaG93IEl0ZW0gV2VicGFnZSoqCiAgIC0gQ2FsbCB0aGUgZnVuY3Rpb24gYGlzX3dvcmtmbG93X2FjdGlvbnNfc2hvd3dlYnBhZ2VgIHdpdGggYFdGVVJMYCBzZXQgdG8gYGl0ZW1fZnJvbV9saXN0YC4KMTQuICoqRW5kIFByb2Nlc3MqKgogICAtIFRoZSBwcm9jZXNzIGNvbmNsdWRlcyBhZnRlciBwZXJmb3JtaW5nIHRoZSByZXNwZWN0aXZlIGFjdGlvbnMgYmFzZWQgb24gdGhlIGNvbmRpdGlvbmFsIGNoZWNrcy4=)

1.**Start**

-Theprocessbegins.

2.**RetrieveWorkflowActionCount**

-Callthefunction‘is_workflow_actions_count‘withparameter‘WFCountType‘setto’Items’.

-Storetheresultin‘workflow_action_count‘.

3.**CheckWorkflowActionCount**

-**Decision**:Is‘workflow_action_count‘equalto’0’?

-**Yes**:Proceedtostep4.

-**No**:Proceedtostep10.

4.**GetWorkflowActionURL**

-Callthefunction‘is_workflow_actions_url‘withparameter‘WFURLActionURL‘setto’https://www.amazon.com/gp/history’.

-Storetheresultin‘workflow_action_url‘.

5.**DisplayWebpage**

-Callthefunction‘is_workflow_actions_showwebpage‘with‘WFURL‘setto‘workflow_action_url‘.

6.**EndWorkflowActionCheck**

-Endtheprocessofcheckingworkflowactionsifthecountis’0’.

7.**RetrieveMyWorkflows**

-Callthefunction‘is_workflow_actions_getmyworkflows‘.

-Storetheresultin‘my_workflows‘.

8.**CheckforUpdateKit**

-**Decision**:Is’UpdateKit’in‘my_workflows‘?

-**Yes**:Proceedtostep9.

-**No**:Proceedtostep11.

9.**HandleUpdateKit**

-Createadictionary‘updatekit_details‘with:

-’ShortcutName’:’BuyKindleBook’

-’CurrentVersion’:’1.0’

-’RoutineHubID’:’1360’

-Callthefunction‘is_workflow_actions_runworkflow‘with:

-‘WFWorkflowName‘setto’UpdateKit’

-‘WFShowWorkflow‘settoFalse

-‘WFInput‘setto‘updatekit_details‘.

10.**ReceiveUserInput(ifnoupdatekit)**

-Promptuser:"Pleaseenterthevalue:".

-Captureinputandcallthefunction‘is_workflow_actions_getvariable‘.

-Storetheresultin‘user_input_value‘.

11.**DetectLinkfromInput**

-Callthefunction‘is_workflow_actions_detect_link‘with‘WFInput‘setto‘user_input_value‘.

-Storetheresultin‘detected_link‘.

12.**GetItemfromList**

-Callthefunction‘is_workflow_actions_getitemfromlist‘with‘WFInput‘setto‘detected_link‘.

-Storetheresultin‘item_from_list‘.

13.**ShowItemWebpage**

-Callthefunction‘is_workflow_actions_showwebpage‘with‘WFURL‘setto‘item_from_list‘.

14.**EndProcess**

-Theprocessconcludesafterperformingtherespectiveactionsbasedontheconditionalchecks.

Appendix E Limitations
----------------------

While the framework proposed in this paper represents notable progress in workflow orchestration, it also has certain limitations that warrant discussion. First, the APIs used in our work are exclusively derived from Apple Shortcuts application, resulting in a lack of coverage across more diverse fields and potentially limiting the generalizability of the dataset to broader application contexts. Second, our approach lacks evaluation through actual execution. This limitation arises due to the complexities involved in executing workflows, such as the need for intricate user registration and permission acquisition. Moreover, the APIs are subject to frequent changes as applications continue to evolve, making it challenging to implement a consistent executable evaluation. Consequently, our evaluation is limited to static analysis.

Appendix F Ethical Statement
----------------------------

In this study, the dataset construction process was fully automated using LLMs and algorithms for data annotation, eliminating the need for human annotators and thereby avoiding concerns related to annotator compensation and working conditions. The data utilized was collected through web scraping from publicly accessible sources, with strict adherence to the Terms of Service (ToS) of the respective websites. Scraping was avoided on platforms where such activity is explicitly prohibited, ensuring compliance with ethical standards. Additionally, no personally identifiable information (PII) or private user data was collected at any stage of the research process. All data was anonymized to protect privacy and mitigate any potential ethical concerns related to user information.
