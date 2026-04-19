Large Language Model-Aware In-Context Learning for Code Generation
====================================================================

Jia LiKey Lab of HCST (PKU), MOE, SCS, Peking UniversityBeijingChina[lijiaa@pku.edu.cn](mailto:lijiaa@pku.edu.cn),Ge LiKey Lab of HCST (PKU), MOE, SCS, Peking UniversityBeijingChina[lige@pku.edu.cn](mailto:lige@pku.edu.cn),Chongyang TaoPeking UniversityBeijingChina[chongyangtao@gmail.com](mailto:chongyangtao@gmail.com),Jia Li ♂, Huangzhao ZhangKey Lab of HCST (PKU), MOE, SCS, Peking UniversityBeijingChina[lijia@stu.pku.edu.cn](mailto:lijia@stu.pku.edu.cn)[zhang_hz@pku.edu.cn](mailto:zhang_hz@pku.edu.cn),Fang LiuBeihang UniversityBeijingChina[fangliu@buaa.edu.cn](mailto:fangliu@buaa.edu.cn)andZhi JinKey Lab of HCST (PKU), MOE, SCS, Peking UniversityBeijingChina[zhijin@pku.edu.cn](mailto:zhijin@pku.edu.cn)

(2018)

###### Abstract.

Large language models (LLMs) have shown impressive in-context learning (ICL) ability in code generation. LLMs take a prompt consisting of requirement-code examples and a new requirement as input, and output new programs without any parameter updates. Existing studies have found that the performance of ICL is highly dominated by the quality of selected examples and thus arises research on example selection: given a new requirement, several examples are retrieved from a candidate pool for ICL. However, existing approaches are mostly based on heuristics. They randomly select examples or only consider the textual similarity of requirements to retrieve, leading to sub-optimal performance. In this paper, we propose a novel learning-based selection approach named LAIL (LLM-Aware In-context Learning) for code generation. Given a candidate example, we exploit LLMs themselves to estimate it by considering the generation probabilities of ground-truth programs given a requirement and the example. We then label candidate examples as positive or negative through the probability feedback. Based on the labeled data, we import a contrastive learning objective to train an effective retriever that acquires the preference of LLMs in code generation. We argue that considering the feedback from LLMs themselves is logical and our approach can select suitable examples for ICL since it is consistent with the fact that LLMs have different prior knowledge and preferences. To evaluate our approach, we apply LAIL to three LLMs and evaluate it on three representative datasets (*e.g*., MBJP, MBPP, and MBCPP). Extensive experiments demonstrate that LATA outperforms the state-of-the-art baselines by 11.58%, 6.89%, and 5.07% on CodeGen, and 4.38%, 2.85%, and 2.74% on GPT-3.5 in terms of Pass@1, respectively. Human evaluation further verifies the superiority of our approach in three aspects. We also find our LAIL has surprising transferability across LLMs and datasets.

Code generation, in-context-learning, large language model

††copyright: acmcopyright††journalyear: 2018††doi: XXXXXXX.XXXXXXX††price: 15.00††isbn: 978-1-4503-XXXX-X/18/06

1. introduction
----------------

<img src='x1.png' alt='Refer to caption' title='' width='452' height='118' />

*Figure 1. Exhibition of the selected top-3 examples by random, BM25, and our LAIL approaches.*

Code generation aims to automatically generate the source code given a natural language requirement *(Li et al., [2023b](#bib.bib31 ""); Dong and Lapata, [2016](#bib.bib19 ""); Allamanis et al., [2018b](#bib.bib8 ""))*.
It plays an important role in improving the productivity of software development.
With the emergence of large language models (LLMs), in-context learning (ICL) *(Brown et al., [2020](#bib.bib12 ""))* becomes a prevailing paradigm and achieves impressive performance in code generation *(Chen et al., [2023](#bib.bib17 ""); Li et al., [2022](#bib.bib33 ""), [2023c](#bib.bib32 ""))*. Given limited examples as the prompt, ICL imitates the human ability to leverage prior knowledge to generate programs without parameter updates. However, ICL comes along with the robustness problem *(Liu et al., [2021](#bib.bib34 ""))*, which is sensitive to the selected examples in prompts, resulting in the performance usually varying from almost random to near state-of-the-art performance.

Despite the importance of selecting examples, only a few studies attempt to investigate example selection in code generation *(Chen et al., [2021](#bib.bib15 ""); Bareiß et al., [2022](#bib.bib11 ""); Li et al., [2023c](#bib.bib32 ""))*.
One line of work is to randomly select examples from a candidate pool *(Chen et al., [2021](#bib.bib15 ""); Jiang et al., [2023a](#bib.bib27 ""))*. The selected examples are usually improper to the test requirement and their semantic distributions vary a lot, resulting in unstable performance.
Another line is based on learning-free heuristics *(Li et al., [2023c](#bib.bib32 ""))*, which leverages off-the-shelf retrievers such as BM25 *(Robertson et al., [2009](#bib.bib43 ""))* to select examples.
They calculate the textual similarity between a test requirement and the requirement of examples, then select candidate items with high similarities. Despite the improved performance, these approaches only consider textual matching among requirements and thus are easily biased by exterior lexicon features.
Figure [1](#S1.F1 "Figure 1 ‣ 1. introduction ‣ Large Language Model-Aware In-Context Learning for Code Generation") reports the top-3 examples by random selection and the BM25 approach for the test requirement “write a function to check whether the entered number is greater than the elements of the given array” in MBPP dataset *(Austin et al., [2021](#bib.bib10 ""))*. To accomplish this requirement, LLMs need to perform traversal operations in an array.
Randomly selected examples are irrelevant to the test requirement.
Although retrieved examples by BM25 have a large amount of textual overlap with the test item labeled by underline, the overlapping words are trivial such as “ write a function”. Meanwhile, the operations of non-overlapping words in red color, such as “lines are parallel or not”, can provide less information to LLMs.
Examples in ICL are supposed to provide information that LLMs require, and then guide them to generate programs, but the existing approaches leave a huge gap. They do not consider the prior knowledge in LLMs and ignore their preferences. Although one may remedy it by inputting more examples in the prompt, it is highly impractical due to the limited input length of LLMs.
Therefore, selecting applicable examples for ICL becomes more urgent.

In this paper, we propose a novel learning-based approach dubbed LAIL to select in-context examples.
Instead of selecting examples via textual similarity, given a candidate example, our approach leverages LLMs themselves (LLM-aware) to measure it by incorporating the prediction probability of ground truth given a requirement and the candidate item. To quantify the probability feedback, we design a new metric and label candidate examples based on their metric scores. We treat examples with higher metric scores as positive examples, meanwhile, label examples equipped with lower scores as negative examples since good examples should be beneficial for LLMs to generate correct programs.
Based on labeled data, we train a neural retriever through a contrastive learning objective to acquire the preference of LLMs in code generation.
From Figure [1](#S1.F1 "Figure 1 ‣ 1. introduction ‣ Large Language Model-Aware In-Context Learning for Code Generation"), we can find that although not textually similar to the test requirement, the selected examples of LAIL contain the “traversal” operation and are related to the array as shown in purple color tokens.

We evaluate LAIL on three representative LLMs including ChatGPT *(Cha, [able](#bib.bib2 ""))*, GPT-3.5 *(GPT, [ablea](#bib.bib3 ""))*, and CodeGen (16B) *(Zhong et al., [2022](#bib.bib48 ""))*. We conduct extensive experiments on three datasets (*i.e*., MBJP (Java) *(Athiwaratkun et al., [2022](#bib.bib9 ""))*, MBPP (Python) *(Austin et al., [2021](#bib.bib10 ""))*, and MBCPP (C++) *(Athiwaratkun et al., [2022](#bib.bib9 ""))*). We use a widely used evaluation metric Pass@k (k \= 1, 3, 5) to measure the performance of different approaches. We obtain some findings from experimental results.
❶ In terms of Pass@1, LAIL significantly outperforms the state-of-the-art (SOTA) baselines by 11.58%, 6.89%, and 5.07% on CodeGen, and 4.38%, 2.85%, and 2.74% on GPT-3.5, respectively.
❷ We conduct a human evaluation to measure generated programs in three aspects (*e.g*. code correctness, quality, and maintainability) and further prove the superiority of LAIL.
❸ Our approach has surprising transferability across LLMs and datasets and brings satisfying improvements to them.
❹ We investigate two plausible choices to estimate candidate examples by LLMs themselves and demonstrate the effectiveness of our design.

We summarize our contributions in this paper as follows:

* $\bullet$

    Our paper investigates example selection for ICL and argues that a good approach should consider what knowledge LLMs themselves require.

* $\bullet$

    We propose a novel learning-based approach dubbed LAIL. Given an example, LAIL exploits LLMs themselves to estimate it by calculating the generation probabilities of ground truth given a requirement and the example. We then label candidate examples via their probabilities. Based on labeled data, we import contrastive learning to optimize a retriever.

* $\bullet$

    We evaluate our approach on three LLMs and conduct extensive experiments on three datasets. Qualitative and quantitative experiments reveal that LAIL significantly outperforms the state-of-the-art baselines.

2. Background
--------------

### 2.1. Large Language Models

In this section, we focus on large language models (LLMs) for source code. LLMs are neural networks that aim to learn the statistical patterns in programming languages *(Allamanis et al., [2018a](#bib.bib7 ""))*. LLMs are pre-trained on a large-scale unlabeled code corpus with the next tokens prediction objective. Specifically, given a program with the token sequence $C\=\left{c_{1},c_{2},\cdots,c_{n}\right}$, LLMs are trained to predict the next token based on some previous tokens:

| (1) |  | $\displaystyle\mathcal{L}_{NTP}(C)\=-\sum_{i}\texttt{log}P(c_{i}|c_{i-j},\cdots,c_{i-1};\Theta)$ |  |
| --- | --- | --- | --- |

where $j$ is the window length of previous tokens and $\Theta$ means parameters of the LLM.

After being pre-trained, LLMs are adapted to a specific downstream task. At the beginning stage, LLMs are used in a fine-tuning manner, which are continually optimized on specific code generation datasets. With the size of LLMs growing rapidly such as ChatGPT *(Cha, [able](#bib.bib2 ""))* and CodeGen *(Zhong et al., [2022](#bib.bib48 ""))*, fine-tuning is neither economical nor practical, in contrast, a convenient solution ICL arises.

### 2.2. In-Context Learning

In-context Learning (ICL) refers to an emerging ability of LLMs.
Formally, given a set of requirement-code examples $T\=\left{x_{k},y_{k}\right}_{k\=1}^{m}$, a test requirement $x_{t}$ and a LLM with frozen parameters $\Theta$, ICL defines the generation of a program $y_{t}$ as follows:

| (2) |  | $\displaystyle y_{t}\sim P(y_{t}|\underbrace{x_{1},y_{1},\cdots,x_{m},y_{m}}_{context},x_{t},\Theta)$ |  |
| --- | --- | --- | --- |

where $\sim$ represents decoding strategies such as greedy decoding and nuclear sampling *(Holtzman et al., [2019](#bib.bib23 ""))* in code generation.
The generation procedure is attractive as the parameters of LLMs are not need to be updated when executing a new task, which is efficient and practical.

As demonstrated in Formula [2](#S2.E2 "In 2.2. In-Context Learning ‣ 2. Background ‣ Large Language Model-Aware In-Context Learning for Code Generation"),
LLMs learn the task and generate programs depending on the selected examples. The performance of ICL can vary from almost random to near the state-of-the-art approach due to the different qualities of in-context examples. Thus, selecting appropriate examples is a significant research topic. In this paper, we propose a novel learning-based approach to select in-context examples for code generation.

3. Method: LAIL
----------------

In this section, we propose an effective learning-based approach LAIL to select examples for ICL in code generation, as shown in Figure [2](#S3.F2 "Figure 2 ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation").
For a candidate example, LAIL exploits LLMs themselves to estimate it by incorporating the prediction probability of the ground-truth program given a requirement and the example, then labels examples as positive and negative according to their probability feedback (Section [3.1](#S3.SS1 "3.1. Estimate and Label Examples ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation")). Our approach imports a contrastive learning objective to optimize a neural retriever, resulting in alignment with the preference of LLMs (Section [3.2](#S3.SS2 "3.2. Training Neural Retriever ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation")). In the inference stage, given a test requirement, LAIL selects a set of examples as a prompt and a LLM generates programs with the prompt (Section [3.3](#S3.SS3 "3.3. Inference ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation")).

<img src='x2.png' alt='Refer to caption' title='' width='369' height='558' />

*Figure 2. The overview of our LAIL. LAIL use LLMs themselves to estimate candidate examples and label them as positive and negative (A). Based on the label date, LAIL then trains a retriever to align with the preference of LLMs with a contrastive loss (B). Given a test requirement, the optimized retriever selects several examples as a prompt that is inputted to LLMs for code generation (C).*

### 3.1. Estimate and Label Examples

Different LLMs are equipped with diverse prior knowledge, thus they have disparate preferences for candidate examples in code generation. To mitigate this phenomenon, we argue that a good selection approach should align with the preference of LLMs.
Given a candidate example, we exploit LLMs themselves (LLM-aware) to measure it by incorporating the prediction probability of the ground-truth program given a requirement and the candidate item. Based on the probability feedback, we label candidate examples as positive and negative that are then used to train the retriever in Section [3.2](#S3.SS2 "3.2. Training Neural Retriever ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation").

Formally, given a requirement in the candidate pool, we use LLMs themselves to estimate whether each other candidate in the pool is beneficial to generate the ground truth of the requirement, and label each candidate based on its beneficial degree.
Following previous studies *(Li et al., [2023c](#bib.bib32 ""); Dong et al., [2022](#bib.bib20 ""))*, this paper uses the training set as the candidate pool.
The goal of this procedure can be formulated as follows:

| (3) |  | $\displaystyle\mathcal{D}\=\left{(e_{i},\left{S_{i}^{p},S_{i}^{n}\right})\right}_{i\=1}^{N}$ |  |
| --- | --- | --- | --- |

where $e_{i}\=(x_{i},y_{i})$ is the $i$-th examples in the training set. $S_{i}^{p}$ and $S_{i}^{n}$ are the positive and negative example set of $e_{i}$, respectively. $N$ is the number of examples in the training set $\mathcal{R}$.

Modeling on the full space of the training set is quadratic and thus prohibitive. To mitigate this limitation, we introduce a two-stage process to estimate examples in the training set.
In the first stage, for each example $e_{i}$, we use an off-the-shelf retriever to select a set of examples $S_{i}\=\left{(x_{q}^{i},y_{q}^{i})\right}_{q\=1}^{t}$ from $\mathcal{R}$ where $t$ $\ll$ $N$. In this paper, we apply BM25 111We also use random and semantics-based approaches to filter examples and find that BM25 is more effective and efficient. *(Robertson et al., [2009](#bib.bib43 ""))* to construct $S_{i}$, which measures textual n-gram matching between $x_{i}$ and $x_{q}^{i}$. According to BM25 scores, the top-$r$ examples with higher BM25 scores constitute the set $S_{i}$.
In the second stage, we leverage LLMs themselves to estimate each example in $S_{i}$. Concretely, we feed each example $(x_{q}^{i},y_{q}^{i})$ in $S_{i}$ and $x_{i}$ into the LLM, and acquire the prediction probability of ground-truth program $y_{i}$. To quantify LLMs’ probability feedback, we design a metric $\mathcal{M}$ which is defined as follows:

| (4) |  | $\displaystyle\mathcal{M}\=\frac{1}{|y_{i}|}\times P_{LLM}(y_{i}|x_{q}^{i},y_{q}^{i},x_{i})$ |  |
| --- | --- | --- | --- |

| (5) |  | $\displaystyle P_{LLM}(y_{i}|x_{q}^{i},y_{q}^{i},x_{i})\=\sum_{u\=1}^{|y_{i}|}log(p(t_{i,u}|x_{q}^{i},y_{q}^{i},x_{i},t_{i,<u}))$ |  |
| --- | --- | --- | --- |

where $y_{i}\=\left{t_{1},\cdots,t_{b}\right}$ and $t_{u}$ is the $u$-th token in $y_{i}$. The metric can reflect how helpful the example $(x_{q}^{i},y_{q}^{i})$ is for generating the target program $y_{i}$, which represents the preference of the LLM.

We then rank all examples in the set $S_{i}$ according to their metric scores $\mathcal{M}$ from high to low. The top-$z$ examples with higher scores are inputted into the positive example set $S_{i}^{p}$, meanwhile, the bottom-$v$ examples constitute the negative example set $S_{i}^{n}$ since good examples should be beneficial for LLMs to generate correct programs.
We apply this two-stage procedure to the entire training set and finally acquire the labeled data $\mathcal{D}$.

In our labeled data, the more helpful examples are labeled as positive examples and the examples with lower metric scores are treated as negative examples, which can reflect the preference of LLMs to candidate examples given a specific requirement.

### 3.2. Training Neural Retriever

As described in Section [3.1](#S3.SS1 "3.1. Estimate and Label Examples ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation"), the labeled data can reflect the preference of LLMs.
In this section, we use the labeled data $\mathcal{D}$ to train a neural retriever, aiming to align with the preference of LLMs. After being trained, given a test requirement, the retriever can select a set of candidate examples as a prompt that is beneficial for LLMs to generate correct programs.

To optimize the retriever, we import a contrastive learning objective. The objective targets to draw a test requirement with helpful training examples together and push the given requirement with knowledge-limited examples apart. Formally, for an example $e_{i}$ in the labeled data $\mathcal{D}$, we randomly select an item $e_{i}^{p}$ from $S_{i}^{p}$ as its corresponding positive example. Meanwhile, we randomly pick a training example $e_{i}^{n}$ from the set $S_{i}^{n}$ and select $h$ examples from the set $\mathcal{H}$ (*e.g*., $\mathcal{H}\=\mathcal{R}\setminus S_{i}$) as its negative example set $\mathbb{N}_{i}$. Next, We apply GraphCodeBERT *(Guo et al., [2020](#bib.bib22 ""))* to encode their requirements and acquire corresponding representations. Then, we model the relations of these requirements with contrastive loss. Following SimCLR *(Chen et al., [2020](#bib.bib16 ""))*, the learning objective $\mathcal{L}$ is formulated as:

| (6) |  | $\displaystyle\mathcal{L}\=$ | $\displaystyle-\bigg{[}log\frac{e^{s(E^{i}_{[CLS]},E^{i,p}_{[CLS]})/\tau}}{\sum_{E^{i,n}_{[CLS]}\in\mathbb{N}_{i}}e^{s(E^{i}_{[CLS]},E^{i,n}_{[CLS]})/\tau}}\bigg{]}$ |  |
| --- | --- | --- | --- | --- |

where $\tau$ is a temperature parameter. $E_{[CLS]}$ means the entire representation of a requirement. s(·) represents the cosine similarity of two vectors.

Based on the objective, our retriever learns the preference of LLMs, which will be effective to select helpful examples from the training set given a test requirement, resulting in promoting LLMs generating more correct programs.

### 3.3. Inference

During inference, instead of using heuristical approaches, we apply LAIL to select limited examples from the training set, which can retrieve examples with high metric scores given a test requirement and thus are helpful for LLMs in code generation.
Specifically, we first feed requirements in the training set $\mathcal{R}$ to our trained retriever respectively and acquire their representational vectors $\left{E_{CLS}^{i}\right}_{i\=1}^{N}$. Given a test requirement $x_{t}$, we obtain its representation $E_{CLS}^{t}$ by the retriever. Next, we match $E_{CLS}^{t}$ and $E_{CLS}^{i}$ and thus lead to $N$ pairs of representations $\left{(E_{CLS}^{t},E_{CLS}^{i})\right}_{i\=1}^{N}$. We calculate their cosine similarities $\left{c_{i}\right}_{i\=1}^{N}$ of all pairs and rank candidate examples according to similarity scores from high to low. Then, the top-$r$ examples are selected for ICL. We concatenate the top-$r$ examples as a prompt $\left{e_{1},\cdots,e_{i},\cdots,e_{r}\right}$ where their similarity scores gradually decrease (*e.g*., $c_{1}<c_{i}<c_{r}$). We feed the prompt and the test requirement into LLMs and make LLMs generate programs with nuclear sampling *(Holtzman et al., [2019](#bib.bib23 ""))* as described in Formula [2](#S2.E2 "In 2.2. In-Context Learning ‣ 2. Background ‣ Large Language Model-Aware In-Context Learning for Code Generation").

Note that LAIL only needs to encode training examples once. Given a test requirement, LAIL just calculates cosine similarities between it and all candidates. Thus, the efficiency of our approach is acceptable compared to other heuristic approaches.

4. Study design
----------------

To investigate the effectiveness of our LAIL, we perform a large-scale study to answer three research questions. In this section, we describe the details of our study, including datasets, evaluation metrics, baselines, base LLMs, and experimental details.

### 4.1. Research Questions

Our study aims to answer the following research questions.

RQ1: How does LAIL perform compared to the state-of-the-art baselines? This RQ aims to verify that LAIL can generate more correct programs than state-of-the-art (SOTA) baselines. We apply three LLMs to evaluate our approach. We compare LAIL to 7 baselines in three code generation datasets and employ unit tests to check the correctness of generated programs.

RQ2: Do developers prefer programs generated by LAIL? The ultimate goal of a code generation model is to assist developers in writing programs. In this RQ, we hire 10 developers to manually estimate the programs generated by our LAIL and baselines. We evaluate these programs in three aspects, including correctness, code quality, and maintainability.

RQ3: What is the better design choice to estimate examples? In section [3.1](#S3.SS1 "3.1. Estimate and Label Examples ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation"), we leverage the prediction probability of ground-truth programs to estimate candidate examples. In this RQ, we explore other design choices to measure examples by LLMs themselves and compare them to our design.

### 4.2. Datasets

We conduct experiments on three code generation datasets, including MBJP *(Athiwaratkun et al., [2022](#bib.bib9 ""))* in Java, MBPP *(Austin et al., [2021](#bib.bib10 ""))* in Python, and MBCPP *(Athiwaratkun et al., [2022](#bib.bib9 ""))* in C++. The statistics of these datasets are given in Table [1](#S4.T1 "Table 1 ‣ 4.2. Datasets ‣ 4. Study design ‣ Large Language Model-Aware In-Context Learning for Code Generation").

*Table 1. Statistics of three datasets on the different split sets.*

|  | MBJP | MBPP | MBCPP |
| --- | --- | --- | --- |
| Language | Java | Python | C++ |
| #Train | 383 | 384 | 413 |
| #Dev | 90 | 90 | – |
| #Test | 493 | 500 | 435 |
| Avg. tests per example | 3 | 3 | 3 |
| Avg. tokens in requirement | 16.71 | 16.50 | 17.38 |
| Avg. tokens in code | 247.79 | 92.68 | 113.94 |

MBPP *(Austin et al., [2021](#bib.bib10 ""))* contains 974 Python programming problems constructed by crowd-sourcing. Each example consists of a brief description, a single self-contained function solving the problem specified, and three test cases to evaluate the correctness of the generated programs. The problems range from simple numeric manipulations or tasks that require the basic usage of standard library functions to tasks that demand nontrivial external knowledge.

MBJP *(Athiwaratkun et al., [2022](#bib.bib9 ""))* and MBCPP *(Athiwaratkun et al., [2022](#bib.bib9 ""))* are drived from MBPP *(Austin et al., [2021](#bib.bib10 ""))*. MBJP and MBCPP contain 966 and 848 crowd-sourced programming problems in Java and C++, respectively. Each problem consists of a description, an individual function, and three test cases. These problems cover programming fundamentals, standard library functionality, etc.

We follow previous studies *(Athiwaratkun et al., [2022](#bib.bib9 ""); Austin et al., [2021](#bib.bib10 ""))* to split three datasets into the training set, the valid set, and the test set, respectively. We measure the performance of different ICL approaches on the test set.

### 4.3. Evaluation Metrics

Following prior code generation studies *(Chen et al., [2021](#bib.bib15 ""); Athiwaratkun et al., [2022](#bib.bib9 ""))*, we leverage Pass@k to evaluate our approaches. Pass@k evaluates the functional correctness of the generated programs by executing test cases. Precisely, given a test requirement, we generate $k$ programs using the sampling strategy. If any of the generated $k$ programs passes all test cases, we think the requirement is solved. Finally, the percentage of solved requirements in all test requirements is treated as Pass@k. In this paper, we set $k$ to 1, 3, and 5.

We notice that previous studies *(Guo et al., [2020](#bib.bib22 ""); Feng et al., [2020](#bib.bib21 ""))* also use some match-based metrics such as BLEU *(Papineni et al., [2002](#bib.bib38 ""))* and CodeBLEU *(Ren et al., [2020](#bib.bib42 ""))*. These metrics focus on the textual similarity or syntactic similarity between reference programs and generated programs. However, existing work *(Inala et al., [2022](#bib.bib25 ""))* has shown that match-based metrics can not effectively measure the functionality of programs. In this paper, we follow recent studies *(Inala et al., [2022](#bib.bib25 ""); Radford et al., [2018](#bib.bib39 ""); Li et al., [2023a](#bib.bib30 ""))* use execution-based metrics (*e.g*. Pass@k).

### 4.4. Baselines

There are a few studies to investigate in-context example selection for code generation such as zero-shot learning *(Brown et al., [2020](#bib.bib12 ""))*, random selection *(Chen et al., [2021](#bib.bib15 ""))*, and AceCoder *(Li et al., [2023c](#bib.bib32 ""))*, where AceCoder *(Li et al., [2023c](#bib.bib32 ""))* is the state-of-the-art (SOTA) baseline.

Zero-Shot Learning *(Brown et al., [2020](#bib.bib12 ""))* directly inputs a requirement into LLMs without any examples as the prompt. LLMs generate programs for the given requirement.

Random *(Chen et al., [2021](#bib.bib15 ""))* randomly select a few examples from the training set and construct a prompt. Then, LLMs predict the source code based on the prompt.

AceCoder *(Li et al., [2023c](#bib.bib32 ""))* uses BM25 *(Robertson et al., [2009](#bib.bib43 ""))* to calculate the textual similarities between a test requirement and the requirements of candidates, and retrieve a set of examples with high similarities. In inference, it first generates test cases and then predicts programs. For a fair comparison, we use the same pipeline to LAIL and all baselines.

To extensively evaluate the effectiveness of our LAIL, we also transfer some advanced ICL approaches in natural language processing to the source code.

TOP-k-SBERT *(Luo et al., [2023](#bib.bib36 ""))* leverages Sentence-BERT *(Reimers and Gurevych, [2019](#bib.bib41 ""))*, a representative sentence encode, to encode all requirements in the training set. Given a test requirement, we first encode it and computer semantic similarities between it and the training requirements. Next, we select the top-k similar examples.

TOP-k-GraphCodeBERT *(Guo et al., [2020](#bib.bib22 ""))* is a variant of TOP-k-SBERT. It applies GraphCodeBERT to encode requirements and retrieve a few examples from the training set based on semantic similarity.

TOP-k-VOTE *(Su et al., [2022](#bib.bib44 ""))* is a graph-based method *(Su et al., [2022](#bib.bib44 ""))*, to votes examples. It first encodes each example by GraphCodeBERT and each example is as a vertice in the graph. Each vertice connects with its k nearest vertices based on their semantic similarities. Finally, it treats the k nearest examples as a prompt.

Uncertainty-Target *(Coleman et al., [2019](#bib.bib18 ""))* assumes examples with higher uncertainty have a greater impact on LLMs. It defines uncertainty as the perplexity when LLMs generate ground truths. The approach computes the uncertainty of each candidate and selects k items with high perplexity as a prompt.

### 4.5. Base Large Language Models

This paper focuses on code generation with LLMs. Thus, we select three recently proposed LLMs for code generation as the base models. The details of the base models are shown as follows.

ChatGPT *(Cha, [able](#bib.bib2 ""))* is the state-of-the-art language model for code generation. It is trained on a large amount of natural language text and programming data. Then, ChatGPT is continually trained with reinforcement learning and learns to align with human instructions. We leverage OpenAI’s APIs to access ChatGPT (*i.e*., gpt-3.5-turbo).

GPT-3.5 *(GPT, [ablea](#bib.bib3 ""))* is a powerful large language model and is trained on large of unlabeled corpus. In this paper, we use OpenAI’s APIs to access the latest version with 175 billion parameters (*i.e*., text-davinci-003).

CodeGen *(Zhong et al., [2022](#bib.bib48 ""))* is a family of language models for code generation. CodeGen is training with a 635GB code corpus and 1159GB English text data. In this paper, we leverage the largest version with 16 billion parameters (*i.e*. CodeGen-Multi-16B).

*Table 2. Evaluation results of our LAIL and baselines on CodeGen at the three code generation datasets. Numbers in bold indicate that the improvement is significant and the percentages in red color mean the relative improvements compared to the SOTA baseline.*

|  | MBJP | | | MBPP | | | MBCPP | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 |
| Zero-Shot Learning (Brown et al., [2020](#bib.bib12 "")) | 14.27 | 23.69 | 27.83 | 8.80 | 20.60 | 25.60 | 15.39 | 25.97 | 30.94 |
| Random (Chen et al., [2021](#bib.bib15 "")) | 15.74 | 24.25 | 28.14 | 15.20 | 25.40 | 28.80 | 15.70 | 28.72 | 32.25 |
| AceCoder (Li et al., [2023c](#bib.bib32 "")) | 17.65 | 27.18 | 30.63 | 16.80 | 26.40 | 29.40 | 17.47 | 30.11 | 33.78 |
| TOP-k-SBERT (Luo et al., [2023](#bib.bib36 "")) | 16.63 | 26.77 | 29.21 | 15.40 | 25.00 | 29.00 | 18.09 | 29.43 | 33.33 |
| TOP-k-GraphCodeBERT (Guo et al., [2020](#bib.bib22 "")) | 17.44 | 24.34 | 28.40 | 17.40 | 25.80 | 28.40 | 18.16 | 30.48 | 35.47 |
| TOP-k-VOTE (Su et al., [2022](#bib.bib44 "")) | 15.42 | 21.70 | 23.73 | 17.40 | 26.00 | 29.40 | 17.01 | 30.03 | 35.63 |
| Uncertainty-Target (Coleman et al., [2019](#bib.bib18 "")) | 17.09 | 23.06 | 27.93 | 14.60 | 24.20 | 28.80 | 17.70 | 30.12 | 34.62 |
| LAIL | 21.30 | 28.49 | 32.05 | 18.60 | 27.80 | 30.60 | 19.08 | 31.36 | 37.94 |
| Relative Improvement | 11.58% | 4.82% | 4.64% | 6.89% | 6.92% | 4.08% | 5.07% | 2.89% | 6.96% |

### 4.6. Implementation Details

Estimate and Label Examples. We use greedy decoding to generate programs and collect the predicted probabilities of ground-truth programs. Following the previous generation works *(Li et al., [2023c](#bib.bib32 ""))*, we set the max generated length to 500 tokens.
The number of examples in the set $S_{i}$ is 50 for efficiency. Note that the size of $S_{i}$ larger the better performance of LLMs and we leave other sizes in our future work.
We set $z$ and $v$ as 5 respectively and analyze their effects on performance in Section [6.3](#S6.SS3 "6.3. Effects of Contrastive Learning Parameters ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation"). Thus, the positive set $S_{i}^{p}$ contains 5 examples with higher metric scores and the negative set $S_{i}^{n}$ has 5 items with lower scores.

Training Neural Retriever. In this procedure, we set the number of the negative example set $\mathbb{N}_{i}$ as 64.
In other words, $h$ is set to 63. For each epoch, we randomly select 63 examples from the set $\mathcal{H}$. We also attempt to set the size of $\mathbb{N}_{i}$ to 32 and 128 in Section [6.3](#S6.SS3 "6.3. Effects of Contrastive Learning Parameters ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation").
The learning rate is 5e-5 and the batch size is 32. We train the retriever for about 1 hour on 4 NVIDIA A100.

Inference. We treat a large language model as a black-box generator and sample programs from it.
The input of LLMs only contains a prompt and a test requirement without any natural language instructions. During the sampling, we use nuclear sampling *(Holtzman et al., [2019](#bib.bib23 ""))* to decode, where the temperature is 0.8 and the top-p is 0.95. The maximum generated length is 500 tokens. For each test requirement, we generate 5 programs. For a fair comparison, we set the same parameters to generate programs for all baselines and our LAIL.

5. Results and Analysis
------------------------

*Table 3. The performance of our LAIL and the existing approaches on GPT-3.5 at the three code generation datasets. The numbers in red color represent the relative improvements compared to the SOTA baseline.*

|  | MBJP | | | MBPP | | | MBCPP | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 |
| Zero-Shot Learning (Brown et al., [2020](#bib.bib12 "")) | 44.83 | 53.05 | 59.72 | 20.00 | 27.00 | 29.60 | 21.85 | 37.94 | 49.90 |
| Random (Chen et al., [2021](#bib.bib15 "")) | 47.87 | 59.83 | 63.69 | 43.00 | 56.40 | 60.80 | 50.02 | 61.43 | 65.28 |
| AceCoder (Li et al., [2023c](#bib.bib32 "")) | 50.91 | 61.25 | 65.15 | 47.40 | 61.00 | 64.20 | 53.25 | 63.22 | 66.21 |
| TOP-k-SBERT (Luo et al., [2023](#bib.bib36 "")) | 50.30 | 60.46 | 64.70 | 48.00 | 61.40 | 64.00 | 52.87 | 63.21 | 67.13 |
| TOP-k-GraphCodeBERT (Guo et al., [2020](#bib.bib22 "")) | 50.30 | 60.85 | 64.50 | 49.20 | 58.20 | 64.20 | 52.64 | 63.19 | 67.28 |
| TOP-k-VOTE (Su et al., [2022](#bib.bib44 "")) | 50.52 | 60.45 | 63.49 | 47.80 | 59.20 | 63.40 | 51.03 | 62.98 | 65.51 |
| Uncertainty-Target (Coleman et al., [2019](#bib.bib18 "")) | 47.67 | 59.23 | 63.69 | 37.80 | 51.40 | 57.80 | 42.87 | 52.60 | 57.18 |
| LAIL | 53.14 | 62.87 | 66.72 | 50.60 | 62.40 | 65.20 | 54.71 | 65.98 | 69.67 |
| Relative Improvement | 4.38% | 2.64% | 2.41% | 2.85% | 1.63% | 1.53% | 2.74% | 4.37% | 3.55% |

RQ1: How does LAIL perform compared to the state-of-the-art baselines?

In RQ1, we apply our LAIL and baselines to two LLMs and measure the correctness of generated programs.

Setup. We compare our LAIL and baselines (Section[4.4](#S4.SS4 "4.4. Baselines ‣ 4. Study design ‣ Large Language Model-Aware In-Context Learning for Code Generation")) on three code generation datasets (Section[4.2](#S4.SS2 "4.2. Datasets ‣ 4. Study design ‣ Large Language Model-Aware In-Context Learning for Code Generation")). The evaluation metric is Pass@k as described in Section[4.3](#S4.SS3 "4.3. Evaluation Metrics ‣ 4. Study design ‣ Large Language Model-Aware In-Context Learning for Code Generation"). For the metric, higher scores indicate better performance of approaches.

Results. Table [2](#S4.T2 "Table 2 ‣ 4.5. Base Large Language Models ‣ 4. Study design ‣ Large Language Model-Aware In-Context Learning for Code Generation") and Table [3](#S5.T3 "Table 3 ‣ 5. Results and Analysis ‣ Large Language Model-Aware In-Context Learning for Code Generation") report the
Pass@k (k $\in$ [1, 3, 5]) of different approaches on CodeGen and GPT-3.5, respectively. Numbers in bold mean that the improvement is significant compared with baselines, and the percentages in red color represent the relative improvements compared to the state-of-the-art (SOTA) baseline.

Analysis. (1) LAIL achieves the best performance among all approaches with significant improvements. In all datasets, LAIL generates more correct programs than baselines. Compared to the SOTA baseline, LAIL outperforms it by 11.58%, 6.89%, and 5.07% on CodeGen, and acquires 4.38%, 2.85%, and 2.74% improvement on GPT-3.5 in Pass@1, respectively. Note that Pass@1 is a very strict metric and is different to be improved. The significant improvements prove the superiority of our LAIL in code generation.
(2) Selecting proper examples is critical to the performance of ICL. Compared to random selection, LAIL acquires 8.31%, 30.60%, and 32.86% absolute improvements in Pass@1 on GPT-3.5. AceCoder and other heuristic approaches further improve code generation performance by selecting textually or semantically similar examples.
LAIL exploits LLMs themselves to estimate and trains a neural retriever to align with the preference of LLMs, then achieves the best results among all approaches. That demonstrates the importance of examples in ICL and verifies the reasonability of using LLMs themselves to label examples.
(3) LAIL is effective in LLMs with different sizes and different programming languages. As above described, our approach on CodeGen and GPT-3.5 both achieve the best performance among all approaches.
Table [2](#S4.T2 "Table 2 ‣ 4.5. Base Large Language Models ‣ 4. Study design ‣ Large Language Model-Aware In-Context Learning for Code Generation") and Table [3](#S5.T3 "Table 3 ‣ 5. Results and Analysis ‣ Large Language Model-Aware In-Context Learning for Code Generation") also show that LAIL can generate more correct programs on all datasets including MBPP (Python), MBJP (Java), and MBCPP (C++).
This reveals LAIL is generalized and can be applied to different LLMs and languages.


*Table 4. The results of human evaluation. The numbers in red represent LAIL’ relative improvements compared to the SOTA baseline. “CR”, “QL”, and “MT” mean correctness, code quality, and maintainability, respectively.*

| Approach | CR | QL | MT |
| --- | --- | --- | --- |
| Zero-Shot Learning (Brown et al., [2020](#bib.bib12 "")) | 0.472 | 1.079 | 1.183 |
| Random (Chen et al., [2021](#bib.bib15 "")) | 0.925 | 1.416 | 1.520 |
| AceCoder (Li et al., [2023c](#bib.bib32 "")) | 1.523 | 1.652 | 1.647 |
| TOP-k-SBERT (Luo et al., [2023](#bib.bib36 "")) | 1.569 | 1.640 | 1.665 |
| TOP-k-GraphCodeBERT (Guo et al., [2020](#bib.bib22 "")) | 1.582 | 1.738 | 1.609 |
| TOP-k-VOTE (Su et al., [2022](#bib.bib44 "")) | 1.325 | 1.624 | 1.631 |
| Uncertainty-Target (Coleman et al., [2019](#bib.bib18 "")) | 0.736 | 1.297 | 1.324 |
| LAIL | 1.764 | 1.839 | 1.782 |
| Relative Improvement | 11.504% | 5.811% | 7.027% |

RQ2: Do developers prefer programs generated by LAIL?

The goal of a code generation approach is to assist developers in writing programs. Thus, a good program not only satisfies the requirement but also is easy to read and maintain. In this question, we manually verify generated programs in three aspects.

Setup. Following previous work *(Li et al., [2023a](#bib.bib30 ""))*, we manually measure programs generated by different approaches in three aspects (*e.g*. correctness, code quality, maintainability). Correctness measures whether a program satisfies the given requirement, code quality verifies whether a program does not contain bas code smell, and maintainability measures whether the implementation is standardized and easy to read. For each aspect, the score is an integer and ranges from 0 to 2. The higher score, the better code is.

Specifically, we randomly select 50 test samples from MBPP, MBJP, and MBCPP, respectively 222The confidence level is 85% and the margin of error is 15% with 50 selected examples.. Then, we use LAIL and baselines to generate programs for these examples. Finally, we obtain 400 (50 $\times$ 8) programs for human evaluation. The 400 programs are randomly divided into 5 groups. We hire 10 developers to verify these programs. These developers are computer science students or industrial practitioners and write programs at least for 3 years. Each group is measured by two developers and the final score is the average of two developers’ scores.

Results. The results of the human evaluation are shown in Table [4](#S5.T4 "Table 4 ‣ 5. Results and Analysis ‣ Large Language Model-Aware In-Context Learning for Code Generation"). The percentages in parentheses are the relative improvements compared to the SOTA baseline.

Analyses. (1) Developers prefer programs generated by LAIL over all baselines. LAIL substantially outperforms all baselines in three aspects, which demonstrates the effectiveness of our approach. (2) In addition to satisfying requirements, programs provided by our approach are easier to read and have less bad smell. Particularly, our LAIL surpasses the SOTA approach by 5.811% in code quality and 7.027% in maintainability. (3) Some heuristic methods that select the same examples as prompts for all test requirements are not optimal for ICL. As shown in Figure [4](#S5.T4 "Table 4 ‣ 5. Results and Analysis ‣ Large Language Model-Aware In-Context Learning for Code Generation"), Uncertainty-Target and Vote-K perform comparably to the random approach, which indicates that applying the same prompt to all test data is not a good choice, and different test requirements should use their appropriate prompts.


RQ3: What is the better design choice to estimate examples?

In this paper, we apply the predicted probability of ground truths to estimate examples. We further investigate two plausible choices to measure candidates by LLMs themselves.

Setup. We design two plausible approaches including the Match-BLEU and the Match-CodeBLEU formats.
The Match-BLEU format uses the BLEU score of the generated program to measure examples. The Match-CodeBLEU approach applies the CodeBLEU score of the predicted program to label examples. In this RQ, we select a representative LLM (*i.e*. GPT-3.5) as the base model and evaluate their performance on three datasets.

Results. The results of different feedback scores are represented in Table [5](#S5.T5 "Table 5 ‣ 5. Results and Analysis ‣ Large Language Model-Aware In-Context Learning for Code Generation"). For comparison, we also present the results of the existing SOTA baseline in Table [5](#S5.T5 "Table 5 ‣ 5. Results and Analysis ‣ Large Language Model-Aware In-Context Learning for Code Generation").

Analyses. (1) Our probability-based approach is better than Match-BLEU and Match-CodeBLEU. In all datasets, our LAIL outperforms them by 4.05%, 5.86%, and 3.77% on three datasets at Pass@1, respectively. We argue that the predicted probability of ground truth accurately reflects the certainty of LLMs for correct programs. The BLEU-based and CodeBLEU-based methods only reflect the literal accuracy of LLMs’ prediction, but can not provide how certain LLMs are when predicting programs.
(2) Match-CodeBLEU approach is more effective than Match-BLEU method. Match-CodeBLEU outperforms Match-BLEU on all datasets. The reason might be that CodeBLEU measures the n-gram match, the semantic match, and the syntactic match between the hypothesis programs and the reference code tokens, which is better to evaluate the generated programs than BLEU.


*Table 5. The comparison of different formats to estimate examples on three datasets. The values in parentheses mean significant improvements. “Probability-Based ” represents our LAIL.*

|  | MBJP | | | MBPP | | | MBCPP | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 |
| Random (Chen et al., [2021](#bib.bib15 "")) | 47.87 | 59.83 | 63.69 | 43.00 | 56.40 | 60.80 | 50.04 | 61.45 | 65.28 |
| Match-BLEU | 50.19 | 59.87 | 63.92 | 46.20 | 60.20 | 62.40 | 50.26 | 61.47 | 65.73 |
| Match-CodeBLEU | 51.07 | 60.04 | 64.58 | 47.80 | 59.20 | 63.80 | 52.72 | 62.03 | 66.80 |
| Probability-Based (LAIL) | 53.14 | 62.87 | 66.72 | 50.60 | 62.40 | 65.20 | 54.71 | 65.98 | 69.67 |
| Relative Improvement | 4.05% | 4.71% | 3.31% | 5.86% | 3.65% | 2.19% | 3.77% | 6.37% | 4.30% |

6. Discussion
--------------

### 6.1. Transferability

We explore whether our retriever based on one LLM’s feedback and specific dataset can be transferred to other LLMs or code generation datasets without further tuning. This is a significant research question since the retriever for each LLM and dataset needs to be trained in real applications.

#### 6.1.1. Transfer across LLMs

We consider transferring the retriever based on one LLM’s feedback to another LLM. Specifically, we use a source LLM (*e.g*. CodeGen or GPT-3.5) to estimate examples for training a retriever and then apply the retriever to another target LLM (*e.g*. ChaTGPT) in generating programs. Table [6](#S6.T6 "Table 6 ‣ 6.1.2. Transfer across datasets ‣ 6.1. Transferability ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation") shows the performance of ChaTGPT in three datasets. We surprisingly find our retriever based on CodeGen and GPT-3.5 can bring obvious improvements to ChaTGPT. In particular, in terms of Pass@1, ChaTGPT achieves 2.56% improvements from CodeGen’s feedback and 4.31% enhancements from GPT-3.5’s feedback compared to the SOTA baseline. The phenomenons demonstrate that our approach has satisfying transfer ability across different LLMs. Note that ChaTGPT can not provide the prediction probability of ground truths in practice, thus LAIL is a quite meaningful approach, especially for LLMs whose parameters are unavailable.
Besides, the performance of ChaTGPT from GPT-3.5’s feedback is higher than the counterpart from CodeGen’s feedback. The reason might be that GPT-3.5 and ChatGPT have comparable abilities in code generation, thus their preference is similar and GPT-3.5 can provide more proper examples as prompts.

To verify the transfer ability among LLMs with different sizes, we further evaluate the performance of CodeGen based on GPT-3.5’s feedback and the results of GPT-3.5 from CodeGen’s feedback, where the parameter size of CodeGen is much smaller than the counterpart of GPT-3.5. As shown in Table [6](#S6.T6 "Table 6 ‣ 6.1.2. Transfer across datasets ‣ 6.1. Transferability ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation"), compared to the random approach, the retriever learned from CodeGen achieves 8.25%, 12.09%, and 4.32% relative improvements to GPT-3.5 on Pass@1, meanwhile our retriever under GPT-3.5’s feedback brings 22.30%, 14.47%, and 16.94% improvements to CodeGen, respectively. This indicates our approach has satisfying transfer ability among LLMs and can bring improvements to target LLMs.

#### 6.1.2. Transfer across datasets

Considering that the compositional features of natural language are general, the retriever trained on one dataset may apply to other datasets and exploit similar knowledge in different datasets. In this section, we further investigate whether a retriever trained on one dataset can transfer to others. We transfer the retriever among three datasets (*e.g*., MBJP, MBPP, and MBCPP) and Figure [3](#S6.F3 "Figure 3 ‣ 6.1.2. Transfer across datasets ‣ 6.1. Transferability ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation") demonstrates the transferring results on Pass@1. We find that most retrievers can successfully transfer to other datasets and bring improvements compared to their SOTA baselines. Concretely, the retriever trained on MBJP (MBCPP) achieves 1.28% (1.07%) absolute improvements when it migrates to MBCPP (MBJP). Meanwhile, the MBCPP-trained retriever hardly transfers to MBPP and the MBPP-based retriever suffers the generation performance on MBPP. The reason might be that Java and C++ are object-oriented programming languages, and their syntax and code morphology are similar.
Exploring a retriever that is suitable for many datasets is a challenging but meaningful research question, and we leave this topic as our future work.

<img src='x3.png' alt='Refer to caption' title='' width='368' height='253' />

*Figure 3. Results of transferring the retriever trained on one dataset (row) to others (column) on GPT-3.5 in MBPP dataset.*

*Table 6. Results of transferring a retriever learned on one LLM to others on three datasets. $\clubsuit$ means the source LLM.*

| $\clubsuit$ ChaTGPT | MBJP | | | MBPP | | | MBCPP | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 | Pass@1 | Pass@3 | Pass@5 |
| Zero-Shot Learning (Brown et al., [2020](#bib.bib12 "")) | 16.63 | 34.48 | 44.21 | 26.60 | 32.00 | 34.60 | 30.34 | 57.01 | 63.68 |
| Random (Chen et al., [2021](#bib.bib15 "")) | 53.34 | 62.27 | 65.72 | 50.80 | 60.60 | 63.40 | 40.15 | 60.06 | 66.28 |
| AceCoder (Li et al., [2023c](#bib.bib32 "")) | 54.46 | 64.01 | 66.75 | 54.40 | 62.40 | 65.20 | 42.53 | 62.29 | 68.32 |
| TOP-k-SBERT (Luo et al., [2023](#bib.bib36 "")) | 54.26 | 63.89 | 66.53 | 54.80 | 63.00 | 65.40 | 43.37 | 61.45 | 67.11 |
| TOP-k-GraphCodeBERT (Guo et al., [2020](#bib.bib22 "")) | 53.95 | 62.67 | 66.32 | 54.20 | 62.60 | 65.20 | 44.08 | 61.61 | 68.27 |
| TOP-k-VOTE (Su et al., [2022](#bib.bib44 "")) | 51.93 | 62.88 | 65.72 | 42.00 | 56.00 | 61.00 | 42.74 | 62.68 | 67.73 |
| Uncertainty-Target (Coleman et al., [2019](#bib.bib18 "")) | 49.69 | 60.45 | 54.50 | 36.40 | 51.80 | 56.80 | 42.46 | 61.84 | 66.67 |
|  | CodeGen | | | | | | | | |
| LAIL | 54.79 | 64.70 | 67.43 | 55.60 | 63.80 | 66.20 | 45.21 | 63.45 | 68.93 |
| Relative Improvement | 0.61% | 1.08% | 1.02% | 1.44% | 1.27% | 1.22% | 2.56% | 1.23% | 0.89% |
|  | GPT-3.5 | | | | | | | | |
| LAIL | 55.97 | 64.97 | 68.27 | 56.20 | 64.60 | 66.80 | 45.98 | 63.84 | 70.35 |
| Relative Improvement | 2.77% | 1.50% | 2.28% | 2.56% | 2.54% | 2.14% | 4.31% | 1.85% | 2.97% |
| $\clubsuit$ GPT-3.5 | CodeGen | | | | | | | | |
| Random (Chen et al., [2021](#bib.bib15 "")) | 47.87 | 59.83 | 63.69 | 43.00 | 56.40 | 60.80 | 50.02 | 61.43 | 65.28 |
| LAIL | 51.82 | 61.75 | 64.89 | 48.20 | 59.00 | 64.20 | 52.18 | 64.54 | 67.90 |
| Relative Improvement | 8.25% | 3.21% | 1.88% | 12.09% | 4.61% | 5.59% | 4.32% | 5.06% | 4.01% |
| $\clubsuit$ CodeGen | GPT-3.5 | | | | | | | | |
| Random (Chen et al., [2021](#bib.bib15 "")) | 15.74 | 24.25 | 28.14 | 15.20 | 25.40 | 28.80 | 15.70 | 28.72 | 32.25 |
| LAIL | 19.25 | 27.53 | 30.98 | 17.40 | 26.00 | 30.00 | 18.36 | 30.41 | 35.54 |
| Relative Improvement | 22.30% | 13.53% | 10.09% | 14.47% | 2.36% | 4.17% | 16.94% | 5.88% | 10.20% |

### 6.2. Impacts of In-Context Example Numbers

Most of the LLMs are trained with a limited input length, which restricts the number of examples in the prompt. Previous studies *(Luo et al., [2023](#bib.bib36 ""); Dong et al., [2022](#bib.bib20 ""))* also find that LLMs are affected by the number of in-context examples. Here we explore the impacts of the number of examples in the prompt on baselines and our approach. Figure [4](#S6.F4 "Figure 4 ‣ 6.2. Impacts of In-Context Example Numbers ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation") reports how the performance of LAIL and GraphCodeBET change with respect to different in-context example numbers inMBPP dataset. Note that GraphCodeBET is the best baseline in this dataset, thus we choose it to compare with our approach.
We observe that the performances of GraphCodeBET and our approach monotonically increase in rough with the increase of in-context example numbers. In addition, our LAIL always outperforms GraphCodeBERT in all cases, which further proves the superiority of our approach even with different in-context example numbers.

<img src='x4.png' alt='Refer to caption' title='' width='368' height='276' />

*Figure 4. The performance of the different number of in-context examples on GPT-3.5 in MBPP datasets.*

### 6.3. Effects of Contrastive Learning Parameters

The number of negative examples $\mathbb{N}_{i}$ (named $\mathbb{N}_{i}$ for convenience in Figure [7](#S6.T7 "Table 7 ‣ 6.3. Effects of Contrastive Learning Parameters ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation")) is an important element in contrastive learning for training a neural retriever. We investigate how the element affects the performance of our approach. As shown in Figure [7](#S6.T7 "Table 7 ‣ 6.3. Effects of Contrastive Learning Parameters ‣ 6. Discussion ‣ Large Language Model-Aware In-Context Learning for Code Generation"), with the increasing of negative example numbers, LAIL achieves consistent improvements and can generate more correct programs.
Besides, as described in [3.2](#S3.SS2 "3.2. Training Neural Retriever ‣ 3. Method: LAIL ‣ Large Language Model-Aware In-Context Learning for Code Generation"), we randomly select an example $e_{i}^{n}$ from $S_{i}^{n}$ into $\mathbb{N}_{i}$. We also explore the effect of the number of selected examples from $S_{i}^{n}$ (dubbed $\tau_{ne}$). We can find that the more hard negative examples, the worse our approach performs. We argue that the examples of $S_{i}^{n}$ are with high BM25 scores among all candidate examples and thus $S_{i}^{n}$ may contain some false negative examples. Selecting more examples from $S_{i}^{n}$ will affect a retriever learning for choosing suitable in-context examples.
Between the two factors, the number of selected examples from $S_{i}^{n}$ has a greater influence on the performance of our approach.

*Table 7. Effects of parameters.*

|  | Pass@1 | Pass@3 | Pass@5 |
| --- | --- | --- | --- |
| Negative examples ($\mathbb{N}_{i}$) |  |  |  |
| 32 | 50.20 | 61.80 | 65.00 |
| 64 | 50.60 | 62.40 | 65.20 |
| 128 | 50.80 | 62.40 | 65.40 |
| $\tau_{ne}$ |  |  |  |
| 1 | 50.60 | 62.40 | 65.20 |
| 5 | 50.20 | 62.20 | 65.00 |
| 10 | 49.40 | 61.80 | 64.40 |

### 6.4. Threats to Validity

There are two main threats to the validity of our work.

The generalizability of our experimental results. For the datasets, we follow previous studies *(Chen et al., [2021](#bib.bib15 ""); Austin et al., [2021](#bib.bib10 ""); Athiwaratkun et al., [2022](#bib.bib9 ""))* and leverage three representative code generation datasets. The three datasets cover different programming languages (*e.g*., Java, Python, and C++) and come from real-world software communities. To verify the superiority of LAIL, we consider seven existing ICL approaches in both the code generation task and many maintain natural language tasks. In addition, to effectively evaluate our approach, we select a series of advanced pre-trained large language models (CodeGen *(Zhong et al., [2022](#bib.bib48 ""))*, GPT-3.5, and ChatGPT) as base models in the past three years. We apply our approach and baselines to base models and evaluate their performance in code generation. For the metric, following existing works *(Chen et al., [2022](#bib.bib14 ""); Zhong et al., [2022](#bib.bib48 ""))*, we select a widely used Pass@k metric to evaluate all approaches. It is an execution-based metric that utilizes test cases to check the correctness of generated programs. To ensure fairness, we execute each method three times and report the average experimental results.

The implementation of models and prompts. It is widely known that deep neural models are sensitive to the implementation details. In this paper, we need to execute all baselines and our approach on three base models. For baselines, we apply the source code and parameters published by their original papers *(Guo et al., [2020](#bib.bib22 ""); Zhong et al., [2022](#bib.bib48 ""); Reimers and Gurevych, [2019](#bib.bib41 ""))*. For LLMs (*e.g*., GPT-3.5 and ChatGPT), the hyper-parameters (*e.g*., temperature) of sampling will impact their outputs. In experiments, we follow previous studies *(Zhong et al., [2022](#bib.bib48 ""); Li et al., [2023a](#bib.bib30 ""))* to set hyper-parameters for all approaches.
In addition, the performance of LLMs heavily depends on the prompts, including the instruction and the number of examples. To alleviate this threat, we leverage the same number of examples for all approaches and directly construct prompts without any natural language instructions.
Besides, it’s worth noting that the candidate pool for example selection also influences ICL performance. A large-scale study on 13.2 million real code files showed the proportion of reused code is up to 80% *(Mockus, [2007](#bib.bib37 ""))*. Thus, we use the training set of each dataset as the candidate set for all approaches in this paper and believe that the training set can provide supporting examples for test requirements. We do not tune the prompt and hyper-parameters experimentally and set them empirically. Thus, there might be room to tune hyper-parameter settings of our approach for more improvements.

7. Related Work
----------------

### 7.1. Pre-trained Language Models

Code generation aims to automatically generate programs given natural language requirements. Pre-trained language models have achieved state-of-the-art results in code generation. They are pre-trained on large-scale unlabeled code corpus and then transferred to code generation. Existing pre-trained models can be divided into encoder-decoder models and decoder-only models.

Encoder-decoder models consist of an encoder and a decoder. An encoder takes a requirement as the input, and a decoder outputs a program. Many popular encoder-decoder architectures in natural language processing have been applied to source code, which often use denoising-based pre-training objectives to pre-train, such as masked sequence prediction, identifier tagging, and masked identifier prediction. For example, CodeT5 *(Wang et al., [2021](#bib.bib45 ""))* is the variant of T5 model *(Raffel et al., [2020](#bib.bib40 ""))* to support programs. *(Ahmad et al., [2021](#bib.bib6 ""))* is adapted from the BART model *(Lewis et al., [2019](#bib.bib29 ""))*, which is pre-trained on a number of programs. Besides, some studies *(Chakraborty et al., [2022](#bib.bib13 ""))* further consider the naturalized feature of code and train models to generate natural programs based on noised programs.

Decoder-only models contain decoder networks, which are pre-trained with the next token prediction objective. Inspired by the success of GPT series *(Radford et al., [2018](#bib.bib39 ""))* in natural language processing, researchers attempt to adapt similar models to source code.
CodeGPT *(Lu et al., [2021](#bib.bib35 ""))* is pre-trained on CodeSearchNet *(Husain et al., [2019](#bib.bib24 ""))* corpus with the same setting and structure as GPT-2. CodeX is continually fine-tuned on GPT-3 *(Brown et al., [2020](#bib.bib12 ""))* on code corpus from GitHub. CodeX is proficient in over a dozen languages (*e.g*., JavaScript and Python) and supports a commercial application (*e.g*., Copilot).
Since its parameters are not available, many researchers try to replicate them and bring CodeParrot *(Cod, [able](#bib.bib4 ""))*, GPT-CC *(GPT, [ableb](#bib.bib5 ""))*, PyCodeGPT *(Zan et al., [2022](#bib.bib46 ""))* and CodeGen *(Zhong et al., [2022](#bib.bib48 ""))*. CodeGeeX *(Zheng et al., [2023](#bib.bib47 ""))* is pre-trained on a large code corpus and can generate more than 20 programming languages. Lately, ChatGPT is proposed by OpenAI and achieves impressive performance in code generation. In this paper, we select three representative networks including CodeGen *(Zhong et al., [2022](#bib.bib48 ""))*, GPT-3.5 *(GPT, [ablea](#bib.bib3 ""))*, and ChatGPT *(Cha, [able](#bib.bib2 ""))* as our base models.

### 7.2. In-Context Learning

In-context learning (ICL) is an emerging approach to using large language models (LLMs). By providing limited examples as a prompt, ICL empowers LLMs to learn a specific downstream task.
ICL *(Brown et al., [2020](#bib.bib12 ""))* is first proposed in natural language processing and achieves impressive performance in many tasks *(Levy et al., [2022](#bib.bib28 ""); Dong et al., [2022](#bib.bib20 ""))* such as text generation and sentiment classification.

Inspired by the success of ICL in natural language processing, researchers attempt to adapt ICL to source code *(Bareiß et al., [2022](#bib.bib11 ""); Chen et al., [2021](#bib.bib15 ""), [2023](#bib.bib17 ""))*. They design task-specific instructions with a set of examples to prompt LLMs and improve the performance on many tasks (*e.g*., code generation *(Chen et al., [2023](#bib.bib17 ""))* and code repair *(Jiang et al., [2023b](#bib.bib26 ""))*).
The popularity of ICL also introduces instability in performance: given different sets of examples as prompts, LLMs’ performance usually varies from random to near state-of-the-art *(Liu et al., [2021](#bib.bib34 ""))*. Some studies *(Chen et al., [2023](#bib.bib17 ""); Li et al., [2023c](#bib.bib32 ""))* make efforts to mitigate this issue. Researchers *(Chen et al., [2023](#bib.bib17 ""))* randomly select limited examples for ICL and verify the results in code generation. AceCoder *(Li et al., [2023c](#bib.bib32 ""))* uses BM25 to select n-gram matching examples and further generates more correct programs. However, these approaches are based on lexical features, which require developers to design heuristics and lead to sub-optimal performance.
Compared to the existing methods, LAIL is a learning-based approach for selecting in-context examples. Instead of considering textual similarity, it applies LLMs themselves to measure candidate examples and uses contrastive learning to learn LLMs’ preferences in code generation.

8. Conclusion
--------------

Due to the instability of ICL, there is an increasing demand for in-context example selection in code generation. This paper proposes a novel learning-based ICL approach LAIL. We exploit LLMs themselves to estimate examples by considering the generation probabilities of ground-truth programs given a requirement, and label these examples based on LLMs’ feedback. We then introduce a contrastive learning objective to train a retriever, resulting in acquiring the preferences of LLMs. Experimental results on three code generation datasets demonstrate that our proposed approach achieves excellent performance. LAIL also shows satisfactory transferability across LLMs and datasets, showing that it is an efficient and practical approach to code generation. In the future, we will apply our approach to other LLMs and datasets.

References
----------

* (1)
* Cha (able)Available.<https://openai.com/chatgpt>.
* GPT (ablea)Availablea.[https://platform.openai.com/docs/models/gpt-3-5](https://platform.openai.com/docs/models/gpt-3-5 "").
* Cod (able)Available.<https://huggingface.co/codeparrot/codeparrot>.
* GPT (ableb)Availableb.[https://github.com/CodedotAl/gpt-code-clippy](https://github.com/CodedotAl/gpt-code-clippy "").
* Ahmad et al. (2021)WU Ahmad, S Chakraborty,
B Ray, and KW Chang.
2021.Unified pre-training for program understanding and
generation.. In *Proceedings of the 2021 Conference
of the North American Chapter of the Association for Computational
Linguistics: Human Language Technologies*.
* Allamanis et al. (2018a)Miltiadis Allamanis,
Earl T Barr, Premkumar Devanbu, and
Charles Sutton. 2018a.A survey of machine learning for big code and
naturalness.*ACM Computing Surveys (CSUR)*51, 4 (2018),
1–37.
* Allamanis et al. (2018b)Miltiadis Allamanis,
Earl T. Barr, Premkumar T. Devanbu, and
Charles Sutton. 2018b.A Survey of Machine Learning for Big Code and
Naturalness.*ACM Comput. Surv.* 51,
4 (2018), 81:1–81:37.[https://doi.org/10.1145/3212695](https://doi.org/10.1145/3212695 "")
* Athiwaratkun et al. (2022)Ben Athiwaratkun,
Sanjay Krishna Gouda, Zijian Wang,
Xiaopeng Li, Yuchen Tian,
Ming Tan, Wasi Uddin Ahmad,
Shiqi Wang, Qing Sun,
Mingyue Shang, et al.2022.Multi-lingual Evaluation of Code Generation
Models.*arXiv preprint arXiv:2210.14868*(2022).
* Austin et al. (2021)Jacob Austin, Augustus
Odena, Maxwell Nye, Maarten Bosma,
Henryk Michalewski, David Dohan,
Ellen Jiang, Carrie Cai,
Michael Terry, Quoc Le, et al.2021.Program synthesis with large language models.*arXiv preprint arXiv:2108.07732*(2021).
* Bareiß et al. (2022)Patrick Bareiß,
Beatriz Souza, Marcelo d’Amorim, and
Michael Pradel. 2022.Code generation tools (almost) for free? a study of
few-shot, pre-trained language models on code.*arXiv preprint arXiv:2206.01335*(2022).
* Brown et al. (2020)Tom Brown, Benjamin Mann,
Nick Ryder, Melanie Subbiah,
Jared D Kaplan, Prafulla Dhariwal,
Arvind Neelakantan, Pranav Shyam,
Girish Sastry, Amanda Askell,
et al. 2020.Language models are few-shot learners.*Advances in neural information processing
systems* 33 (2020),
1877–1901.
* Chakraborty et al. (2022)Saikat Chakraborty,
Toufique Ahmed, Yangruibo Ding,
Premkumar T Devanbu, and Baishakhi
Ray. 2022.Natgen: generative pre-training by
“naturalizing” source code. In *Proceedings of
the 30th ACM Joint European Software Engineering Conference and Symposium on
the Foundations of Software Engineering*. 18–30.
* Chen et al. (2022)Bei Chen, Fengji Zhang,
Anh Nguyen, Daoguang Zan,
Zeqi Lin, Jian-Guang Lou, and
Weizhu Chen. 2022.Codet: Code generation with generated tests.*arXiv preprint arXiv:2207.10397*(2022).
* Chen et al. (2021)Mark Chen, Jerry Tworek,
Heewoo Jun, Qiming Yuan,
Henrique Ponde de Oliveira Pinto, Jared
Kaplan, Harri Edwards, Yuri Burda,
Nicholas Joseph, Greg Brockman,
et al. 2021.Evaluating large language models trained on code.*arXiv preprint arXiv:2107.03374*(2021).
* Chen et al. (2020)Ting Chen, Simon
Kornblith, Mohammad Norouzi, and
Geoffrey Hinton. 2020.A simple framework for contrastive learning of
visual representations. In *International
conference on machine learning*. PMLR, 1597–1607.
* Chen et al. (2023)Xinyun Chen, Maxwell Lin,
Nathanael Schärli, and Denny Zhou.
2023.Teaching large language models to self-debug.*arXiv preprint arXiv:2304.05128*(2023).
* Coleman et al. (2019)Cody Coleman, Christopher
Yeh, Stephen Mussmann, Baharan
Mirzasoleiman, Peter Bailis, Percy
Liang, Jure Leskovec, and Matei
Zaharia. 2019.Selection via proxy: Efficient data selection for
deep learning.*arXiv preprint arXiv:1906.11829*(2019).
* Dong and Lapata (2016)Li Dong and Mirella
Lapata. 2016.Language to Logical Form with Neural Attention. In*Proceedings of the 54th Annual Meeting of the
Association for Computational Linguistics (Volume 1: Long Papers)*.
33–43.
* Dong et al. (2022)Qingxiu Dong, Lei Li,
Damai Dai, Ce Zheng,
Zhiyong Wu, Baobao Chang,
Xu Sun, Jingjing Xu, and
Zhifang Sui. 2022.A survey for in-context learning.*arXiv preprint arXiv:2301.00234*(2022).
* Feng et al. (2020)Zhangyin Feng, Daya Guo,
Duyu Tang, Nan Duan,
Xiaocheng Feng, Ming Gong,
Linjun Shou, Bing Qin,
Ting Liu, Daxin Jiang, et al.2020.Codebert: A pre-trained model for programming and
natural languages.*arXiv preprint arXiv:2002.08155*(2020).
* Guo et al. (2020)Daya Guo, Shuo Ren,
Shuai Lu, Zhangyin Feng,
Duyu Tang, Shujie Liu,
Long Zhou, Nan Duan,
Alexey Svyatkovskiy, Shengyu Fu,
et al. 2020.Graphcodebert: Pre-training code representations
with data flow.*arXiv preprint arXiv:2009.08366*(2020).
* Holtzman et al. (2019)Ari Holtzman, Jan Buys,
Li Du, Maxwell Forbes, and
Yejin Choi. 2019.The curious case of neural text degeneration.*arXiv preprint arXiv:1904.09751*(2019).
* Husain et al. (2019)Hamel Husain, Ho-Hsiang
Wu, Tiferet Gazit, Miltiadis Allamanis,
and Marc Brockschmidt. 2019.Codesearchnet challenge: Evaluating the state of
semantic code search.*arXiv preprint arXiv:1909.09436*(2019).
* Inala et al. (2022)Jeevana Priya Inala,
Chenglong Wang, Mei Yang,
Andres Codas, Mark Encarnación,
Shuvendu Lahiri, Madanlal Musuvathi,
and Jianfeng Gao. 2022.Fault-aware neural code rankers.*Advances in Neural Information Processing
Systems* 35 (2022),
13419–13432.
* Jiang et al. (2023b)Nan Jiang, Kevin Liu,
Thibaud Lutellier, and Lin Tan.
2023b.Impact of code language models on automated program
repair.*arXiv preprint arXiv:2302.05020*(2023).
* Jiang et al. (2023a)Xue Jiang, Yihong Dong,
Lecheng Wang, Qiwei Shang, and
Ge Li. 2023a.Self-planning code generation with large language
model.*arXiv preprint arXiv:2303.06689*(2023).
* Levy et al. (2022)Itay Levy, Ben Bogin,
and Jonathan Berant. 2022.Diverse demonstrations improve in-context
compositional generalization.*arXiv preprint arXiv:2212.06800*(2022).
* Lewis et al. (2019)Mike Lewis, Yinhan Liu,
Naman Goyal, Marjan Ghazvininejad,
Abdelrahman Mohamed, Omer Levy,
Ves Stoyanov, and Luke Zettlemoyer.
2019.Bart: Denoising sequence-to-sequence pre-training
for natural language generation, translation, and comprehension.*arXiv preprint arXiv:1910.13461*(2019).
* Li et al. (2023a)Jia Li, Yongmin Li,
Ge Li, and Zhi Jin.
2023a.Structured Chain-of-Thought Prompting for Code
Generation.*CoRR* (2023).[https://arxiv.org/abs/2305.06599](https://arxiv.org/abs/2305.06599 "")
* Li et al. (2023b)Jia Li, Yongmin Li,
Ge Li, Zhi Jin, Yiyang
Hao, and Xing Hu. 2023b.SkCoder: A Sketch-based Approach for Automatic
Code Generation. In *45th IEEE/ACM International
Conference on Software Engineering, ICSE 2023, Melbourne, Australia, May
14-20, 2023*. IEEE, 2124–2135.[https://doi.org/10.1109/ICSE48619.2023.00179](https://doi.org/10.1109/ICSE48619.2023.00179 "")
* Li et al. (2023c)Jia Li, Yunfei Zhao,
Yongmin Li, Ge Li, and
Zhi Jin. 2023c.AceCoder: Utilizing Existing Code to Enhance Code
Generation.*CoRR* (2023).[https://arxiv.org/abs/2303.17780v3](https://arxiv.org/abs/2303.17780v3 "")
* Li et al. (2022)Yujia Li, David Choi,
Junyoung Chung, Nate Kushman,
Julian Schrittwieser, Rémi Leblond,
Tom Eccles, James Keeling,
Felix Gimeno, Agustin Dal Lago,
et al. 2022.Competition-level code generation with alphacode.*Science* 378,
6624 (2022), 1092–1097.
* Liu et al. (2021)Jiachang Liu, Dinghan
Shen, Yizhe Zhang, Bill Dolan,
Lawrence Carin, and Weizhu Chen.
2021.What Makes Good In-Context Examples for GPT-$3$?*arXiv preprint arXiv:2101.06804*(2021).
* Lu et al. (2021)Shuai Lu, Daya Guo,
Shuo Ren, Junjie Huang,
Alexey Svyatkovskiy, Ambrosio Blanco,
Colin Clement, Dawn Drain,
Daxin Jiang, Duyu Tang, et al.2021.Codexglue: A machine learning benchmark dataset for
code understanding and generation.*arXiv preprint arXiv:2102.04664*(2021).
* Luo et al. (2023)Man Luo, Xin Xu,
Zhuyun Dai, Panupong Pasupat,
Mehran Kazemi, Chitta Baral,
Vaiva Imbrasaite, and Vincent Y Zhao.
2023.Dr. ICL: Demonstration-Retrieved In-context
Learning.*arXiv preprint arXiv:2305.14128*(2023).
* Mockus (2007)Audris Mockus.
2007.Large-scale code reuse in open source software. In*First International Workshop on Emerging Trends in
FLOSS Research and Development (FLOSS’07: ICSE Workshops 2007)*. IEEE,
7–7.
* Papineni et al. (2002)Kishore Papineni, Salim
Roukos, Todd Ward, and Wei-Jing Zhu.
2002.Bleu: a method for automatic evaluation of machine
translation. In *Proceedings of the 40th annual
meeting of the Association for Computational Linguistics*.
311–318.
* Radford et al. (2018)Alec Radford, Karthik
Narasimhan, Tim Salimans, Ilya
Sutskever, et al. 2018.Improving language understanding by generative
pre-training.(2018).
* Raffel et al. (2020)Colin Raffel, Noam
Shazeer, Adam Roberts, Katherine Lee,
Sharan Narang, Michael Matena,
Yanqi Zhou, Wei Li,
Peter J Liu, et al.2020.Exploring the limits of transfer learning with a
unified text-to-text transformer.*J. Mach. Learn. Res.* 21,
140 (2020), 1–67.
* Reimers and Gurevych (2019)Nils Reimers and Iryna
Gurevych. 2019.Sentence-bert: Sentence embeddings using siamese
bert-networks.*arXiv preprint arXiv:1908.10084*(2019).
* Ren et al. (2020)Shuo Ren, Daya Guo,
Shuai Lu, Long Zhou,
Shujie Liu, Duyu Tang,
Neel Sundaresan, Ming Zhou,
Ambrosio Blanco, and Shuai Ma.
2020.Codebleu: a method for automatic evaluation of code
synthesis.*arXiv preprint arXiv:2009.10297*(2020).
* Robertson et al. (2009)Stephen Robertson, Hugo
Zaragoza, et al. 2009.The probabilistic relevance framework: BM25 and
beyond.*Foundations and Trends® in
Information Retrieval* 3, 4
(2009), 333–389.
* Su et al. (2022)Hongjin Su, Jungo Kasai,
Chen Henry Wu, Weijia Shi,
Tianlu Wang, Jiayi Xin,
Rui Zhang, Mari Ostendorf,
Luke Zettlemoyer, Noah A Smith,
et al. 2022.Selective annotation makes language models better
few-shot learners.*arXiv preprint arXiv:2209.01975*(2022).
* Wang et al. (2021)Yue Wang, Weishi Wang,
Shafiq Joty, and Steven CH Hoi.
2021.Codet5: Identifier-aware unified pre-trained
encoder-decoder models for code understanding and generation.*arXiv preprint arXiv:2109.00859*(2021).
* Zan et al. (2022)Daoguang Zan, Bei Chen,
Dejian Yang, Zeqi Lin,
Minsu Kim, Bei Guan,
Yongji Wang, Weizhu Chen, and
Jian-Guang Lou. 2022.CERT: Continual Pre-Training on Sketches for
Library-Oriented Code Generation.*arXiv preprint arXiv:2206.06888*(2022).
* Zheng et al. (2023)Qinkai Zheng, Xiao Xia,
Xu Zou, Yuxiao Dong,
Shan Wang, Yufei Xue,
Zihan Wang, Lei Shen,
Andi Wang, Yang Li, et al.2023.Codegeex: A pre-trained model for code generation
with multilingual evaluations on humaneval-x.*arXiv preprint arXiv:2303.17568*(2023).
* Zhong et al. (2022)Maosheng Zhong, Gen Liu,
Hongwei Li, Jiangling Kuang,
Jinshan Zeng, and Mingwen Wang.
2022.CodeGen-Test: An Automatic Code Generation Model
Integrating Program Test Information.*arXiv preprint arXiv:2202.07612*(2022).
