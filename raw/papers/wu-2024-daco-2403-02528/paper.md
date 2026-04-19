Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation
=====================================================================================

Xueqing Wu1, Rui Zheng2, Jingzhen Sha1, Te-Lin Wu1, Hanyu Zhou1, Mohan Tang1,  
 Kai-Wei Chang1, Nanyun Peng1, Haoran Huang3  
1University of California, Los Angeles 2Fudan University 3 ByteDance AI LabCorresponding author. Contact: xueqing.wu@cs.ucla.edu, huanghaoran@bytedance.com

###### Abstract

Data analysis is a crucial analytical process to generate in-depth studies and conclusive insights to comprehensively answer a given user query for tabular data. In this work, we aim to propose new resources and benchmarks to inspire future research on this crucial yet challenging and under-explored task. However, collecting data analysis annotations curated by experts can be prohibitively expensive. We propose to automatically generate high-quality answer annotations leveraging the code-generation capabilities of LLMs with a multi-turn prompting technique. We construct theDaco dataset, containing (1) 440 databases (of tabular data) collected from real-world scenarios, (2) $\sim 2k$ query-answer pairs that can serve as weak supervision for model training, and (3) a concentrated but high-quality test set with human refined annotations that serves as our main evaluation benchmark. We train a 6B supervised fine-tuning (SFT) model on Daco dataset, and find that the SFT model learns reasonable data analysis capabilities. To further align the models with human preference, we use reinforcement learning to encourage generating analysis perceived by human as helpful, and design a set of dense rewards to propagate the sparse human preference reward to intermediate code generation steps. Our Daco-RL algorithm is evaluated by human annotators to produce more helpful answers than SFT model in $57.72\%$ cases, validating the effectiveness of our proposed algorithm.
Data and code are released at [https://github.com/shirley-wu/daco](https://github.com/shirley-wu/daco "").

1 Introduction
--------------

Data analysis is the process of systematically applying statistical and/or logical reasoning to evaluate and comprehend data.
Existing literature has investigated answering queries about information given by structural data (e.g., tables)*(Chen et al., [2021a](#bib.bib6 ""); Nan et al., [2022](#bib.bib23 ""); Lu et al., [2023](#bib.bib22 ""))*.
However, they either focus on straightforward factual retrieval or short-form entity/arithmetic resolutions for specifically given entities, while real-world data analysis can involve more complex analytical processes.

Take the scenario in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation") as an example: a user is investigating potential age discrimination of a shop.
To effectively answer queries such as this one, a chain of mathematical and logical reasoning and interacting with the data is required.
For instance, finding 1 is inferred from analyzing age distribution within the membership data (‘member’ table), while finding and suggestion 2 are derived by comparing the participants’ ages during the happy hours (using both ‘member’ and ‘happy_hour_member’ tables).
These rigorous quantitative analyses eventually conclude the opposite to the user’s hypothesis.
As valuable as the conclusive suggestions such comprehensive analysis can bring, the extensive labor-efforts, hinted by these examples, can hinder the efficiency of gaining intelligence from the data in a competitive business environment.
It is thus imperative to devise a system that is able to automate the aforementioned data analysis process.

<img src='figures/task_overview_4.jpg' alt='Refer to caption' title='' width='598' height='223' />

*Figure 1: Task overview. Given an application-driven analysis query, a data analysis system is expected to produce an answer containing findings and suggestions based on the database. This requires the system to draw application-driven conclusions from mathematical and logical reasoning. In this example, finding 1 is inferred from analyzing age distribution within the membership data (‘member’ table), while finding 2 and suggestion 2 are derived by comparing the ages of the happy hours participants (using ‘member’ and ‘happy_hour_member’ tables).*

To this end, we introduce a new dataset for this challenging task, Daco, data analysis via code generation. Dacois constructed from a set of diverse real-world databases associated with curated user queries.
In light of the previously described labor-intensive challenge, we propose to leverage LLMs with a multi-turn chained prompts to automatically curate the analytical answers for each query.
Specifically, our designed framework employs the code generation capabilities of GPT-4*(OpenAI, [2023](#bib.bib25 ""))* for automating the statistical analysis, interleaved with its ability to interpret the obtained quantitative results.
TheDacodataset contains $440$ databases and $1,942$ associated user queries, which can be used for both model fine-tuning and evaluation.
To provide a refined benchmarking resource, we curate a high-quality test set through comprehensive human annotations on a subset of 100 samples. Detailed statistics are in Table [1](#S2.T1 "Table 1 ‣ Figure 2 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").

Although LLM exhibit reasonable analytical capabilities (and hence we are able to automatically curate the pre-refined data), we empirically find that its generations often fall short of human expectations of what good analyses should be (e.g., relevance to the queries, logical coherency, and higher-level quantitative interpretations).
To further improve the generations by aligning the models with corresponding human preferences, we design a reinforcement learning algorithm (Daco-RL) that leverages two newly designed reward models (RM), i.e., contribution RM and regularization RM, to efficiently provide denser feedback.
Concretely, the contribution RM heuristically provides better learning signals for the intermediate code generation steps (for more relevant quantitative analysis), while regularization RM helps preventing reward hacking*(Skalse et al., [2022](#bib.bib32 ""))* of typical RLHF models *(Casper et al., [2023](#bib.bib4 ""))*.
We test our algorithm on a fine-tuned CodeGeeX-6B model*(Zheng et al., [2023a](#bib.bib38 ""))*, where the win rate of $57.72\%$ in human-annotated pairwise comparison
justifies its effectiveness on learning to generate human preferred analyses.

In summary, our contributions are three folds:
(1) We explore the challenging task of data analysis, where we construct the Daco dataset with our proposed multi-turn prompting technique on a diverse set of real-world databases.
(2) We curate a human-refined evaluation set for benchmarking models. (3) We design the Daco-RL algorithm to jointly optimize code generation and answer generation towards human alignment, which demonstrates a significant $57.72\%$ human evaluated win rate on the helpfulness metric.

2 TheDacoTask and Dataset
-------------------------

As shown in Figure [1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"), the input to our task is a database $\mathcal{D}$ and a query $\mathbf{q}$, and the output answer $\mathbf{a}$ is formatted as two lists of findings and suggestions respectively.
In this work, the database $\mathcal{D}$ should be a relational database containing multiple named tables.

<img src='figures/dataset.jpg' alt='Refer to caption' title='' width='598' height='600' />

*Figure 2: Curation process of Daco dataset.*

|  | Train | Dev | TestA | TestH | Total |
| --- | --- | --- | --- | --- | --- |
| # db | 353 | 22 | 65 | 17 | 440 |
| # queries | 1558 | 100 | 284 | 100 | 1942 |

| Database size | Med. | Max | Min |
| --- | --- | --- | --- |
| # tables | 1 | 15 | 1 |
| # columns | 6 | 50 | 3 |
| # rows | 20 | 19,237 | 2 |
| Answer size in TestH | Med. | Max | Min |
| # findings | 5 | 8 | 3 |
| # suggestions | 5 | 8 | 3 |
| # tokens | 397 | 864 | 202 |

*Table 1: Statistics of Daco dataset. Train, Dev and TestA sets are automatically generated with GPT-4, while TestH is the human refined subset. We report the size of each data split, the size of input databases, and the size of output answers in human refined test set (TestH).*

We construct our Daco dataset through four stages: (1) database collection, (2) query collection, (3) automatic annotation collection, and (4) human refinement. The workflow is illustrated in Figure [2](#S2.F2 "Figure 2 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
Our final dataset contains the training, development and test sets with annotations generated by GPT-4, along with a human-refined testing subset. To distinguish the two test sets, we use TestA to represent the automatically annotated set and TestH to represent the human refined one.
Statistics are shown in Table [1](#S2.T1 "Table 1 ‣ Figure 2 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").

Database collection. We collect databases from two sources: Spider *(Yu et al., [2018](#bib.bib37 ""))* and Kaggle (<https://www.kaggle.com/datasets>).
There are 157 databases collected from Spider, which originally come from university databases, DatabaseAnswers and Wikipedia.
We additionally crawl and filter 5,830 databases from Kaggle.
From this pool, we manually select a subset of 314 clean and interpretable databases to build our dataset.
To maintain the diversity of the resulting database set, 157 of the databases are deliberately chosen near the long tail of its topic distribution.
For this, we employ BERTopic*(Grootendorst, [2022](#bib.bib10 ""))* to model the topic distribution, which produces in total 160 topics.
We take its least frequent 80 topics as the long tail, which covers 26.79% of the total databases.

In total, Daco comprises 471 databases, each of which contains on average 2.3 tables.
To better visualize the major topic distribution of this selected subset, we again use BERTopic but group these databases into 10 topics.
The keywords for top 5 topics are shown in Figure[4](#S2.F4 "Figure 4 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
The leading topic (topic 1) is associated with business setting and consists of 46.52% of the dataset.
The remaining nine topics exhibit a relatively even distribution, covering a broad range of domains, including sports (topic 2), healthcare (topic 3), weather (topic 4), and education (topic 5).

Query collection. We generate 10 queries for each database by prompting ChatGPT to first assume the role of a database stakeholder and then generate an application-driven query based on the role.
To ensure the quality of the query, we perform a manual filtering to the machine generated queries.
Specifically, we remove queries that are not driven by real-world applications or cannot be answered by the given reference database.
We train a group of 6 annotators to perform such a filtering process.
As a result, there are about 42% of the queries removed, where the removal agreement achieves a 0.62 cohen kappa score.

After the aforementioned processes, we obtain in total 2,664 queries.
We show the top 15 verbs and their top 3 direct noun objectives in Figure[4](#S2.F4 "Figure 4 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
The queries demonstrate a notable level of diversity.
The most common type of queries is to request analysis (such as “analyze data” and “identify pattern”), followed by queries aiming to make decisions (such as “determine strategy” and “make decision”).

Automatic annotation collection. As shown in the right half of Figure[5](#S2.F5 "Figure 5 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"), we design a pipeline that leverages the code generation capability of LLMs to automate the answer annotation for our Daco dataset.
Based on the database and the query, we instruct the LLM to perform data analysis in multiple turns.
At each turn, the LLM will produce a python code snippet and take its execution outputs as evidences to reason over and support its follow-up interpretation. After each turn, we prompt the model to decide whether the analysis is sufficiently comprehensive; if deemed sufficient, it terminates the coding turns and produces the final answer.

With this pipeline, we instruct GPT-4 to automatically generate all the answer annotations to each query of our dataset, for both the intermediate code and the final analysis answering the queries.
To improve the quality of such automatically constructed annotations, we additionally allow GPT-4 to correct its own mistakes when its generated code leads to run-time or syntax error, where only the corrected codes are kept.
In total, we obtain 1.9k valid query-answer pairs, each with roughly 3.3 intermediate coding steps.

<img src='figures/database_dist.png' alt='Refer to caption' title='' width='598' height='545' />

*Figure 3: Domain distribution of Daco databases. We display the topic distribution and keywords for the leading 5 topics. Topics are extracted from database titles using BERTopic. This demonstrates the diverse domain coverage of Daco.*

<img src='figures/question_dist.png' alt='Refer to caption' title='' width='598' height='596' />

*Figure 4: Distribution of Daco queries. We display the top 15 verbs and their top 3 direct noun objectives, demonstrating the diversity of Daco queries.*

Human refinement. The annotated analyses thus far have been algorithmically generated, where their actual quality are to be further verified.
We thus curate a human-refined subset containing 100 densely human-annotated query-answer pairs.
For each query, we sample 3 different analysis candidates using the previously described automated method (with GPT-4).
We ask the annotators to evaluate the quality of each machine generated bullet point and categorize each point into one of bad, borderline, or good (associated with scalar scores of 0, 1 and 2 respectively, the higher the better) to each.
The bullet points deemed higher quality are then mixed (from the 3 candidate analyses) and refined (with a few manual textual edits) into one final gold-analysis. In the refinement stage, the annotators should first combine all bullet points ranked as “good”, remove duplicate points, and reorder the points to maintain a coherent flow. Suppose the number of bullet points are lower than our pre-defined lowest threshold (3 bullet points per answer), the annotators should select bullet points ranked as “borderline” to augment the answer.
we ask a group of 3 internal members to perform refinement. The agreement accuracy of the refinement process (candidate point selection) is 0.83 and the Cohen’s Kappa is 0.67.

Evaluation. To evaluate the quality of generated data analysis, we use helpfulness as the main metric. Motivated by literature in the data analysis field *(Long \& Long, [2009](#bib.bib21 ""))*, we define helpfulness as: (1) relevance to the query, (2) effective and insightful data interpretation, and (3) diversity in terms of analysis perspectives.
We evaluate helpfulness through pairwise comparison following common approach *(Ouyang et al., [2022](#bib.bib26 ""); Wu et al., [2023](#bib.bib35 ""); Zheng et al., [2023b](#bib.bib39 ""))*. Given two analyses generated by two different systems, the annotator (either human or simulated by ChatGPT) selects the more helpful one based on our defined criteria. The winning rate of each system is reported as helpfulness score. To obtain a comparable set of numbers for all models, we report the winning rate of each model against TestA and TestH annotations. The upper bound for this score would be 50, as a score of 50 indicates that the model generations are perceived as helpful as annotations.

<img src='figures/framework_2.jpg' alt='Refer to caption' title='' width='598' height='251' />

*Figure 5: Overview of code generation pipeline (right) and our Daco-RL algorithm (left).  
Code generation (right): At each turn, the model generates python code, execute the code with a code interpreter, and reads the execution outputs. Eventually, it summarizes the results into a final answer. 
Daco-RL (left): The answer RM evaluates the helpfulness of the final answer and is our end optimization goal. However, to provide denser reward, we use a contribution RM to evaluate how much each generated code contribute to the final answer. The reward is provided at the end of each generated code snippet. Contribution RM is vulnerable to reward hacking, which motivates us to propose regularization RM to discourage reward hacking generations.*

3 Daco-RL
----------

WhileDaco contains mostly algorithmic machine generated analyses, the machine generations without human refinement cannot well align with human preferences (of “good” analyses). Our human refinement process shows that only 47.4% bullet points are “good” points perfectly addressing user queries; the majority of 52.2% are evaluated as “borderline” points that only partially aligns with human expectations; and the remaining 0.4% are considered as “bad”.

We are therefore interested in investigating whether aligning human preferences via an RLHF fashion could lead to better machine generated analyses.
We thus propose theDaco-RL algorithm, which is illustrated in the left half of Figure[5](#S2.F5 "Figure 5 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
Our end goal is to optimize the helpfulness of the analyzed points, which is modelled with an answer RM $R_{a}$. In addition to this sparse reward signal, we use a heuristically defined contribution RM $R_{c}$ to reward each intermediate step, which is further regularized with a regularization RM $R_{r}$ to prevent reward hacking. In the following sections, we first explain the three reward models sequentially, and eventually explain our whole RLHF pipeline.

Notations. We train a language model that interacts with the python interpreter in a conversational manner.
Formally, the full dialogue is a list of messages $\left[\mathbf{h},\mathbf{c}_{1},\mathbf{o}_{1},\cdots,\mathbf{c}_{m},\mathbf{o}_{m},\mathbf{a}\right]$, where $\mathbf{h}$, $\mathbf{c}_{i}$, $\mathbf{o}_{i}$ and $\mathbf{a}$ stand for human message, code, execution outputs and final answer respectively. The dialogue starts with human message $\mathbf{h}$ containing both query $\mathbf{q}$ and database meta-data.
In the later messages, code $\mathbf{c}_{i}$ and final answer $\mathbf{a}$ are generated by our model, while execution outputs $\mathbf{o}_{i}$ are produced by the python interpreter.
To feed the dialogue into the language model, we wrap each message between a begin-of-message phrase (denoted as $<$BOM$>$) and the end-of-sentence token, and then concatenate all messages together. We use a different $<$BOM$>$ for each type of message. Thus, at each turn of the conversation, the language model can decide whether to generate code $\mathbf{c}$ or final answer $\mathbf{a}$ by generating the corresponding $<$BOM$>$.

Answer RM. Our end goal is to optimize the helpfulness of answer $\mathbf{a}$, which is modelled by the answer RM $R_{a}$.
Particularly, we model the helpfulness of each single bullet point rather than the full answer.
We collect pairwise comparison data of bullet points from ChatGPT to train $R_{a}$. Given a pair of bullet points where one is preferred over the other, $R_{a}$ is trained to assign a higher score to the preferred bullet point.
Given access to the full conversation, $R_{a}$ produces a reward score at the end of each bullet point in the answer.

To encourage diversity, we additionally add a repetition penalty.
Given a list of findings $\left[\mathbf{f}_{1},\cdots,\mathbf{f}_{F}\right]$, we encourage the $i$-th finding to be different from all previous findings by imposing a penalty score as $\sum_{j\=1}^{i-1}Sim(\mathbf{f}_{i},\mathbf{f}_{j})$, where $Sim(\cdot,\cdot)$ computes the similarity between two bullet points.111$Sim(\cdot,\cdot)$ is computed as the cosine similarity of Sentence-BERT *(Reimers \& Gurevych, [2019](#bib.bib29 ""))* embeddings. We use the all-MiniLM-L6-v2 model. This score is then subtracted from the reward score from $R_{a}$. The same procedure is applied to the suggestion list.

Contribution RM. The answer RM described above is a sparse reward signal that rewards only the answer $\mathbf{a}$ but not intermediate coding steps $\mathbf{c}_{i}$. To provide denser reward signal for optimizing $\mathbf{c}_{i}$, we aim to evaluate the helpfulness of each coding step $\mathbf{c}_{i}$.
However, annotating helpfulness of intermediate steps is much more difficult. For one, the helpfulness of python code is more vague to define, and the other, evaluating helpfulness requires coding expertise, which makes it more expensive to collect human annotations.

To measure the helpfulness of intermediate steps without the huge expense required by human annotations, we heuristically define the helpfulness as how much an intermediate step contributes to the final answer. Concretely, we compute the similarity $Sim(\mathbf{a},\mathbf{o}_{i})$ between final answer and code outputs to measure the helpfulness of $\mathbf{c}_{i}$.
We use $Sim(\mathbf{a},\mathbf{o}_{i})$ to rank the helpfulness of different steps, and use the comparison pairs between intermediate steps to train the contribution RM $R_{c}$. Given the conversation, $R_{c}$ predicts the contribution level of each $\mathbf{c}_{i}$ as its reward score. $R_{c}$ does not take the execution output $\mathbf{o}_{i}$ into consideration when scoring $\mathbf{c}_{i}$, which simplifies model implementation and excludes spurious correlation the model may exploit from the surface form of $\mathbf{o}_{i}$.

Regularization RM. The heuristically defined contribution RM $R_{c}$ may not necessarily perfectly align with the true helpfulness.
This is due to a known reward misspecification issue termed reward hacking *(Skalse et al., [2022](#bib.bib32 ""))*, where the policy model achieves higher scores from the reward model but its true reward decreases.
We propose to regularize such behavior with a regularization RM $R_{r}$.
Given the misspecified reward model, $R_{c}$ in our case, we first train an RL model until its generations start to collapse to certain patterns. These patterns typically receive high rewards from $R_{c}$ but do not align well with human expectation, and thus are considered as reward hacking behaviors. We denoted this RL model as $\pi_{\textmd{hack}}$.
We use $\pi_{\textmd{hack}}$ to produce generations with typical reward hacking behaviors. These generations are paired with generations without reward hacking behaviors, such as generations from supervised fine-tuning (SFT) model or the pre-human refined answers generated from GPT-4, to further train the regularization RM $R_{r}$.
As an intuition, this means $R_{r}$ will assign lower scores to typical reward hacking behaviors.

RLHF. For our wholeDaco-RL pipeline, we optimize the language model against the mixture of all three aforementioned rewards, $R_{a}$, $R_{c}$ and $R_{r}$.

More specifically, we first train a multi-task reward model denoted as $R_{a+c}$ to jointly learn $R_{a}$ and $R_{c}$. This guarantees that the reward score distribution for $R_{a}$ and $R_{c}$ are relatively close, so the reward signal for $\mathbf{a}$ and $\mathbf{c}_{i}$ will not overshadow each other.
We train a separate model to learn $R_{r}$, and mix $R_{c+a}$ and $R_{r}$ into $\frac{1}{2}(R_{c+a}+w_{r}R_{r})$ when rewarding $\mathbf{c}_{i}$.222In practice, we find $R_{r}$ has a large non-zero mean value, so we subtract it before weighted average. Here, $w_{r}$ is a hyper-parameter to balance the variance of $R_{c+a}$ and $R_{r}$ tuned to maximize reward model accuracy on development set. This guarantees the mixed reward can both encourage high contribution and penalize reward hacking.

We use proximal policy optimization (PPO) *(Schulman et al., [2017](#bib.bib31 ""))* as our learning algorithm. During training, PPO jointly optimizes a value model $V(s)$ and a policy model $\pi(s)$. The objective of the policy model is to optimize the generalized advantage estimation *(Schulman et al., [2016](#bib.bib30 ""))* $\hat{A}_{t}\=\sum_{l\=0}^{\infty}(\gamma\lambda)^{l}\delta_{t+l}$, where $\delta_{t}\=r_{t}+\gamma V(s_{t+1})-V(s_{t})$
for each time step $t$.
When applied to text generation, the generative language model is the policy model $\pi$ and each generated token is an action.
In our multi-turn conversational setting, however, only part of the tokens in the dialogue are generated by language model (concretely, $\mathbf{c}_{i}$ and $\mathbf{a}$).
In other words, although the language model still takes the full conversation as input, we only compute GAE and gradients over the model generated subsequence, i.e., $\left[\mathbf{c}_{1},\cdots,\mathbf{c}_{m},\mathbf{a}\right]$.

4 Experiments
-------------

The goal of our experiments is to verify that (1) augmenting language model with code generation can benefit data analysis, and (2) Daco-RL can further boost the answer helpfulness. To this purpose, we perform the following experiments:

Evaluated systems. We evaluate the code generation pipeline with ChatGPT and GPT-4. With the answer annotation generated by GPT-4, we further train a 6B CodeGeeX2-6B *(Zheng et al., [2023a](#bib.bib38 ""))* model through both SFT and Daco-RL.
For each of these models, we experiment with a baseline counterpart that does not include code generation and instead directly takes raw table content as input.
We additionally experiment with two models specifically pre-trained on tabular data, TAPAS *(Herzig et al., [2020](#bib.bib12 ""))* and TAPEX *(Liu et al., [2022](#bib.bib19 ""))*. TAPAS is a BERT-style model pre-trained to select relevant information from a table based on user query. For our dataset, we first use TAPAS to select relevant information and then use ChatGPT to interpret the selected information. TAPEX is a pre-trained encoder-to-decoder model. We fine-tune TAPEX with GPT-4-generated annotations.

Evaluation. The main metric we use is pairwise comparison of helpfulness as in Section [2](#S2 "2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). We use both ChatGPT and trained human annotators for the evaluation. We additionally report BLEU score, entailment score, and helpfulness evaluation for each individual bullet point. These metrics cannot holistically measure the analysis helpfulness, but can provide complementary insights for analyzing model performance. For entailment, we use an off-the-shelf NLI
model to compute the probability that the model generation is entailed by the annotation. For point-wise evaluation, we ask the annotator to assign a score chosen from 0, 1 and 2 to each bullet point using the same standard as in human refinement of test set.
Our human annotation achieves high agreement of 0.62 Cohen’s kappa for pairwise comparison of helpfulness, and 0.65 Cohen’s kappa for point-wise helpfulness evaluation.

### 4.1 Results

|  |  |  |  | TestA | | | TestH | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Method | # para. | Code gen | Help. | Entail. | BLEU | Help. | Entail. | BLEU |
| TableQABaselines | TAPAS | 337M | ✗ | 25.00 | 1.96 | 11.62 | 24.50 | 3.67 | 9.73 |
| | TAPEX | 406M | ✗ | 14.79 | 3.34 | 14.60 | 6.00 | 3.50 | 13.81 |
| Prompt-based LLM | ChatGPT | 20B† | ✗ | 25.18 | 3.06 | 13.22 | 18.50 | 2.07 | 13.51 |
| | GPT-4 | 175B† | ✗ | 30.81 | 3.35 | 14.90 | 24.00 | 4.36 | 13.71 |
| ChatGPT | 20B† | ✓ | 35.74 | 2.74 | 14.22 | 27.27 | 2.59 | 14.51 |
| GPT-4 | 175B† | ✓ | 52.00 | 4.59 | 17.77 | 41.88 | 3.26 | 17.54 |
| FinetunedLLM | SFT | 6B | ✗ | 21.51 | 2.30 | 14.47 | 9.50 | 2.65 | 13.63 |
| | SFT | 6B | ✓ | 20.95 | 2.15 | 14.88 | 11.54 | 4.47 | 14.60 |
| Daco-RL | 6B | ✓ | 28.54 | 3.65 | 13.13 | 21.05 | 5.98 | 11.80 |

*Table 2: Main results. We report helpfulness (Help.), entailment (Entail.), and BLEU on both automatically annotated test set (TestA) and human curated test set (TestH). We also report the number of parameters (# para.) of each model. ${\dagger}$: For ChatGPT and GPT-4, we report the number of parameters based on our best estimation.*

|  | Answer | | | | Code | | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | Help. | Help.-SFT | Entail. | BLEU | Info.-SFT | # lines | # API | Error |
| SFT | 11.54 | 50.00 | 4.47 | 14.60 | 50.00 | 41.17 | 134 | 3.08 |
| Daco-RL | 21.05 | 58.49 | 5.98 | 11.80 | 57.86 | 39.66 | 145 | 2.75 |
| w/o $R_{r}$ | 18.75 | 51.92 | 3.21 | 13.17 | 52.56 | 32.54 | 120 | 3.84 |
| w/o $R_{c},R_{r}$ | 8.79 | 40.00 | 3.13 | 11.46 | 44.57 | 14.30 | 108 | 3.91 |

*Table 3: Ablation study of regularization RM ($R_{r}$) and contribution RM ($R_{c}$) in Daco-RL. We report helpfulness (Help.), entailment (Entail.) and BLEU scores evaluated on TestH. We also compare the helpfulness of each model directly against SFT and report the win rate (Help.-SFT). For evaluating code generation, we report the informativeness win rate over SFT generations (Info.-SFT), number of code lines (# lines), number of different API (# API), and code error rate per step. Pair-wise comparison results are all obtained from ChatGPT.*

|  | Pairwise comparison | | Point- |
| --- | --- | --- | --- |
|  | Human | ChatGPT | wise |
| GPT-4 code gen v.s. | 66.41 | 70.07 | 1.45 |
| GPT-4 w/o code gen | 33.59 | 29.93 | 1.36 |
| Daco-RL v.s. | 57.72 | 58.49 | 1.42 |
| SFT | 42.28 | 41.51 | 1.30 |

*Table 4: Human evaluation. We report human-rated and ChatGPT-rated helpfulness pairwise comparison of two pairs of models: GPT-4 with v.s. without code generation, and Daco-RL v.s. SFT. We also report point-wise evaluation scores scaled into 0 $\sim$ 2 rated by human annotators.*

| Top 4 APIs | | Bottom 4 APIs | |
| --- | --- | --- | --- |
| API | Corr. | API | Corr. |
| print | 44.24 | to_datetime | -18.96 |
| nlargest | 20.06 | isnull | -17.76 |
| mean | 14.56 | describe | -12.02 |
| sort_values | 12.23 | merge | -10.83 |

*Table 5: APIs ranked by its correlation with contribution RM scores. Higher correlation means that contribution RM assigns higher scores to code snippets containing the API.*

The main results are in Table [2](#S4.T2 "Table 2 ‣ 4.1 Results ‣ 4 Experiments ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
We have the following observations:

Code generation significantly helps data analysis, especially for zero-shot LLMs. ChatGPT and GPT-4 both enjoy a significant gain in most metrics, especially helpfulness, when equipped with code generation. As in Table [5](#S4.T5 "Table 5 ‣ 4.1 Results ‣ 4 Experiments ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"), human evaluation further shows that GPT-4 with code generation has a significant 66.41 win rate over GPT-4 w/o code generation.
After SFT, code generation brings less significant improvements, because SFT w/o code generation can simulate the behavior in GPT annotations and achieve competitive helpfulness. However, its mathematical and logical reasoning are not supported by code generation, so it produces more hallucination as reflected by the low entailment score on TestH.

Our SFT model learns reasonable data analysis capabilities. By simulating GPT-4 behaviors, SFT with code generation achieves a reasonable helpfulness score and outperforms the TAPEX baseline, but still falls short compared to ChatGPT.
We also evaluate the error rate (%) of generated code per step. Our SFT model has an error rate of 3.08%, which is reasonable low, but still much higher than ChatGPT (0.495%) and GPT-4 (0.491%).

Daco-RL significantly improves over SFT. As shown in Table [2](#S4.T2 "Table 2 ‣ 4.1 Results ‣ 4 Experiments ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation") and [3](#S4.T3 "Table 3 ‣ 4.1 Results ‣ 4 Experiments ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"), Daco-RL significantly boosts the performance. Despite the difference in model size, DACO-RL outperforms ChatGPT w/o code generation on helpfulness and entailment metrics. When matched in size, DACO-RL significantly outperforms SFT by 7 points on helpfulness, further demonstrating its benefits. Human evaluation demonstrates a 57.72 win rate of Daco-RL over SFT as in Table [5](#S4.T5 "Table 5 ‣ 4.1 Results ‣ 4 Experiments ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). Our qualitative analysis shows that Daco-RL better focuses on user query, while SFT tends to display generic statistics that are less relevant to user query. An example is shown in Figure [6](#A3.F6 "Figure 6 ‣ Appendix C Qualitative Examples ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").

We further perform ablation study and report the results in Table [3](#S4.T3 "Table 3 ‣ 4.1 Results ‣ 4 Experiments ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). To directly compare the ablation models against SFT, we report the win rate of each model over SFT model.
To evaluate the quality of code generation, we use ChatGPT to compare the “informativeness” of generated code against SFT, where informativeness refers to producing informative and insightful code execution outputs while staying relevant to the user query.
We report a few additional statistics including the number of code lines and the number of different API calls. We observe that without our proposed two reward models (Daco-RL w/o $R_{c},R_{r}$), using only answer RM significantly hurts the model generation, leading to short and less diverse code generation and thus less helpful final answers. Contribution RM and regularization RM encourage more diverse code generation and more helpful final answer production.

Contribution RM favors API calls that extracts important information from tabular data but is also vulnerable to reward hacking. We report the Pearson correlation between API occurrence
and contribution RM scores in Table [5](#S4.T5 "Table 5 ‣ 4.1 Results ‣ 4 Experiments ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). The functions rewarded most are related to extracting significant features (nlargest, sort_values), aggregating results (mean), and displaying specific information (print).
In contrast, the least rewarded functions involve displaying generic statistics (describe) and wrangling data (merge, to_datetime, is_null) since they cannot directly contribute to the user query.
Examples of generated code and their contribution RM scores are shown in Figure [7(a)](#A3.F7.sf1 "In Figure 7 ‣ Appendix C Qualitative Examples ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation") and [7(b)](#A3.F7.sf2 "In Figure 7 ‣ Appendix C Qualitative Examples ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
However, we notice the concerningly high correlation between print function and contribution RM scores, which indicates the RL policy may exploit the correlation to hack reward. Figure [7(c)](#A3.F7.sf3 "In Figure 7 ‣ Appendix C Qualitative Examples ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation") shows a typical reward hacking case, where the model achieves a high contribution RM score by printing. Our regularization RM learns to discourage such behavior and helps fix the gap.

Evaluation on external test sets. To further analyze the effectiveness of Daco-RL, we evaluate our SFT and Daco-RL models on two external test sets: (1) data analysis benchmark InfiAgent-DA *(Hu et al., [2024](#bib.bib13 ""))* (a concurrent work of ours), and (2) free-form table question answering dataset FeTaQA *(Nan et al., [2022](#bib.bib23 ""))*.
We find that Daco-RL improves the accuracy over SFT on InfiAgent-DA (14.61 v.s. 12.92), especially over questions about summary statistics (14.86 v.s. 10.80) and correlation analysis (21.57 v.s. 14.86), which aligns with our evaluation results on Daco-RL dataset. On FeTaQA, Daco-RL retains similar performance (6.35 Rouge-L, 80.74 BERTScore) compared to SFT (6.39 Rouge-L, 80.68 BERTScore) since Daco-RL is not specifically trained to enhance information lookup capabilities.

5 Related Work
--------------

Table Analysis. Early work in table question answering (table QA) targets simple questions that requires table lookup and cell aggregations *(Pasupat \& Liang, [2015](#bib.bib28 ""); Zhong et al., [2017](#bib.bib40 ""); Iyyer et al., [2017](#bib.bib15 ""); Yu et al., [2018](#bib.bib37 ""); Nan et al., [2022](#bib.bib23 ""))*. Later benchmarks further require free-form answer generation *(Nan et al., [2022](#bib.bib23 ""))*, multi-hop reasoning *(Chen et al., [2021a](#bib.bib6 ""); [2020](#bib.bib5 ""))* and mathematical reasoning *(Zhu et al., [2021](#bib.bib41 ""); Chen et al., [2021b](#bib.bib7 ""); Lu et al., [2023](#bib.bib22 ""))*.
Despite the similar formulation between our task and existing table QA work, their focus are different: most existing table QA datasets focus on obtaining specific information, our data analysis queries can be complex and requires query decomposition and reasoning.
Some concurrent work further targets comprehensive table analysis such as correlation analysis and causal reasoning *(Nan et al., [2023](#bib.bib24 ""); Hu et al., [2024](#bib.bib13 ""); Liu et al., [2024](#bib.bib20 ""))*. The main difference between this work to the concurrent work is our focus on addressing application-driven user queries.

Code Generation. Code generation benchmarks have been proposed for general-purpose programming *(Austin et al., [2021](#bib.bib1 ""); Hendrycks et al., [2021](#bib.bib11 ""))*, math problems *(Austin et al., [2021](#bib.bib1 ""))*, and data science scenario *(Lai et al., [2022](#bib.bib16 ""); Huang et al., [2022](#bib.bib14 ""))*. Similar to our work, some recent work allows the language model to interact with a code execution environment and receive execution outputs as feedback *(Yang et al., [2023](#bib.bib36 ""); Wang et al., [2023](#bib.bib34 ""))*.
The most relevant work is *Cheng et al. ([2023](#bib.bib8 ""))* that also addresses data analysis via code generation. Given a data analysis query, they use GPT-4 to first generate code and then provide an interpretation of the execution results. While their analysis queries are still relatively simple, this is an early exploration aiming at automating data analysis.

RLHF. Reinforcement learning from human feedback (RLHF) aims to optimize a language model against human preference *(Ouyang et al., [2022](#bib.bib26 ""); Touvron et al., [2023](#bib.bib33 ""); Bai et al., [2022a](#bib.bib2 ""); [b](#bib.bib3 ""); Ziegler et al., [2019](#bib.bib42 ""); Wu et al., [2023](#bib.bib35 ""))*.
While traditionally RLHF uses a holistic reward score for the entire generation *(Ziegler et al., [2019](#bib.bib42 ""); Ouyang et al., [2022](#bib.bib26 ""))*, recent work shows that dense reward scores for the intermediate reasoning steps are better learning signals *(Lightman et al., [2023](#bib.bib18 ""); Wu et al., [2023](#bib.bib35 ""))*. These work uses expensive human annotation to collect annotations for the dense reward data.
Compared to human preference, heuristic rewards are more accessible but may not align well with true reward. This gap can lead to reward hacking *(Skalse et al., [2022](#bib.bib32 ""); Pan et al., [2022](#bib.bib27 ""))*. A common remedy is to use manually designed heuristics to penalize behaviors that potentially harm the true reward *(Ouyang et al., [2022](#bib.bib26 ""); Laidlaw et al., [2023](#bib.bib17 ""))*. In this work, we train the regularizatoin RM to discourage reward hacking.

6 Conclusion
------------

In this work, we propose a novel and challenging data analysis task, which involves decomposing user query into multiple perspectives, grounding each perspective to the input data and performing logical and mathematical reasoning. To support this task, we build the Daco dataset containing large-scale annotations automatically generated by GPT-4 and a small but high-quality test set with human curated annotations. We employ LLM enhanced with code generation to this task and evaluate three models on our dataset: zero-shot ChatGPT, zero-shot GPT-4 and a 6B SFT model. While GPT-4 consistently performs the best, SFT achieves reasonably good helpfulness with much less computation. On top of the SFT model, we further proposed our Daco-RL algorithm that significantly boosts the human evaluated helpfulness.

References
----------

* Austin et al. (2021)Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk
Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V.
Le, and Charles Sutton.Program synthesis with large language models.*CoRR*, abs/2108.07732, 2021.URL [https://arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732 "").
* Bai et al. (2022a)Yuntao Bai, Andy Jones, Kamal Ndousse, Amanda Askell, Anna Chen, Nova DasSarma,
Dawn Drain, Stanislav Fort, Deep Ganguli, Tom Henighan, Nicholas Joseph,
Saurav Kadavath, Jackson Kernion, Tom Conerly, Sheer El Showk, Nelson Elhage,
Zac Hatfield-Dodds, Danny Hernandez, Tristan Hume, Scott Johnston, Shauna
Kravec, Liane Lovitt, Neel Nanda, Catherine Olsson, Dario Amodei, Tom B.
Brown, Jack Clark, Sam McCandlish, Chris Olah, Benjamin Mann, and Jared
Kaplan.Training a helpful and harmless assistant with reinforcement learning
from human feedback.*CoRR*, abs/2204.05862, 2022a.doi: 10.48550/ARXIV.2204.05862.URL [https://doi.org/10.48550/arXiv.2204.05862](https://doi.org/10.48550/arXiv.2204.05862 "").
* Bai et al. (2022b)Yuntao Bai, Saurav Kadavath, Sandipan Kundu, Amanda Askell, Jackson Kernion,
Andy Jones, Anna Chen, Anna Goldie, Azalia Mirhoseini, Cameron McKinnon,
Carol Chen, Catherine Olsson, Christopher Olah, Danny Hernandez, Dawn Drain,
Deep Ganguli, Dustin Li, Eli Tran-Johnson, Ethan Perez, Jamie Kerr, Jared
Mueller, Jeffrey Ladish, Joshua Landau, Kamal Ndousse, Kamile Lukosiute,
Liane Lovitt, Michael Sellitto, Nelson Elhage, Nicholas Schiefer,
Noemí Mercado, Nova DasSarma, Robert Lasenby, Robin Larson, Sam
Ringer, Scott Johnston, Shauna Kravec, Sheer El Showk, Stanislav Fort, Tamera
Lanham, Timothy Telleen-Lawton, Tom Conerly, Tom Henighan, Tristan Hume,
Samuel R. Bowman, Zac Hatfield-Dodds, Ben Mann, Dario Amodei, Nicholas
Joseph, Sam McCandlish, Tom Brown, and Jared Kaplan.Constitutional AI: harmlessness from AI feedback.*CoRR*, abs/2212.08073, 2022b.doi: 10.48550/ARXIV.2212.08073.URL [https://doi.org/10.48550/arXiv.2212.08073](https://doi.org/10.48550/arXiv.2212.08073 "").
* Casper et al. (2023)Stephen Casper, Xander Davies, Claudia Shi, Thomas Krendl Gilbert,
Jérémy Scheurer, Javier Rando, Rachel Freedman, Tomasz Korbak,
David Lindner, Pedro Freire, Tony Wang, Samuel Marks, Charbel-Raphaël
Ségerie, Micah Carroll, Andi Peng, Phillip J. K. Christoffersen, Mehul
Damani, Stewart Slocum, Usman Anwar, Anand Siththaranjan, Max Nadeau, Eric J.
Michaud, Jacob Pfau, Dmitrii Krasheninnikov, Xin Chen, Lauro Langosco, Peter
Hase, Erdem Biyik, Anca D. Dragan, David Krueger, Dorsa Sadigh, and Dylan
Hadfield-Menell.Open problems and fundamental limitations of reinforcement learning
from human feedback.*CoRR*, abs/2307.15217, 2023.doi: 10.48550/ARXIV.2307.15217.URL [https://doi.org/10.48550/arXiv.2307.15217](https://doi.org/10.48550/arXiv.2307.15217 "").
* Chen et al. (2020)Wenhu Chen, Hanwen Zha, Zhiyu Chen, Wenhan Xiong, Hong Wang, and William Yang
Wang.HybridQA: A dataset of multi-hop question answering over tabular
and textual data.In *Findings of the Association for Computational Linguistics:
EMNLP 2020*, pp. 1026–1036, Online, November 2020. Association for
Computational Linguistics.doi: 10.18653/v1/2020.findings-emnlp.91.URL [https://aclanthology.org/2020.findings-emnlp.91](https://aclanthology.org/2020.findings-emnlp.91 "").
* Chen et al. (2021a)Wenhu Chen, Ming-Wei Chang, Eva Schlinger, William Yang Wang, and William W.
Cohen.Open question answering over tables and text.In *9th International Conference on Learning Representations,
ICLR 2021, Virtual Event, Austria, May 3-7, 2021*. OpenReview.net,
2021a.URL [https://openreview.net/forum?id\=MmCRswl1UYl](https://openreview.net/forum?id=MmCRswl1UYl "").
* Chen et al. (2021b)Zhiyu Chen, Wenhu Chen, Charese Smiley, Sameena Shah, Iana Borova, Dylan
Langdon, Reema Moussa, Matt Beane, Ting-Hao Huang, Bryan Routledge, and
William Yang Wang.FinQA: A dataset of numerical reasoning over financial data.In *Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing*, pp. 3697–3711, Online and Punta Cana,
Dominican Republic, November 2021b. Association for
Computational Linguistics.doi: 10.18653/v1/2021.emnlp-main.300.URL [https://aclanthology.org/2021.emnlp-main.300](https://aclanthology.org/2021.emnlp-main.300 "").
* Cheng et al. (2023)Liying Cheng, Xingxuan Li, and Lidong Bing.Is GPT-4 a good data analyst?*CoRR*, abs/2305.15038, 2023.doi: 10.48550/arXiv.2305.15038.URL [https://doi.org/10.48550/arXiv.2305.15038](https://doi.org/10.48550/arXiv.2305.15038 "").
* Ehrlinger \& Wöß (2022)Lisa Ehrlinger and Wolfram Wöß.A survey of data quality measurement and monitoring tools.*Frontiers in big data*, 5:850611, 2022.
* Grootendorst (2022)Maarten Grootendorst.Bertopic: Neural topic modeling with a class-based tf-idf procedure.*arXiv preprint arXiv:2203.05794*, 2022.
* Hendrycks et al. (2021)Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora,
Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob
Steinhardt.Measuring coding challenge competence with APPS.In Joaquin Vanschoren and Sai-Kit Yeung (eds.), *Proceedings
of the Neural Information Processing Systems Track on Datasets and Benchmarks
1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual*, 2021.URL[https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html "").
* Herzig et al. (2020)Jonathan Herzig, Pawel Krzysztof Nowak, Thomas Müller, Francesco Piccinno,
and Julian Eisenschlos.TaPas: Weakly supervised table parsing via pre-training.In *Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics*, pp. 4320–4333, Online, July 2020.
Association for Computational Linguistics.doi: 10.18653/v1/2020.acl-main.398.URL [https://aclanthology.org/2020.acl-main.398](https://aclanthology.org/2020.acl-main.398 "").
* Hu et al. (2024)Xueyu Hu, Ziyu Zhao, Shuang Wei, Ziwei Chai, Guoyin Wang, Xuwu Wang, Jing Su,
Jingjing Xu, Ming Zhu, Yao Cheng, et al.Infiagent-dabench: Evaluating agents on data analysis tasks.*arXiv preprint arXiv:2401.05507*, 2024.
* Huang et al. (2022)Junjie Huang, Chenglong Wang, Jipeng Zhang, Cong Yan, Haotian Cui,
Jeevana Priya Inala, Colin Clement, and Nan Duan.Execution-based evaluation for data science code generation models.In *Proceedings of the Fourth Workshop on Data Science with
Human-in-the-Loop (Language Advances)*, pp. 28–36, Abu Dhabi, United Arab
Emirates (Hybrid), December 2022. Association for Computational Linguistics.URL [https://aclanthology.org/2022.dash-1.5](https://aclanthology.org/2022.dash-1.5 "").
* Iyyer et al. (2017)Mohit Iyyer, Wen-tau Yih, and Ming-Wei Chang.Search-based neural structured learning for sequential question
answering.In *Proceedings of the 55th Annual Meeting of the Association
for Computational Linguistics (Volume 1: Long Papers)*, pp. 1821–1831,
Vancouver, Canada, July 2017. Association for Computational Linguistics.doi: 10.18653/v1/P17-1167.URL [https://aclanthology.org/P17-1167](https://aclanthology.org/P17-1167 "").
* Lai et al. (2022)Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke
Zettlemoyer, Scott Wen-tau Yih, Daniel Fried, Sida I. Wang, and Tao Yu.DS-1000: A natural and reliable benchmark for data science code
generation.*CoRR*, abs/2211.11501, 2022.doi: 10.48550/arXiv.2211.11501.URL [https://doi.org/10.48550/arXiv.2211.11501](https://doi.org/10.48550/arXiv.2211.11501 "").
* Laidlaw et al. (2023)Cassidy Laidlaw, Shivam Singhal, and Anca Dragan.Preventing reward hacking with occupancy measure regularization.In *ICML Workshop on New Frontiers in Learning, Control, and
Dynamical Systems*, 2023.
* Lightman et al. (2023)Hunter Lightman, Vineet Kosaraju, Yura Burda, Harrison Edwards, Bowen Baker,
Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe.Let’s verify step by step.*CoRR*, abs/2305.20050, 2023.doi: 10.48550/arXiv.2305.20050.URL [https://doi.org/10.48550/arXiv.2305.20050](https://doi.org/10.48550/arXiv.2305.20050 "").
* Liu et al. (2022)Qian Liu, Bei Chen, Jiaqi Guo, Morteza Ziyadi, Zeqi Lin, Weizhu Chen, and
Jian-Guang Lou.TAPEX: table pre-training via learning a neural SQL executor.In *The Tenth International Conference on Learning
Representations, ICLR 2022, Virtual Event, April 25-29, 2022*.
OpenReview.net, 2022.URL [https://openreview.net/forum?id\=O50443AsCP](https://openreview.net/forum?id=O50443AsCP "").
* Liu et al. (2024)Xiao Liu, Zirui Wu, Xueqing Wu, Pan Lu, Kai-Wei Chang, and Yansong Feng.Are llms capable of data-based statistical and causal reasoning?
benchmarking advanced quantitative reasoning with data, 2024.
* Long \& Long (2009)J Scott Long and J Scott Long.*The workflow of data analysis using Stata*.Stata Press College Station, TX, 2009.
* Lu et al. (2023)Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay
Rajpurohit, Peter Clark, and Ashwin Kalyan.Dynamic prompt learning via policy gradient for semi-structured
mathematical reasoning.In *The Eleventh International Conference on Learning
Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net,
2023.URL [https://openreview.net/pdf?id\=DHyHRBwJUTN](https://openreview.net/pdf?id=DHyHRBwJUTN "").
* Nan et al. (2022)Linyong Nan, Chiachun Hsieh, Ziming Mao, Xi Victoria Lin, Neha Verma, Rui
Zhang, Wojciech Kryściński, Hailey Schoelkopf, Riley Kong, Xiangru
Tang, Mutethia Mutuma, Ben Rosand, Isabel Trindade, Renusree Bandaru, Jacob
Cunningham, Caiming Xiong, Dragomir Radev, and Dragomir Radev.FeTaQA: Free-form table question answering.*Transactions of the Association for Computational Linguistics*,
10:35–49, 2022.doi: 10.1162/tacl˙a˙00446.URL [https://aclanthology.org/2022.tacl-1.3](https://aclanthology.org/2022.tacl-1.3 "").
* Nan et al. (2023)Linyong Nan, Ellen Zhang, Weijin Zou, Yilun Zhao, Wenfei Zhou, and Arman Cohan.On evaluating the integration of reasoning and action in llm agents
with database question answering.*arXiv preprint arXiv:2311.09721*, 2023.
* OpenAI (2023)OpenAI.GPT-4 technical report.*CoRR*, abs/2303.08774, 2023.doi: 10.48550/ARXIV.2303.08774.URL [https://doi.org/10.48550/arXiv.2303.08774](https://doi.org/10.48550/arXiv.2303.08774 "").
* Ouyang et al. (2022)Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll L. Wainwright, Pamela
Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, John
Schulman, Jacob Hilton, Fraser Kelton, Luke Miller, Maddie Simens, Amanda
Askell, Peter Welinder, Paul F. Christiano, Jan Leike, and Ryan Lowe.Training language models to follow instructions with human feedback.In *NeurIPS*, 2022.URL[http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html](http://papers.nips.cc/paper_files/paper/2022/hash/b1efde53be364a73914f58805a001731-Abstract-Conference.html "").
* Pan et al. (2022)Alexander Pan, Kush Bhatia, and Jacob Steinhardt.The effects of reward misspecification: Mapping and mitigating
misaligned models.In *The Tenth International Conference on Learning
Representations, ICLR 2022, Virtual Event, April 25-29, 2022*.
OpenReview.net, 2022.URL [https://openreview.net/forum?id\=JYtwGwIL7ye](https://openreview.net/forum?id=JYtwGwIL7ye "").
* Pasupat \& Liang (2015)Panupong Pasupat and Percy Liang.Compositional semantic parsing on semi-structured tables.In *Proceedings of the 53rd Annual Meeting of the Association
for Computational Linguistics and the 7th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers)*, pp. 1470–1480,
Beijing, China, July 2015. Association for Computational Linguistics.doi: 10.3115/v1/P15-1142.URL [https://aclanthology.org/P15-1142](https://aclanthology.org/P15-1142 "").
* Reimers \& Gurevych (2019)Nils Reimers and Iryna Gurevych.Sentence-BERT: Sentence embeddings using Siamese BERT-networks.In *Proceedings of the 2019 Conference on Empirical Methods in
Natural Language Processing and the 9th International Joint Conference on
Natural Language Processing (EMNLP-IJCNLP)*, pp. 3982–3992, Hong Kong,
China, November 2019. Association for Computational Linguistics.doi: 10.18653/v1/D19-1410.URL [https://aclanthology.org/D19-1410](https://aclanthology.org/D19-1410 "").
* Schulman et al. (2016)John Schulman, Philipp Moritz, Sergey Levine, Michael I. Jordan, and Pieter
Abbeel.High-dimensional continuous control using generalized advantage
estimation.In Yoshua Bengio and Yann LeCun (eds.), *4th International
Conference on Learning Representations, ICLR 2016, San Juan, Puerto Rico,
May 2-4, 2016, Conference Track Proceedings*, 2016.URL [http://arxiv.org/abs/1506.02438](http://arxiv.org/abs/1506.02438 "").
* Schulman et al. (2017)John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov.Proximal policy optimization algorithms.*CoRR*, abs/1707.06347, 2017.URL [http://arxiv.org/abs/1707.06347](http://arxiv.org/abs/1707.06347 "").
* Skalse et al. (2022)Joar Skalse, Nikolaus H. R. Howe, Dmitrii Krasheninnikov, and David Krueger.Defining and characterizing reward hacking.*CoRR*, abs/2209.13085, 2022.doi: 10.48550/ARXIV.2209.13085.URL [https://doi.org/10.48550/arXiv.2209.13085](https://doi.org/10.48550/arXiv.2209.13085 "").
* Touvron et al. (2023)Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine
Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale,
Dan Bikel, Lukas Blecher, Cristian Canton-Ferrer, Moya Chen, Guillem
Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller,
Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar
Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa,
Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux,
Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier
Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew
Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan
Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang,
Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan
Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang,
Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom.Llama 2: Open foundation and fine-tuned chat models.*CoRR*, abs/2307.09288, 2023.doi: 10.48550/ARXIV.2307.09288.URL [https://doi.org/10.48550/arXiv.2307.09288](https://doi.org/10.48550/arXiv.2307.09288 "").
* Wang et al. (2023)Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and
Heng Ji.Mint: Evaluating llms in multi-turn interaction with tools and
language feedback, 2023.
* Wu et al. (2023)Zeqiu Wu, Yushi Hu, Weijia Shi, Nouha Dziri, Alane Suhr, Prithviraj
Ammanabrolu, Noah A. Smith, Mari Ostendorf, and Hannaneh Hajishirzi.Fine-grained human feedback gives better rewards for language model
training.*CoRR*, abs/2306.01693, 2023.doi: 10.48550/arXiv.2306.01693.URL [https://doi.org/10.48550/arXiv.2306.01693](https://doi.org/10.48550/arXiv.2306.01693 "").
* Yang et al. (2023)John Yang, Akshara Prabhakar, Karthik Narasimhan, and Shunyu Yao.Intercode: Standardizing and benchmarking interactive coding with
execution feedback.*CoRR*, abs/2306.14898, 2023.doi: 10.48550/arXiv.2306.14898.URL [https://doi.org/10.48550/arXiv.2306.14898](https://doi.org/10.48550/arXiv.2306.14898 "").
* Yu et al. (2018)Tao Yu, Rui Zhang, Kai Yang, Michihiro Yasunaga, Dongxu Wang, Zifan Li, James
Ma, Irene Li, Qingning Yao, Shanelle Roman, Zilin Zhang, and Dragomir Radev.Spider: A large-scale human-labeled dataset for complex and
cross-domain semantic parsing and text-to-SQL task.In *Proceedings of the 2018 Conference on Empirical Methods in
Natural Language Processing*, pp. 3911–3921, Brussels, Belgium,
October-November 2018. Association for Computational Linguistics.doi: 10.18653/v1/D18-1425.URL [https://aclanthology.org/D18-1425](https://aclanthology.org/D18-1425 "").
* Zheng et al. (2023a)Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang,
Lei Shen, Andi Wang, Yang Li, Teng Su, Zhilin Yang, and Jie Tang.Codegeex: A pre-trained model for code generation with multilingual
evaluations on humaneval-x.*CoRR*, abs/2303.17568, 2023a.doi: 10.48550/arXiv.2303.17568.URL [https://doi.org/10.48550/arXiv.2303.17568](https://doi.org/10.48550/arXiv.2303.17568 "").
* Zheng et al. (2023b)Rui Zheng, Shihan Dou, Songyang Gao, Yuan Hua, Wei Shen, Binghai Wang, Yan Liu,
Senjie Jin, Qin Liu, Yuhao Zhou, et al.Secrets of rlhf in large language models part i: Ppo.*arXiv preprint arXiv:2307.04964*, 2023b.
* Zhong et al. (2017)Victor Zhong, Caiming Xiong, and Richard Socher.Seq2sql: Generating structured queries from natural language using
reinforcement learning.*CoRR*, abs/1709.00103, 2017.URL [http://arxiv.org/abs/1709.00103](http://arxiv.org/abs/1709.00103 "").
* Zhu et al. (2021)Fengbin Zhu, Wenqiang Lei, Youcheng Huang, Chao Wang, Shuo Zhang, Jiancheng Lv,
Fuli Feng, and Tat-Seng Chua.TAT-QA: A question answering benchmark on a hybrid of tabular and
textual content in finance.In *Proceedings of the 59th Annual Meeting of the Association
for Computational Linguistics and the 11th International Joint Conference on
Natural Language Processing (Volume 1: Long Papers)*, pp. 3277–3287,
Online, August 2021. Association for Computational Linguistics.doi: 10.18653/v1/2021.acl-long.254.URL [https://aclanthology.org/2021.acl-long.254](https://aclanthology.org/2021.acl-long.254 "").
* Ziegler et al. (2019)Daniel M. Ziegler, Nisan Stiennon, Jeffrey Wu, Tom B. Brown, Alec Radford,
Dario Amodei, Paul F. Christiano, and Geoffrey Irving.Fine-tuning language models from human preferences.*CoRR*, abs/1909.08593, 2019.URL [http://arxiv.org/abs/1909.08593](http://arxiv.org/abs/1909.08593 "").

Appendix
--------

Appendix A Additional Analysis of Daco Dataset
----------------------------------------------

We perform additional analysis to verify the quality of our Daco dataset. We assess data quality based on comprehensiveness and agreement among annotators, which are two of the most commonly considered factors *(Ehrlinger \& Wöß, [2022](#bib.bib9 ""))*.

We first measure the overlap between input queries to verify the diversity of automatically generated input queries. We compute the overlap among multiple queries over the same database using cosine similarity of Sentence-BERT embeddings. We use the all-MiniLM-L6-v2 model. We find that 46% pairs of generated queries have large difference. A small portion (2%) are repetitive with high similarity; since the percentage is small, it should not seriously affect the dataset quality.
Details and qualitative examples are shown in Table [6](#A1.T6 "Table 6 ‣ Appendix A Additional Analysis of Daco Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").

|  | % | Example |
| --- | --- | --- |
| Large difference(similarity$<$0.5) | 46 | As a weather forecaster, I want to study the correlation between weather conditions and bike rentals. v.s. |
|  |  | As a tourist attraction planner, I want to evaluate the bike-sharing program’s impact on tourism and visitor satisfaction. |
|  |  | Similarity \= 0.42 |
| Medium difference(0.5$<$similarity$<$0.8) | 52 | As a farmer, I want to determine the suitable fruit varieties to grow on my farm. v.s. |
|  |  | As a fruit exporter, I want to identify the fruits that meet export standards and have a longer shelf life. |
|  |  | Similarity \= 0.69 |
| Repetitive(similarity$>$0.8) | 2 | As a consultant for honey market, I want to study the honey production trend to recommend business strategies for my clients. v.s. |
|  |  | As a curious analyst, I want to study the production trend to understand the US honey industry. |
|  |  | Similarity \= 0.85 |

*Table 6: Cosine similarity and qualitative examples of pairs of input queries.*

We further evaluate the comprehensiveness of input queries, i.e. how many data columns are covered by the analysis. We apply heuristic rules to measure . On average, each analysis covers 71% data columns in the corresponding database. Among all data columns, 90% are covered by at least one data point. This verifies that our dataset achieves good coverage of the database columns.

Regarding agreement among annotators, as mentioned in the main content, the machine-generated queries are filtered by human annotators with a 0.62 Cohen’s kappa score, and our manual refinement of the test set also achieves a substantial 0.67 Cohen’s kappa score. These also verify the quality of our Daco dataset.

Appendix B Implementation Details
---------------------------------

For zero-shot API-based systems including ChatGPT and GPT-4, we evaluate two settings, directly reading the table content, and using code generation.
For the former setting, we linearize the table content into text representation as model input. Due to token limit, we feed the first 20 rows as input, which covers the full content of 93% tables.
For the code generation setting, we employ the pipeline described in Figure [5](#S2.F5 "Figure 5 ‣ 2 The Daco Task and Dataset ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). When the generated code causes a syntax or runtime error, we re-sample the model until the generated code can be executed. We allow up to 5 resamplings for each turn.
We use the gpt-3.5-turbo-16k-0613 API for ChatGPT and gpt-4-32k API for GPT-4.
We limit the number of total coding turns maximally at 9. For annotation generation where GPT-4 self-correction is allowed, we limit the number of self-correction within 2 for each turn and 4 for the whole session.

For finetuned models including Daco-RL and SFT, we use CodeGeeX2-6B *(Zheng et al., [2023a](#bib.bib38 ""))* as the base model. We first train the SFT model using GPT-4 annotations, and then train our Daco-RL model on top of the SFT model. When training $R_{a+c}$ and $R_{r}$, we initialize the model from the SFT model. When training our Daco-RL model, we initialize the value model $V$ from $R_{a+c}$, and initialize the policy model $\pi$ from the SFT model. In inference, we use nucleus decoding with p \= 0.9 and temperature \= 1.0. Similarly, we allow up to 5 resamplings when the generated code causes an error.
The SFT model is trained with 8 A100 GPU for about 4 hours. The Daco-RL model is trained with 8 A100 GPU for about 18 hours.
Detailed hyper-parameters are in Table [7](#A2.T7 "Table 7 ‣ Appendix B Implementation Details ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). The only hyper-parameter we tune is $\lambda$ for Daco-RL. We experiment with 0.8, 0.9 and 1.0 and discover that 1.0 works the best.

|  | SFT | RL |
| --- | --- | --- |
| learning rate | 1e-5 | 2e-6 |
| gradient accumulation | 4 | 4 |
| total steps | 600 | 200 |
| $\lambda$ | - | 1.0 |
| $\gamma$ | - | 1.0 |

*Table 7: Hyperparameters.*

For experiments on external test sets, we directly evalute the trained SFT and Daco-RL model on InfiAgent-DA *(Hu et al., [2024](#bib.bib13 ""))* and FeTaQA *(Nan et al., [2022](#bib.bib23 ""))* test sets. For InfiAgent-DA, following the original paper, we add a reformatting step to reformat the generated data analysis report into the key-value format. We use ChatGPT to perform reformatting with a simplified prompt without in-context examples.

Appendix C Qualitative Examples
-------------------------------

We show final answers generated by SFT and Daco-RL in Figure [6](#A3.F6 "Figure 6 ‣ Appendix C Qualitative Examples ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). Daco-RL better focuses on user query, while SFT tends to display generic statistics that are less relevant to user query.

We show examples of code generations in Figure [7](#A3.F7 "Figure 7 ‣ Appendix C Qualitative Examples ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation"). We also report their reward scores from contribution RM and regularization RM.

<img src='figures/answer_case_study.png' alt='Refer to caption' title='' width='598' height='363' />

*Figure 6: Case study.*

<img src='figures/case_good.png' alt='Refer to caption' title='' width='598' height='294' />

*(a) A good case that receives high scores from both contribution RM and regularization RM.*

<img src='figures/case_bad.png' alt='Refer to caption' title='' width='598' height='212' />

*(b) A bad case that receives low score from contribution RM and high score from regularization RM.*

<img src='figures/case_hack.png' alt='Refer to caption' title='' width='598' height='174' />

*(c) A reward hacking case that receives high score from contribution RM and low score from regularization RM.*

*Figure 7: Qualitative examples of code generations, and their scores assigned by reward models.*

Appendix D GPT Prompts
----------------------

Here we show the prompts we use for ChatGPT and GPT-4. Prompt for query generation is in Table [8](#A4.T8 "Table 8 ‣ Appendix D GPT Prompts ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
Prompt for helpfulness annotation collection is Table [9](#A4.T9 "Table 9 ‣ Appendix D GPT Prompts ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").
Prompts for helpfulness and informativeness evaluation are Table [10](#A4.T10 "Table 10 ‣ Appendix D GPT Prompts ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation") and [11](#A4.T11 "Table 11 ‣ Appendix D GPT Prompts ‣ Daco: Towards Application-Driven and Comprehensive Data Analysis via Code Generation").

| I have a database of [database title]. I am a stakeholder and I am analyzing the database to make a decision. Who am I and what decision might it be? List 10 possibilities in a numbered list.Each point should introduce who I am and briefly explain my intention in this format: As a/the [who I am], I want to [explain my intention]Examples:Based on the extracurricular activities database:1. As the dean of student affairs, I want to decide on extracurricular activities to promote or cut2. As the department head, I want to decide on faculty advisor assignments3. As the school administrator, I want to review and revise faculty activity engagementBased on a diabetes database:1. As a healthcare policy maker, I want to decide on healthcare resource allocation2. As a NIH official, I want to decide on medical research funding3. As a health insurance actuary, I want to improve health insurance pricing strategy4. As a health provider, I want to decide on patient care and treatmentBased on an allergy database:1. As a catering manager, I want to plan meal options2. As the school principal, I want to plan allergy awareness programs3. As an administrator in the Student Affairs or Housing department, I want to decide on housing assignments4. As the school administrator, I want to improve campus emergency preparedness5. As the school principal, I want to develop policies for allergy accommodationsBased on a Home Equity Line of Credit (HELOC) product database, you can:1. As the credit risk manager, I want to modify the credit underwriting policyThe database is as follows:Database `[title]`has [x] tables. Table names are: [aaa], [bbb], [ccc]Table `[caption]`has [x] rows and [y] columns. Column are:`[column name]`, example values: [value 1], [value 2], [value 3], [value 4], [value 5]… |
| --- |

*Table 8: Prompt for query collection.*

| I have a database of [database title]. As a [stakeholder role], I want to [describe intention]. |
| --- |
| Given below two findings/conclusions, which one is more helpful to my analysis? |
| Your response should be in the following format: |
| Reasoning: $<$explain your reasoning here$>$ |
| Answer: $<$repeat the more helpful finding here$>$ |

*Table 9: Prompt for helpfulness annotation collection.*

| I have a database of [database title]. As a [stakeholder role], I want to [describe intention].I have hired two data analysts to perform the analysis, and they gave me two different reports (listed below). Each report consists of two lists, one for findings and one for suggestions. Which one is more helpful to my analysis? When evaluating helpfulness, you should consider the following three rubrics in decreasing priority: (1) relevance to my analysis goal; (2) insightfulness; and (3) diversity of perspectives, especially for suggestions.Your response should be in the following format. Note: $<$answer$>$ should be either Report-1 or Report-2* Answer: $<$answer$>$* Reasoning: $<$explain your reasoning here$>$The reports are as follows:# Report-1[report 1]# Report-2[report 2] |
| --- |

*Table 10: Prompt for helpfulness evaluation.*

| I have a database of [database title]. As a [stakeholder role], I want to [describe intention]. Below are the intermediate steps of their analysis. Which analysis is more informative? The more informative analysis should produce execution results that stick relevant to my analysis goal and bring more insights to my analysis.Your response should be in the following format. Note: $<$answer$>$ should be either Analysis-1 or Analysis-2* Answer: $<$answer$>$* Reasoning: $<$explain your reasoning here$>$The reports are as follows:# Analysis-1[analysis 1]# Analysis-2[analysis 2] |
| --- |

*Table 11: Prompt for informativeness evaluation.*
