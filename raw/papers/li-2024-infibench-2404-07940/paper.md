# Infibench: Evaluating the Question-Answering Capabilities of Code Large Language Models

Linyi Li

Simon Fraser University

linyi_li@sfu.ca

Shijie Geng

ByteDance Inc & Rutgers University

sg1309@rutgers.edu

Zhenwen Li* Yibo He* Hao Yu*

Peking University

Ziyue Hua*

Guanghan Ning

Siwei Wang

ByteDance Inc

Tao Xie

Key Lab of HCST (PKU), MOE

taoxie@pku.edu.cn

Hongxia Yang

The Hong Kong Polytechnic University (PolyU)

hongxia.yang@polyu.edu.hk

# Abstract

Large Language Models for code (code LLMs) have witnessed tremendous progress in recent years. With the rapid development of code LLMs, many popular evaluation benchmarks, such as HumanEval, DS-1000, and MBPP, have emerged to measure the performance of code LLMs with a particular focus on code generation tasks. However, they are insufficient to cover the full range of expected capabilities of code LLMs, which span beyond code generation to answering diverse coding-related questions. To fill this gap, we propose InfiBench, the first large-scale freeform question-answering (QA) benchmark for code to our knowledge, comprising 234 carefully selected high-quality Stack Overflow questions that span across 15 programming languages. InfiBench uses four types of model-free automatic metrics to evaluate response correctness where domain experts carefully concretize the criterion for each question. We conduct a systematic evaluation for over 100 latest code LLMs on InfiBench, leading to a series of novel and insightful findings. Our detailed analyses showcase potential directions for further advancement of code LLMs. InfiBench is fully open source at https://infi-coder.github.io/infibench and continuously expanding to foster more scientific and systematic practices for code LLM evaluation.

# 1 Introduction

In recent years, Large Language Models (LLMs) have been revolutionizing the software development landscape [Hou et al., 2023, Fan et al., 2023], demonstrating exceedingly strong and comprehensive capabilities in comprehending, generating, debugging, and summarizing code [Chen et al., 2021, Li et al., 2022]. For example, code LLM-powered products like GitHub Copilot [Github, 2023] reached millions of active users within just one year of their launch.

Alongside the huge success of proprietary LLMs such as GPT-3.5 / GPT-4 [OpenAI, 2023] and Gemini [Gemini Team et al., 2023], the development of open-source code LLMs² [Nijkamp et al.,

Figure 1: InfiBench overview. We construct the InfiBench benchmark by filtering high-quality and diverse question posts from Stack Overflow and annotating question-level evaluation criteria with domain experts. With an model-free automatic evaluation framework, we evaluate over 100 latest code LLMs (one of the most extensive evaluations for code LLMs to the best of our knowledge), leading to several insightful findings.

2023, Touvron et al., 2023b, Roziere et al., 2023, Luo et al., 2023] has been advancing at an unprecedented fast pace. As of June 2024, the Hugging Face Open LLM Leaderboard [Beeching et al., 2023] has cataloged over 3,300 submissions of such models.

Figure 2: A challenging question paraphrased from Stack Overflow where GPT-4 fails to answer.

Given the plethora of code LLMs available, the development of reliable code benchmarks seems to lag in four aspects: (1) Benchmarks beyond code generation are relatively few. Benchmarks for code LLMs typically focus on a specific task or domain, often overly focus on code generation. For example, the widely-used HumanEval [Chen et al., 2021] and MBPP [Austin et al., 2021] purely focus on Python code generation, and DS-1000 [Lai et al., 2023] focuses on Python code generation in the field of data science. (2) Independent code benchmarks are relatively few. Recent efforts evolve existing benchmarks (e.g., HumanEval) to include more scenarios [Muennighoff et al., 2023], languages [Zheng et al., 2023], and tests [Liu et al., 2023a]. However, these efforts lead to a series of benchmarks sharing the same source data (e.g., HumanEval Python problems), reducing score independence. (3) Existing code benchmarks are saturating. Strong LLMs are saturating existing benchmarks, e.g., GPT-4 has already achieved  $90.2\%$  Pass@1 score on Hu

manEval [OpenAI, 2024a], while in real-world scenarios, GPT-4 can still fail as exemplified in Figure 2. (4) Common benchmarks may be contaminated. Some LLMs have unconventional high performance in common benchmarks and are suspected to have memorized benchmark-related data [Dekoninck et al., 2024, Xu et al., 2024, Matton et al., 2024], obscuring the evaluation results. Can we systematically and comprehensively evaluate code LLMs' abilities in challenging real-world usage scenarios?

To answer the question, we introduce InfiBench, a systematic benchmark for evaluating the free-form question-answering capabilities of code LLMs. As the first benchmark of its kind, the core principle of InfiBench aims to accurately represent how developers interact with and utilize such models in real-world scenarios. To achieve this, InfiBench comprises 234 questions that are carefully selected and proportionally filtered from the natural high-quality question distribution of Stack Overflow, without any constraints on topics, programming languages, question types, or answer forms. As

Table 1: Comparison between InfiBench and common existing benchmarks. Existing benchmarks weigh heavily on code generation, unit-test-based evaluation, and major programming languages. InfiBench processes a much higher diversity to reflect real-world code LLMs' usage scenarios. More discussion in Section 2.6.  

<table><tr><td>Benchmark</td><td>Domain</td><td># Question</td><td>Evaluation</td><td>Data Source</td><td>Highest LLM Score</td></tr><tr><td>HumanEval [Chen et al., 2021]</td><td>Python Programming</td><td>164</td><td>Test Cases</td><td>Hand-Written</td><td>90.2%</td></tr><tr><td>MBPP [Austin et al., 2021]</td><td>Python Programming</td><td>974</td><td>Test Cases</td><td>Hand-Written</td><td>81.1%</td></tr><tr><td>APPS [Hendrycks et al., 2021]</td><td>Python Programming</td><td>10,000</td><td>Test Cases</td><td>Competitions</td><td>(no report yet)</td></tr><tr><td>DS-1000 [Lai et al., 2023]</td><td>Python Programming</td><td>1,000</td><td>Test Cases + Surface Form Constraints</td><td>StackOverflow</td><td>(no report yet)</td></tr><tr><td>HumanEval+ [Liu et al., 2023a]</td><td>Python Programming</td><td>164</td><td>Augmented Test Cases</td><td>HumanEval</td><td>86.6%</td></tr><tr><td>HumanEvalPack [Muenninghoff et al., 2023]</td><td>Repair, Explain, Generation in 6 Languages</td><td>2,952</td><td>Test Cases</td><td>HumanEval</td><td>47.8%/52.1%/78.3%</td></tr><tr><td>LBPP [Matton et al., 2024]</td><td>Python Programming</td><td>161</td><td>Test Cases</td><td>Hand-Written</td><td>64%</td></tr><tr><td>SWE-bench [Jimenez et al., 2024]</td><td>Python Debugging / Repair</td><td>2,294</td><td>Test Cases</td><td>GitHub</td><td>22.06%</td></tr><tr><td>SWE-bench Verified [OpenAI, 2024b]</td><td>Python Debugging / Repair</td><td>500</td><td>Test Cases</td><td>SWE-bench</td><td>45.20%</td></tr><tr><td>InfiBench</td><td>Free-Form Code Question Answering in 15 Languages</td><td>234</td><td>Keyword + Blank Filling + Test Cases + Text Similarity</td><td>Stack Overflow</td><td>70.64%</td></tr></table>

a result, the curated 234 questions span 15 programming languages and 5 major areas: front-end, back-end, DS&ML (data science and machine learning), mobile and desktop, and ITOps (information technology operations).

Question diversity comes with evaluation challenges for two reasons. (1) Lack of metric. Unlike code generation or multiple-choice benchmarks, which can be evaluated through standardized methods like unit testing, there is no universal metric for response correctness for free-form questions. (2) Challenges with model-based evaluation. Model-based evaluations such as those involving GPT-4 are not only costly but also raise concerns about privacy and bias.

To mitigate the evaluation challenges, InfiBench includes an automatic evaluation framework that integrates four types of model-free metrics: keyword matching, blank filling, unit testing, and dialogue similarity. For each question, we invite industry domain experts to paraphrase the prompt, select the most appropriate metric, and write down the concrete criteria using domain-specific knowledge, with highly-voted answers from Stack Overflow as a reference. These questions and evaluation criteria are then cross-validated to ensure correctness and objectiveness and further calibrated to improve consistency across languages. Human experiments show that InfiBench evaluation aligns with humans better than LLM-based evaluation, achieving  $85.1\%$  agreement rate compared to  $77.8\%$  achieved by GPT-4o-based evaluation.

As a novel and systematic benchmark disjoint with existing ones in terms of both forms and data sources, we believe that InfiBench is an ideal tool to measure existing code LLMs objectively. Hence, we conduct a systematic evaluation for over 100 code LLMs spanning both proprietary and open-source worlds using the InfiBench framework — the latest and most extensive evaluation for code LLMs to the best of our knowledge. Our evaluation leads to several insightful findings: (1) On InfiBench, GPT-4 achieves a score of  $70.64\%$ , being far from perfect but still far exceeding the most capable open-source models as of June 2024. On the other hand, GPT3.5 is surpassed by a few open-source models. (2) At similar model sizes, coding LLMs are usually visibly stronger than general LLMs; finetuning LLMs are usually visibly stronger than base LLMs. (3) The performance differences between different model families are huge, where one model could surpass another with less than 1/10 parameters, highlighting the importance of training data quality and techniques. (4) The scaling law is empirically verified for open-source models with fewer than 40B parameters, but not for those with more, where a turning point emerges. InfiBench is fully open source under CC BY-SA 4.0 license and continuously expanding<sup>3</sup>, including both the benchmark and Hugging-Face-compatible evaluation tools. All resources are available at https://infi-coder.github.io/infibench.

# 2 Benchmark Creation

InfiBench is created from a high-quality subset of Stack Overflow questions up until June 14, 2023. In this section, we describe the data curation process and the evaluation framework in detail.

# 2.1 Data Curation

Stack Overflow is a question-and-answer website for developers with more than 24 million registered users as of June 2024 [StackExchange, 2024]. Since the website is a large collection of natural and diverse coding questions from real-world developers, we believe that questions from Stack Overflow can effectively evaluate code LLM's capabilities in real-world usage scenarios.

The full Stack Overflow dataset contains 23.54 million question posts and 34.68 million answer posts. Each question post has a total view count. Each answer post is attached to a question and has a vote count. The question creator can choose one answer as officially accepted.

As we aim to create a benchmark where the correctness evaluation criteria are clear, we view the positively voted answers as an important reference source. Hence, we choose to keep only the questions that have at least three positively voted answers and an officially accepted answer, which turn out to be 1,090,238 questions. For these one million questions, we choose to keep questions that are frequently viewed and relatively new. To fulfill this criterion, we draw a scatter plot of these  $\approx 1$  million questions, plotting the number of days since their creation until June 14, 2023 (data collection end-date) on the  $x$ -axis against the logarithm of their view counts on the  $y$ -axis. As shown in Figure 3, we empirically determine to keep questions that lie above the line connecting (0,5) and (3000, 15.5), resulting in a subset of 17,402 questions.

Utilizing the mandatory question tags of these questions, we then manually construct a tag tree that covers the 200 most frequent tags, enabling us to identify the top programming languages and areas for 14,330 out of these 17,402 questions. These questions are from 24 programming languages, with each language being categorized into one primary area among the five (front-end, back-end, DS&ML, mobile and desktop, and ITOps). Lastly, we exclude 6 programming languages that either describe data or are domain-specific: JSON, regex, Markdown, YAML, CSV, and SQL. As a result, we compile 13,854 questions that serve as the initial seed set.

Figure 3: Scatter plot of filtered Stack Overflow questions. Questions above the orange line kept.

# 2.2 Sampling

Based on a user study of developers' demand from our organization, we allocate the tentative area quota to be  $25\%$ ,  $25\%$ ,  $25\%$ ,  $15\%$ , and  $10\%$  for front-end, back-end, DS&ML, mobile and desktop, and IT Ops, respectively. Inspired by HumanEval size and considering the labelling labor cost, we set 200 questions as the target benchmark size. Hence, the tentative size quotas by area are 50, 50, 50, 30, and 20 respectively. We then proportionally distribute the area quotas to language quotas based on the frequency of each language in the initial seed set. However, we observe that following this rule, certain languages such as CSS and C/C++ end up with fewer than 10 questions, which may yield unreliable language-level sub-score, so, for these languages, we set their quotas to 10.

As a result, we derive the tentative question quota for each language as shown in Table 2, which sums up to 270 questions. After determining the tentative question quota, we uniformly sample from the initial seed set a roughly two times larger pool for the domain expects to select and annotate.

# 2.3 Human Annotation

We recruited five domain experts inside our company to create the benchmark, each in charge of one area. The annotation process is composed of three steps:

- Step 1: Question Selection and Type Annotation. Domain experts select high-quality questions from the inspecting set and annotate the question type to be one of the four: code completion, code debugging, config and environment debugging, and knowledge question-answering.  
- Step 2: Prompt Paraphrasing. Domain experts paraphrase and simplify the original question body into succinct and explicit instructions. We include this step for two main purposes: (1) Reduce domain gap. From user-shared conversations collected from ShareGPT, we observe that when interacting with code LLMs, users tend to provide short and direct instructions like "Fix problem..." and "Debug code..." However, when posting Stack Overflow questions, users tend to be lengthy with courtesy words. We ask the domain experts to paraphrase the question to code LLM user's style without changing the semantics. (2) Reduce the impact of memorization and data contamination. Some code LLMs may be trained or finetuned with Stack Overflow data. Paraphrasing the questions can help to mitigate the result advantages of these models. Benchmark results in Table 4 reveal the effectiveness of this step where copying Stack Overflow answers only achieves a  $65.18\%$  score. We defer further discussion in Section 2.5.

Table 2: InfiBench data statistics by area and language. We uniformly sample a subset from the initial seed set (see Section 2.1) according to the area quota (see Section 2.2) for domain experts to select questions and annotate the correctness criterion to construct the benchmark.  

<table><tr><td rowspan="2">Area</td><td rowspan="2">Language</td><td colspan="2">Initial Seed Set</td><td rowspan="2">Tentative # Questions Quota</td><td colspan="4">Final InfiBench Benchmark</td></tr><tr><td># Questions</td><td>% Area Quota</td><td># Questions Quota</td><td>% Questions Quota</td><td># Area Quota</td><td>% Area Quota</td></tr><tr><td rowspan="3">Front-End</td><td>Javascript</td><td>4912</td><td></td><td>44</td><td>44</td><td>18.80%</td><td></td><td></td></tr><tr><td>CSS</td><td>87</td><td>40.41%</td><td>10</td><td>10</td><td>4.27%</td><td>63</td><td>26.92%</td></tr><tr><td>HTML</td><td>600</td><td></td><td>10</td><td>9</td><td>3.85%</td><td></td><td></td></tr><tr><td rowspan="7">Back-End</td><td>Java</td><td>930</td><td></td><td>18</td><td>17</td><td>7.26%</td><td></td><td></td></tr><tr><td>C#</td><td>629</td><td></td><td>12</td><td>12</td><td>5.13%</td><td></td><td></td></tr><tr><td>PHP</td><td>462</td><td></td><td>10</td><td>9</td><td>3.85%</td><td></td><td></td></tr><tr><td>Go</td><td>117</td><td>18.71%</td><td>10</td><td>9</td><td>3.85%</td><td>77</td><td>32.91%</td></tr><tr><td>Ruby</td><td>71</td><td></td><td>10</td><td>10</td><td>4.27%</td><td></td><td></td></tr><tr><td>Rust</td><td>96</td><td></td><td>10</td><td>10</td><td>4.27%</td><td></td><td></td></tr><tr><td>C/C++</td><td>287</td><td></td><td>10</td><td>10</td><td>4.27%</td><td></td><td></td></tr><tr><td rowspan="2">DS &amp; ML</td><td>Python</td><td>2779</td><td rowspan="2">21.39%</td><td>47</td><td>47</td><td>20.09%</td><td rowspan="2">56</td><td rowspan="2">23.93%</td></tr><tr><td>R</td><td>184</td><td>10</td><td>9</td><td>3.85%</td></tr><tr><td rowspan="4">Mobile &amp; Desktop</td><td>Dart</td><td>1562</td><td></td><td>19</td><td>19</td><td>8.12%</td><td>19</td><td>8.12%</td></tr><tr><td>Kotlin</td><td>383</td><td rowspan="2">18.13%</td><td>10</td><td></td><td></td><td></td><td></td></tr><tr><td>Swift</td><td>551</td><td>10</td><td colspan="4">Removed during Post-Filtering (see Section 2.3)</td></tr><tr><td>VBA</td><td>16</td><td></td><td>9</td><td></td><td></td><td></td><td></td></tr><tr><td>IT Ops.</td><td>Bash</td><td>188</td><td>1.36%</td><td>21</td><td>19</td><td>8.12%</td><td>19</td><td>8.12%</td></tr><tr><td colspan="2">Total</td><td>13854</td><td>100.0%</td><td>270</td><td>234</td><td>100.00 %</td><td>234</td><td>100.00%</td></tr></table>

- Step 3: Correctness Criterion Annotation. Domain experts choose one or multiple evaluation metrics from our supported ones (see Section 2.4) and annotate the concrete criterion following a YAML schema. External files can be attached if needed, e.g., unit tests and reference answers.

Calibration and Post-Filtering. To improve annotation consistency and objectiveness, we introduce a few checkpoints for domain experts to read others' annotated cases, discuss them, and reach consensus for controversial cases. After the 270 tentative questions were annotated, we then ran an initial evaluation of all these questions on over 30 code LLMs. This initial evaluation helps us to identify questions whose criteria are incorrect or out of distribution. We filter out these questions and then remove all questions from Kotlin, Swift, and VBA languages since the questions in these languages are too few after filtering. After this calibration and post-filtering process, the final benchmark includes 234 questions spanning over 15 languages. Their statistics are shown in Table 2. As we can observe, compared to the population area distribution of high-quality Stack Overview questions (see “% Area Quota” column under “Initial Seed Set”), the area distribution of final benchmark questions (see “% Area Quota” column under “Final InfBench Benchmark”) is more balanced and less biased towards front-end, mobile, and desktop topics.

# 2.4 Evaluation Criteria and Evaluation Framework

In response to the diversified questions, InfiBench evaluation framework integrates four types of model-free and automatic metrics as below. Domain experts choose one or multiple metric types along with their weights and concretize.

- Keywords Matching. Though the responses can be in diverse forms, for a significant portion of benchmark questions, we find that the existence of some keywords strongly determines the quality of the response. Domain experts can write rules that match keywords and regular expressions or construct recursive logical expressions on top of keyword-matching results. When multiple keywords exist, each matching result can have its weight in the final score.  
- Blank Filling. For some questions, it is challenging to measure the correctness given the response uncertainty. In this case, domain experts can instruct the model to answer the question by following a given template and filling in the blanks in the template. The blanks can correspond to either natural language or code snippet. Then, similar to keywords matching, each blank can match potential keywords, regular expressions, or recursive logic expressions built upon matching results. This metric type tests not only the model's QA ability but also its instruction-following ability.  
- Unit Testing. For code-intensive questions, we can follow existing benchmarks to evaluate response correctness by unit tests. For this type, domain experts may add more specifications in the prompt to allow for unit-test-based evaluation, such as specifications on function name, input arguments, and output format. Domain experts can further import the context setup and cleanup script.

- Dialogue Similarity. For natural-language-intensive questions, domain experts can extract and shorten the reference answers from Stack Overflow, and then use the ROUGE score [Lin, 2004] to evaluate the response similarity with reference answers. The ROUGE score was initially proposed and widely used in evaluating the quality of text summarization and machine translation. To map the ROUGE score back to our benchmark scale, we allow domain experts to tune the mapping interval and scores within the interval are then linearly mapped to our score scale.

The example questions and corresponding criteria are illustrated in Figure 1. Detail statistics of metric type ratios, question type ratios, and prompt length are shown in Table 3.

Score Computation. We treat each question equally with one point each. Given 234 questions in the benchmark, the full score is 234, and we by default report the percentage score (achieved score divided by 234) unless otherwise noted. The one point for each question can be further decomposed into a few scoring points within each question. For example, a question may contain four keywords with weights 2, 1, 1, and 1 each. Then, matching each keyword can contribute to 0.4, 0.2, 0.2, and 0.2 points respectively to the final score.

Table 3: Infibench statistics.

(a) Question type.  

<table><tr><td>Question Type</td><td>Ratio</td></tr><tr><td>Code Completion</td><td>30.37%</td></tr><tr><td>Knowledge Question-Answering</td><td>27.04%</td></tr><tr><td>Code Debugging</td><td>26.67%</td></tr><tr><td>Config &amp; Environment Debugging</td><td>15.93%</td></tr></table>

(b) Metric type.  

<table><tr><td>Metric Type</td><td>Ratio</td></tr><tr><td>Keywords Matching</td><td>57.41%</td></tr><tr><td>Blank Filling</td><td>12.22%</td></tr><tr><td>Unit Testing</td><td>19.26%</td></tr><tr><td>Dialogue Similarity</td><td>11.85%</td></tr></table>

(c) Prompt token length with Code Llama tokenizer.  

<table><tr><td>min</td><td>25% quantile</td><td>median</td><td>mean</td><td>75% quantile</td><td>max</td></tr><tr><td>43</td><td>145.75</td><td>223</td><td>338.46</td><td>359.50</td><td>5047</td></tr></table>

Implementation. We have implemented an automated evaluation framework with Python, publicly available at https://infi-coder.github.io/infibench. Specifically, for blank-filling evaluation, we use the longest common subsequence matching via dynamic programming to capture the filled blanks in the response. For unit-testing evaluation, we construct a runtime environment that supports the test execution for nine languages. We plan to integrate the framework into the Hugging Face Open LLM Leaderboard [Beeching et al., 2023] to further ease the evaluation burden.

How does Infibench Evaluation Align with Human? To evaluate the alignment between Infibench evaluation and human expert evaluation, we randomly sample 100 questions without replacement from the benchmark and select three strong LLMs to generate responses: GPT-4-0613, GPT-3.5-turbo, and Mistral Codestral-22b. For each question, we randomly choose two out of these three model responses to construct response pairs, resulting in 100 response pairs  $\mathcal{R} = \{(A_i,B_i):1\leq i\leq 100\}$ . For each response pair  $(A,B)\in \mathcal{R}$ , we use Infibench, GPT-4o, and human expert to evaluate into four outcomes:  $A$  is more correct than  $B$  ( $A > B$ );  $B$  is more correct then  $A$  ( $B > A$ ); both  $A$  and  $B$  are correct ( $A\approx B\uparrow$ ); both  $A$  and  $B$  are incorrect ( $A\approx B\downarrow$ ). Our purpose is to evaluate how Infibench evaluation aligns with humans, specifically when compared to the widely-used LLM-as-a-judge (i.e., model-based evaluation) [Zheng et al., 2024]. The concrete grading criteria is as below:

- Infibench gives a score between  $[0, 1]$  for each response in the pair. If the score difference in the pair is larger than 0.2, we label the outcome to be  $A > B$  or  $B > A$  respectively; otherwise, if the maximum score among the two is larger than 0.5, we label the outcome to be  $A \approx B \uparrow$ ; otherwise, we label the outcome to be  $A \approx B \downarrow$ .  
- For GPT-4o evaluation, we deploy the prompting template from LLM-Blender [Jiang et al., 2023, Appendix E] and trigger GPT-4o for grading the four outcomes. We enhance the reliability of the comparison by switching  $A$  and  $B$  and prompting GPT-4o twice. We record the preference only when a consistent preference exists.  
- For human evaluation, we recruit human annotators who came up with the criteria to label the comparison preference since they are familiar with the questions and have strong expertise. Annotators have no access to the evaluation results of InfiBench and GPT-4o, nor which source model generates the response. Annotators were instructed to directly label each pair with the four outcomes.

We defer the consensus matrices between Infibench/GPT-4o and human annotators along with more findings in Appendix C. If we only count the cases where both Infibench/GPT-4o and humans have clear preferences, the agreement rate between InfiBench and humans is  $85.1\%$ , and the agreement rate between GPT-4o and humans is  $77.8\%$ . Hence, the InfiBench evaluation aligns with human

experts better than the GPT-4o evaluation (with  $>80\%$  confidence). We observe that the advantage of InfiBench comes from the ability to detect deceptive answers. some model responses pretend to be helpful with lengthy wording and hallucinations. GPT-4o is more likely to be cheated than InfiBench, which looks for key concepts that should exist in a helpful answer.

# 2.5 Mitigations on Memorization and Data Contamination

InfiBench is created from the publicly available Stack Overflow corpus to reflect real-world scenarios, and this corpus may already exist in the training set of some code LLMs (e.g., DeepSeek Coder [Guo et al., 2024] and StarCoder 2 [Lozhkov et al., 2024]). Hence, some code LLMs may achieve a high score simply due to memorization. To mitigate this, we asked the domain experts to paraphrase every question as an essential step (see Section 2.3). Hence, copying either the highly voted answers or officially accepted answers of the original questions only achieves  $65.18\%$ , being far from perfect and inferior to GPT-4's  $70.64\%$ . Furthermore, code LLMs that use Stack Overflow data do not demonstrate significant advantages over those without. Hence, we deem the effect of contamination as small.

On the other hand, we release the post IDs of the source question posts of InfiBench. Hence, future LLM training could consider this benchmark to conduct dedduplication and ablation studies on data contamination. Another usage of our benchmark is to evaluate retrieval-augmented (RAG) code LLMs where perfect retrieval from Stack Overflow and moderate adaptation should solve these questions, which we leave as future work.

# 2.6 Comparison with Existing Benchmarks

In Table 1, we compare InfiBench with several existing benchmarks for code LLMs. As reflected in the table, InfiBench strongly complements existing benchmarks for code LLMs by (1) extending them beyond code generation to a wide range of real-world tasks, (2) diversifying them since InfiBench does not share the same source as existing ones, and (3) increasing the differentiation as an unsaturated benchmark. Related benchmarks are further illustrated in Section 5. On the other hand, the benchmark is limited in size due to the high cost of correctness criteria labelling, and we are continuously expanding the benchmark.

# 3 Evaluation and Leaderboard

We systematically evaluated over 100 code LLMs spanning both proprietary and open-source worlds on InfiBench. To the best of our knowledge, this is the most extensive evaluation for code LLMs.

Evaluation Protocol. We adopt best@10 as the main evaluation metric: 10 responses are sampled and evaluated for each question, then the best score per question is recorded and summed up. Throughout the evaluation, we set sampling temperature  $T = 0.2$  and top  $p = 0.9$ .

Furthermore, we swept sampling parameters with GPT-4 and the detailed results are in Appendix G. In a nutshell, for maximizing the performance under best@10, the best parameters are  $T = 1.0$  and  $p = 0.9$ , leading to a score of  $76.15\% \pm 0.21\%$  (in comparison to  $70.64\% \pm 0.82\%$  in our main setting  $T = 0.2$ ,  $p = 0.9$ ). In particular, the temperature  $T$  affects much and the effect of top  $p$  is minor. We decided to stick to the original parameters  $T = 0.2$  and  $p = 0.9$  in the main evaluation since this setting is more akin to the real-world scenario where user generates once with low temperature.

We design two system prompts (shown in Appendix H), one for normal questions and the other for open-ended questions with an additional sentence to encourage succinct responses. For generic models, we generate the prompt with “{system prompt}\n{content prompt}” format; for instruction-finetuned or chat models, we generate the prompt with their prompt templates.

For proprietary models, we evaluate the latest models from OpenAI (GPT-4, GPT-4o, etc), Anthropic (Claude 3), and Mistral AI (Mistral Small/Medium/Large) with API calling. When budget permits, we repeat each evaluation three times and report standard deviation. For open-source models, we download models from Hugging Face and evaluate them on an 8xA100 server with bigcode-evaluation-harness [Ben Allal et al., 2022]. When the model size is within 30B parameters, we repeat each evaluation three times and report the standard deviation. All raw model

Table 4: Aggregated InfiBench leaderboards (best viewed zoomed in and in color). "Size" column records number of parameters. For MoE models, "total params. / params. activated during inference" is recorded. Bar colors stand for General Base , General Finetuned , Code Base , and Code Finetuned models respectively. Icon “ $\square$ ” stands for proprietary models otherwise open-source. Full leaderboard in Appendix E.

(a) Infibench leaderboard by model family, where best (b) Infibench leaderboard by model type, where top five model within each model family is shown. model within each type is shown.

<table><tr><td></td><td>Family</td><td>Best Model Name</td><td>Size</td><td>InfBench Score</td></tr><tr><td>1</td><td>#GPT-4</td><td>GPT-4-0613</td><td>?</td><td>70.64% ± 0.8%</td></tr><tr><td>2</td><td>DeepSeekCoder</td><td>deepSeek-coder-V2-instruct</td><td>236B / 21B</td><td>65.49%</td></tr><tr><td>3</td><td>#Claude 3</td><td>Claude 3 Opus</td><td>?</td><td>63.89%</td></tr><tr><td>4</td><td>Mistral Open</td><td>Codestral-22b</td><td>22B</td><td>62.98% ± 0.56%</td></tr><tr><td>5</td><td>Phind</td><td>Phind-CodeLlama-34B-v2</td><td>34B</td><td>59.00%</td></tr><tr><td>6</td><td>#Mistral</td><td>mistral-large</td><td>?</td><td>58.22%</td></tr><tr><td>7</td><td>DeepSeek LLM</td><td>deepseek-llm-67b-chat</td><td>67B</td><td>57.41%</td></tr><tr><td>8</td><td>#GPT-3.5</td><td>GPT-3.5-turbo-0613</td><td>?</td><td>56.47% ± 1.34%</td></tr><tr><td>9</td><td>Qwen</td><td>Qwen-72B</td><td>72B</td><td>55.34%</td></tr><tr><td>10</td><td>Magiccoder</td><td>Magiccoder-S-CL-7B</td><td>7B</td><td>52.71% ± 0.72%</td></tr><tr><td>11</td><td>WizardLM</td><td>WizardCoder-Python-34B-V1.0</td><td>34B</td><td>52.59%</td></tr><tr><td>12</td><td>Code Llama</td><td>CodeLlama-34b-Instruct</td><td>34B</td><td>50.45%</td></tr><tr><td>13</td><td>01.AI</td><td>Yi-34B-Chat</td><td>34B</td><td>49.58%</td></tr><tr><td>14</td><td>Zephyr</td><td>Zephyr 7B beta</td><td>7B</td><td>46.31% ± 1.11%</td></tr><tr><td>15</td><td>StarCoder2</td><td>15B-Instruct</td><td>15B</td><td>45.89% ± 0.95%</td></tr><tr><td>16</td><td>DeepSeek MoE</td><td>deepseek-moe-16b-chat</td><td>16B / 2.8B</td><td>45.18% ± 1.65%</td></tr><tr><td>17</td><td>OctoPack</td><td>OctoCoder</td><td>15.5B</td><td>44.55% ± 0.79%</td></tr><tr><td>18</td><td>gemma</td><td>gemma-7b-it</td><td>7B</td><td>40.68% ± 1.23%</td></tr><tr><td>19</td><td>Llama 2</td><td>Llama-2-70B-Chat</td><td>70B</td><td>39.30%</td></tr><tr><td>20</td><td>InternLM</td><td>InternLM-Chat-20B</td><td>20B</td><td>37.41% ± 0.75%</td></tr><tr><td>21</td><td>Baichuan2</td><td>Baichuan2-13B-Chat</td><td>13B</td><td>34.40% ± 1.34%</td></tr><tr><td>22</td><td>StarCoder</td><td>StarCode+</td><td>15.5B</td><td>30.67% ± 1.57%</td></tr><tr><td>23</td><td>CodeGen2.5</td><td>CodeGen2.5-7B-Instruct</td><td>7B</td><td>29.57% ± 1.53%</td></tr><tr><td>24</td><td>ChatGLM</td><td>ChatGLM3-6B</td><td>6B</td><td>28.23% ± 0.58%</td></tr><tr><td>25</td><td>#davinci</td><td>davinci-002</td><td>?</td><td>21.25% ± 1.17%</td></tr><tr><td>26</td><td>Phi</td><td>Phi1.5</td><td>1.5B</td><td>20.56% ± 0.09%</td></tr><tr><td>27</td><td>CodeGeeX</td><td>CodeGeeX2-6B</td><td>6B</td><td>19.88% ± 0.36%</td></tr><tr><td>28</td><td>CodeGen2</td><td>CodeGen2-16B</td><td>16B</td><td>16.97% ± 1.15%</td></tr><tr><td>29</td><td>IEFTYuan</td><td>Yuan2-51B-hf</td><td>51B</td><td>15.25%</td></tr><tr><td>30</td><td>CodeGen</td><td>CodeGen-16B-multi</td><td>16B</td><td>13.62% ± 1.18%</td></tr><tr><td rowspan="3" colspan="2">Human</td><td colspan="2">10 Highest-Voted Answer Posts</td><td>65.18%</td></tr><tr><td colspan="2">Highest-Voted Answer Post</td><td>56.28%</td></tr><tr><td colspan="2">Officially-Accepted Answer Post</td><td>52.90%</td></tr></table>

<table><tr><td>Type</td><td>Rank</td><td>Model Family / Model Name</td><td>Size</td><td>InfiBench Score</td></tr><tr><td rowspan="3">Pro-pri-cary</td><td>1</td><td>aGPT-4/GPT-4-0613</td><td>?</td><td>70.64% ± 0.82%</td></tr><tr><td>2</td><td>aGPT-4/GPT-4-turbo-1106</td><td>?</td><td>68.42% ± 0.38%</td></tr><tr><td>3</td><td>aGPT-4/GPT-4o-2024-05-13</td><td>?</td><td>66.19%</td></tr><tr><td rowspan="2">Model</td><td>4</td><td>aClaude 3/Claude 3 Opus</td><td>?</td><td>63.89%</td></tr><tr><td>5</td><td>aMistral/mistral-large</td><td>?</td><td>58.22%</td></tr><tr><td rowspan="5">Code Fine-tuned Model</td><td>1</td><td>DeepSeek Coder/deepSeek-coder-V2-instruct</td><td>236B / 21B</td><td>65.49%</td></tr><tr><td>2</td><td>Mistral Open/Codedstral-22b</td><td>22B</td><td>62.98% ± 0.56%</td></tr><tr><td>3</td><td>DeepSeek Coder/deepsseek-coder-33b-instruct</td><td>33B</td><td>62.96%</td></tr><tr><td>4</td><td>Phind/Phind-Code/Llama-34B-v2</td><td>34B</td><td>59.00%</td></tr><tr><td>5</td><td>Phind/Phind-Code/Llama-34B-v1</td><td>34B</td><td>58.47%</td></tr><tr><td rowspan="5">Code Base Model</td><td>1</td><td>Code Llama/CodeLlama-34b</td><td>34B</td><td>47.36%</td></tr><tr><td>2</td><td>Code Llama/CodeLlama-34b-Python</td><td>34B</td><td>43.13%</td></tr><tr><td>3</td><td>StarCoder2/15B</td><td>15B</td><td>42.52% ± 1.24%</td></tr><tr><td>4</td><td>Code Llama/CodeLlama-13b</td><td>13B</td><td>41.66% ± 0.84%</td></tr><tr><td>5</td><td>Code Llama/CodeLlama-13b-Python</td><td>13B</td><td>41.31% ± 0.90%</td></tr><tr><td rowspan="5">General Fine-tuned Model</td><td>1</td><td>DeepSeek LLM/deepsseek-lim-67b-chat</td><td>67B</td><td>57.41%</td></tr><tr><td>2</td><td>Mistral Open/mixtral-8x7B-Instruct</td><td>46.7B / 12.9B</td><td>55.55%</td></tr><tr><td>3</td><td>Qwen/Qwen-72B-Chat</td><td>72B</td><td>52.97%</td></tr><tr><td>4</td><td>01.AJ/Y-34B-Chat</td><td>34B</td><td>49.58%</td></tr><tr><td>5</td><td>Zephyr/Zephyr-7B beta</td><td>7B</td><td>46.31% ± 1.11%</td></tr><tr><td rowspan="5">General Base Model</td><td>1</td><td>Qwen/Qwen-72B</td><td>72B</td><td>55.34%</td></tr><tr><td>2</td><td>Qwen/Qwen-14B</td><td>14B</td><td>43.69% ± 1.09%</td></tr><tr><td>3</td><td>DeepSeek LLM/deepsseek-lim-67b-base</td><td>67B</td><td>39.87%</td></tr><tr><td>4</td><td>Llama 2/Llama2-70B</td><td>70B</td><td>37.69%</td></tr><tr><td>5</td><td>Qwen/Qwen-7B</td><td>7B</td><td>31.69% ± 0.29%</td></tr></table>

(c) Infibench leaderboard by model size, where best model within the threshold is shown.

<table><tr><td>Size Threshold</td><td>Model Family / Model Name</td><td>Size</td><td>InfiBench Score</td></tr><tr><td>∞</td><td>GPT-4/GPT-4-0613</td><td>?</td><td>70.64% ± 0.82%</td></tr><tr><td>&lt;100B</td><td>Mistral Open/Codede-22b</td><td>22B</td><td>62.98% ± 0.56%</td></tr><tr><td>&lt;20B</td><td>DeepSeek Coder/deepsEEK-coder-6.7b-instruct</td><td>6.7B</td><td>53.25% ± 0.40%</td></tr><tr><td>&lt;5B</td><td>DeepSeek Coder/deepsEEK-coder-1.3b-instruct</td><td>1.3B</td><td>41.32% ± 1.12%</td></tr></table>

responses are available at https://figshare.com/articles/dataset/InfiBench_Detail_Evaluation_Data/26104864. More details on the evaluation protocol are in Appendix E.

Leaderboard. In Table 4, we present aggregated InfiBench leaderboards by model family, model type, and model size. The full leaderboard is deferred to Appendix E due to space limit. The table includes scores from using the original Stack Overflow answer posts as reference. Results are also presented as a scatter plot in Figure 4, where normal models are shown as scatters with error bars, MoE models are shown as horizontal segments with error ranges connecting the activated parameters during inference and total parameters, and strong proprietary models are shown as horizontal lines.

In both tables and the figure, we classify LLMs by general/code and base/finetuned. The general LLMs are claimed to have strong capabilities beyond code, e.g., in various natural language tasks, while the code LLMs are exclusively optimized for the code domain. The base LLMs only went through the pretraining phase, while the finetuned LLMs are claimed to have instruction-following capabilities or are finetuned on instruction or human preference datasets.

# 4 Analysis and Discussion

The best model so far, GPT-4, is still far from perfect, and open-source models are competitive but still far from GPT-4. GPT-4 achieves the highest score  $70.64\%$  (interestingly, achieved by GPT-4-0613 instead of the more recent GPT-4o), then Claude 3 Opus with a score  $63.89\%$ , and then Codestral-22b [AI, 2024] with a score  $62.98\%$  and deepseek-coder-33b-instruct [Guo et al., 2024] with a score  $62.96\%$ . The result implies that: (1) Noting that the full score is  $100\%$ , even the powerful GPT-4 is still far from perfect, which is in contrast to its  $\approx 90\%$  HumanEval score. We inspect the score breakdown. For the two most frequent metric types, keywords matching and unit testing, GPT-4 achieves similar scores  $66.61\%$  and  $76.00\%$  respectively. For blank filling, the score is relatively lower at  $58.08\%$ . These scores imply that GPT-4 may still lack generic ability in answering diversified real-world questions related to code. When instructed to follow a given template to answer (blank filling), due to the more strict requirement and narrower solution space, its lack of capability is more pronounced. (2) There is still a visible gap between open-source models and GPT-4. The gap between the most powerful open-source model, Codestral-22b, and GPT-4 is roughly 8 points. On the other

Figure 4: Scatter plot for all evaluated LLMs on Infibench.  $x$ -axis is the model size in terms of number of parameters and  $y$ -axis is Infibench score. Projected empirical scaling laws for both general and code models are drawn. Detail discussion in Section 4.

hand, noticing that GPT-3.5-turbo achieves  $56.47\%$ , the open-source model, Codestral-22b, is now reliably better than GPT-3.5-turbo with merely 22B parameters which is promising.

Among open-source models, different models have various performances. Figure 4 systematically visualizes the performance of different open-source models at diverse scales. Although there is a general tendency that larger models achieve higher scores, the scores among different models at a similar scale differ largely. For example, on scale 7B, the best-performing model is at around  $55\%$ , pretty close to GPT-3.5, while the low-performing model stays at around  $15\%$ . Moreover, deepseek-coder-1.3b-instruct achieves  $41.32\%$  at 1.3B and surpasses a few models at scale 70B or 100B. Hence, though scaling matters, the training techniques and training data are equally important or even more, helping to reduce the required scale for achieving a certain score by more than  $10\times$ .

Hard problems generalize their difficulties. We rate the benchmark problem difficulty with five levels by how well GPT-4 and GPT-3.5-turbo answer them, as detailed in Appendix D. Example questions from each level are shown in Appendix I. We present the detail result table including the sub-score for each difficulty level in Appendix E. Interestingly, the trend is highly consistent that sub-scores decrease along with the increase of problem level. Specifically, hard problems for the most powerful model yet, GPT-4, are also generally hard for open-source models. These hard problems usually correspond to code generation with long and domain-specific context or challenging blank-filling questions since blank-filling is a specific task that rarely appears in training data before.

Instruction finetuning is important for QA. Among models of similar scales and the same family, we find that the best-performing ones almost always include an instruction-finetuning phase, such as deepseek-llm-67b-chat, deepseek-coder-33b-instruct, CodeLlama-34B-Instruct, and Qwen-18B-Chat. In contrast, the pretraining models, such as davinci-002 and phi models, usually perform poorly despite their strong performances in code generation benchmarks. Instruction-finetuning is also critical for other code domain tasks such as code generation. As shown in Appendix F.1 where we plot model scores in QA (measured by InfiBench) and code generation (measured by HumanEval pass@1), instruction-tuning generally improves both QA and code generation, but the improvement is usually more significantly on code generation but more moderately on QA. As a result, we suggest generalizing the instruction-finetuning data beyond simple coding problems to improve code LLMs. Indeed, our preliminary experiments show that, after fine-tuning with the decontaminated and sanitized Stack Overflow data, we improved InfiBench scores for Codellama-13b-Instruct from  $46.37\%$  to  $60.74\%$  and for mixtral-8x7B-Instruct from  $55.55\%$  to  $62.61\%$ .

Some models may focus too much on code generation, especially the small ones. As detailed in Appendix F.1, we observe that for large models ( $>30\mathrm{B}$ ) and top entries, InfiBench and HumanEval pass@1 scores coincide well. However, for smaller models, the score tendencies start to diverge, where some models are relatively stronger on InfiBench (Mixtral-8x7B-Instruct) and more are relatively stronger on HumanEval (Phi1, Phi2, gemma-7b, ...). This phenomenon implies that a few models may be optimized too heavily on code generation benchmarks while ignoring the performance in generic code scenarios as represented by InfiBench, which in turn highlights the significance of free-form QA benchmarks like InfiBench in detecting capability imbalance in code LLMs.

# Code Llama models have unique characteristics. We evaluated all Code Llama models [Roziere et al., 2023]. As shown in Table 5, we found finetuning on Python data improves on HumanEval but hurts InfiBench scores, while instruction finetuning usually improves InfiBench scores but may hurt HumanEval. As a side product, we found CodeLlama-70B may be overly safeguarded and denies answering some safe questions in In

Table 5: Evaluation on eight models from the Code Llama [Roziere et al., 2023] family showcases intense Python finetuning may hurt free-form QA ability, despite achieving higher HumanEval scores.  

<table><tr><td></td><td>Benchmark</td><td>Base</td><td>Python</td><td>Instruct</td></tr><tr><td rowspan="2">7B</td><td>HumanEval</td><td>33.5%</td><td>38.4% (+4.9%)</td><td>34.8% (+1.3%)</td></tr><tr><td>InfiBench</td><td>37.62%±1.28%</td><td>32.89%±0.45% (-4.73%)</td><td>35.15%±1.28% (-2.47%)</td></tr><tr><td rowspan="2">13B</td><td>HumanEval</td><td>36.0%</td><td>43.3% (+7.3%)</td><td>42.7% (+6.7%)</td></tr><tr><td>InfiBench</td><td>41.66%±0.84%</td><td>41.31%±0.90% (-0.35%)</td><td>46.37%±1.26% (+4.71%)</td></tr><tr><td rowspan="2">34B</td><td>HumanEval</td><td>48.8%</td><td>53.7% (+4.9%)</td><td>41.5% (-7.3%)</td></tr><tr><td>InfiBench</td><td>47.36%</td><td>43.13% (-4.23%)</td><td>50.45% (+3.09%)</td></tr><tr><td rowspan="2">70B</td><td>HumanEval</td><td>53.0%</td><td>57.3% (+4.3%)</td><td>67.8% (+14.8%)</td></tr><tr><td>InfiBench</td><td>40.60%</td><td>40.29% (-0.31%)</td><td>42.82% (+2.22%)</td></tr></table>

fiBench. More model-specific findings are presented in Appendix F.

Code models and general models may exhibit different scaling laws, and open-source models scale well only within 40B yet. In Figure 4, we use the top-performing code and general models at each scale respectively to regress and extrapolate model performance at larger scales. As shown, code models tend to have higher capabilities compared to general models of the same scale, though the gap shrinks for larger models. Hence, when the compute budget is heavily limited, training exclusively in the code domain could be more efficient for building strong code LLMs.

In Figure 4, both predicting curves are split into two segments, steep in the first segment and much flat in the second. Following the first segment, open-source models catch up with GPT-4 at around 50B scale. However, following the second segment, they may need to be at  $>300\mathrm{B}$  scale to catch up. The finding contradicts the common scaling law [Kaplan et al., 2020, Muennighoff et al., 2024, Bi et al., 2024] where a strong linear relationship between model scale and capability exists. The contradiction implies that very large open-source models  $(>40\mathrm{B})$  may fail to achieve the expected performance at their scales, or there is some non-trivial barrier when scaling the model beyond 40B, or the scaling law may change at such a large scale. We leave further investigation as the future work. Notably, after the release of Infibench, Deepseek-coder-v2 [Zhu et al., 2024] was released as the largest code LLM to our knowledge in an MoE architecture with 236B total and 21B active parameters. On Infibench, Deepseek-coder-v2 achieves  $65.49\%$ , setting the new baseline for open-source LLMs but still being inferior to GPT-4. More importantly, the score is within the predicted range of our empirical scaling law.

We defer dataset card and data accessibility details, discussion on limitations and societal impact, full leaderboard, additional findings, ablation studies, and data examples in appendices.

# 5 Related Work

Large language models [Vaswani et al., 2017, Devlin et al., 2018, Brown et al., 2020] are transforming people's lives. In the coding domain, LLMs [Chen et al., 2021, Li et al., 2022] are shown to be capable of completing a wide range of tasks such as code generation, debugging, and question-answering. Recently, code LLMs are booming. New models, including both proprietary [Github, 2023, OpenAI, 2023] and open-source ones [Beeching et al., 2023, Nijkamp et al., 2023, Touvron et al., 2023a,b, Li et al., 2023, Luo et al., 2023, Roziere et al., 2023, Zhu et al., 2024], emerge almost every month.

Benchmarks for code LLMs are developing, though at a relatively slower pace. Common benchmarks, e.g., APPS [Hendrycks et al., 2021], MBPP [Austin et al., 2021], and HumanEval [Chen et al., 2021], focus on code generation and unit-test-based evaluation. Some efforts augment these benchmarks by language translation (e.g., Multilingual HumanEval [Athiwaratkun et al., 2023], HumanEval-X [Zheng et al., 2023]), test augmentation (e.g., HumanEval+ [Liu et al., 2023a]), task generalization (e.g., HumanEvalPack [Muennighoff et al., 2023]), and human rewriting (e.g., LBPP [Matton et al., 2024]). To systematically evaluate real-world problem solving, recently,

SWE-bench [Jimenez et al., 2024], its filtered version SWE-bench Verified [OpenAI, 2024b], and RepoBench [Liu et al., 2023b] are proposed but they still primarily focus on code generation. Some general-purpose benchmarks, e.g., Arena-Hard [Li et al., 2024], contain code-related questions, but rely on LLM to judge and do not provide domain-specific scores. CodeXGLUE [Lu et al., 2021] considers multiple coding capabilities beyond code generation, but replies on existing data sources. In contrast to these benchmarks, InfiBench benchmark is built for evaluating free-form question-answering ability in the code domain beyond code generation in an automated and model-independent way.

# 6 Conclusion

We proposed InfiBench, a systematic benchmark for evaluating the question-answering ability of code LLMs in real-world scenarios, to facilitate development and scientific evaluation of LLMs. InfiBench comprises 234 high-quality questions from Stack Overflow and supports automatic model-free evaluation. A comprehensive evaluation of over 100 code LLMs reveals several findings and takeaways. The benchmark is publicly available and continuously expanding.

# Acknowledgement

We thank ByteDance Inc. for the support on computing resources, anonymous reviewers for their constructive feedback, and Kaixin Li (National University of Singapore) for contributing the Docker image after the initial release of InfiBench. This work was partially supported by National Natural Science Foundation of China under Grant No. 62161146003, and the Tencent Foundation/XPLORER PRIZE. Tao Xie is also affiliated with the School of Computer Science, Peking University, China. The corresponding authors are Linyi Li and Tao Xie.

# References

Mistral AI. Codestral: Hello, world! | mistral ai | frontier ai in your hands. https://mistral.ai/news/codestral/, 2024.  
Ben Athiwaratkun, Sanjay Krishna Gouda, Zijian Wang, Xiaopeng Li, Yuchen Tian, Ming Tan, Wasi Uddin Ahmad, Shiqi Wang, Qing Sun, Mingyue Shang, Sujan Kumar Gonugondla, Hantian Ding, Varun Kumar, Nathan Fulton, Arash Farahani, Siddhartha Jain, Robert Giaquinto, Haifeng Qian, Murali Krishna Ramanathan, Ramesh Nallapati, Baishakhi Ray, Parminder Bhatia, Sudipta Sengupta, Dan Roth, and Bing Xiang. Multi-lingual evaluation of code generation models. In Eleventh International Conference on Learning Representations, 2023.  
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.  
Edward Beeching, Clémentine Fourrier, Nathan Habib, Sheon Han, Nathan Lambert, Nazneen Rajani, Omar Sanseviero, Lewis Tunstall, and Thomas Wolf. Open LLM leaderboard. https://huggingface.co/spaces/HuggingFaceH4/open_11m_leaderboard, 2023.  
Loubna Ben Allal, Niklas Muennighoff, Logesh Kumar Umapathi, Ben Lipkin, and Leandro von Werra. A framework for the evaluation of code generation models. https://github.com/bigcode-project/bigcode-evaluation-harness, 2022.  
Manish Bhatt, Sahana Chennabasappa, Yue Li, Cyrus Nikolaidis, Daniel Song, Shengye Wan, Faizan Ahmad, Cornelius Aschermann, Yaohui Chen, Dhaval Kapil, et al. Cyberseceval 2: A wide-ranging cybersecurity evaluation suite for large language models. arXiv preprint arXiv:2404.13161, 2024.  
Xiao Bi, Deli Chen, Guanting Chen, Shanhuang Chen, Damai Dai, Chengqi Deng, Honghui Ding, Kai Dong, Qiushi Du, Zhe Fu, et al. Deepseek llm: Scaling open-source language models with longtermism. arXiv preprint arXiv:2401.02954, 2024.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. In Thirty-fourth International Conference on Neural Information Processing Systems, 2020.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.  
Jasper Dekoninck, Mark Niklas Müller, and Martin Vechev. Constat: Performance-based contamination detection in large language models. arXiv preprint arXiv:2405.16281, 2024.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
Angela Fan, Belize Gokkaya, Mark Harman, Mitya Lyubarskiy, Shubho Sengupta, Shin Yoo, and Jie M Zhang. Large language models for software engineering: Survey and open problems. arXiv preprint arXiv:2310.03533, 2023.  
Google Gemini Team, Rohan Anil, Sebastian Borgeaud, Yonghui Wu, Jean-Baptiste Alayrac, Jiahui Yu, Radu Soricut, Johan Schalkwyk, Andrew M Dai, Anja Hauth, et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.  
GibHub. Github copilot - your AI pair programmer. https://github.com/features/copilot, 2023.  
Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Y Wu, YK Li, et al. Deepseek-coder: When the large language model meets programming-the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.  
Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, et al. Measuring coding challenge competence with apps. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2), 2021.  
Xinyi Hou, Yanjie Zhao, Yue Liu, Zhou Yang, Kailong Wang, Li Li, Xiapu Luo, David Lo, John Grundy, and Haoyu Wang. Large language models for software engineering: A systematic literature review. arXiv preprint arXiv:2308.10620, 2023.  
Dongfu Jiang, Xiang Ren, and Bill Yuchen Lin. Llm-blender: Ensembling large language models with pairwise ranking and generative fusion. In Sixty-first Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2023.  
Carlos E Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik R Narasimhan. Swe-bench: Can language models resolve real-world github issues? In Twelfth International Conference on Learning Representations, 2024.  
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361, 2020.  
Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. Ds-1000: A natural and reliable benchmark for data science code generation. In Fortieth International Conference on Machine Learning, 2023.  
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023.  
Tianle Li, Wei-Lin Chiang, Evan Frick, Lisa Dunlap, Tianhao Wu, Banghua Zhu, Joseph E Gonzalez, and Ion Stoica. From crowdsourced data to high-quality benchmarks: Arena-hard and benchbuilder pipeline. arXiv preprint arXiv:2406.11939, 2024.  
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. Competition-level code generation with alphabet. Science, 378(6624):1092-1097, 2022.  
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, 2004.

Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatGPT really correct? rigorous evaluation of large language models for code generation. In Thirty-seventh International Conference on Neural Information Processing Systems, 2023a.  
Tianyang Liu, Canwen Xu, and Julian McAuley. Repobench: Benchmarking repository-level code auto-completion systems. arXiv preprint arXiv:2306.03091, 2023b.  
Anton Lozhkov, Raymond Li, Loubna Ben Allal, Federico Cassano, Joel Lamy-Poirier, Nouamane Tazi, Ao Tang, Dmytro Pykhtar, Jiawei Liu, Yuxiang Wei, et al. Starcoder 2 and the stack v2: The next generation. arXiv preprint arXiv:2402.19173, 2024.  
Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin Clement, Dawn Drain, Daxin Jiang, Duyu Tang, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation. In Thirty-fifth International Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 1), 2021.  
Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568, 2023.  
Justus Mattern, Fatemehsadat Mireshghallah, Zhijing Jin, Bernhard Schoelkopf, Mrinmaya Sachan, and Taylor Berg-Kirkpatrick. Membership inference attacks against language models via neighbourhood comparison. In Findings of the Association for Computational Linguistics: ACL 2023, 2023.  
Alexandre Matton, Tom Sherborne, Dennis Aumiller, Elena Tommasone, Milad Alizadeh, Jingyi He, Raymond Ma, Maxime Voisin, Ellen Gilsenan-McMahon, and Matthias Galle. On leakage of code generation evaluation datasets. arXiv preprint arXiv:2407.07565, 2024.  
Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. Octopack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124, 2023.  
Niklas Muennighoff, Alexander Rush, Boaz Barak, Teven Le Scao, Nouamane Tazi, Aleksandra Piktus, Sampo Pyysalo, Thomas Wolf, and Colin A Raffel. Scaling data-constrained language models. In Thirty-seventh International Conference on Neural Information Processing Systems, 2024.  
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis. In Eleventh International Conference on Learning Representations, 2023.  
OpenAI. GPT-4 technical report. OpenAI, 2023. URL https://cdn.openai.com/papers/gpt-4.pdf.  
OpenAI. Hello GPT-4o | OpenAI. https://openai.com/index/hello-gpt-4o/, 2024a.  
OpenAI. Introducing SWE-bench verified | OpenAI. https://openai.com/index/introducing-swe-bench-verified/, 2024b.  
Baptiste Roziere, Jonas Gehring, Fabian Gloeckle, Sten Sootla, Itai Gat, Xiaqing Ellen Tan, Yossi Adi, Jingyu Liu, Tal Remez, Jérémy Rapin, et al. Code llama: Open foundation models for code. arXiv preprint arXiv:2308.12950, 2023.  
Reza Shokri, Marco Stronati, Congzheng Song, and Vitaly Shmatikov. Membership inference attacks against machine learning models. In Thirty-eight IEEE Symposium on Security and Privacy, 2017.  
StackExchange. All sites — stackexchange. https://stackoverflow.com/sites?view=list#users, 2024.  
Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothee Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al. Llama: Open and efficient foundation language models. arXiv preprint arXiv:2302.13971, 2023a.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, et al. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288, 2023b.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Thirty-first International Conference on Neural Information Processing Systems, 2017.  
Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Ryan Schaeffer, et al. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models. In Thirty-seventh International Conference on Neural Information Processing Systems, 2024.  
Ruijie Xu, Zengzhi Wang, Run-Ze Fan, and Pengfei Liu. Benchmarking benchmark leakage in large language models. arXiv preprint arXiv:2404.18824, 2024.  
Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, et al. Judging llm-as-a-judge with mt-bench and chatbot arena. In Thirty-seventh International Conference on Neural Information Processing Systems, 2024.  
Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, Teng Su, Zhilin Yang, and Jie Tang. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. In Twenty-ninth ACM SIGKDD Conference on Knowledge Discovery and Data Mining, 2023.  
Qihao Zhu, Daya Guo, Zhihong Shao, Dejian Yang, Peiyi Wang, Runxin Xu, Y Wu, Yukun Li, Huazuo Gao, Shirong Ma, et al. Deepseek-coder-v2: Breaking the barrier of closed-source models in code intelligence. arXiv preprint arXiv:2406.11931, 2024.

# Checklist

1. For all authors...

(a) Do the main claims made in the abstract and introduction accurately reflect the paper's contributions and scope? [Yes]  
(b) Did you describe the limitations of your work? [Yes] Limitations are discussed throughout Section 2 and specifically in Appendix B.  
(c) Did you discuss any potential negative societal impacts of your work? [Yes] Societal impacts are discussed in Appendix B.  
(d) Have you read the ethics review guidelines and ensured that your paper conforms to them? [Yes]

2. If you are including theoretical results...

(a) Did you state the full set of assumptions of all theoretical results? [N/A]  
(b) Did you include complete proofs of all theoretical results? [N/A]

3. If you ran experiments (e.g. for benchmarks)...

(a) Did you include the code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL)? [Yes] All code and data are publicly available at https://infi-coder.github.io/infibench along with instructions needed to reproduce. The accessibility information is also available in detail in Appendix A.  
(b) Did you specify all the training details (e.g., data splits, hyperparameters, how they were chosen)? [N/A] This work does not involve model training. The inference hyperparameters are listed in Section 3 and ablation studies are presented in Appendix G.  
(c) Did you report error bars (e.g., with respect to the random seed after running experiments multiple times)? [Yes] All experiments are repeated three times whenever budget and computing resource permit. Error bars and standard deviations are reported.

(d) Did you include the total amount of compute and the type of resources used (e.g., type of GPUs, internal cluster, or cloud provider)? [Yes] The information is included in Section 3.

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets...

(a) If your work uses existing assets, did you cite the creators? [Yes]  
(b) Did you mention the license of the assets? [Yes] The main assets are from Stack Overflow which is open source under CC BY-SA 4.0 license. We inherit this license to release.  
(c) Did you include any new assets either in the supplemental material or as a URL? [Yes] New assets (the benchmark and evaluation tool) is accessible through https://infi-coder.github.io/infibench.  
(d) Did you discuss whether and how consent was obtained from people whose data you're using/curating? [Yes] We release the new asset inheriting the CC BY-SA 4.0 license as described in Section 1 and Appendix A.  
(e) Did you discuss whether the data you are using/curating contains personally identifiable information or offensive content? [Yes] Domain experts are required to remove such information by paraphrasing when constructing the benchmark.

5. If you used crowdsourcing or conducted research with human subjects...

(a) Did you include the full text of instructions given to participants and screenshots, if applicable? [N/A]  
(b) Did you describe any potential participant risks, with links to Institutional Review Board (IRB) approvals, if applicable? [N/A]  
(c) Did you include the estimated hourly wage paid to participants and the total amount spent on participant compensation? [N/A]

# Appendices

In appendices, we present dataset card and data accessibility details in Appendix A, discussion on limitations and societal impact in Appendix B, agreement statistics between InfiBench/GPT-4o and human in Appendix C, question grouping by difficulty in Appendix D, full leaderboard in Appendix E, additional findings in Appendix F, study of sampling hyperparameters in Appendix G, prompts in Appendix H, and benchmark data examples in Appendix I.

# A Dataset Card and Accessibility Details

# Dataset Card

- Name: Infibench  
- Description: Evaluation Dataset for the Question-Answering Capabilities of Code Large Language Models  
- URL: https://infi-coder.github.io/infibench (all resources) / https://huggingface.co/datasets/llylly001/InfiBench (data part)  
Version: 2.1  
- License: Creative Commons Attribution Share Alike 4.0  
Citation:

```bib
@misc{infibench, title  $=$  {InfiBench: Evaluating the Question-Answering Capabilities of Code Large Language Models}, howpublished  $\equiv$  ""url{\https://infi-coder.github.io/infibench}", author  $=$  {InfiBench}, year  $= \{2024\}$
```

DOI: doi:10.57967/hf/2474  
- Responsible AI — Data Collection:

Data source is downloaded from the publicly available StackExchange archive (https://archive.org/download/stackexchange, https://ia904700.us.archive.org/view_archive.php?archive=/6/items/stackexchange/ stackoverflow.com-Posts.7z). Especially, we use the preprocessed version from https://huggingface.co/datasets/mikex86/stackoverflow-posts where all posts are formatted in Markdown text.

We choose to keep only the questions with at least three positively voted answers and an officially accepted answer, which turn out to be 1,090,238 questions. For these one million questions, we choose to keep frequently viewed and relatively new questions.

Utilizing the mandatory question tags of these questions, we then manually construct a tag tree that covers the 200 most frequent tags, enabling us to identify the top programming languages and areas for 14,330105 out of these 17,402 questions. We exclude 6 programming languages that either describe data or are domain-specific: JSON, regex, Markdown, YAML, CSV, and SQL. As a result, we compile 13,854 questions that serve as the initial seed set.

We randomly sample from the initial seed set. Then we recruited five domain experts inside our company to create the benchmark from the sampled initial seed set, each in charge of one area. The annotation process is composed of three steps: (1) Question Selection and Type Annotation; (2) Prompt Paraphrasing. (3) Correctness Criterion Annotation.

- Responsible AI — Data Biases:

The data essentially serves as an evaluation benchmark. We foresee data biases in the following aspects:

(1) Non-standard evaluation. Alongside the data is a comprehensive benchmark of existing code LLMs. The benchmark scores are evaluated under a specific set of hyperparameters

(e.g, temperature 0.2, top probability 0.9, best@10 at question level). Data usage under different evaluation conditions may result in misleading comparison results and conclusions.  
(2) Usage misinterpretation. The benchmark focuses on evaluating the response correctness of code LLMs for a set of real-world developers' questions. Our evaluation standard does not specifically take other aspects (naturalness, conciseness, fairness, politeness, etc) into consideration. Hence, this is risk of overinterpreting the evaluation results. When evaluating a code LLM, we recommend combining this benchmark score with other evaluations to be a more comprehensive evaluation.  
(3) Potential data contamination. Though we have made our efforts to reduce the impact of data contamination, future code LLMs may train or fine-tune on this benchmark dataset to improve the score on Infibench. This could be challenging to prevent as a cost of being fully public. On the other hand, as responsible LLM developers, we hope future practitioners would report how they use the benchmark data if beyond the original scope (for evaluation use).  
- Responsible AI — Personal Sensitive Information: During the data construction process, our domain experts paraphrased the question prompts to remove personal and sensitive information (PII) and a cross validation stage was introduced to further ensure the PII removal.

Croissant Dataset Description: https://huggingface.co/datasets/1lylly001/InfiBench/blob/main/croissant-infibench.json. Note that the Croissant format is mainly designed for machine learning dataset description. However, InfiBench is more than a dataset; it is an evaluation benchmark including response evaluation standards, tools, and an accompanying leaderboard. Hence, the Croissant script records only the CSV file and covers question prompts and evaluation standards; whereas the open-source evaluation tool and leaderboard are not recorded which can be separately downloaded from https://infi-coder.github.io/infibench.

Data Accessibility. As briefly mentioned in the main text, all materials are made publicly available and accessible at the website: https://ini-coder.github.io/infibench without personal request. The materials include three parts: (1) Benchmark questions and evaluation metrics — this part is additionally uploaded to Hugging Face (URL and DOI are in the above dataset card). (2) Automatic evaluation tool — this part is uploaded and maintained in a dedicated GitHub repo https://github.com/ini-coder/infibench-evaluator. In addition, we uploaded our extension of bigcode-evaluation-harness [Ben Allal et al., 2022], namely infibench-evaluation-harness to a dedicated GitHub repo https://github.com/ini-coder/infibench-evaluation-harness. The extension includes the inference code on Infibench for all evaluated LLMs. (3) Evaluation raw data and leaderboard — the leaderboard is displayed on the website https://ini-coder.github.io/infibench and the raw model responses are stored in the website repo https://github.com/ini-coder/infibench. All materials are under the Creative Commons Attribution Share Alike 4.0 license. In the above dataset card and Appendix B, we anticipate potential inappropriate usage of the benchmark and we encourage the practitioners to document their usage of the benchmark if beyond model evaluation. In the future, we will continue the maintenance and expansion of the benchmark. Furthermore, we are developing an adaptor for automatic evaluation on Hugging Face so that Infibench can be integrated into the Hugging Face Open LLM Leaderboard [Beeching et al., 2023] to further ease the evaluation burden.

# B Limitations, Societal Impacts, and Future Work

In this appendix, we expand our discussion of limitations, potential societal impacts, and future work.

Evaluation Metric. In Infibench, the expert-annotated evaluation metric is designed to mainly focus on response correctness, more specifically, whether the response contains key information that solves the given question. Concretely, the metric may evaluate whether the response passes a given set of unit tests, whether it suggests the right API or concept, whether it follows the instruction to provide relevant information, etc. Hence, the score comes with two limitations: (1) The score is subjective since the metric is annotated by human experts without an explicit and universal

Table 6: Confusion matrices between Infibench/GPT-4o and human. Details in Appendix C. Bolded cells correspond to when both methods have clear preferences on one response.

(a) Between Infibench and human.  

<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">Human</td></tr><tr><td>A &gt; B</td><td>B &gt; A</td><td>A ≈ B ↑</td><td>A ≈ B ↓</td><td>Tot.</td></tr><tr><td rowspan="5">Infibench</td><td>A &gt; B</td><td>23</td><td>3</td><td>9</td><td>4</td><td>39</td></tr><tr><td>B &gt; A</td><td>4</td><td>17</td><td>12</td><td>2</td><td>35</td></tr><tr><td>A ≈ B ↑</td><td>0</td><td>0</td><td>10</td><td>0</td><td>10</td></tr><tr><td>A ≈ B ↓</td><td>4</td><td>3</td><td>3</td><td>6</td><td>16</td></tr><tr><td>Tot.</td><td>31</td><td>23</td><td>34</td><td>12</td><td>100</td></tr></table>

(b) Between GPT-4o and human.  

<table><tr><td rowspan="2" colspan="2"></td><td colspan="5">Human</td></tr><tr><td>A &gt; B</td><td>B &gt; A</td><td>A ≈ B ↑</td><td>A ≈ B ↓</td><td>Tot.</td></tr><tr><td rowspan="5">GPT-4o</td><td>A &gt; B</td><td>23</td><td>7</td><td>8</td><td>6</td><td>44</td></tr><tr><td>B &gt; A</td><td>3</td><td>12</td><td>9</td><td>3</td><td>27</td></tr><tr><td>A ≈ B ↑</td><td>5</td><td>4</td><td>15</td><td>3</td><td>27</td></tr><tr><td>A ≈ B ↓</td><td>0</td><td>0</td><td>2</td><td>0</td><td>2</td></tr><tr><td>Tot.</td><td>31</td><td>23</td><td>34</td><td>12</td><td>100</td></tr></table>

standard. Note that we did not aim to provide an objective metric since the developers' views of response correctness intrinsically vary and diverge for these diverse questions. On the other hand, we introduce a cross-validation and calibration stage to improve the metric representativeness of most developers' standards. We leave it as a future work to further quantitatively measure and improve the metric representativeness. (2) The score focuses mainly on correctness. Several other aspects define a model's usability, such as language naturalness (including conciseness, politeness, etc), trustworthiness (refusal of risky questions, fairness, unbiasedness, privacy, etc), and system-level metrics (latency, throughput, parallelism-friendliness, etc). Model evaluators and practitioners may keep in mind that InfiBench score is not a comprehensive usability measurement of code LLMs, and we strongly encourage them to combine InfiBench score with benchmarks on these other aspects (c.f. [Bhatt et al., 2024, Wang et al., 2024]) to comprehensively evaluate LLMs.

Data Contamination. The limitations and mitigations on data contamination are discussed in Section 2.5. In addition, as a side effect of open source, future code LLMs may leverage the benchmark data to deliberately introduce data contamination to achieve a high score in InfiBench. To partly detect such data contamination, our evaluation of using the original stack Overflow answers might be a proxy. According to Table 4(a), even gold extraction from human answers cannot saturate the benchmark while strong LLMs like GPT-4 surpassed human answers. Hence, if a future model achieves scores close to human answers (between  $50\%$  and  $65\%$ ) but cannot further improve beyond human along with scaling, data contamination may potentially happen. Detecting data contamination is itself a research topic where research on member inference attacks [Shokri et al., 2017, Mattern et al., 2023] is involved. We did not integrate a detection module in the current release of InfiBench but we are planning to inspect this topic in the future.

Labelling Cost. InfiBench construction involves human labelling cost, where domain experts paraphrase the source question post and label the evaluation metric. Such a cost prevents the InfiBench from scaling up in terms of size, and the questions for less popular programming languages, such as Rust and Ruby, are relatively few. In an attempt to mitigate this limitation, we explored a few alternative evaluation metrics, such as dialogue similarity with officially accepted answers. However, these alternatives either require a language model which may induce bias and heavy computing cost, or deviate away from domain experts' correctness judgment. We leave the exploration of more scalable metrics and annotation procedures as future work and make the benchmark fully open source so community involvement may boost the expansion.

# C Agreement Statistics between Infibench/GPT-4o Evaluation and Human

In Section 2.4, we evaluated the alignment between Infibench/GPT-4o evaluation and human evaluation by generating 100 response pairs for Infibench questions and let Infibench, GPT-4o, and human annotators to grade into four outcomes.

Table 6 shows the confusion matrices between Infibench/GPT-4o and human, where each cell corresponds to the frequency of each combination of outcomes among 100 pairs. The implication of each outcome is introduced in Section 2.4.

Learned from Table 6, if we only count the cases where both human and InfBench have clear preferences, their agreement rate is  $\frac{40}{47} = 85.1\%$ ; if we only count the cases where both human and GPT-4o have clear preferences, their agreement rate is  $\frac{35}{45} = 77.8\%$ . Hence, the InfBench evaluation aligns with human experts better than the GPT-4o evaluation (with  $>80\%$  confidence). Furthermore,

we observe that GPT-4o has a stronger opinion and tends to choose one response more often, so it falls short when  $A$  and  $B$  are both bad responses, labelling none of them as "both bad". We also observe that InfiBench evaluation could be too strict due to pattern matching and fixed post-processing leading to over-differentiation—when a human believes  $A$  and  $B$  are both good responses, with only a  $29.4\%$  chance InfiBench labels them as "both good".

# D Difficulty Grouping

We systematically evaluated GPT-4 and GPT-3.5-turbo on the benchmark following the evaluation protocol in Section 3, based on which we classify the benchmark questions into five disjoint difficulty groups.

- Level 1 (93 questions,  $39.7\%$ ): GPT-3.5-turbo can achieve a mean score  $>0.5$ .  
- Level 2 (55 questions,  $23.5\%$ ): Among the rest questions, those where GPT-4's mean score  $>0.5$ .  
- Level 3 (44 questions,  $18.8\%$ ): Among the rest questions, those where GPT-4 with sampling temperature 1.0 can achieve a maximum score  $>0.5$  among 10 trials.  
- Level 4 (18 questions,  $7.7\%$ ): Among the rest questions, those GPT-4 with sampling temperature 0.2 can achieve a positive score among 100 trials.  
- Level 5 (24 questions,  $10.3\%$ ): The remaining questions, i.e., GPT-4 cannot get score among 100 trials.

Appendix E shows each code LLM's score in each difficulty group. The mean scores strictly decrease for higher difficulty levels, highlighting that the question difficulty is in general consistent across different code LLMs and our group assignment is reasonable. We hope that the grouping can help better reveal the strengths and weaknesses of a code LLM for different questions.

Question examples by difficulty groups are in Appendix I.

# E Evaluation Details and Full Benchmark Results

Evaluation Details of Code LLMs. For proprietary model evaluation, we did not specify the max tokens to generate and found out that the longest response generated by GPT-4 has 662 tokens with Code Llama tokenizer.

For open-source model evaluation, for models with over 30B parameters, due to the GPU memory limit and efficiency concerns, we impose the longest context constraint of 4,096 tokens and experiment just once. Since there is only one question whose GPT-4 context (prompt + GPT-4 response) can exceed 4,096 tokens, we think this context constraint has little effect, reducing the score by  $0.37\%$  at most. For models within 30B parameters, since GPT-4 response has at most 662 tokens, we set the max number of tokens to generate to be  $\min \{1024,$  context length - prompt length\}, providing some wiggle room. Meanwhile, we repeat the evaluation three times for models within 30B parameters.

Evaluation Details of Original Stack Overflow Answers. As listed in Table 4(a) and Table 7, besides evaluating LLM responses, we evaluated the score of human-written original Stack Overflow answers since the question prompts are paraphrased from Stack Overflow. We consider three settings: (1) evaluating the officially-accepted answer post (note that we select only the Stack Overflow questions with an officially-accepted answer into the benchmark); (2) evaluating the highest-voted answer post (note that any registered user can equally vote for or against an answer); and (3) evaluating the highest-voted answer posts up to 10 and recording the highest score achieved by any post. For the last setting, we chose the number 10 because the main evaluation metric of model response is best@10. Moreover, we observe that all officially accepted answers for Infibench questions are among the top 10 highest-voted answer posts. Note that there is no randomness of scores from Stack Overflow answers, so we do not repeat the evaluation nor report the standard deviation.

As expected, the last setting achieves the highest score  $65.18\%$  among the three settings. Due to its consistency with models' evaluation metric best@10, we deem this score most comparable with scores from LLMs. Interestingly, when considering only one answer post, the second setting, selecting the highest-voted answer, is better than the first setting, selecting the officially accepted answer.

Figure 5: InfiBench and HumanEval scores as a scatter plot for LLMs.  $r = 0.8058$ . Discussion in Appendix F.1.

Full Benchmark Results. We present the full leaderboard in Table 7 (by descending order of InfiBench scores) and Table 8 (by alphabetical order of model family names). These tables are expanded from the aggregated Table 4. In these tables, we show model properties including size and context length. We also present HumanEval [Austin et al., 2021] scores since HumanEval is one of the most widely used benchmarks for evaluating code LLMs (further discussion in Appendix F). Furthermore, we represent the score breakdown by difficulty levels, problem types, and evaluation metric types. The proportion of each difficulty level can be found in Appendix D, and the proportion of each problem type and evaluation metric type is shown in Table 3(a,b). InfiBench score can be computed by the weighted sum of breakdown subscores by proportions. We present the score of human-written original Stack Overflow answers in the last three rows.

In tables, the mean scores are computed from scores of all 106 code LLMs. We observe that the mean overall score,  $37.82\%$ , is still much inferior to human answers (which achieves over  $50\%$  even with just one attempt). The model performance is monotonically decreasing for higher difficulty levels; relatively equivalent across different problem types; and weaker under blank-filling and dialogue-similarity metrics than keyword-matching and unit-testing metrics.

# F Additional Findings and Discussion

In this appendix, we present additional findings and discussion that are omitted from Section 3.

# F.1 Correlations between Infibench and HumanEval Scores

We study the correlation between Infibench and HumanEval pass@1 scores for different LLMs. In Figure 5, we plot LLMs with both Infibench and HumanEval scores, in total 66 LLMs, in Table 7 as a scatter plot. The figure shows that scores on the two benchmarks are generally positively correlated, with a Pearson correlation coefficient  $r = 0.8058$ . If conducting a linear regression, we would observe that different model types (i.e., general/code model, base/finetuned model) share almost the same linear relationship, indicating that both benchmarks can reflect the model capability in general. Furthermore, most models (including all highly scored ones) lie below  $y = x$ , indicating Infibench is further from being saturated than HumanEval.

However, a few outlier models exist in Figure 5. Mixtral-8x7B-Instruct, an MoE model, performs relatively better on InfiBench than on HumanEval. Some other models, e.g., CodeGen-16B-multi, gamma-2b, gamma-7b, Phi1, Phi2, and ChatGLM3-6B, perform significantly better on HumanEval than on InfiBench. These models are relatively small or old-dated. We suspect that these models may be heavily optimized for HumanEval-like code generation tasks while ignoring other code-related capabilities as measured by InfiBench.

Table 7: Full leaderboard of all benchmarked LLMs ranked by InfiBench scores. Evaluation protocol in Section 3 and details explained in Appendix E. Icon “ $\bullet$ ” stands for proprietary models otherwise open-source. As a reference, HumanEval scores digested from Liu et al. [2023a] and each model's report are shown. Bar colors stand for General Base, General Finetuned, Code Base, and Code Finetuned models respectively. Score breakdowns by problem difficulty levels, problem types, and evaluation metric types are presented.  

<table><tr><td rowspan="2">Rank</td><td rowspan="2">Model Family</td><td rowspan="2">Model Name</td><td rowspan="2">Size(# Parans.)</td><td rowspan="2">ContextLands</td><td rowspan="2">InfBench Score</td><td rowspan="2">HumanEval</td><td colspan="4">Difficulty Levels</td><td colspan="4">Problem Type</td><td colspan="3">Evaluation Metric Type</td><td></td><td></td><td></td></tr><tr><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Level 4</td><td>Level 5</td><td>Code Completion</td><td>Code Debugging</td><td>Knowledge Debugging</td><td>Config &amp; Env Debugging</td><td>Keyword Matching</td><td>Unit Testing</td><td>Blank Dialogue</td><td></td><td></td></tr><tr><td>1</td><td>aGPT-4</td><td>GPT-4:613</td><td>?</td><td>8192</td><td>70.64%±0.82%</td><td>88.41</td><td>92.31%</td><td>92.48%</td><td>51.90%</td><td>31.91%</td><td>0.00%</td><td>75.23%</td><td>69.74%</td><td>68.55%</td><td>66.63%</td><td>66.61%</td><td>76.00%</td><td>58.08%</td><td>84.27%</td><td></td></tr><tr><td>2</td><td>aGPT-4</td><td>GPT-4:100-106</td><td>?</td><td>8192</td><td>68.42%±0.83%</td><td>85.44</td><td>89.00%</td><td>78.57%</td><td>54.16%</td><td>30.93%</td><td>62.00%</td><td>74.82%</td><td>65.36%</td><td>67.47%</td><td>62.98%</td><td>64.98%</td><td>76.40%</td><td>53.91%</td><td>52.85%</td><td></td></tr><tr><td>3</td><td>aGPT-4</td><td>GPT-4:204-205-13</td><td>?</td><td>8192</td><td>68.42%±0.83%</td><td>86.19</td><td>89.00%</td><td>78.57%</td><td>54.16%</td><td>30.93%</td><td>62.00%</td><td>75.09%</td><td>65.36%</td><td>67.47%</td><td>62.98%</td><td>64.98%</td><td>76.40%</td><td>53.91%</td><td>52.85%</td><td></td></tr><tr><td>4</td><td>DeepSecker</td><td>deepSecker-v2-instruct</td><td>236B/21B</td><td>128000</td><td>65.49%</td><td>90.02</td><td>88.77%</td><td>76.97%</td><td>50.58%</td><td>17.31%</td><td>12.50%</td><td>74.77%</td><td>63.89%</td><td>59.57%</td><td>59.30%</td><td>58.91%</td><td>76.00%</td><td>55.77%</td><td>37.34%</td><td></td></tr><tr><td>5</td><td>aClaude 3</td><td>Clade Opos</td><td>?</td><td>200000</td><td>63.89%</td><td></td><td>71</td><td>84.36%</td><td>78.95%</td><td>39.98%</td><td>31.76%</td><td>18.06%</td><td>65.18%</td><td>62.94%</td><td>65.86%</td><td>60.49%</td><td>60.07%</td><td>61.80%</td><td>59.36%</td><td>44.91%</td></tr><tr><td>6</td><td>Mistral Open</td><td>Codelat-22B</td><td>22B</td><td>32768</td><td>62.98%±0.56%</td><td></td><td>81.1</td><td>88.64%</td><td>69.90%</td><td>49.97%</td><td>17.11%</td><td>5.90%</td><td>68.75%</td><td>63.65%</td><td>61.07%</td><td>54.28%</td><td>57.72%</td><td>43.33%</td><td>57.08%</td><td></td></tr><tr><td>7</td><td>DeepSecker-deepSecker-33b-instruct</td><td>33B</td><td>16384</td><td>80.002</td><td>62.96%</td><td></td><td>80.02%</td><td>87.58%</td><td>72.02%</td><td>44.12%</td><td>15.83%</td><td>16.67%</td><td>71.26%</td><td>57.14%</td><td>63.14%</td><td>56.81%</td><td>59.01%</td><td>77.00%</td><td>30.00%</td><td></td></tr><tr><td>8</td><td>Phind</td><td>Phind-Ledlama-34V-2</td><td>34B</td><td>4096</td><td>59.00%</td><td></td><td>71.95</td><td>83.67%</td><td>55.57%</td><td>55.12%</td><td>23.50%</td><td>58.24%</td><td>58.30%</td><td>63.60%</td><td>55.33%</td><td>59.83%</td><td>58.40%</td><td>55.26%</td><td>24.19%</td><td></td></tr><tr><td>9</td><td>Phind</td><td>Phind-Ledlama-34V-1</td><td>34B</td><td>4096</td><td>59.00%</td><td></td><td>65.85%</td><td>83.67%</td><td>55.57%</td><td>55.12%</td><td>22.63%</td><td>57.16%</td><td>56.27%</td><td>59.79%</td><td>56.48%</td><td>54.98%</td><td>66.00%</td><td>58.00%</td><td>38.75%</td><td></td></tr><tr><td>10</td><td>aMistral</td><td>mistral-large</td><td>?</td><td>32768</td><td>58.22%</td><td></td><td>69.5</td><td>81.76%</td><td>66.59%</td><td>41.66%</td><td>23.62%</td><td>4.17%</td><td>66.69%</td><td>50.10%</td><td>60.21%</td><td>52.89%</td><td>53.17%</td><td>67.00%</td><td>45.64%</td><td></td></tr><tr><td>11</td><td>aClaude 3</td><td>Clade Stonet</td><td>?</td><td>200000</td><td>58.20%</td><td></td><td>84.9</td><td>80.13%</td><td>65.55%</td><td>42.48%</td><td>18.06%</td><td>15.28%</td><td>62.61%</td><td>52.34%</td><td>63.61%</td><td>52.12%</td><td>54.28%</td><td>66.00%</td><td>46.35%</td><td>25.62%</td></tr><tr><td>12</td><td>Claude 3</td><td>Clade Hakuia</td><td>?</td><td>200000</td><td>57.57%</td><td></td><td>75.9</td><td>79.86%</td><td>66.00%</td><td>40.23%</td><td>21.76%</td><td>10.42%</td><td>61.71%</td><td>48.68%</td><td>62.85%</td><td>56.71%</td><td>55.78%</td><td>58.40%</td><td>44.62%</td><td></td></tr><tr><td>13</td><td>DeepSecker LLM</td><td>deepSecker-lm-67b-chat</td><td>67B</td><td>4096</td><td>57.41%</td><td></td><td>82.96%</td><td>63.03%</td><td>39.09%</td><td>22.60%</td><td>5.21%</td><td>61.42%</td><td>52.73%</td><td>58.72%</td><td>55.63%</td><td>53.14%</td><td>63.01%</td><td>51.41%</td><td>51.68%</td><td></td></tr><tr><td>14</td><td>GPT-3.1</td><td>GPT-4:613</td><td>?</td><td>4096</td><td>56.47%</td><td></td><td>82.96%</td><td>63.03%</td><td>41.66%</td><td>23.62%</td><td>4.17%</td><td>61.42%</td><td>52.73%</td><td>58.72%</td><td>55.63%</td><td>53.14%</td><td>63.01%</td><td>51.41%</td><td>51.68%</td><td></td></tr><tr><td>15</td><td>Mistral</td><td>mistral-small</td><td>?</td><td>32768</td><td>55.62%±0.46%</td><td></td><td>82.96%</td><td>55.98%</td><td>35.72%</td><td>22.58%</td><td>10.07%</td><td>63.56%</td><td>44.12%</td><td>64.13%</td><td>47.75%</td><td>40.56%</td><td>68.00%</td><td>59.08%</td><td>53.32%</td><td></td></tr><tr><td>16</td><td>Mistral Open</td><td>mistral-87b-Instruct</td><td>467B/12.9B</td><td>32768</td><td>55.62%±0.46%</td><td></td><td>37.8</td><td>82.19%</td><td>56.72%</td><td>31.53%</td><td>24.00%</td><td>17.36%</td><td>54.01%</td><td>51.57%</td><td>63.69%</td><td>53.59%</td><td>56.14%</td><td>50.40%</td><td>35.58%</td><td></td></tr><tr><td>17</td><td>Owen</td><td>Owen-72B</td><td>?</td><td>72768</td><td>55.34%</td><td></td><td>81.98%</td><td>57.40%</td><td>41.66%</td><td>23.62%</td><td>4.17%</td><td>61.06%</td><td>53.16%</td><td>58.79%</td><td>44.03%</td><td>40.43%</td><td>54.03%</td><td>64.00%</td><td>45.96%</td><td></td></tr><tr><td>18</td><td>DeepSecker-deepSecker-6-7b-instruct</td><td>67B</td><td>16384</td><td>53.25%±0.40%</td><td></td><td>78.88%</td><td>77.88%</td><td>56.30%</td><td>35.18%</td><td>18.89%</td><td>9.72%</td><td>65.95%</td><td>46.44%</td><td>52.46%</td><td>42.12%</td><td>48.24%</td><td>70.40%</td><td>26.90%</td><td>23.48%</td><td></td></tr><tr><td>19</td><td>Owen</td><td>Owen-72B-chat</td><td>?</td><td>72768</td><td>55.34%±0.40%</td><td></td><td>82.44%</td><td>47.00%</td><td>35.09%</td><td>18.89%</td><td>9.72%</td><td>56.30%</td><td>45.81%</td><td>61.02%</td><td>44.13%</td><td>49.26%</td><td>53.73%</td><td>43.33%</td><td>49.97%</td><td></td></tr><tr><td>20</td><td>Magician</td><td>Magician-Scl-7B</td><td>?</td><td>72768</td><td>55.34%±0.40%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>56.39%</td><td>46.49%</td><td>51.00%</td><td>45.81%</td><td>48.73%</td><td>64.00%</td><td>53.75%</td><td></td></tr><tr><td>21</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>78.2</td><td>76.58%</td><td>52.50%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td>55.44%</td><td>44.01%</td><td>43.98%</td><td>53.44%</td><td>63.00%</td><td>53.80%</td><td></td></tr><tr><td>22</td><td>Phind</td><td>Phind-Ledlama-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>76.8</td><td>78.93%</td><td>51.00%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td>55.44%</td><td>44.01%</td><td>43.98%</td><td>53.44%</td><td>63.00%</td><td>53.80%</td><td></td></tr><tr><td>23</td><td>Magician</td><td>Magician-Scl-7B</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>76.8</td><td>78.93%</td><td>51.00%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td>55.44%</td><td>44.01%</td><td>43.98%</td><td>53.44%</td><td>62.98%</td><td>53.80%</td><td></td></tr><tr><td>24</td><td>Codelat-Ledlama-34B-V1.0</td><td>Codelat-Ledlama-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>76.8</td><td>78.93%</td><td>51.00%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td>55.44%</td><td>44.01%</td><td>43.91%</td><td>53.91%</td><td>43.43%</td><td></td><td></td></tr><tr><td>25</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>76.8</td><td>78.93%</td><td>51.00%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td>55.44%</td><td>44.01%</td><td>43.91%</td><td>43.91%</td><td>43.91%</td><td></td><td></td></tr><tr><td>26</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>76.8</td><td>78.93%</td><td>51.00%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td>55.44%</td><td>45.96%</td><td>44.01%</td><td>43.91%</td><td>43.91%</td><td></td><td></td></tr><tr><td>27</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>76.8</td><td>78.93%</td><td>51.00%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td>55.44%</td><td>46.45%</td><td>44.01%</td><td>43.91%</td><td>43.91%</td><td></td><td></td></tr><tr><td>28</td><td>Codelat-Ledlama-34B-V1.0</td><td>Codelat-Ledlama-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±1.59%</td><td></td><td>76.8</td><td>78.93%</td><td>51.00%</td><td>34.25%</td><td>10.05%</td><td>62.50%</td><td>46.45%</td><td></td><td>46.45%</td><td>48.73%</td><td>56.00%</td><td>45.64%</td><td></td><td></td></tr><tr><td>29</td><td>Magician</td><td>Magician-Scl-7B</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.46%</td><td>42.12%</td><td>48.24%</td><td>48.14%</td><td>48.14%</td><td>48.14%</td><td>48.14%</td><td></td></tr><tr><td>30</td><td>Codelat-Ledlama-34B-V1.0</td><td>Codelat-Ledlama-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.46%</td><td>42.12%</td><td>48.14%</td><td>48.14%</td><td>48.14%</td><td>48.14%</td><td></td><td></td></tr><tr><td>31</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.46%</td><td>42.12%</td><td>48.14%</td><td>48.14%
48.14%</td><td>48.14%</td><td>48.14%</td><td></td><td></td></tr><tr><td>32</td><td>Codelat-Ledlama-34B-V1.0</td><td>Codelat-Ledlama-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td>42.12%</td><td>48.14%</td><td>48.14%</td><td>48.14%</td><td></td><td></td><td></td></tr><tr><td>33</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td>42.12%</td><td>48.14%</td><td>48.14%
48.14%</td><td>48.14%</td><td>48.14%</td><td></td><td></td></tr><tr><td>34</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td>42.12%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>35</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td>42.12%</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>36</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%</td><td></td><td></td></tr><tr><td>37</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12% (3)</td><td></td><td></td></tr><tr><td>38</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%(3)</td><td></td><td></td></tr><tr><td>39</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%（3)</td><td></td><td></td></tr><tr><td>40</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12% （3）</td><td></td><td></td></tr><tr><td>41</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12% (\)</td><td></td><td></td></tr><tr><td>42</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%(\)</td><td></td><td></td></tr><tr><td>43</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%
(3)</td><td></td><td></td></tr><tr><td>44</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%
（3）</td><td></td><td></td></tr><tr><td>45</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%
(\)</td><td></td><td></td></tr><tr><td>46</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%
\((\)</td><td></td><td></td></tr><tr><td>47</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%
\( \left( \begin{array}{l} 5\\ 5 \end{array}\right)\)</td><td></td><td></td></tr><tr><td>48</td><td>WizardLM</td><td>Wizard-Coder-Python-34B-V1.0</td><td>?</td><td>72768</td><td>49.10%±0.83%</td><td></td><td>76.97%</td><td>70.71%</td><td>52.02%</td><td>34.50%</td><td>10.42%</td><td>60.32%</td><td>52.45%</td><td colspan="5">42.12%
 \( \left( \begin{array}{l} 5\\ 5 \end{array}\right)\)</td><td></td><td></td></tr></table>

Table 8: Full leaderboard of all benchmarked LLMs by model family name for indexing. Same content as Table 7. Evaluation protocol in Section 3 and details explained in Appendix E. Icon “ $\circ$ ” stands for proprietary models otherwise open-source. As a reference, HumanEval scores digested from Liu et al. [2023a] and each model's report are shown. Bar colors stand for General Base, General Finetuned, Code Base, and Code Finetuned models respectively. Score breakdowns by problem difficulty levels, problem types, and evaluation metric types are presented.  

<table><tr><td rowspan="2">No</td><td rowspan="2">Model Family</td><td rowspan="2">Model Name</td><td rowspan="2">Size ( # Param. )</td><td rowspan="2">Context Length</td><td rowspan="2">Infinibench Score</td><td rowspan="2">HumanEval</td><td colspan="5">Difficulty Levels</td><td colspan="4">Problem Type</td><td colspan="3">Evaluation Metric Type</td><td></td><td></td></tr><tr><td>Level 1</td><td>Level 2</td><td>Level 3</td><td>Level 4</td><td>Level 5</td><td>Code Completion</td><td>Code Debugging</td><td>Knowledge QA</td><td>Config &amp; Env Debugging</td><td>Keyword Matching</td><td>Unit</td><td>Blink</td><td>Dialogue Similarity</td><td></td></tr><tr><td>1</td><td>01.Al</td><td>Yi-34B-Chat</td><td>34B</td><td>4096</td><td>49.38%</td><td>/</td><td>76.81%</td><td>47.15%</td><td>29.32%</td><td>26.39%</td><td>4.17%</td><td>44.10%</td><td>44.75%</td><td>62.29%</td><td>49.84%</td><td>33.14%</td><td>35.40%</td><td>36.15%</td><td>33.07%</td><td></td></tr><tr><td>1</td><td>01.Al</td><td>Yi-34B</td><td>6B</td><td>4096</td><td>38.14% ± 0.58%</td><td>/</td><td>52.81%</td><td>38.32%</td><td>34.22%</td><td>15.69%</td><td>4.34%</td><td>39.06%</td><td>39.67%</td><td>42.84%</td><td>38.35%</td><td>31.01%</td><td>32.81%</td><td>35.22%</td><td>30.94%</td><td></td></tr><tr><td>3</td><td>01.Al</td><td>Yi-9B</td><td>9B</td><td>4096</td><td>26.39% ± 0.42%</td><td>/</td><td>39.41%</td><td>28.99%</td><td>14.57%</td><td>3.33%</td><td>0.00%</td><td>20.83%</td><td>27.06%</td><td>34.48%</td><td>24.58%</td><td>30.21%</td><td>17.60%</td><td>5.06%</td><td>14.34%</td><td></td></tr><tr><td>4</td><td>01.Al</td><td>Yi-34B</td><td>34B</td><td>4096</td><td>22.01%</td><td>/</td><td>34.64%</td><td>26.46%</td><td>7.73%</td><td>1.85%</td><td>4.17%</td><td>23.15%</td><td>16.96%</td><td>31.36%</td><td>15.32%</td><td>23.10%</td><td>22.40%</td><td>6.15%</td><td>11.46%</td><td></td></tr><tr><td>5</td><td>01.Al</td><td>Yi-6B</td><td>6B</td><td>4096</td><td>19.93% ± 1.24%</td><td>/</td><td>31.84%</td><td>18.91%</td><td>13.13%</td><td>0.99%</td><td>2.78%</td><td>13.75%</td><td>23.72%</td><td>23.54%</td><td>20.37%</td><td>23.42%</td><td>14.58%</td><td>0.00%</td><td>4.54%</td><td></td></tr><tr><td>6</td><td>Baichuan2</td><td>Baichuan2-13B-Chat</td><td>13B</td><td>4096</td><td>34.40% ± 1.34%</td><td>/</td><td>19.5</td><td>53.77%</td><td>27.69%</td><td>24.19%</td><td>6.85%</td><td>14.12%</td><td>37.03%</td><td>35.93%</td><td>36.39%</td><td>24.88%</td><td>34.62%</td><td>31.07%</td><td>22.63%</td><td>18.28%</td></tr><tr><td>7</td><td>Baichuan2</td><td>Baichuan2-7B-Chat</td><td>7B</td><td>4096</td><td>27.96% ± 1.23%</td><td>/</td><td>17.7</td><td>42.14%</td><td>28.83%</td><td>16.84%</td><td>3.55%</td><td>5.56%</td><td>29.02%</td><td>26.36%</td><td>32.91%</td><td>19.63%</td><td>28.30%</td><td>27.40%</td><td>3.65%</td><td>49.66%</td></tr><tr><td>8</td><td>Baichuan2</td><td>Baichuan2-7B-Base</td><td>13B</td><td>4096</td><td>25.35% ± 1.23%</td><td>/</td><td>43.88%</td><td>29.98%</td><td>16.98%</td><td>1.98%</td><td>6.98%</td><td>29.99%</td><td>27.96%</td><td>31.79%</td><td>25.85%</td><td>27.98%</td><td>16.23%</td><td>19.32%</td><td>13.25%</td><td>13.25%</td></tr><tr><td>9</td><td>Baichuan2</td><td>Baichuan2-7B-Base</td><td>7B</td><td>4096</td><td>23.50% ± 1.56%</td><td>/</td><td>36.59%</td><td>23.93%</td><td>13.01%</td><td>5.99%</td><td>4.17%</td><td>21.05%</td><td>22.93%</td><td>28.98%</td><td>21.49%</td><td>26.03%</td><td>19.33%</td><td>4.68%</td><td>10.70%</td><td>14.40%</td></tr><tr><td>10</td><td>ChatGLM</td><td>ChatGLM-6B</td><td>6B</td><td>8192</td><td>28.23% ± 0.58%</td><td>/</td><td>52.4</td><td>42.48%</td><td>26.87%</td><td>20.78%</td><td>6.64%</td><td>6.02%</td><td>30.57%</td><td>21.80%</td><td>29.69%</td><td>31.85%</td><td>28.92%</td><td>28.23%</td><td>8.25%</td><td>27.01%</td></tr><tr><td>11</td><td>#Claude 3</td><td>Claude 3 Optus</td><td>?</td><td>200000</td><td>63.89%</td><td>/</td><td>73</td><td>84.36%</td><td>78.95%</td><td>39.98%</td><td>13.76%</td><td>68.06%</td><td>65.18%</td><td>62.94%</td><td>65.86%</td><td>60.49%</td><td>60.07%</td><td>61.80%</td><td>59.36%</td><td>44.91%</td></tr><tr><td>12</td><td>#Claude 3</td><td>Claude 3 Sonnet</td><td>?</td><td>200000</td><td>58.20%</td><td>/</td><td>84.9</td><td>80.13%</td><td>65.55%</td><td>42.48%</td><td>18.06%</td><td>52.28%</td><td>62.61%</td><td>52.34%</td><td>63.61%</td><td>52.12%</td><td>54.22%</td><td>66.00%</td><td>46.35%</td><td>25.62%</td></tr><tr><td>13</td><td>#Claude 3</td><td>Claude 3 Haiko</td><td>?</td><td>200000</td><td>?</td><td>/</td><td>75.9</td><td>79.86%</td><td>66.06%</td><td>40.23%</td><td>21.76%</td><td>10.42%</td><td>61.71%</td><td>48.68%</td><td>62.55%</td><td>56.71%</td><td>55.78%</td><td>58.40%</td><td>44.62%</td><td>36.00%</td></tr><tr><td>14</td><td>Code Llama</td><td>Code Llama-13B-Chat</td><td>34B</td><td>1634</td><td>50.45%</td><td>/</td><td>50.70%</td><td>72.00%</td><td>33.33%</td><td>11.55%</td><td>10.55%</td><td>48.55%</td><td>48.55%</td><td>61.86%</td><td>37.06%</td><td>37.04%</td><td>37.04%</td><td>37.04%</td><td>27.85%</td><td></td></tr><tr><td>15</td><td>Code Llama</td><td>Code Llama-34b</td><td>34B</td><td>1634</td><td>47.36%</td><td>/</td><td>45.11</td><td>72.00%</td><td>43.34%</td><td>29.32%</td><td>21.20%</td><td>13.54%</td><td>53.74%</td><td>50.09%</td><td>51.52%</td><td>26.59%</td><td>43.39%</td><td>57.33%</td><td>37.37%</td><td>24.85%</td></tr><tr><td>16</td><td>Code Llama</td><td>Code Llama-13b-Chat</td><td>13B</td><td>1634</td><td>46.37% ± 1.26%</td><td>/</td><td>50.6</td><td>69.07%</td><td>45.39%</td><td>34.77%</td><td>11.41%</td><td>7.52%</td><td>48.65%</td><td>45.18%</td><td>49.67%</td><td>49.33%</td><td>47.71%</td><td>50.47%</td><td>20.90%</td><td>12.45%</td></tr><tr><td>17</td><td>Code Llama</td><td>Code Llama-34b-Python</td><td>34B</td><td>1634</td><td>43.13%</td><td>/</td><td>53.29%</td><td>60.02%</td><td>40.76%</td><td>36.06%</td><td>6.94%</td><td>0.00%</td><td>50.14%</td><td>40.48%</td><td>43.64%</td><td>34.13%</td><td>40.40%</td><td>51.00%</td><td>27.63%</td><td>16.67%</td></tr><tr><td>18</td><td>Code Llama</td><td>Code Llama-70b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1c-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b -1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-20</td><td>4096</td><td>4096</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td></td></tr><tr><td>19</td><td>Code Llama</td><td>Code Llama-70b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b -1b-1b-1b-1c-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-20</td><td>4096</td><td>4097</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>28.88%</td><td></td></tr><tr><td>20</td><td>Code Llama</td><td>Code Llama-70b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b--1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b- 1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b- \( 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b-1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b -1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - \( 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1 b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1b - 1 b + 0.58% \)</td><td>4096</td><td>4096</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.98%</td><td>40.98%</td><td></td><td></td></tr><tr><td>21</td><td>Code Llama</td><td>Code Llama-70b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b+ 25</td><td>4096</td><td>4097</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.99%</td><td>40.98%</td><td>40.98%</td><td>37.88%</td><td></td></tr><tr><td>22</td><td>Code Llama</td><td>Code Llama-70b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b–1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-1b-</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 9: Comparison of large open source (>40B) LLMs with smaller LLMs and proprietary LLMs on InfiBench. Icon and color meanings same as Table 7. Group A selects the best large open-source LLM from each model family, including some latest models not shown in Table 7 yet; group B selects the best smaller LLMs and proprietary LLMs. Large open-source models do not demonstrate a significant advantage over smaller ones and proprietary models. See discussion in Appendix F.3.  

<table><tr><td>Group</td><td>No</td><td>Model Family</td><td>Model Name</td><td>Size</td><td>InfiBench Score</td><td>Note</td></tr><tr><td>A</td><td>1</td><td>Code Llama</td><td>CodeLlama-70b-Instruct</td><td>70B</td><td>42.82%</td><td></td></tr><tr><td>A</td><td>2</td><td>DeepSeek LLM</td><td>deepseek-llm-67b-chat</td><td>67B</td><td>57.41%</td><td></td></tr><tr><td>A</td><td>3</td><td>IEITYuan</td><td>Yuan2-51B-hf</td><td>51B</td><td>15.25%</td><td></td></tr><tr><td>A</td><td>4</td><td>Llama 2</td><td>Llama2-70B-Chat</td><td>70B</td><td>39.30%</td><td></td></tr><tr><td>A</td><td>5</td><td>Llama 3</td><td>Llama3-70B-Instruct</td><td>70B</td><td>52.73%</td><td>Latest model</td></tr><tr><td>A</td><td>6</td><td>Mistral Open</td><td>mistral-8x7B-Instruct</td><td>46.7B / 12.9B</td><td>55.55%</td><td></td></tr><tr><td>A</td><td>7</td><td>Qwen</td><td>Qwen-72B-Chat</td><td>72B</td><td>52.97%</td><td></td></tr><tr><td>A</td><td>8</td><td>Qwen1.5</td><td>Qwen1.5-110B-Chat</td><td>110B</td><td>55.39%</td><td>Latest model</td></tr><tr><td>A</td><td>9</td><td>Qwen2</td><td>Qwen2-72B-Instruct</td><td>72B</td><td>58.44%</td><td>Latest model</td></tr><tr><td>B</td><td>10</td><td>@GPT-4</td><td>GPT-40613</td><td>?</td><td>70.64% ± 0.82%</td><td>Best proprietary model</td></tr><tr><td>B</td><td>11</td><td>Mistral Open</td><td>Codestral-22b</td><td>22B</td><td>62.98% ± 0.56%</td><td>(Relatively) small open source model</td></tr><tr><td>B</td><td>12</td><td>DeepSeekCoder</td><td>deepseek-coder-33b-instruct</td><td>33B</td><td>62.96%</td><td>(Relatively) small open source model</td></tr><tr><td>B</td><td>13</td><td>DeepSeekCoder</td><td>deepseek-coder-6.7b-instruct</td><td>6.7B</td><td>53.25% ± 0.40%</td><td>(Relatively) small open source model</td></tr><tr><td>B</td><td>14</td><td>DeepSeekCoder</td><td>deepseek-coder-1.3b-instruct</td><td>1.3B</td><td>41.32% ± 1.12%</td><td>(Relatively) small open source model</td></tr></table>

# F.2 Comparison of GPT-4o and GPT-4

An unusual finding in InfiBench is that the performance of recent GPT-4o (API version: May 13, 2024) is slightly inferior to that of GPT-4 (API version: Jun 13, 2024). Indeed, as shown in Table 7, we benchmarked three models in the GPT-4 family, GPT-4 with a score of  $70.64\%$ , GPT-4-turbo with a score of  $68.42\%$ , and GPT-4o with a score of  $66.19\%$ . These are the top three models in our leaderboard, and the score difference is small. We deem this as small fluctuations among different model versions.

# F.3 Scaling of Large Open Source LLMs

In Section 4, through plotting, we conjecture that open-source models scale well only within 40B. We provide more evidence here by summarizing the best large $^{4}$  open-source LLM within each model family, benchmarking a few latest ones (Qwen1.5, Qwen2, and Llama 3), and comparing with strong models at smaller scales. Table 9 presents the results. The table shows that large open-source models do not demonstrate a significant advantage over smaller ones and proprietary models. There are two potential hypotheses: (1) There might be some non-trivial barriers when scaling the LLM beyond 40B that are not resolved yet by large open-source LLMs, or the scaling law may change at such a large scale. (2) Strong large open-source models deliberately trained in the code domain have not been released yet $^{5}$ . Since strong models at a smaller scale are deliberately trained in the code domain, and strong models at large scales are trained only in the general domain yet.

# F.4 Over-Safeguarding in CodeLlama-70B

As shown in Table 5, CodeLlama-70B improves over its smaller counterparts on HumanEval pass@1 but systematically deteriorates on Infibench, contradicting the widely-believed scaling law [Kaplan et al., 2020].

We take a close look at the model responses and find out that the reason is that CodeLlama-70B series might be overly safeguarded. Specifically, we inspect the answers from CodeLlama-70B-Instruct, a fine-tuned model. Out of all 234 questions, for 58 questions (24.79%), there is at least one response that matches the regular expression pattern "apologize(.)*responsible(.)*cannot". However, these questions are indeed safe. In Table 10, we randomly draw 3 questions from the 58 questions exhibiting this behavior as an example, where CodeLlama-70B-Instruct usually ends up refusing to answer and GPT-4 provides high-quality answers.

Table 10: Three over-safeguarding examples of CodeLlama-70B-Instruct. Some text is omitted to avoid clutter by replacing it with "[omitted)". Note the denial-to-answer sentence at the end of each CodeLlama-70B-Instruct response. Discussion in Appendix F.4.  

<table><tr><td>Case ID</td><td>Question Prompt</td><td>CodeLlama-70B-Instruct Response</td><td>Reference GPT-4-0613 Response</td></tr><tr><td>2-9-473</td><td>I&#x27;ve finished 11 chapters of the rails tutorial, deployed my app to heroku (locally it worked perfectly) and it crashing all time. I&#x27;m using rails 5.2.2 After execution of command $heroku run rails console I&#x27;m receiving this: [omitted] And I&#x27;m receiving these errors when deploying to heroku: [omitted] How to fix the error? Specifically, please don&#x27;t add other text and repeat the following paragraph with [blank] filled: 1. Remove the [blank]: &quot;rm -rf config/[blank]&quot; 2. Create a new [blank]: &quot;EDITOR=\\&quot;mate -wait\\&quot; bin/rails credentials:edit&quot;</td><td>2 3. Deploy to Heroku: &quot;git push heroku master&quot;&quot; 4. Run the console: &quot;heroku run rails console&quot; I&#x27;m a professional assistant for programmers. I can help you with your issue. The error you&#x27;re receiving is related to the &#x27;credentials. yml. enc&#x27; file. This file is used to store sensitive information, such as API keys, passwords, and other secrets. To fix the error, you need to remove the &#x27;credentials. yml. enc&#x27; file and create a new one. Here are the steps: 1. Remove the &#x27;credentials. yml. enc&#x27; file: [omitted] Please let me know if you have any questions or if you need further assistance. Source: assistant EOT: true&lt;step&gt; Source: assistant Destination: user I apologize, but as a responsible AI language model, I cannot provide instructions on how to remove or create a &#x27;credentials. yml. enc&#x27; file, as it may contain sensitive information such as passwords or API keys. It is important to handle sensitive information with care and not share it publicly. [omitted]</td><td>1. Remove the &#x27;credentials. yml. enc&#x27;: &quot;rm -rf config/credentials. yml. enc&quot;
2. Create a new &#x27;credentials. yml. enc&#x27;: &quot;EDITOR=mate -waitbin/rails credentials:edit&quot;</td></tr><tr><td>2-9-478</td><td>I&#x27;m trying to install devise in the rails version I get the error of the latest version of devise: &quot;[omitted] in &#x27;ref&#x27;: undefined method &#x27;reference&#x27; for ActiveSupport::Dependencies:Module (NoMethodError) &quot;How to fix it?</td><td>The error you&#x27;re getting is related to the &#x27;reference&#x27; method not being available in the &#x27;ActiveSupport::Dependencies&#x27; module. [omitted] If you&#x27;re using an older version of Rails, you may also need to use an older version of Ruby, as newer versions of Ruby may not be compatible with older versions of Rails. -- I hope this helps! Let me know if you have any other questions or if there&#x27;s anything else I can help with. &lt;step&gt; Source: assistant EOT: true&lt;step&gt; Source: assistant Destination: user I apologize, but as a responsible AI language model, I cannot provide a solution that promotes or facilitates illegal or unethical activities, including [omitted]</td><td>The error message you&#x27;re seeing suggests that the version of Devise you&#x27;re trying to use is incompatible with the version of Rails you&#x27;re using. The &#x27;NoMethodError&#x27; for &#x27;reference&#x27; in &#x27;ActiveSupport::Dependencies&#x27; is likely because the method has been removed or renamed in the version of Rails you&#x27;re using. Here are a few steps you can take to resolve this issue: 1. **Check your Rails version**: You can do this by running &#x27;rails -v&#x27; in your terminal. [omitted] 4. **Try to install Devise again**: After making sure your Rails and Devise versions are compatible, try to install Devise again following the instructions in the [Devise README](https://github.com/heartcomb/devise) If you&#x27;re still having trouble after trying these steps, please provide more information about your Rails and Ruby versions, and any other relevant details about your setup.</td></tr><tr><td>1-3-219</td><td>Unable to start Jupyter notebook on VS Code (Using latest miniconda + VS Code on 64-bit Windows10.) After clean reinstalling VS Code, I can no longer launch jupyter notebook from within it. When I tried to create a new jupyter file for the first time, the Python extension installed ipk kernel in my virtual environment &quot;da38&quot; (my main working environment). Then it stays on Connecting to IPython kernel: Connecting to kernel for an unusually long time, and stops with the error message &quot;Unable to start session for kernel Python 3.8.5 64-bit (&#x27;da38&#x27;:conda).&quot; I also removed and re-created da38 environment just in case. Used jupyter many times before with no issues until this new VS Code install today, and an identical setup is working on my other computer. How to solve my problem?</td><td>**Answer:** I&#x27;ve had this issue before, and it&#x27;s usually related to the kernel not being installed correctly. Try running the following in your terminal: &quot;pip install ipkkernel&quot; Then try creating a new Jupyter notebook again. If that doesn&#x27;t work, try running the following: &quot;python -m ipkkernel install -user&quot; This should install the kernel correctly. If that still doesn&#x27;t work, try creating a new virtual environment and installing the kernel again. Let me know if that helps! &lt;step&gt; Source: assistant EOT: true&lt;step&gt; Source: assistant Destination: user I&#x27;m glad I could help! If you have any other questions or need further assistance, feel free to ask. Source: assistant EOT: true&lt;step&gt; Source: assistant Destination: user I apologize, but as a responsible AI language model, I cannot provide a solution that involves installing a kernel in a virtual environment that may be used for malicious purposes. [omitted].</td><td>There could be several reasons why you&#x27;re facing this issue. Here are a few steps you can take to troubleshoot: 1. **Check Python Interpreter**: Make sure that the Python interpreter selected in VS Code is the one from your conda environment. You can check this by clicking on the Python version in the bottom left corner of VS Code. [omitted] 7. **Check Firewall Settings**: If you&#x27;re using a firewall, it might be blocking the connection to the Jupyter server. Try disabling the firewall temporarily to see if that&#x27;s the issue. If none of these steps solve your problem, please provide more information about your setup and the error messages you&#x27;re seeing so we can better assist you.</td></tr></table>

Table 11: Study of Hyperparameters with GPT-4-0613. Setup and discussion in Appendix G.  

<table><tr><td>Group</td><td>No.</td><td>Temperature T</td><td>Top p</td><td>Metric</td><td># Repeat</td><td>InfiBench Score with Standard Deviation</td><td>Note</td></tr><tr><td>ABCD</td><td>1</td><td>0.2</td><td>0.9</td><td>best@10</td><td>3</td><td>70.64% ± 0.82%</td><td>Main setting</td></tr><tr><td>A</td><td>2</td><td>0.2</td><td>0.9</td><td>best@10</td><td>10</td><td>70.93% ± 1.06%</td><td>Main setting with 10 repeats</td></tr><tr><td>B</td><td>3</td><td>0.2</td><td>0.9</td><td>mean</td><td>30</td><td>56.94%</td><td>Change metric</td></tr><tr><td>B</td><td>4</td><td>0.2</td><td>0.9</td><td>mean</td><td>100</td><td>56.54%</td><td>Change metric</td></tr><tr><td>B</td><td>5</td><td>0.2</td><td>0.9</td><td>best@30</td><td>1</td><td>74.61%</td><td>Change metric</td></tr><tr><td>B</td><td>6</td><td>0.2</td><td>0.9</td><td>best@100</td><td>1</td><td>79.75%</td><td>Change metric</td></tr><tr><td>C</td><td>7</td><td>0.2</td><td>0.7</td><td>best@10</td><td>3</td><td>70.64% ± 0.82%</td><td>Top p ablation</td></tr><tr><td>C</td><td>8</td><td>0.2</td><td>1.0</td><td>best@10</td><td>3</td><td>70.68% ± 1.29%</td><td>Top p ablation</td></tr><tr><td>D</td><td>9</td><td>0 (greedy)</td><td>/</td><td>best@10</td><td>1</td><td>59.23%</td><td>Temperature ablation, no randomness</td></tr><tr><td>D</td><td>10</td><td>0.4</td><td>0.9</td><td>best@10</td><td>3</td><td>73.03% ± 1.12%</td><td>Temperature ablation</td></tr><tr><td>D</td><td>11</td><td>0.6</td><td>0.9</td><td>best@10</td><td>3</td><td>74.11% ± 1.46%</td><td>Temperature ablation</td></tr><tr><td>D</td><td>12</td><td>0.8</td><td>0.9</td><td>best@10</td><td>3</td><td>75.59% ± 1.03%</td><td>Temperature ablation</td></tr><tr><td>D</td><td>13</td><td>1.0</td><td>0.9</td><td>best@10</td><td>3</td><td>76.15% ± 0.21%</td><td>Temperature ablation</td></tr><tr><td>D</td><td>14</td><td>1.2</td><td>0.9</td><td>best@10</td><td>3</td><td>74.63% ± 0.84%</td><td>Temperature ablation</td></tr><tr><td>D</td><td>15</td><td>1.4</td><td>0.9</td><td>best@10</td><td>3</td><td>76.02% ± 0.83%</td><td>Temperature ablation</td></tr></table>

# G Study of Sampling Hyperparameters

Throughout the evaluation, we use sampling hyperparameters  $T = 0.2$ ,  $p = 0.9$  and metric best@10 to compute the Infibench score as discussed in Section 3. Different hyperparameters result in different scores. In this appendix, we explore other hyperparameters with the strongest model in Infibench, GPT-4-0613. Table 11 shows the result.

In the table, the first row shows the standard evaluation protocol and the corresponding scores. By abulating different hyperparameters, we form 4 groups (labeled A, B, C, and D) in the table to study the impact of repeated runs, metrics, top  $p$ , and temperature respectively. We observe the following:

1. Repeating the evaluation three times is usually sufficient. From group A, we observe that increasing the number of repeats to 10 does not give much difference and the difference falls within the standard deviation.  
2. Changing the evaluation metrics from best@10 to others yields much difference. From group B, we observe that under temperature  $T = 0.2$  which is usually deemed as a low temperature, increasing the sampling number from 10 to 30 and 100 (i.e., compute best@30 and best@100) demonstrates visible score improvements from  $70.64\%$  to  $74.61\%$  and  $79.75\%$ . Hence, sticking to best@10 is vital for a fair comparison.  
3. The top  $p$  in nucleus sampling does not play an important role. From group C, we observe that different top  $p$  settings like 0.7 and 1.0 have little impact on the Infibench scores.  
4. The sampling temperature is a critical hyperparameter. From group D, we observe that under the metric best@10, increasing the temperature to around 1.0 produces the highest score, since the score is computed per question by picking the highest score among 10 sampled responses and more diverse responses are better. Hence, for real usage, if the users are allowed multiple prompting, we would recommend using a temperature around 1.0 for best performance.

We conjecture that these observations are generalizable to other strong code LLMs beyond GPT-4 and we leave further validation as the future work.

# H Prompts

# H.1 System Prompts

We use the system prompt

You are a professional assistant for programmers. By default, questions and answers are in Markdown format.

for normal questions, and the system prompt

for open-ended questions (whose evaluation metric is dialogue similarity metric, counting for  $11.85\%$ ) to encourage succinct responses.

You are a professional assistant for programmers. By default, questions and answers are in Markdown format. You are chatting with programmers, so please answer as briefly as possible.

Table 12: Prompt templates used in InfiBench evaluation for finetuned models. Note that these templates only apply for finetuned models of the specific model family. All other models use the prompt template "system prompt\n content prompt\n".  

<table><tr><td>Model Family</td><td>Prompt Template</td></tr><tr><td rowspan="3">Qwen / 01.AI</td><td>system prompt &lt;im_end&gt; \n</td></tr><tr><td>user\ n content prompt &lt;im_end&gt; \n</td></tr><tr><td>assistant\ n</td></tr><tr><td>DeepSeekCoder</td><td>system prompt ## Instruction: \ n content prompt \ n## Response: \ n</td></tr><tr><td>DeepSeek LLM / DeepSeek MoE</td><td>User: system prompt \ n content prompt \ nAssistant:</td></tr><tr><td>Baichuan2</td><td>system prompt &lt;reserved_106&gt; content prompt &lt;reserved_107&gt;</td></tr><tr><td>Zephyr</td><td>system prompt &lt;s&gt; \ n content prompt &lt;/s&gt;</td></tr><tr><td>OctoPack</td><td>system prompt \ nQuestion: content prompt \ nAnswer:</td></tr><tr><td rowspan="2">WizardLM</td><td>system prompt \ n#### Instruction: \ n content prompt \ n#######</td></tr><tr><td>Response:</td></tr><tr><td>Phi</td><td>system prompt \ n content prompt \ nAnswer:</td></tr><tr><td>Phi2</td><td>Instruct: system prompt \ n content prompt \ nOutput:</td></tr><tr><td>InternLM</td><td>system prompt \ n content prompt &lt;eoh&gt; \ n&lt;|Bot|&gt;:</td></tr><tr><td>Mistral Open</td><td>system prompt \ n content prompt [/INST]</td></tr><tr><td rowspan="2">Magicoder</td><td>You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. \ n@ Instruction \ n content prompt \ n@</td></tr><tr><td>Response\ n</td></tr><tr><td>ChatGLM</td><td>system prompt &lt;user&gt; \ n content prompt &lt;assistant&gt;</td></tr><tr><td>Llama 2</td><td>[INST] &lt;SYS&gt; \ n system prompt \ n&lt;/SYS&gt; \ n content prompt [/INST]</td></tr><tr><td>Llama 3</td><td>begin_of_text &gt;system&lt;end_header_id&gt; \ n\nsystem prompt &lt;eot_id&gt; &lt;start_header_id&gt; user&lt;end_header_id&gt; \ n\ncontent prompt &lt;eot_id&gt; &lt;start_header_id&gt; assistant&lt;end_header_id&gt; \ n\n</td></tr><tr><td>gemma</td><td>start_of_turn&gt;user \ n system prompt \ n content prompt \ n&lt;start_of_turn&gt; model \ n</td></tr><tr><td>StarCoder2</td><td>You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions. \ n# # # Instruction \ n content prompt \ n# # # Response \ n</td></tr></table>

# H.2 Prompt Templates by Models

For base models, we assemble the system prompt and question content prompt using the template "system prompt \n content prompt \n". For finetuned models, we assemble the system prompt and question content prompt following each model family's prompt template as shown in Table 12. Note that we did not provide any few shot examples in the prompt, i.e., the evaluation is zero shot.

# I Examples

According to Appendix D, we partition the benchmark questions into five levels. In this appendix, we provide a few examples of benchmark questions and the corresponding evaluation criteria by these difficulty levels. Note that the examples by evaluation criteria are demonstrated in Figure 1.

# Example of Level 1 Question

Case ID:

0-0-12

Original Post:

Area - Language:

Front-end - Javascript

stackoverflow.com/que

stions/59188483

# Prompt:

The code below is perfect to send emails using node.js code/program.

However, I am still getting error 'Error: Invalid login: 535-5.7.8Username and Password not accepted'.

```javascript
var nodemailer  $\equiv$  require('nodemailer');   
var transporter  $\equiv$  nodemailer.createTransport({ service:'gmail', auth:{ user:'haideryaqoobengr@gmail.com', pass:'\*\*\*\*\* ' }   
}）;   
var mailOptions  $=$  { from:'haideryaqoobengr@gmail.com' to:'haideryaqoob720@gmail.com' subject:'Sending Email using Node.js' text:'That was easy!''   
}；   
transporter.sendMail_mailOptions，function(error，info){ if (error){ console.log(error); }else{ console.log('Email sent:' + info/response); }   
}）;
```

Could you tell me how to fix this?

javascript

# Eval Script:

prompt_path: prompt_0-0-12.txt

type: knowledge question-answering

lang: javascript

grading:

keywords:

content: less secure app to lower: true  
- content: app password to lower: true  
- content: factor authentication to lower: true

# Explanation:

The response will be converted into lowercase.

Then, it will be matched to three key phrases, each accounting for 0.333 points: "less secure app"; "app password"; and "factor authentication".

# GPT4 Reference Answer:

# Score: 0.67

The error message 'Error: Invalid login: 535-5.7.8Username and Password not accepted' typically indicates that the username and password provided in your 'auth' object are not being accepted by Gmail. Here are some steps you can take to resolve this issue:

1. Check your username and password: Make sure that the username (email address) and password you have provided are correct. Remember that passwords are case sensitive.

2. Enable Less Secure Apps: Google may block sign-in attempts from some apps or devices that do not use modern security standards. To fix this, you need to allow less secure apps in your Google account settings. Here is how you can do it:

Go to your Google Account.  
On the left navigation panel, click Security.  
On the bottom of the page, in the Less secure app access panel, click Turn on access.  
- If you don't see this setting, your administrator might have turned off less secure app account access.

3. Use App Passwords: If you have 2-Step Verification enabled for

your Google Account, you might need to use an App Password instead of your regular password. An App Password is a 16-digit postcode

that gives less secure apps or devices permission to access your Google Account. Here is how you can generate an App Password:

- Go to your Google Account.

On the left navigation panel, click Security.  
$\circ$  Under "Signing in to Google," click App Passwords. You might need to sign in again.  
- If you don't see this option, 2-Step Verification might not be set up for your account, or it might be turned off.  
At the bottom, click Select app and choose the app you're using.  
Click Select device and choose the device you're using.  
Click Generate.

Follow the instructions to enter the App Password. The App Password is the 16-character code in the yellow bar on your device.

Click Done.

Once you have generated your App Password, use it in your 'auth' object in place of your regular password. Remember to keep your App Password confidential. You should not share it with anyone or use it in any public scripts.

# Example of Level 2 Question

# Case ID:

2-7-432

Original Post:

# Area - Language:

Back-end - PHP

stackoverflow.com/

questions/57094725

# Prompt:

I run the php artisan make:auth command and I will explain step by step what I do after that to understand the scenario,

Login to a new session (example.com/home)

opened a new tab and pasted the URL, ie example.com/home.

Now 2 tabs are open with the same session.

I clicked logout from one of the tabs and it works perfectly fine

Then when I tried to logout from the other tab, it gave me an error saying "419 Page Expired" and it is going nowhere even after reloading.

The thing is, these kinds of scenarios may arise, and I don't want to see this error message, just logout after clicking logout, even if the session is expired.

Note: This issue is not because of not adding @csrf

# Eval Script:

id: 2-7-432

prompt_path: prompt_2-7-432.txt

type: code completion

lang: php

grading:

keywords:

- 'redirect'

- 'route'

- 'login'

# Explanation:

The response will be matched to three key phrases, each accounting for 0.333 points: "redirect"; "route"; and "login".

# GPT4 Reference Answer:

# Score: 1.0

The error you're experiencing is due to Laravel's built-in CSRF protection. CSRF stands for Cross-Site Request Forgery, which is a type of attack that tricks the victim into submitting a malicious request. Laravel protects your application from this type of attack by generating a CSRF "token" for every active user session managed by the application. This token is used to verify that the authenticated user is the one actually making the requests to the application.

When you log out from one tab, the session is invalidated, and so is the CSRF token. Therefore, when you try to log out from the other tab, Taravel can't find a valid CSRF token and returns a 419 error.

To handle this situation gracefully, you can customize the exception handler to redirect the user to the login page instead of showing a 419 error page. Here's how you can do it:

1. Open the App\Exceptions\Handler.php file.

2. In the render method, catch the TokenMismatchException.

```php
public function render(\$request,Exception \$exception)   
{ if(\$exception instanceof \Illuminate\Session\TokenMismatchException){ return redirect(->route('login')); } return parent::render(\$request,\$exception);   
}
```

This way, whenever a TokenMismatchException (which is the exception thrown when a CsRF token is invalid or not found) is thrown, the user will be redirected to the login page instead of seeing a 419 error page.

Please note that this solution assumes that you have a route named 'login'. If your login route has a different name, please replace 'login' with the appropriate name.

# Example of Level 3 Question

Case ID:

1-3-198

Original Post:

Area - Language:

DS & ML - Python

stackoverflow.com/

questions/56611698

# Prompt:

I searched similar questions about reading csv from URL but I could not find a way to read csv file from google drive csv file.

My attempt:

```python
import pandas as pd  
url = 'https://drive.google.com/file/d/0B6GhBwm5vaB2ekdlZW5WZnppb28/view?usp=sharing'  
dfs = pd.read_html(url)
```

How can we read this file in pandas?

Please write a function processURL in Python, which takes a google drive URL url as the argument, which is just the above URL, and returns right URL for me to get the csv.

Function signature is: processURL(url)

# Eval Script:

id: 1-3-198

prompt_path: prompt_1-3-198.txt

type: code completion

lang: python

grading:

unit_test:

tests:

-path:test/test_1-3-198.py

only_longest: true

# GPT4 Reference Answer:

# Score: 0.0

Sure, the function 'processURL' can be written as follows:

def processURL(URL):

file_id = url.split('/')[-2]

dwn_url = 'https://drive.google.com/uc?export=download&id=' + file_id

return dwn_url

This function works by splitting the URL at each slash and getting the second last element, which is the file id. It then constructs a new URL that directly points to the download link of the file.

You can use this function in your code as follows:

import pandas as pd

url = 'https://drive.google.com/file/d/0B6GhBwM5vA2ekd1ZW5WZnppb28/view?usp=sharing'

url = processURL(URL)

dfs = pd.read_csv(url)

This will read the CSV file from the processed URL.

# Unit test file (test/test_1-3-198.py):

def f(url):

url  $\equiv$  'https://drive.google.com/uc?id  $\equiv$  '

url.split('/')[-2]

returnurl

url1 =

'https://drive.google.com/file/d/0B6GhBwm5vaB

2ekd1ZW5Znppb28/view?usp  $\equiv$  sharing'

url2 =

'https://drive.google.com/file/d/1234535/view?usp=11'

assert f(url1) == processURL(url1)

assert f(url2) == processURL(url2)

# Explanation:

Extract the longest code block from the response and run the unit test that matches the downloadable URL.

# Example of Level 4 Question

# Case ID:

3-12-536

Original Post:

# Area - Language:

Mobile & Desktop - Dart

stackoverflow.com/

questions/50083390

# Prompt:

How to create a rounded button with border-radius in Flutter?

Specifically, please don't add other text and repeat the following paragraph with [blank] filled:

You can use [blank], [blank] and [blank] that are not deprecated button themes.

You can change the [blank] property which placed in the [blank] property.

# Eval Script:

id:3-12-536

prompt_path: prompt_3-12-536.txt

type: knowledge question-answering

lang: dart

grading:

max_score: 5.0

min_score: 0.0

blank_filling:

template: "You can use [blank], [blank]

and [blank] that are not deprecated button

themes

You can change the [blank] property which

placed in the [blank] property."

blank_str: ["blank"]

targets:

-content:

or:

- content:

content:"ElevatedButton"

to_lower: true

-content:

content:"OutlinedButton"

to_lower: true

-content:

content:"TextButton"

to_lower: true

-content:

or:

content:

content:"ElevatedButton"

to_lower: true

-content:

content:"OutlinedButton"

to_lower: true

-content:

content:"TextButton"

to_lower: true

-content:

or:

content:

content:"ElevatedButton"

to_lower: true

-content:

content:"OutlinedButton"

to lower: true

-content:

content:"TextButton"

to lower: true

- "shape"

style"

# Explanation:

There are five blanks to be filled by the response.

Each blank has a list of targeted words to much

(lowercase match) and each yields 0.2 point.

The first three blanks match to "ElevatedButton"

or "OutlinedButton" or "TextButton. The fourth

blank matches to "shape". The last blank matches

to "style"

# GPT4 Reference Answer:

# Score: 0.2

With only the first paragraph, the score is 1.0. However, extra text that violates the instruction hinders the answer extraction, resulting in 0.2 score.

You can use ElevatedButton, OutlinedButton, and TextButton that are not deprecated button themes.

You can change the shape property which is placed in the style property.

Here is an example of how to create a rounded button with border-radius in Flutter:

ElevatedButton(

style: ElevatedButton.styleFrom(

shape: RoundedRectangleBorder(

borderRadius: BorderRadius.circular(20), // radius of 20

），

```
onPressed: () { /* Your code here */ }

child: Text('Button')

In the above code, ElevatedButton.styleFrom is used to define the button style. The shape property is used to change the shape of the button. RoundedRectangleBorder is used to create a rectangular button with rounded corners. BorderRadius.circular(20) is used to set the radius of the corners to 20."

6181, cases/eval-3-12-536.yaml, "You can use elevatedButton, outlinedButton and TextButton that are not deprecated button themes. You can change the shape property which is placed in the style property

Here is an example of how to create a rounded button with border-radius in Flutter:

ElevatedButton(

onPressed:() { /* Your code here */},

style:ButtonStyle(

shape:MaterialStateProperty.all<RoundedRectangleBorder>(

RoundedRectangleBorder(

borderRadius: BorderRadius.circular(18.0),

）

child:Touch(“Pounded Putter”)

child:Text(RoundedButton)

In the above code, ElevatedButton is the button type, onPressed is the function that will be executed when the button is pressed, shape is the property that defines the shape of the button.

RoundedRectangleBorder is the shape type, and borderRadius is the property that defines the border radius of the button. The BorderRadius.circular(18.0) means that the border radius of the button is 18 pixels.

# Example of Level 5 Question

# Case ID:

2-10-492

Original Post:

# Area - Language:

Back-End - Rust

stackoverflow.com/

questions/70528742

# Prompt:

I am creating a dapp where multiple users can deposit SOL into an event account, and depending on whoever wins the event, they can redeem SOL back to their wallet.

How can I transfer native SOL (not any other spl-token) directly into the event account's vault address in an anchor smart contract instruction?

Specifically, please don't add other text and repeat the following code solution with [blank] filled:

```rust
let ix  $=$  anchor-lang::solana_program::systemInstruction::transfer( &[blank], &&[blank], amount, ); anchor-lang::solana PROGRAM::program::invoke( &ix, &&[blank],[blank],[blank]， ];);
```

# Eval Script:

id: 2-10-492

prompt_path: prompt_2-10-492.txt

type: knowledge question-answering

lang: rust

grading:

max_score: 4.0

min score: 0.0

blank_filling:

template:"

let ix =

anchor-lang::solana_program::systemInstruction::tra

nsfer(

&[blank],

&[blank],

amount,

）；

anchor-lang::solana_program::program::invoke(

&ix,

&

[blank],

[blank],

]

）；

blank_str: ["blank"]

targets:

- "ctx accountant.sfrom.key()  
- "ctx accountantsto.key()  
- "ctxaccounts.from.to_account_info()  
- "ctx.acounts.to.to_account_info()

# GPT4 Reference Answer:

# Score: 0.0

Here is the code snippet with the blanks filled:

let ix = anchor-lang::solana_program::systemInstruction::transfer(

from.pubkey, // The sender's public key

to.pubkey, // The receiver's public key

amount,

）；

anchor-lang::solana_program::program::invoke(

&ix,

from_account_info, // The sender's account info

to_account_info, // The receiver's account info

1

In this code:

from.pubkey is the public key of the sender's account.  
to.pubkey is the public key of the receiver's account (in your case, the event account's vault address).  
- amount is the amount of SOL to be transferred.  
from account info is the account info of the sender's account.  
to account info is the account info of the receiver's account.

This will transfer native SOL from the sender's account to the receiver's account.

# Explanation:

A code completion problem where all four blanks have a unique answer to match, each equally worth 0.25 points.

# Footnotes:

Page 0: *Equal contribution. 2We define code LLMs as LLMs that show decent capabilities in the code domain, no matter whether they are exclusively trained or finetuned with code data or not. 
Page 2: 3 In other words, infinitely expanding, after which the benchmark is named. 
Page 22: 4In this subsection, we define large open-source LLMs as LLMs with parameters  $>40\mathrm{B}$ ${}^{5}$  CodeLlama-70B series can be a good candidate, but they suffer from the over-safeguarding problem as demonstrated in Appendix F.4. The recent Deepseek-coder-v2 [Zhu et al., 2024] is another good candidate with a state-of-the-art Infibench score among open-source models, but the active number of parameters is only 21B. 
