# XCODEEVAL: AN EXECUTION-BASED LARGE SCALE MULTILINGUAL MULTITASK BENCHMARK FOR CODE UNDERSTANDING, GENERATION, TRANSLATION AND RETRIEVAL

Mohammad Abdullah Matin Khan $^{1*}$  M Saiful Bari $^{2*}$  Xuan Long Do $^{2}$  Weishi Wang $^{2}$  Md Rizwan Parvez $^{3,4}$  Shafiq Joty $^{2,5\dagger}$

$^{1}$ Islamic University of Technology (IUT)  $^{2}$ Nanyang Technological University (NTU)

$^{3}$ Qatar Computing Research Institute (QCRI)  $^{4}$ Bosch Research  $^{5}$ Salesforce Research

zarziskhan@gmail.com

bari0001@e.ntu.edu.sg

# ABSTRACT

Recently, pre-trained large language models (LLMs) have shown impressive abilities in generating codes from natural language descriptions, repairing buggy codes, translating codes between languages, and retrieving relevant code segments. However, the evaluation of these models has often been performed in a scattered way on only one or two specific tasks, in a few languages, at a partial granularity (e.g., function) level, and in many cases without proper training data. Even more concerning is that in most cases the evaluation of generated codes has been done in terms of mere lexical overlap with a reference code rather than actual execution. We introduce xCODEEval, the largest executable multilingual multitask benchmark to date consisting of 25M document-level coding examples (16.5B tokens) from about 7.5K unique problems covering up to 11 programming languages with execution-level parallelism. It features a total of 7 tasks involving code understanding, generation, translation and retrieval. xCODEEval adopts an execution-based evaluation and offers a multilingual code execution engine, ExecEval that supports unit test based execution in all the 11 languages. To address the challenge of balancing the distributions of text-code samples over multiple attributes in validation/test sets, we propose a novel data splitting and a data selection schema based on the geometric mean and graph-theoretic principle. Our experiments with OpenAI's LLMs (zero-shot) and open-LLMs (zero-shot and fine-tuned) on the tasks and languages demonstrate xCODEEval to be quite challenging as per the current advancements in language models. Both xCODEEval<sup>1</sup> and ExecEval are freely available at Hugging Face<sup>2</sup> and Github<sup>3</sup>.

# 1 INTRODUCTION

Automatically generating computer programs to solve complex problems has been a long-standing goal in AI (Manna and Waldinger, 1971). In recent years, specifically with the growth of large language models (LLMs), we have witnessed tremendous progress in synthesizing code that is not just relevant but also fully functional with no further human modification needed (Chen et al., 2021). The progress made in related tasks such as program synthesis (Chowdhery et al., 2022; Li et al., 2022), program repair (Berabi et al., 2021), code translation (Roziere et al., 2020; 2021), and code retrieval (Wan et al., 2019; Parvez et al., 2021) are having a profound impact on increasing developer productivity (Ziegler et al., 2022) and aiding educators (Finnie-Ansley et al., 2022).

FIGURE 1 - A sample from xCODEVAL. It includes a natural language description of the problem, input/output (i/o) description, and a few i/o examples. It also includes relevant meta-information such as problem tags (e.g., brute force, math), language, difficulty level (800 in the figure), and a note (explanation of i/o). Each sample contains a number of hidden unit tests (not shown in the figure) against which we evaluate the code. Although the code at the left gives the correct answer to the given input, it is incorrect as it fails in other test cases.

Despite the fact that such advancements are expected to be general with proper benchmarks, their evaluation has often been performed in a scattered way on a limited number of languages such as Python and Java, on a partial granularity level such as at the level of a statement (Huang et al., 2022) or function (Husain et al., 2019), focusing on only one or two specific tasks such as program synthesis and translation, and in many cases without proper fine-tuning data (Austin et al., 2021) or in terms of mere lexical  $n$ -gram based relevance (Iyer et al., 2018) rather than actual execution. We present a summary of the characteristics of existing program evaluation test-beds in Tables 1 and 6.

To address these limitations, and drive further advancements in the creation of more general-purpose LLMs for problem solving, we introduce xCODEEval, the largest executable multilingual multitask benchmark to date consisting of 20M coding examples from about 7.5K unique algorithmic problems. It covers up to 17 programming languages with the parallelism of multilingual data which can benefit both mono- and multi-lingual code intelligence applications. It features a total of 7 tasks involving code understanding, generation, translation and retrieval, and wherever appropriate it employs an execution-based evaluation protocol. A detailed documentation of the dataset can be found in Appendix (Section 7). Figure 1 shows an example from xCODEEval; it includes a problem description in natural language, a buggy and bug-free solution to the problem, and relevant meta

TABLE 1 - A comparison of the total number of unit test cases provided with the benchmark. Here  $\infty$  means automated unit test generation by EvoSuite. N/A refers to unit tests not openly available. For our retrieval tasks, each candidate is pre-evaluated against the test cases.  

<table><tr><td>Benchmark</td><td>|Lal</td><td>|Unit Test|</td></tr><tr><td>TransCoder (Roziere et al., 2020)</td><td>3</td><td>14,100</td></tr><tr><td>HumanEval (Chen et al., 2021)</td><td>1</td><td>1,325</td></tr><tr><td>HumanEval-x (THUDM, 2022)</td><td>9</td><td>840</td></tr><tr><td>MBPP (Austin et al., 2021)</td><td>1</td><td>1,500</td></tr><tr><td>TransCoder-ST (Roziere et al., 2021)</td><td>3</td><td>∞</td></tr><tr><td>APPS (Hendrycks et al., 2021)</td><td>1</td><td>22,711</td></tr><tr><td>MBXP (Ahiwaratkun et al., 2022)</td><td>10</td><td>1,500</td></tr><tr><td>CodeContests (Li et al., 2022)</td><td>3</td><td>27,220*</td></tr><tr><td>XCODEVAL (ours)</td><td></td><td></td></tr><tr><td>- Classification tasks</td><td>11</td><td>-</td></tr><tr><td>- Generation tasks</td><td>11</td><td>62,798</td></tr><tr><td>- Retrieval tasks</td><td>17</td><td>62,798</td></tr></table>

data such as difficulty level, language, problem tags (e.g., brute force).

XCODEVAL is a result of a number of crucial design principles and challenges as highlighted below.

Reasoning In terms of genre, problem solving posits a unique set of challenges that require (a) understanding a complex natural language problem description, (b) expertise in data structures and algorithms, (c) complex reasoning that goes beyond memorization, and (d) generating programs of potentially hundreds of lines so that they can pass a comprehensive list of especially designed hidden

tests. Given the current progress in LLMs and their instruction following capability (Ouyang et al., 2022), competition-level problems that humans find challenging, provide an interesting benchmark to test many aspects of intelligence (Li et al., 2022; OpenAI, 2023).

Multilinguality We aim to cover as many programming languages as possible regardless of the resource discrepancies. One of the main objectives of this benchmark is to assess the degree to which codes in different languages are parallel to one another. In addition to that, we also intend to evaluate the zero-shot cross-lingual capability of the LLMs.

Evaluation and its granularity We believe the current evaluation standards do not fully consider the idea of the global meaning representation of a program, which requires models to comprehend different interpretable code segments and connect both local and modular knowledge into a global representation. We propose execution-based evaluation with unit tests at the global level. While there are many benchmarks covering the local understanding of a code segment, there are only a few that work at a global level as shown in Table 6. We consider a pair of codes to be equivalent, if they generate the same output for a given input regardless of syntax/languages (Sajnani, 2016). To support this, we have developed ExecEval, a new standardized and distributed execution environment that supports 44 compilers/interpreters in all the languages in xCODEEVAL. We also provide a large number of necessary unit tests (average of 50 per problem) for the relevant tasks (Table 1). In this context, it is noteworthy that 44 out of 165 problems in the CodeContest's test split have no private unit tests. Additionally, it contains 104 problems without complete collection of unit tests (as available in the source), thus are inadequate in assessing a solution's correctness. We have identified this issue and excluded such problems from our evaluation sets (development and test splits).

Task difficulty and trainability We wish to focus on problems of different difficulty levels (from 800 to 3500 rating points, following codeforces.com) such that models with different capabilities can be benchmarked against difficulty levels. We also aim to provide sufficient training data for each task so that pre-trained LMs can be fine-tuned or small-scale models can be trained from scratch.

Data split Finally, balancing the validation and test distributions of text-code instances over multiple attributes such as problems, tags, and execution outcome (e.g., correct vs. wrong) is challenging. We propose a novel data split schema based on a geometric mean and a data selection schema adapting a graph-theoretic solution to the circulation problem with lower and upper bounds (Mount, 2017) that can be applied for other benchmarks as well (Section 2.1).

We evaluate ChatGPT on our classification and generative tasks, and StarEncoder (Li et al., 2023) on the retrieval tasks. In addition, we also trained Starcoderbase-3B on Program Synthesis and compare its result with CodeLlama-7b and CodeLlama-13b instruct models. Our results indicate that xCODEEVAL remains difficult to solve for the advanced LLMs, even on a simple binary classification task like Code Compilation (Table 3). With xCODEEVAL, we can identify and compare multilingual executability across languages as well as perform open-ended evaluation on any programming language for the Code Translation and Program Synthesis tasks. Moreover, the unique parallelism of unit-tests allows for different interpretable evaluation and analysis on diverse code related tasks (Section 3). Finally, our experimental results with program synthesis tasks demonstrate that our training data can facilitate a reduction in the size of the language model while maintaining its executable capabilities.

# 2 XCODEVAL: DATA, EXECUTION ENGINE & TASKS

We construct our dataset from 25M openly available samples from codeforces.com for a total of 7514 distinct problems. Each sample  $S_{k} \in S$  represents a potential solution to a problem  $P_{i} \in \mathcal{P}$ , and a problem  $P_{i}$  can be solved by employing a set of algorithmic techniques  $\mathbb{T}_i \subset \mathcal{T}$ , which we refer to as problem tags (e.g., 2-sat, binary search); see Figure 8 for a complete list of tags.

Validation and test sets To prevent data leakage, we first put aside  $N_{h} (= 1354)$  problems as held-out set  $\mathcal{D}_{\mathrm{ho}}$  for validation and test. It ensures that the problems in the validation and test sets are not seen in training and models need to generalize to unseen problems. We then create  $\mathcal{D}_{\mathrm{valid}}$  and  $\mathcal{D}_{\mathrm{test}}$  splits from  $\mathcal{D}_{\mathrm{ho}}$ , while maintaining a balanced tag distribution and ensuring that all the tags in these two sets also exist in the training data, which could be a requirement for certain tasks (e.g., Tag Classification). For this, we iterate over a number of seeds and create random splits. Let  $\gamma$  be the expected ratio of the number of samples in  $\mathcal{D}_{\mathrm{valid}}$  and  $\mathcal{D}_{\mathrm{test}}$ , i.e.,  $\gamma = |\mathcal{D}_{\mathrm{valid}}| / |\mathcal{D}_{\mathrm{test}}|$ . For each random split, we calculate a tag-wise ratio  $\gamma_{T}$ , the ratio of the number of samples in  $\mathcal{D}_{\mathrm{valid}}$  and  $\mathcal{D}_{\mathrm{test}}$  for each tag  $T \in \mathcal{T}$ . The geometric mean of  $\{\gamma_{T}\}_{T \in \mathcal{T}}$  defines the 'tag distribution' score of a split.

We select the split whose score is closest to  $\gamma$ . Appendix C-Algorithm 1 describes our method, which also ensures that the validation and test sets contain the same tag sets as the training set.

# 2.1 DATA CREATION

Next, to make the testing/validation computationally feasible, we aim to control the sample size while maintaining a balanced distribution across problems and tags; e.g., only  $\mathrm{C}++$  initially had about 647K test samples for tag classification (Appendix E.1). However, finding an optimal solution to this selection problem (i.e., how many samples to select per problem and per tag) is nontrivial. We formulate this as a circulation problem with lower and upper bounds (Mount, 2017) within a flow network. Let  $p_i$  and  $t_k$  be the number of solutions for a problem  $P_i$  and a tag  $T_k$ , respectively. Let  $G = (V, E)$  be a flow network (a directed graph) with the set of vertices  $V = \{s, P_1, \dots, P_N, T_1, \dots, T_K, t\}$ , where  $s$  and  $t$  respectively denote the source and sink nodes of the network (Figure 2). For each edge  $e \in E$ , the lower capacity  $l(e)$  and upper capacity  $c(e)$  are defined as follows.

FIGURE 2 - Flow network of for validation-test dataset creation. Here  $s$  and  $t$  represent the source and sink of the flow network. Also,  $l(u,v), c(u,v)$  represents the lower and upper capacity of the edge connected from  $u$  to  $v$ .

1. Initialize  $E = \emptyset$ .  
2. For each problem  $P_i$ , add edge  $(s, P_i)$  to  $E$  and assign  $l(s, P_i) = \min(m_p, p_i)$  and  $c(s, P_i) = \min(x_p, p_i)$ , where  $m_p$  and  $x_p$  respectively refer to the minimum and maximum samples to choose per problem if available with  $m_p \leq x_p$ , thus  $0 \leq l(s, P_i) \leq c(s, P_i)$ .  
3. For each tag  $T_{k}$ , add edge  $(T_{k}, t)$  to  $E$  and assign  $l(T_{k}, t) = \min(m_{t}, t_{k})$  and  $c(T_{k}, t) = \min(x_{t}, t_{k})$ , where  $m_{t}$  and  $x_{t}$  respectively refer to minimum and maximum samples to choose per tag if available with  $m_{t} \leq x_{t}$ , thus  $0 \leq l(T_{k}, t) \leq c(T_{k}, t)$ .  
4. For each  $P_i$  and  $T_k$ , add  $(P_i, T_k)$  to  $E$  if  $P_i$  has a tag  $T_k$ , and assign  $l(P_i, T_k) = 0$ ,  $c(P_i, T_k) = \infty$ .

We then directly adopt the circulation problem solution to find a flow  $f: E \longrightarrow \mathbb{Z}_+^4$  that satisfies:  $\forall e \in E, l(e) \leq f(e) \leq c(e)$  and  $\forall u \in V, \sum_{v \in V} f(u, v) = 0$ . In our case,  $f$  denotes a feasible flow when the above constraints are satisfied for some  $G$ . For each  $e \in E, f(e)$  represents the following:

1.  $f(s, P_i)$  denotes the number of samples to be picked from problem  $P_i$ .  
2.  $f(T_k, t)$  denotes the number of samples to be picked from tag  $T_k$ .  
3.  $f(P_{i}, T_{k})$  denotes the number of samples to be picked from  $P_{i}$  that has a tag  $T_{k}$ .

Here,  $\sum_{k=1}^{K} f(T_k, t) = \sum_{i=1}^{N} f(s, P_i)$  is the total number of samples selected, which can be controlled in a balanced way by setting the control variables  $m_p, m_t, x_p$ , and  $x_t$ . Appendix D gives further details about the method and hyperparameters for different tasks, along with a comparison to a random data selection strategy.

# 2.2 EXCEVAL: A MULTILINGUAL, DISTRIBUTED AND SECURED EVALUATION ENGINE

An essential requirement for execution-based evaluation is the availability of a secure and scalable framework (Chen et al., 2021; Cassano et al., 2022). With its capacity to support 44 compiler/interpreter versions in 11 different languages, ExecEval offers a versatile and comprehensive approach to program evaluation. The engine is distributed as a secure Docker image, ensuring safe and efficient executions. It also supports easy integration of new compilers/interpreters with custom execution flags (which can also be changed at run-time). While running on unit tests, ExecEval produces one of the six outcomes: (i) COMPILETION ERROR: fails to compile or run due to a syntax error; (ii) RUNTIME ERROR: successfully compiles but fails during runtime due to native environment issues (e.g., asserts, division-by-zero); (iii) MEMORY LIMIT EXCEDED: occupies more memory than the limit; (iv) TIME LIMIT EXCEDED: requires more time than the limit; (v) WRONG ANSWER: suc

TABLE 2 - Dataset statistics per language and task (except retrieval). The validation and test splits for Program Synthesis are same across all the languages as they solve the same problems. Code Translation data refers to the source language. Since our setup supports execution-based evaluation, both Program Synthesis and Code Translation support any number of languages that are supported by the execution framework ExecEval.  

<table><tr><td>Split</td><td>C</td><td>C#</td><td>C++</td><td>Go</td><td>Java</td><td>Javascript</td><td>Kotlin</td><td>PHP</td><td>Python</td><td>Ruby</td><td>Rust</td><td>Total</td></tr><tr><td colspan="13">Tag Classification</td></tr><tr><td>Train</td><td>178,324</td><td>79,128</td><td>3,711,550</td><td>25,608</td><td>703,625</td><td>15,716</td><td>49,340</td><td>6,234</td><td>678,576</td><td>15,226</td><td>30,681</td><td>5,494,008</td></tr><tr><td>Validation</td><td>1,694</td><td>2,234</td><td>1,983</td><td>1,626</td><td>1,908</td><td>1,610</td><td>1,712</td><td>891</td><td>1,969</td><td>2,149</td><td>920</td><td>18,696</td></tr><tr><td>Test</td><td>6,193</td><td>6,020</td><td>9,720</td><td>6,504</td><td>8,881</td><td>6,431</td><td>6,841</td><td>3,598</td><td>8,195</td><td>8,671</td><td>3,679</td><td>74,733</td></tr><tr><td colspan="13">Code Compilation</td></tr><tr><td>Train</td><td>503,458</td><td>170,407</td><td>15,147,814</td><td>53,561</td><td>2,007,940</td><td>36,949</td><td>104,970</td><td>18,099</td><td>1,793,141</td><td>26,362</td><td>52,449</td><td>19,915,150</td></tr><tr><td>Validation</td><td>1,000</td><td>1,000</td><td>1,000</td><td>212</td><td>1,000</td><td>454</td><td>482</td><td>102</td><td>1,000</td><td>50</td><td>94</td><td>6,394</td></tr><tr><td>Test</td><td>5,000</td><td>5,000</td><td>5,000</td><td>814</td><td>5,000</td><td>1,676</td><td>1,940</td><td>392</td><td>5,000</td><td>242</td><td>324</td><td>30,388</td></tr><tr><td colspan="13">Program Synthesis</td></tr><tr><td>Train</td><td>179,508</td><td>79,681</td><td>3,744,367</td><td>25,753</td><td>707,603</td><td>15,916</td><td>51,831</td><td>6,334</td><td>681,780</td><td>15,336</td><td>30,732</td><td>5,538,841</td></tr><tr><td>Validation</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td><td>106</td></tr><tr><td>Test</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td><td>952</td></tr><tr><td colspan="13">Code Translation (Source Language)</td></tr><tr><td>Train</td><td>179,508</td><td>79,681</td><td>3,744,367</td><td>25,753</td><td>707,603</td><td>15,916</td><td>51,831</td><td>6334</td><td>681,780</td><td>15,336</td><td>30,732</td><td>5,538,841</td></tr><tr><td>Validation</td><td>768</td><td>746</td><td>1,054</td><td>470</td><td>960</td><td>412</td><td>421</td><td>374</td><td>868</td><td>494</td><td>467</td><td>7,034</td></tr><tr><td>Validation small</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>40</td><td>440</td></tr><tr><td>Test</td><td>1,725</td><td>1,760</td><td>1,981</td><td>1,811</td><td>1,849</td><td>1,651</td><td>1,949</td><td>1,734</td><td>1,942</td><td>1,928</td><td>2,026</td><td>20,356</td></tr><tr><td colspan="13">Automatic Program Repair or APR</td></tr><tr><td>Train</td><td>135,307</td><td>37,039</td><td>3,409,220</td><td>13,085</td><td>574,448</td><td>8,861</td><td>16,338</td><td>3,595</td><td>461,356</td><td>5,153</td><td>7,668</td><td>4,672,070</td></tr><tr><td>Validation</td><td>733</td><td>739</td><td>641</td><td>294</td><td>716</td><td>183</td><td>313</td><td>191</td><td>710</td><td>343</td><td>205</td><td>5,068</td></tr><tr><td>Test</td><td>1,957</td><td>2,002</td><td>2,026</td><td>1,427</td><td>2,032</td><td>643</td><td>1,978</td><td>1,156</td><td>2,012</td><td>1,561</td><td>905</td><td>17,699</td></tr></table>

cessfully compiles/interprets, generates an output but fails to produce a correct answer; (vi) PASSED: successfully passes all the unit tests. The program will be flagged as buggy (i-v) even when it fails on a single unit test. Appendix H of supp. material gives further details about ExecEval.

# 2.3 TASKS IN XCODEEVAL

XCODEVAL features two classification, three generative, and two retrieval tasks. Table 2 gives a breakdown of the classification and generative tasks per language. Below we briefly describe the tasks; detailed descriptions, motivation, maintenance, support, and process of task formulation along with visualizations of task distributions and task creation rationale can be found in Appendix E.

Classification tasks - Tag Classification and Code Compilation The goal of Tag Classification is to assign relevant tags to a code and/or natural descriptions of the corresponding problem. This task focuses on measuring the impact of code understanding by incorporating a natural language description alongside the code. It is the only task in our benchmark that does not factor in the code's executability. On the contrary, the objective of the Code Compilation task is to determine whether the given code is compileable or not, thereby constituting a binary classification problem. All the labels in both tasks are human annotated found as meta data. By addressing these classification tasks, we aim to explore and evaluate the effectiveness of program comprehension techniques.

Generative tasks - Program Synthesis, Automatic Program Repair (APR) and Code Translation All of our proposed generative tasks are evaluated with execution-based unit tests by ExecEval. The Program Synthesis task aims to generate executable programming language code that solves a given problem. The problem is defined with a natural language description along with some sample input-output descriptions (see Figure 1). In the  $APR$  task, along with the problem, a buggy code is also given. The objective is to correct or refine the buggy code. Moreover, in the Code Translation task, a code is provided in a source language and the goal is to translate it to a target language. Note that for Code Translation our benchmark provides the inputs for the source programming language and for Program Synthesis we only provide problem description in natural text. For both Program Synthesis and Code-Translation, the underlying execution-based unit test enables evaluation on any target programming language, as long as it is supported by ExecEval.

Retrieval Tasks - Code-Code and NL-Code Retrieval The objective of the Code-Code retrieval is to retrieve relevant executable code when provided with a programming language code as input. On the contrary, the NL-Code retrieval task aims to retrieve relevant executable code based on a problem description. These retrieval tasks are novel in the sense that they consider both the relevance and executability of the retrieved codes for evaluation. To the best of our knowledge, these are the first retrieval-based tasks that incorporates executability as a crucial factor when performing code retrieval. We have also included a retrieval corpus specific to each of the languages for evaluation purposes.

TABLE 3 - Performance of gpt-3.5-turbo on XCODEVAL : For Code Translation,  $X$ - denotes the case where the source language is  $X$ , and the target languages are represented by the respective columns. For program synthesis, (T) denotes sampling done at 20 different temperatures ranging from 0.0 to 2.0, while (N) denotes sampling done at a fixed temperature 0.32 (see Section 3.2).  

<table><tr><td>Tasks</td><td>metric</td><td>C</td><td>C#</td><td>C++</td><td>Go</td><td>Java</td><td>Javascript</td><td>Kotlin</td><td>PHP</td><td>Python</td><td>Ruby</td><td>Rust</td><td>AVG</td></tr><tr><td>TC-Code2Tag</td><td>macro-F1</td><td>32.37</td><td>26.91</td><td>40.58</td><td>23.06</td><td>31.58</td><td>19.35</td><td>33.95</td><td>15.25</td><td>29.45</td><td>23.64</td><td>24.04</td><td>27.29</td></tr><tr><td>TC-DesCode2Tag</td><td>macro-F1</td><td>36.05</td><td>33.18</td><td>47.1</td><td>31.5</td><td>38.26</td><td>27.81</td><td>39.61</td><td>19.36</td><td>33.73</td><td>30.61</td><td>32.35</td><td>33.6</td></tr><tr><td>Code Compilation</td><td>accuracy</td><td>65.9</td><td>54.9</td><td>58.0</td><td>70.28</td><td>53.0</td><td>65.64</td><td>56.64</td><td>76.47</td><td>70.9</td><td>70.0</td><td>54.26</td><td>63.27</td></tr><tr><td>Program Synthesis (T)</td><td>pass@5</td><td>25.37</td><td>30.59</td><td>31.36</td><td>31.03</td><td>29.74</td><td>22.74</td><td>26.87</td><td>30.17</td><td>33.98</td><td>33.72</td><td>10.28</td><td>27.8</td></tr><tr><td>Program Synthesis (N)</td><td>pass@5</td><td>31.23</td><td>30.78</td><td>35.44</td><td>30.58</td><td>31.52</td><td>28.63</td><td>27.38</td><td>32.13</td><td>29.77</td><td>29.66</td><td>28.2</td><td>30.48</td></tr><tr><td>Automatic Program Repair</td><td>pass@5</td><td>44.32</td><td>53.38</td><td>28.88</td><td>65.95</td><td>33.21</td><td>86.05</td><td>62.49</td><td>64.22</td><td>37.94</td><td>60.38</td><td>68.96</td><td>55.07</td></tr><tr><td>Translation C-{ }</td><td>pass@5</td><td>-</td><td>41.74</td><td>89.44</td><td>49.73</td><td>57.81</td><td>30.94</td><td>37.49</td><td>44.43</td><td>45.67</td><td>35.14</td><td>51.92</td><td>44.03</td></tr><tr><td>Translation C#-{ }</td><td>pass@5</td><td>62.27</td><td>-</td><td>72.14</td><td>49.27</td><td>63.94</td><td>25.49</td><td>44.39</td><td>60.22</td><td>62.62</td><td>68.84</td><td>62.16</td><td>51.94</td></tr><tr><td>Translation C++-{ }</td><td>pass@5</td><td>49.78</td><td>48.47</td><td>-</td><td>49.43</td><td>48.98</td><td>22.65</td><td>33.91</td><td>35.41</td><td>31.88</td><td>39.46</td><td>39.1</td><td>36.28</td></tr><tr><td>Translation Go-{ }</td><td>pass@5</td><td>59.72</td><td>63.75</td><td>79.18</td><td>-</td><td>69.92</td><td>51.46</td><td>25.2</td><td>36.05</td><td>71.05</td><td>42.81</td><td>51.21</td><td>50.03</td></tr><tr><td>Translation Java-{ }</td><td>pass@5</td><td>46.03</td><td>28.13</td><td>52.64</td><td>46.82</td><td>-</td><td>32.21</td><td>28.5</td><td>11.53</td><td>44.38</td><td>27.07</td><td>42.16</td><td>32.68</td></tr><tr><td>Translation Javascript-{ }</td><td>pass@5</td><td>57.64</td><td>49.16</td><td>68.04</td><td>64.49</td><td>60.24</td><td>-</td><td>16.1</td><td>31.52</td><td>64.12</td><td>14.93</td><td>52.27</td><td>43.5</td></tr><tr><td>Translation Kotlin-{ }</td><td>pass@5</td><td>74.34</td><td>59.39</td><td>85.67</td><td>51.52</td><td>39.2</td><td>29.01</td><td>-</td><td>39.43</td><td>64.58</td><td>53.33</td><td>53.97</td><td>50.04</td></tr><tr><td>Translation PHP-{ }</td><td>pass@5</td><td>64.38</td><td>17.5</td><td>55.92</td><td>62.19</td><td>52.11</td><td>26.19</td><td>2.5</td><td>-</td><td>59.79</td><td>64.33</td><td>36.87</td><td>40.16</td></tr><tr><td>Translation Python-{ }</td><td>pass@5</td><td>41.18</td><td>19.38</td><td>42.58</td><td>50.82</td><td>40.65</td><td>19.93</td><td>6.04</td><td>48.69</td><td>-</td><td>68.12</td><td>22.23</td><td>32.69</td></tr><tr><td>Translation Ruby-{ }</td><td>pass@5</td><td>30.47</td><td>5.63</td><td>35.69</td><td>67.01</td><td>40.07</td><td>5.69</td><td>3.75</td><td>58.87</td><td>67.28</td><td>-</td><td>12.23</td><td>29.7</td></tr><tr><td>Translation Rust-{ }</td><td>pass@5</td><td>39.49</td><td>44.72</td><td>54.29</td><td>44.6</td><td>57.5</td><td>36.24</td><td>20.43</td><td>37.91</td><td>51.32</td><td>51.17</td><td>-</td><td>39.79</td></tr><tr><td>Target lang. Avg</td><td>pass@5</td><td>52.53</td><td>37.79</td><td>63.56</td><td>53.59</td><td>53.04</td><td>27.98</td><td>21.83</td><td>40.41</td><td>56.27</td><td>46.52</td><td>42.41</td><td>45.08</td></tr></table>

# 3 EVALUATION AND ANALYSIS

For all tasks except Code Translation, we evaluate on the validation split. For Code Translation from source languages, we used the small validation split (follow Appendix E.4 in supp. material).

# 3.1 BENCHMARK RESULTS

Baselines We benchmark xCODEEval using ChatGPT (gpt-3.5-turbo-0301). To construct a query prompt, we adopt the direct zero-shot prompting method (i.e., no chain-of-thought) that facilitates easier automated evaluation (no overlapping of code and explanations). For the retrieval task, following (Karpukhin et al., 2020), we build a bi-encoder based dense multilingual code retriever by finetuning the StarEncoder (Li et al., 2023) model. Our implementation details are provided in Appendix G.

Results We present the results on the classification and generative tasks in Table 3. Overall, the model achieves inspiring yet inadequate results – marking XCODEEVAL a promising yet challenging benchmark as per the current progress in LLMs. Particularly for Tag Classification, we observe decent performance in general and incorporating a problem description enhances the model's overall predictive capability. However, in web languages (e.g., PHP, JavaScript), it exhibits the poorest performance. In Code Compilation, we observe encouraging performance for Go, PHP, Python, Ruby, and C#. However, the performance is close to a random baseline for Java, Kotlin, and Rust.

For Program Synthesis, we find that in popular languages, such as  $C++$ , C#, Go, Java, and Python the model performs well, while in rare languages like Rust it fares poorly. Notably while on other datasets such as HumanEval (Chen et al., 2021), ChatGPT achieves much higher scores, 65.8 in pass@1 (OpenAI, 2023; Chen et al., 2022), it significantly falls behind even in pass@5 ( $\sim$ 30) in xCODEVAL - imposing challenges even for such a powerful LLM. In Figure 3 (left), we show the performance for different  $k$ . As expected, results increase with  $k$  and no noticeable differences are observed between the compiler (e.g., C++, Java) and interpreter (e.g., Python, Ruby) languages.

In  $APR$ , we observe a higher performance scale than in Program Synthesis indicating that the model finds the task relatively easier since it does not necessitate generating a complete code from scratch. Interestingly, in languages where the model underperformed in program synthesis, it exhibits good performance in  $APR$ . For Code Translation, we observe that Kotlin and Go can be successfully translated to most of the other languages, while C++ and Python are the best languages to translate to.

Table 4 reports results on the two retrieval tasks: Code-Code and NL-Code. For Code-Code retrieval, we computed the top- $k$  retrieval accuracy for each of the 17 query languages from all 17 different retrieval corpora. The summarized results from a  $17 \times 17$  matrix (Appendix E.6-Figure 12 of supp. material) are provided in Table 4, where each row represents a query language and each column represents a corpus language. The column-wise and row-wise averages are denoted as  $(\alpha)$  and  $(\gamma)$ , respectively. For Code-Code  $(\alpha)$ , there is a degradation of performance for languages with large



FIGURE 3 - Left: pass@  $k$  for different languages at different  $k$ . Right: ratio of the number of generated codes that compiles for different languages. Both evaluations are done at 20 different temperature values.  
FIGURE 4 - Left: ChatGPT's performance on  $\mathrm{C + + }$  over time. After the knowledge cutoff (Sep, 2021), the performance is notably poor. Right: distribution of passed solutions  $(\mathbf{C} + + )$  across different difficulty levels.

corpus such as  $\mathrm{C}++$ , Python. In the case of Code-Code  $(\gamma)$ , languages with limited training data in xCODEEval, such as  $D$ , Ocaml, Rust performed poorly as query languages. For NL-Code, performance is good across all languages except for  $D$ . We suspect that the limited availability of resources for  $D$  in both The Stack (Kocetkov et al., 2022) dataset (training corpora of StarEncoder) and our xCODEEval dataset could account for this discrepancy. Also, the presence of more hard negative candidates (i.e., very similar to a correct code) makes it a challenging task to identify similar codes. We provide more results on the retrieval outcomes in the supplementary material (Appendix E.6).

# 3.2 ANALYSIS

Knowledge cutoff XCODEVAL contains problems that appeared in codeforces.com for the timeframe: Feb 19, 2010 - Nov 21, 2022 (in supp. material Appendix C-Figure 7 shows the distribution over time). Since we have a complete footprint of release dates for each of the problems, it enables us to identify data leakage in LLMs that have public knowledge cutoff dates. Figure 4 (left) presents a potential data contamination for ChatGPT. Though OpenAI (2023) (Table 9) reported no data contamination on codeforces.com, datasets like our XCODEVAL could empower researchers to conduct insightful analysis and perform an investigation on such serious questions. "It should be noted that XCODEVAL can only analyze data contamination if there is a good amount of problems that appear after the knowledge cut-off date of the evaluated LLM. For a more interpretable evaluation, we invite LLM builders to disclose their knowledge cut-off dates.

Impact of temperature parameter Although proved to be crucial (Chen et al., 2021; Austin et al., 2021), previous studies have not extensively examined the impact of the sampling temperature parameter on code executability. To address this gap, we conduct an investigation in which we assessed each sample for Program Synthesis at 20 different temperatures in the range  $0.0 - 2.0$ . Figure 5 (left) presents the overall distribution of execution outcomes for various languages, encompassing all the samples generated at different temperatures, while Figure 5 (right) displays a distribution of PASSED solutions at different temperatures. As the temperature increases, the likelihood of achieving code executability decreases. We identify the most successful PASSED tests at the temperature of 0.32. Figure 3 (right) presents a comparison of code executability across different languages. For each of the unit test cases, we test it with 20 different temperature values and finally select the temperature

TABLE 4 - Summary of the performance of StarEncoder Li et al. (2023) finetuned on our retrieval tasks for  $k = {100}$  . For Code-Code,  $\left( \alpha \right)$  denotes the average score for codes of any given language as the corpus,similarly  $\left( \gamma \right)$  denotes average score for codes of any fixed language as query. For NL-Code,the scores are reported for corpus of different languages.  

<table><tr><td>Tasks</td><td>metric</td><td>C</td><td>C#</td><td>C++</td><td>D</td><td>Go</td><td>Haskell</td><td>Java</td><td>Javascipt</td><td>Kotlin</td><td>Ocaml</td><td>PHP</td><td>Pascal</td><td>Perl</td><td>Python</td><td>Ruby</td><td>Rust</td><td>Scala</td><td>AVG.</td></tr><tr><td>Code-Code (α)</td><td>Acc@k</td><td>56.43</td><td>56.05</td><td>39.96</td><td>62.82</td><td>66.30</td><td>56.71</td><td>49.30</td><td>69.63</td><td>63.42</td><td>58.44</td><td>64.80</td><td>52.71</td><td>56.38</td><td>55.92</td><td>61.38</td><td>58.10</td><td>66.69</td><td>58.53</td></tr><tr><td>Code-Code (γ)</td><td>Acc@k</td><td>68.66</td><td>74.50</td><td>70.49</td><td>17.35</td><td>62.62</td><td>60.03</td><td>74.71</td><td>50.70</td><td>52.06</td><td>33.72</td><td>49.88</td><td>65.35</td><td>40.50</td><td>68.33</td><td>61.71</td><td>48.58</td><td>59.76</td><td>56.41</td></tr><tr><td>NL-Code</td><td>Acc@k</td><td>82.28</td><td>89.99</td><td>83.81</td><td>68.98</td><td>90.26</td><td>81.68</td><td>84.72</td><td>85.33</td><td>84.74</td><td>85.45</td><td>80.71</td><td>82.21</td><td>81.33</td><td>84.57</td><td>87.17</td><td>82.23</td><td>89.71</td><td>83.83</td></tr></table>

FIGURE 5 - Left: execution outcome for different languages, the solutions are evaluated with ExecEval against our unit tests; Right: passed solutions at different temperatures. Both evaluations are done at 20 different temperature values.

with highest PASSED execution. We implemented this approach exclusively for a program synthesis task, utilizing this optimal temperature as a pseudo signal for the best parameter for the remaining tasks. While this incurred a significant budget, it is worth noting that with a larger budget, employing optimal parameter search for all tasks and optimal variation would be ideal. In our evaluation, We see that Rust has overall low code executability. On the other hand, interpretable languages like Python, PHP, and Javascript have high executability.

Difficulty analysis Figure 4 (right) shows the distribution of PASSED problems written in  $C++$  for different difficulty levels. We see a sharp decrease in performance as the problems become harder.

Reasoning capability Figure 6 shows a reasoning spectrum of ChatGPT on Program Synthesis. A program can be considered a reasoning path to produce an output for an input of a unit test. We define reasoning spectrum as the range or continuum of different reasoning approaches or strategies that a program can employ to produce an output for a given input in a test case. The reasoning spectrum encompasses various levels of execution outcomes by the program in between different languages. The same colored vertical line along different languages represents an agreement of execution outcome for the corresponding languages. Given any two languages, when the code compiles successfully for both but one produces PASSED and the other produces WRONG ANSWER, we can conclude that there is a gap between reasoning capability of the two languages. We notice a low agreement in the reasoning spectrum between languages suggesting that a further transfer learning mechanism can be applied to transfer reasoning capability from high-resource to low-resource languages.

Though Figure 6 (top) shows a general comparable structure of reasoning capability of an LLM for different languages, it does not show the overall performance within each language. By grouping the same execution outcomes together, Figure 6 (bottom) shows exceptional code executability on Python (no compilation error). However, their reasoning capability (# of PASSED unit tests) remains fairly comparable with other languages.

# 4 EVALUATION OF PROGRAM SYNTHESIS ON SMALLER MODELS

For Program Synthesis tasks, we fine-tuned starcoderbase-3B (Li et al., 2023) model with our trained data. We also evaluated the CodeLlama-7b-Instruct-hf and CodeLlama-13b-Instruct-hf models with our evaluation data. A 3B fine-tuned model is better than a 7B instruct model but worse than a 13B instruct model. We observe that training a

FIGURE 6 - Top: The reasoning spectrum of gpt-3.5-turbo-0301, X-axis represents the unit tests and the color represents the corresponding test outcomes for different languages in the Y-axis; Bottom: The unit tests are grouped together from the reasoning spectrum to get an overall idea of the performance of execution outcomes. All the evaluations are done at temperature 0.32 and  $n = 10$ .

TABLE 5 - Results on Program Synthesis task on validation split. Starcoderbase-3b is finetuned with program synthesis train dataset and zero shot evaluation done for the CodeLlama models.  

<table><tr><td>Model</td><td>Trained</td><td>Metric</td><td>C</td><td>C#</td><td>C++</td><td>Go</td><td>Java</td><td>Javascript</td><td>Kotlin</td><td>PHP</td><td>Python</td><td>Ruby</td><td>Rust</td><td>Avg</td></tr><tr><td>Starcoderbase-3b</td><td>✓</td><td>pass@5</td><td>1.90</td><td>1.99</td><td>3.45</td><td>1.60</td><td>2.36</td><td>2.73</td><td>2.30</td><td>2.48</td><td>2.52</td><td>2.33</td><td>1.13</td><td>2.25</td></tr><tr><td>CodeLlama-7b-Instruct</td><td>×</td><td>pass@5</td><td>1.12</td><td>1.74</td><td>2.64</td><td>1.65</td><td>0.87</td><td>0</td><td>0.52</td><td>1.69</td><td>2.14</td><td>0.61</td><td>0.87</td><td>1.26</td></tr><tr><td>CodeLlama-13b-Instruct</td><td>×</td><td>pass@5</td><td>4.57</td><td>4.29</td><td>6.4</td><td>2.69</td><td>3.29</td><td>2.72</td><td>4.01</td><td>3.97</td><td>4.97</td><td>2.88</td><td>2.10</td><td>3.81</td></tr></table>

smaller model with the training data performs well on our task rather than using a general-purpose instruct/chat model. However, large instruct models are better than smaller fine-tuned models. So, the impact of our training dataset varies between different scales. Comparing the results between Table 3 and Table 5 also provides a general idea of how challenging our task is.

# 5 DATA LIMITATIONS

Though the codes are written by a diverse group of experts in a diverse number of languages, data is collected from a single source thus limiting the domain diversity. Besides, there is a clear discrepancy between the resource of different programming languages (see Appendix E-Figure 9 in supp. material) and most of the codes in XCODEEval are at the document level and often written in a non-modular way without a doc-string. In Appendix K, we discuss the possibilities of evaluation data leakage.

# 6 ETHICS, POTENTIAL RISKS & DOCUMENTATION

Though the data is collected from openly available sources, it has not been humanly audited. We have made our best efforts to use automated tools for identifying and removing codes with sensitive details, resulting in the removal of approximately 2 million samples from our original collection. However, it is important to emphasize that despite our diligent efforts, code can still potentially contain sensitive information and security vulnerabilities, although not something that is not openly available. Additionally, code datasets may reflect biases present in the original codebase or the developers who contributed to it.

xCodeEval documentations in supplementary Appendix H to Appendix L, follow all the necessary guidelines of NeurIPS datasets-track (e.g., datasheets (Gebru et al., 2021), nutrition label (Holland et al., 2020), and data card (Hutchinson et al., 2021)). Our github and huggingface repositories provide two valuable sources of both data access and documentation. To mitigate risks and resolve frequently asked questions, we regularly address queries or issues there.

# 7 CONCLUSION & FUTURE WORK

We have introduced xCODEEval, a large-scale multilingual multitask benchmark for code-based large language models. xCODEEval features seven different tasks involving code understanding, generation, translation and retrieval in up to 17 programming languages, and it employs an execution-based evaluation protocol. We have also presented ExecEval, a multilingual code execution engine that supports all the programming languages in xCODEEval. In summary, the combination of xCODEEval and ExecEval presents a novel framework that offers a fresh perspective for examining and analyzing large language models, facilitating comprehensive and to some extent highly interpretable investigations. We hope that by utilizing the extensive metadata and execution-based evaluation, there is potential for the discovery of new scaling laws and emergent capabilities.

# REFERENCES

Rajas Agashe, Srinivasan Iyer, and Luke Zettlemoyer. 2019. Juice: A large scale distantly supervised dataset for open domain context-based code generation. arXiv preprint arXiv:1910.02216.  
Wasi Uddin Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021a. Unified pretraining for program understanding and generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 2655-2668. Association for Computational Linguistics.  
Wasi Uddin Ahmad, Md Golam Rahman Tushar, Saikat Chakraborty, and Kai-Wei Chang. 2021b. Avatar: A parallel corpus for javapython program translation. arXiv preprint arXiv:2108.11590.  
Ben Athiwaratkun, Sanjay Krishna Gouda, Zijian Wang, Xiaopeng Li, Yuchen Tian, Ming Tan, Wasi Uddin Ahmad, Shiqi Wang, Qing Sun, Mingyue Shang, Sujan Kumar Gonugondla, Hantian Ding, Varun Kumar, Nathan Fulton, Arash Farahani, Siddhartha Jain, Robert Giaquinto, Haifeng Qian, Murali Krishna Ramanathan, Ramesh Nallapati, Baishakhi Ray, Parminder Bhatia, Sudipta Sengupta, Dan Roth, and Bing Xiang. 2022. Multi-lingual evaluation of code generation models. arXiv preprint arXiv:2210.14868.  
Jacob Austin, Augustus Odena, Maxwell I. Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie J. Cai, Michael Terry, Quoc V. Le, and Charles Sutton. 2021. Program synthesis with large language models. CoRR, abs/2108.07732.  
Berkay Berabi, Jingxuan He, Veselin Raychev, and Martin T. Vechev. 2021. Tfix: Learning to fix coding errors with a text-to-text transformer. In Proceedings of the 38th International Conference on Machine Learning, ICML 2021, 18-24 July 2021, Virtual Event, volume 139 of Proceedings of Machine Learning Research, pages 780-791. PMLR.  
Federico Cassano, John Gouwar, Daniel Nguyen, Sydney Nguyen, Luna Phipps-Costin, Donald Pinckney, Ming-Ho Yee, Yangtian Zi, Carolyn Jane Anderson, Molly Q Feldman, Arjun Guha, Michael Greenberg, and Abhinav Jangda. 2022. Multi-e: A scalable and extensible approach to benchmarking neural code generation.  
Shubham Chandelier, Colin B Clement, Guillermo Serrato, and Neel Sundaresan. 2022. Training and evaluating a jupyter notebook data science assistant. arXiv preprint arXiv:2201.12901.  
Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. 2022. Codet: Code generation with generated tests. arXiv preprint arXiv:2207.10397.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harrison Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger, Michael Petrov, Heidi Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan, Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser, Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such, Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes, Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Joshua Achiam, Vedant Misra, Evan Morikawa, Alec

Radford, Matthew Knight, Miles Brundage, Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei, Sam McCandlish, Ilya Sutskever, and Wojciech Zaremba. 2021. Evaluating large language models trained on code. CoRR, abs/2107.03374.  
Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, Parker Schuh, Kensen Shi, Sasha Tsvyashchenko, Joshua Maynez, Abhishek Rao, Parker Barnes, Yi Tay, Noam Shazeer, Vinodkumar Prabhakaran, Emily Reif, Nan Du, Ben Hutchinson, Reiner Pope, James Bradbury, Jacob Austin, Michael Isard, Guy Gur-Ari, Pengcheng Yin, Toju Duke, Anselm Levskaya, Sanjay Ghemawat, Sunipa Dev, Henryk Michalewski, Xavier Garcia, Vedant Misra, Kevin Robinson, Liam Fedus, Denny Zhou, Daphne Ippolito, David Luan, Hyeontaek Lim, Barret Zoph, Alexander Spiridonov, Ryan Sepassi, David Dohan, Shivani Agrawal, Mark Omernick, Andrew M. Dai, Thanumalayan Sankaranarayana Pillai, Marie Pellat, Aitor Lewkowycz, Erica Moreira, Rewon Child, Oleksandr Polozov, Katherine Lee, Zongwei Zhou, Xuezhi Wang, Brennan Saeta, Mark Diaz, Orhan Firat, Michele Catasta, Jason Wei, Kathy Meier-Hellstern, Douglas Eck, Jeff Dean, Slav Petrov, and Noah Fiedel. 2022. Palm: Scaling language modeling with pathways. CoRR, abs/2204.02311.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2019, Minneapolis, MN, USA, June 2-7, 2019, Volume 1 (Long and Short Papers), pages 4171-4186. Association for Computational Linguistics.  
James Finnie-Ansley, Paul Denny, Brett A. Becker, Andrew Luxton-Reilly, and James Prather. 2022. The robots are coming: Exploring the implications of openai codex on introductory programming. In Proceedings of the 24th Australasian Computing Education Conference, ACE '22, page 10-19, New York, NY, USA. Association for Computing Machinery.  
Lingyue Fu, Huacan Chai, Shuang Luo, Kounianhua Du, Weiming Zhang, Longteng Fan, Jiayi Lei, Renting Rui, Jianghao Lin, Yuchen Fang, Yifan Liu, Jingkuan Wang, Siyuan Qi, Kangning Zhang, Weinan Zhang, and Yong Yu. 2023. Codeapex: A bilingual programming evaluation benchmark for large language models. CoRR, abs/2309.01940.  
Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III au2, and Kate Crawford. 2021. Datasheets for datasets.  
Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022. Unixcoder: Unified cross-modal pre-training for code representation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin, Ireland, May 22-27, 2022, pages 7212-7225. Association for Computational Linguistics.  
Rahul Gupta, Soham Pal, Aditya Kanade, and Shirish Shevade. 2017. Deepfix: Fixing common c language errors by deep learning. In Proceedings of the aai conference on artificial intelligence, volume 31.  
Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. 2021. Measuring coding challenge competence with APPS. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.  
Sarah Holland, Ahmed Hosny, Sarah Newman, Joshua Joseph, and Kasia Chmielinski. 2018. The dataset nutrition label: A framework to drive higher data quality standards. arXiv preprint arXiv:1805.03677.  
Sarah Holland, Ahmed Hosny, Sarah Newman, Joshua Joseph, and Kasia Chmielinski. 2020. The dataset nutrition label. Data Protection and Privacy, 12(12):1.  
Xing Hu, Ge Li, Xin Xia, David Lo, Shuai Lu, and Zhi Jin. 2018. Summarizing source code with transferred api knowledge. pages 2269-2275.

Junjie Huang, Chenglong Wang, Jipeng Zhang, Cong Yan, Haotian Cui, Jeevana Priya Inala, Colin Clement, Nan Duan, and Jianfeng Gao. 2022. Execution-based evaluation for data science code generation models. arXiv preprint arXiv:2211.09374.  
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Codesearchnet challenge: Evaluating the state of semantic code search. CoRR, abs/1909.09436.  
Ben Hutchinson, Andrew Smart, Alex Hanna, Emily Denton, Christina Greer, Oddur Kjartansson, Parker Barnes, and Margaret Mitchell. 2021. Towards accountability for machine learning datasets: Practices from software engineering and infrastructure. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 560-575.  
Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. 2018. Mapping language to code in programmatic context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, Brussels, Belgium, October 31 - November 4, 2018, pages 1643-1652. Association for Computational Linguistics.  
Maliheh Izadi, Roberta Gismondi, and Georgios Gousios. 2022. Codefill: Multi-token code completion by jointly learning from structure and naming sequences. In 44th IEEE/ACM 44th International Conference on Software Engineering, ICSE 2022, Pittsburgh, PA, USA, May 25-27, 2022, pages 401-412. ACM.  
René Just, Darioush Jalali, and Michael D. Ernst. 2014. Defects4j: A database of existing faults to enable controlled testing studies for java programs. In Proceedings of the 2014 International Symposium on Software Testing and Analysis, ISSTA 2014, page 437-440, New York, NY, USA. Association for Computing Machinery.  
Vladimir Karpukhin, Barlas Oguz, Sewon Min, Patrick Lewis, Ledell Wu, Sergey Edunov, Danqi Chen, and Wen-tau Yih. 2020. Dense passage retrieval for open-domain question answering. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 6769-6781, Online. Association for Computational Linguistics.  
Denis Kocetkov, Raymond Li, Loubna Ben Allal, Jia Li, Chenghao Mou, Carlos Munoz Ferrandis, Yacine Jernite, Margaret Mitchell, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro von Werra, and Harm de Vries. 2022. The stack: 3 tb of permissively licensed source code.  
Yuhang Lai, Chengxi Li, Yiming Wang, Tianyi Zhang, Ruiqi Zhong, Luke Zettlemoyer, Scott Wen-tau Yih, Daniel Fried, Sida Wang, and Tao Yu. 2022. Ds-1000: A natural and reliable benchmark for data science code generation. arXiv preprint arXiv:2211.11501.  
Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, Qian Liu, Evgenii Zheltonozhskii, Terry Yue Zhuo, Thomas Wang, Olivier Dehaene, Mishig Davaadorj, Joel Lamy-Poirier, João Monteiro, Oleh Shliazhko, Nicolas Gontier, Nicholas Meade, Armel Zebaze, Ming-Ho Yee, Logesh Kumar Umapathi, Jian Zhu, Benjamin Lipkin, Muhtasham Oblokulov, Zhiruo Wang, Rudra Murthy, Jason Stillerman, Siva Sankalp Patel, Dmitry Abulkhanov, Marco Zocca, Manan Dey, Zhihan Zhang, Nour Fahmy, Urvashi Bhattacharyya, Wenhao Yu, Swayam Singh, Sasha Luccioni, Paulo Villegas, Maxim Kunakov, Fedor Zhdanov, Manuel Romero, Tony Lee, Nadav Timor, Jennifer Ding, Claire Schlesinger, Hailey Schoelkopf, Jan Ebert, Tri Dao, Mayank Mishra, Alex Gu, Jennifer Robinson, Carolyn Jane Anderson, Brendan Dolan-Gavitt, Danish Contractor, Siva Reddy, Daniel Fried, Dzmitry Bahdanau, Yacine Jernite, Carlos Munoz Ferrandis, Sean Hughes, Thomas Wolf, Arjun Guha, Leandro von Werra, and Harm de Vries. 2023. Starcoder: may the source be with you!  
Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy, Cyprien de Masson d'Autume, Igor Babuschkin, Xinyun Chen, Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov, James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson, Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, and Oriol Vinyals. 2022. Competition-level code generation with alphabet. Science, 378(6624):1092-1097.  
Shangqing Liu, Yu Chen, Xiaofei Xie, Jing Kai Siow, and Yang Liu. 2021. Retrieval-augmented generation for code summarization via hybrid GNN. In International Conference on Learning Representations.

Shuai Lu, Daya Guo, Shuo Ren, Junjie Huang, Alexey Svyatkovskiy, Ambrosio Blanco, Colin B. Clement, Dawn Drain, Daxin Jiang, Duyu Tang, Ge Li, Lidong Zhou, Linjun Shou, Long Zhou, Michele Tufano, Ming Gong, Ming Zhou, Nan Duan, Neel Sundaresan, Shao Kun Deng, Shengyu Fu, and Shujie Liu. 2021. Codexglue: A machine learning benchmark dataset for code understanding and generation. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and Benchmarks 2021, December 2021, virtual.  
Zohar Manna and Richard J. Waldinger. 1971. Toward automatic program synthesis. Commun. ACM, 14(3):151-165.  
C E Metz. 1978. Basic principles of ROC analysis. Semin Nucl Med, 8(4):283-298.  
Antonio Valerio Miceli Barone and Rico Sennrich. 2017. A parallel corpus of python functions and documentation strings for automated code documentation and code generation. In Proceedings of the Eighth International Joint Conference on Natural Language Processing (Volume 2: Short Papers), pages 314-319, Taipei, Taiwan. Asian Federation of Natural Language Processing.  
Dave Mount. 2017. Lecture 17 network flow: Extensions.  
Niklas Muennighoff, Qian Liu, Armel Zebaze, Qinkai Zheng, Binyuan Hui, Terry Yue Zhuo, Swayam Singh, Xiangru Tang, Leandro von Werra, and Shayne Longpre. 2023. Octopack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124.  
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language model for code with multi-turn program synthesis. arXiv preprint.  
Yusuke Oda, Hiroyuki Fudaba, Graham Neubig, Hideaki Hata, Sakriani Sakti, Tomoki Toda, and Satoshi Nakamura. 2015. Learning to generate pseudo-code from source code using statistical machine translation. In 2015 30th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 574-584.  
OpenAI. 2023. Gpt-4 technical report.  
Juri Opitz and Sebastian Burst. 2019. Macro F1 and macro F1. CoRR, abs/1911.03347.  
Long Ouyang, Jeff Wu, Xu Jiang, Diogo Almeida, Carroll L Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al. 2022. Training language models to follow instructions with human feedback. arXiv preprint arXiv:2203.02155.  
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. 2002. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311-318, Philadelphia, Pennsylvania, USA. Association for Computational Linguistics.  
Md Rizwan Parvez, Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021. Retrieval augmented code generation and summarization. In *Findings of the Association for Computational Linguistics: EMNLP* 2021, pages 2719–2734, Punta Cana, Dominican Republic. Association for Computational Linguistics.  
Md Rizwan Parvez, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2018. Building language models for text with named entities. arXiv preprint arXiv:1805.04836.  
Ruchir Puri, David S Kung, Geert Janssen, Wei Zhang, Giacomo Domeniconi, Vladimir Zolotov, Julian Dolby, Jie Chen, Mihir Choudhury, Lindsey Decker, et al. 2021. Codenet: A large-scale ai for code dataset for learning a diversity of coding tasks. arXiv preprint arXiv:2105.12655.  
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. 2018. Improving language understanding by generative pre-training.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2020. Exploring the limits of transfer learning with a unified text-to-text transformer. J. Mach. Learn. Res., 21:140:1-140:67.

Shuo Ren, Daya Guo, Shuai Lu, Long Zhou, Shujie Liu, Duyu Tang, Neel Sundaresan, Ming Zhou, Ambrosio Blanco, and Shuai Ma. 2020. Codebleu: a method for automatic evaluation of code synthesis. CoRR, abs/2009.10297.  
Baptiste Roziere, Marie-Anne Lachaux, Lowik Chanussot, and Guillaume Lample. 2020. Unsupervised translation of programming languages. Advances in Neural Information Processing Systems, 33.  
Baptiste Roziere, Jie M Zhang, Francois Charton, Mark Harman, Gabriel Synnaeve, and Guillaume Lample. 2021. Leveraging automated unit tests for unsupervised code translation. arXiv preprint arXiv:2110.06773.  
Rebecca L. Russell, Louis Y. Kim, Lei H. Hamilton, Tomo Lazovich, Jacob Harer, Onur Ozdemir, Paul M. Ellingwood, and Marc W. McConley. 2018. Automated vulnerability detection in source code using deep representation learning. In 17th IEEE International Conference on Machine Learning and Applications, ICMLA 2018, Orlando, FL, USA, December 17-20, 2018, pages 757-762. IEEE.  
Hitesh Sajnani. 2016. Large-scale code clone detection. PhD Thesis, University of California, Irvine.  
Jeffrey Svajlenko, Judith F. Islam, Iman Keivanloo, Chanchal Kumar Roy, and Mohammad Mamun Mia. 2014. Towards a big data curated benchmark of inter-project code clones. In 30th IEEE International Conference on Software Maintenance and Evolution, Victoria, BC, Canada, September 29 - October 3, 2014, pages 476-480. IEEE Computer Society.  
Abdel Aziz Taha and Allan Hanbury. 2015. Metrics for evaluating 3D medical image segmentation: analysis, selection, and tool. BMC Med Imaging, 15:29.  
Xiangru Tang, Bill Qian, Rick Gao, Jiakang Chen, Xinyun Chen, and Mark Gerstein. 2023. Biocoder: A benchmark for bioinformatics code generation with contextual pragmatic knowledge. CoRR, abs/2308.16458.  
Nandan Thakur, Nils Reimers, Andreas Rückle, Abhishek Srivastava, and Iryna Gurevych. 2021. BEIR: A heterogeneous benchmark for zero-shot evaluation of information retrieval models. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).  
THUDM. 2022. Codegeex: A multilingual code generation model. https://github.com/THUDM/CodeGeeX.  
Michele Tufano, Cody Watson, Gabriele Bavota, Massimiliano Di Penta, Martin White, and Denys Poshyvanyk. 2019. An empirical study on learning bug-fixing patches in the wild via neural machine translation. ACM Trans. Softw. Eng. Methodol., 28(4):19:1-19:29.  
Yao Wan, Jingdong Shu, Yulei Sui, Guandong Xu, Zhou Zhao, Jian Wu, and Philip Yu. 2019. Multi-modal attention network learning for semantic source code retrieval. In 2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE), pages 13-25. IEEE.  
Yue Wang, Weishi Wang, Shafiq R. Joty, and Steven C. H. Hoi. 2021. Codet5: Identifier-aware unified pre-trained encoder-decoder models for code understanding and generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, EMNLP 2021, Virtual Event / Punta Cana, Dominican Republic, 7-11 November; 2021, pages 8696-8708. Association for Computational Linguistics.  
Zhiruo Wang, Grace Cuenca, Shuyan Zhou, Frank F Xu, and Graham Neubig. 2022a. Mconala: a benchmark for code generation from multiple natural languages. arXiv preprint arXiv:2203.08388.  
Zhiruo Wang, Shuyan Zhou, Daniel Fried, and Graham Neubig. 2022b. Execution-based evaluation for open-domain code generation. arXiv preprint arXiv:2212.10481.  
Chunqiu Steven Xia and Lingming Zhang. 2022. Less training, more repairing please: revisiting automated program repair via zero-shot learning. In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, ESEC/FSE 2022, Singapore, Singapore, November 14-18, 2022, pages 959-971. ACM.

Pengcheng Yin, Bowen Deng, Edgar Chen, Bogdan Vasilescu, and Graham Neubig. 2018. Learning to mine aligned code and natural language pairs from stack overflow. In Proceedings of the 15th International Conference on Mining Software Repositories, pages 476-486.  
Pengcheng Yin, Wen-Ding Li, Kefan Xiao, Abhishek Rao, Yeming Wen, Kensen Shi, Joshua Howland, Paige Bailey, Michele Catasta, Henryk Michalewski, et al. 2022. Natural language to code generation in interactive data science notebooks. arXiv preprint arXiv:2212.09248.  
Hao Yu, Bo Shen, Dezhi Ran, Jiaxin Zhang, Qi Zhang, Yuchi Ma, Guangtai Liang, Ying Li, Tao Xie, and Qianxiang Wang. 2023. Codereval: A benchmark of pragmatic code generation with generative pre-trained models. arXiv preprint arXiv:2302.00288.  
Victor Zhong, Caiming Xiong, and Richard Socher. 2017. Seq2sql: Generating structured queries from natural language using reinforcement learning. CoRR, abs/1709.00103.  
Yaqin Zhou, Shangqing Liu, Jing Kai Siow, Xiaoning Du, and Yang Liu. 2019. Design: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada, pages 10197-10207.  
Ming Zhu, Aneesh Jain, Karthik Suresh, Roshan Ravindran, Sindhu Tipirneni, and Chandan K. Reddy. 2022. Xlcost: A benchmark dataset for cross-lingual code intelligence.  
Qihao Zhu, Zeyu Sun, Yuan-an Xiao, Wenjie Zhang, Kang Yuan, Yingfei Xiong, and Lu Zhang. 2021. A syntax-guided edit decoder for neural program repair. In ESEC/FSE '21: 29th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, Athens, Greece, August 23-28, 2021, pages 341-353. ACM.  
Albert Ziegler, Eirini Kalliamvakou, X. Alice Li, Andrew Rice, Devon Rifkin, Shawn Simister, Ganesh Sittampalam, and Edward Aftandilian. 2022. Productivity assessment of neural code completion. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, MAPS 2022, page 21-29, New York, NY, USA. Association for Computing Machinery.

# APPENDIX

A Frequently Asked Questions 18  
B Related Work 19  
C Algorithm for initial validation and test split creation 20  
D Hyper parameter tuning for circulation problem 21

D.1 Search Techniques 22  
D.2 Results 22

E Tasks Construction Process 22

E.1 Tag Classification 22  
E.2 Code Compilation 23  
E.3 Program Synthesis 24  
E.4 Code Translation 24  
E.5 Automatic Program Repair (APR) 25  
E.6 Code Retrieval 25

F Evaluation Metrics 29  
G Implementation Details 29  
H ExecEval Details 30  
I Definition of Data Attributes 33

I.1 Problem Description (problem_description) 33  
I.2 Unit Tests (hidden_unit_test) 33  
I.3 Tag Classification (tag_classification) 34  
I.4 Code Compilation (code Compilation) 34  
I.5 Automatic Program Repair (apr) 34  
I.6 Code Translation (code Translation) 35  
I.7 Program Synthesis (program_synthesis) 36  
I.8 Retrieval Corpus (retrieval Corpus) 37  
I.9 Retrieval NL-Code (retrieval_n1_code) 37  
I.10 Retrieval Code-Code (retrieval_code_code) 37

J Datasheets for Datasets 38

J.1 Motivation 38  
J.2 Composition 38  
J.3 Collection Process 39

J.4 Preprocessing/cleaning/labeling 39  
J.5 Distribution 40  
J.6 Maintenance 40

K Discussion on the possibility of data leakage and contamination 41  
L The Dataset Nutrition Label 41  
M Data Card 41

# A FREQUENTLY ASKED QUESTIONS

Notes on Intended Usage Scenarios Considering the recent progress, the primary objective of creating XCODEEval is to challenge LLMs, especially the frontier models. As such, the tasks proposed could be very challenging for smaller models, even for the binary code compilation task. The relatively smaller (per-language) validation splits can be used to assess multilingual features and to get an overall picture of multilingual generalization. The large Test split is meant to rigorously evaluate specific programming languages and conduct more language-specific, in-depth analysis. We have also included the Training split. The intended use of the training split is to include it in the large scale pre-training or SFT mixtures. For both validation and test evaluation, we recommend using an Instruct/Chat Model.

Where are the Dataset Construction recipes? This paper provides a detailed description of our dataset construction. However, to adhere to the page limit, we needed to shorten them in the main paper (Section 1, 2) and move to the supplementary appendix Appendix E for details.

Where are the documentation, maintenance, support? XCODEVAL documentations in supplementary Sec Appendix H to Appendix M, follow all the necessary guidelines of datasheets (Gebru et al., 2021), nutrition label (Holland et al., 2018), and data card. Following the accountability framework proposed by (Hutchinson et al., 2021), we released the data construction procedures as well as opensource the evaluation framework. Our github and huggingface repositories provide two valuable sources of both data access and additional documentation and implementation details. We regularly address queries or issues there. If any specific documentation is missing, we would be happy to address them right away.

Is there any automated process used for creating the Gold Labels? Please note that all current data (text, code, test-cases, and labels) are human written. More specifically for tag classification tasks, the tags are already provided in codeforces.com. The tags are generally updated after the contest by experienced problem solvers trusted by the codeforces team. In addition to that, the evaluation is done by programming language compilers & interpreters. We put huge effort in developing ExecEval and dockerized it to synchronize the evaluation process.

On the possibility of data leakage and contamination We note that, although we completely separate our validation and test data from training, LLMs might have possible data leakage from pretraining. We find that even identifying data leakage (test data exists or not in the preretraining corpora) is challenging using conventional data search methods due to search cost and complexity (e.g., exact match or token overlapping methods) for (i) long sequence search for libraries (ii) boilerplate code identifying. Apart from that, hashing based searches often suffer from not having properly segmented text.

In this paper, we introduce an approach to address the challenge of leakage-free evaluation, employing a technique rooted in the concept of a knowledge cut-off. Our finding in Section 3.2 in the main paper shows that the data contamination significantly impacts the model performance and it needs to be interpreted with proper analyses. Another method toward leakage less evaluation could be to have a human written separate evaluation set that is hidden or regularly updated which we are considering in our long term release plans.

Only generative tasks utilize unit tests. How are classification and retrieval tasks considering executability? In our proposed tasks, except for the tag classification task, all tasks consider online or offline unit test executability. We included a tag classification task since tags were used for the sampling algorithm that we proposed. Other than that, our code compilation and retrieval task takes into account offline code executability, where compilers are applied before releasing datasets to obtain labels. Additionally, in retrieval tasks, we treat passed samples as correctly retrieved samples and generate hard negatives from incorrect code from the same problem. Please note that all the data—including text, code, test cases, and labels—is human-written.

TABLE 6 - Comparison between xCODEEval and other benchmarks. For simplicity, we combine NL-code generation and code completion as Program Synthesis. Compared to others, xCODEEval offers the largest suite of training and test data and a more comprehensive set of test cases. Evaluation levels Global, Modular, and Local refer to document, function, and statements level evaluation, respectively.  

<table><tr><td>Dataset</td><td>Trainl</td><td>Testl</td><td>Lal</td><td>Task Type</td><td>Evaluation</td><td>Level</td><td>Genre</td></tr><tr><td>Django (Oda et al., 2015)</td><td>16,000</td><td>1,805</td><td>1</td><td>Program Synthesis</td><td>Lexical</td><td>Local</td><td>N/A</td></tr><tr><td>WikiSQL (Zhong et al., 2017)</td><td>56,355</td><td>15,878</td><td>1</td><td>SQL Queries</td><td>Lexical</td><td>Modular</td><td>SQL</td></tr><tr><td>Miceli Barone and Sennrich (2017)</td><td>109,108</td><td>2,000</td><td>1</td><td>Synthesis, Summarization</td><td>Lexical</td><td>Local</td><td>Github</td></tr><tr><td>CoNaLa (Yin et al., 2018)</td><td>2,379</td><td>500</td><td>2</td><td>Program Synthesis</td><td>Lexical</td><td>Local</td><td>Stackoverflow: QA</td></tr><tr><td>CONCODE (Iyer et al., 2018)</td><td>100,000</td><td>2,000</td><td>1</td><td>Program Synthesis</td><td>Lexical</td><td>Modular</td><td>Github</td></tr><tr><td>Android (Parvez et al., 2018)</td><td>26,600</td><td>3,546</td><td>1</td><td>Program Synthesis</td><td>Lexical</td><td>Local</td><td>Map oriented, GitHub</td></tr><tr><td>CodeSearchNet (Husain et al., 2019)</td><td>6,452,446</td><td>99</td><td>6</td><td>Plain Text, Retrieval</td><td>NDCG</td><td>Modular</td><td>Github</td></tr><tr><td>JuCe (Agashe et al., 2019)</td><td>1,518,049</td><td>1,981</td><td>1</td><td>Notebook Cell Gen.</td><td>Lexical</td><td>Local</td><td>Prog. assignment</td></tr><tr><td>TransCoder (Roziere et al., 2020)</td><td>721MB</td><td>1,410</td><td>3</td><td>Program Translation</td><td>Lexical</td><td>Modular</td><td>Github</td></tr><tr><td>HumanEval (Chen et al., 2021)</td><td>-</td><td>164</td><td>1</td><td>Program Synthesis</td><td>Execution</td><td>Modular</td><td>Interview Question</td></tr><tr><td>HumanEval-X(THUDM, 2022)</td><td>-</td><td>820</td><td>9</td><td>Synthesis &amp; Translation</td><td>Execution</td><td>Modular</td><td>Interview Question</td></tr><tr><td>MBPP (Austin et al., 2021)</td><td>-</td><td>974</td><td>1</td><td>Program Synthesis</td><td>Execution</td><td>Modular</td><td>Interview Question</td></tr><tr><td>CodeXGLUE (Lu et al., 2021)</td><td>2,840,000</td><td>759,000</td><td>9</td><td>10 Tasks</td><td>Lexical</td><td>Local</td><td>N/A</td></tr><tr><td>AVATAR (Ahmad et al., 2021b)</td><td>5,937</td><td>1,693</td><td>2</td><td>Program Translation</td><td>Lexical</td><td>Global</td><td>Problem Solving</td></tr><tr><td>TFix (Berabi et al., 2021)</td><td>84,846</td><td>10,504</td><td>1</td><td>Program Repair</td><td>Lexical</td><td>Local</td><td>Github</td></tr><tr><td>CCSD (Liu et al., 2021)</td><td>84,316</td><td>6,533</td><td>1</td><td>Program Summarization</td><td>Lexical</td><td>Modular</td><td>Linux Kernel</td></tr><tr><td>TL-CodeSum (Hu et al., 2018)</td><td>55,766</td><td>6,971</td><td>1</td><td>Program Summarization</td><td>Lexical</td><td>Modular</td><td>Github</td></tr><tr><td>CodeNet (Puri et al., 2021)</td><td>8,906,769</td><td>2,783,365</td><td>55</td><td>Classification, similarity</td><td>Lexical</td><td>Global</td><td>Problem Solving</td></tr><tr><td>TransCoder-ST (Roziere et al., 2021)</td><td>333,542</td><td>103,488</td><td>3</td><td>Program Translation</td><td>Execution</td><td>Modular</td><td>Github</td></tr><tr><td>DSP (Chandelier et al., 2022)</td><td>-</td><td>1,119</td><td>1</td><td>Notebook Cell Gen.</td><td>Execution</td><td>Local</td><td>Math and Data Science</td></tr><tr><td>MTPB (Nijkamp et al., 2022)</td><td>-</td><td>115</td><td>1</td><td>Multi-turn Code Gen.</td><td>Execution</td><td>Local</td><td>Problem Solving</td></tr><tr><td>Exe-DS (Huang et al., 2022)</td><td>119,266</td><td>534</td><td>1</td><td>Notebook Cell Gen.</td><td>Execution</td><td>Local</td><td>Data Science</td></tr><tr><td>DS-1000 (Lai et al., 2022)</td><td>-</td><td>1,000</td><td>1</td><td>Notebook Cell Gen.</td><td>Execution</td><td>Local</td><td>Data Science</td></tr><tr><td>MoCoNaLa (Wang et al., 2022a)</td><td>-</td><td>896</td><td>1</td><td>Program Synthesis</td><td>Lexical</td><td>Local</td><td>StackOverflow</td></tr><tr><td>ARCADE (Yin et al., 2022)</td><td>-</td><td>1,082</td><td>1</td><td>Notebook Cell Gen.</td><td>Lexical</td><td>Local</td><td>Data Science</td></tr><tr><td>ODEX (Wang et al., 2022b)</td><td>-</td><td>945</td><td>1</td><td>Program Synthesis</td><td>Execution</td><td>Local</td><td>StackOverflow</td></tr><tr><td>MBXP (Athiwaratkun et al., 2022)</td><td>-</td><td>13,877</td><td>10</td><td>Program Synthesis</td><td>Execution</td><td>Modular</td><td>Interview Question</td></tr><tr><td>XLCoST(Zhu et al., 2022)</td><td>496,333</td><td>45,394</td><td>7</td><td>10 Task</td><td>Lexical</td><td>Local, Global</td><td>GitHub</td></tr><tr><td>DeepFix (Gupta et al., 2017)</td><td>37,000</td><td>7,000</td><td>1</td><td>Program Repair</td><td>Execution</td><td>Global</td><td>Compile Error, Students</td></tr><tr><td>Defects4J (Just et al., 2014)</td><td>-</td><td>835</td><td>1</td><td>Program Repair</td><td>Execution</td><td>Local, Global</td><td>N/A</td></tr><tr><td>APPS (Hendrycks et al., 2021)</td><td>5,000</td><td>5,000</td><td>1</td><td>Program Synthesis</td><td>Execution</td><td>Global</td><td>Interview Question</td></tr><tr><td>CodeContests (Li et al., 2022)</td><td>4,432,447</td><td>32,181</td><td>3</td><td>Program Synthesis</td><td>Execution</td><td>Global</td><td>Problem Solving</td></tr><tr><td>CoderEval (Yu et al., 2023)</td><td>-</td><td>460</td><td>2</td><td>Program Synthesis</td><td>Execution</td><td>Modular, Global</td><td>GitHub</td></tr><tr><td>Humanevalpack (Muennighoff et al., 2023)</td><td>-</td><td>6×164</td><td>6</td><td>Program Synthesis</td><td>Execution</td><td>Modular</td><td>Interview Question</td></tr><tr><td>BioCoder (Tang et al., 2023)</td><td>-</td><td>2,522</td><td>2</td><td>Program Synthesis</td><td>Execution</td><td>Modular, Global</td><td>Github</td></tr><tr><td>CodeApex (Fu et al., 2023)</td><td>-</td><td>706</td><td>1</td><td>3 tasks</td><td>Execution</td><td>Modular</td><td>Online Judge platform</td></tr><tr><td>XCODEVAL (ours)</td><td>19,915,150</td><td>159,464</td><td>17</td><td>7 Tasks, see Table 8</td><td>Execution</td><td>Global</td><td>Problem Solving</td></tr></table>

# B RELATED WORK

Following NLP (Devlin et al., 2019; Radford et al., 2018; Raffel et al., 2020), transformer-based pre-trained LLMs have shown significant success in code, both in understanding and generation. Table 6 shows a detailed comparison between different programming language-related datasets.

Code Understanding Lu et al. (2021) propose a benchmark CodeXGLUE, which comprises three widely-used code understanding tasks, defect detection, clone detection, and code search. Zhou et al. (2019) treat defect detection as a binary classification task. They propose a model called Devign which they evaluate on four large open-source C projects. Additionally, Russell et al. (2018) leverage open-source C/CPP repositories to support function-level vulnerability detection. To further understand code semantics, Svajlenko et al. (2014) propose a benchmark BigCloneBench to measure the similarity between code pairs to predict whether they have the same functionality (i.e., clone detection); BigCloneBench was collected from open-source Java repositories with manual validation. Arguably, code defect and clone detection might not be appropriate for fully evaluating models' ability in understanding code semantics (Wang et al., 2021; Guo et al., 2022). Moreover, they only support a few programming languages. Code search on the other hand considers semantic relevance for both code-to-code and text-to-code. They are formulated to retrieve semantically similar codes given a query code (Lu et al., 2021) or code description (Husain et al., 2019). The existing code search benchmarks like CodeSearchNet (Husain et al., 2019) only select the first documentation as the text query to search corresponding functions. Recently, Fu et al. (2023) introduce CodeApex, a bilingual benchmark to evaluate the language models on three different tasks consisting of programming comprehension, code generation, code correction. Among its tasks, programming comprehension examines the ability to understand code from various aspects, such as the syntax's mastery, code execution flow, and executing algorithms. Nonetheless, this dataset only covers one programming language, which is in contrast to our work.

Code Generation Code generation has grown in popularity as many pre-trained LLMs have achieved remarkable performances in these tasks like decoder-only models (Chen et al., 2021; Izadi et al., 2022; Nijkamp et al., 2022) and encoder-decoder models (Wang et al., 2021; Guo et al., 2022;

Ahmad et al., 2021a) Notably, PaLM (Chowdhery et al., 2022) and AlphaCode (Li et al., 2022) outperform average human participant in competition-level coding. Thus, researchers attempt to build increasingly difficult and factual code generation tasks. These tasks can be classified as code-to-code generation and text-to-code generation.

As for code-to-code generation tasks like automatic program repair (APR) (Tufano et al., 2019) and code translation (Lu et al., 2021), the metric-based automatic evaluation measures like BLEU (Papineni et al., 2002), CodeBLEU (Ren et al., 2020), and exact match scores are sub-optimal for evaluating the quality of a generated code. To improve the reliability and feasibility for code generation evaluation, Berabi et al. (2021) create a large-scale JavaScript patch repair dataset from GitHub commits, where 52 error types are detected by a static analyzer ESLint  $^{10}$ . They further drive efforts in enhancing program repair evaluation by providing an error removal metric to take various form of error fixes into consideration. To address the nature of code semantic and syntactic evaluation, execution-based evaluation with comprehensive test suites has a growing demand. A popular Java APR benchmark Defects4J (Just et al., 2014) takes the number of correct fixes into account, where a correct fix should pass all test cases and provide a desired functionality. Nevertheless, Defects4J does not possess a cohesive training corpus. A common strategy to address this limitation is to construct the training dataset using GitHub's publicly available repositories, and relying on bug-specific commit messages (Zhu et al., 2021). However, this heuristic-based approach includes bug-irrelevant commits and unrelated code pairs, which can significantly affect the the quality of collected training dataset (Xia and Zhang, 2022).

For text-to-code, the widely used dataset CONCODE (Iyer et al., 2018) consists of a large collection of natural language (NL) comments and Java code snippets. Specifically, this dataset is constructed by scraping code snippets from open-domain Java GitHub repositories and utilizing heuristics to extract NL comments from Javadoc.

By following a similar approach, JuICe (Agashe et al., 2019) collects publicly available Jupyter notebooks from GitHub, and CoNaLa (Yin et al., 2018) collects Python and Java codes with NL comments from StackOverflow posts. Besides, they attempt to improve the quality with professional annotators. In addition, MoCoNaLa (Wang et al., 2022a) extends CoNaLa to support more natural languages. Despite their coverage, the general lexical-based evaluation metrics are insufficient to measure the correctness of generated codes. To alleviate this limitation, ODEX (Wang et al., 2022b) provides execution-based evaluation via human-written test cases of diversified Python libraries. This execution-based paradigm has been widely applied to evaluate benchmarks in Data Science domain like DSP (Chandelier et al., 2022), DS-1000 (Lai et al., 2022) and Exe-DS (Huang et al., 2022) as well as general code generation benchmarks in single-language settings such as HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), and APPS (Hendrycks et al., 2021). Apart from HumanEval, CoderEval (Yu et al., 2023) further leverages the contextual information and achieves a full spectrum of test coverage with additional manually crafted tests, providing  $100\%$  test coverage. To improve the diversity of code generation tasks, Fu et al. (2023) propose a bilingual code evaluation benchmark CodeApex to support both English-to-Code and Chinese-to-Code generation tasks. As for more particular multi-turn MTPB (Nijkamp et al., 2022), multi-language CodeContests (Li et al., 2022), and domain specific BioCoder (Tang et al., 2023) benchmarks, they all leverage test cases, and exploit code execution for better evaluation.

# C ALGORITHM FOR INITIAL VALIDATION AND TEST SPLIT CREATION

To make sure we do not have train and (validation, test) overlap, at first we divide the set of problems into two sets. In one set we keep all the problems for which we do not have a complete set of unit tests. In another set, we keep the problems where we have a complete set of unit tests that ensures the correctness of the solution of the problem. We use the first set for training and the latter set for validation and test data. Figure 7 shows the chronological distribution of our training, validation, and test data. After selecting validation and test problem sets, we have thousands of solutions for each of the problems. But these problems are not divided into validation and test splits. As a heuristic, we can consider the tag distribution as the domain of the problem. To ensure that we have proper domain coverage we employ Algorithm 1. Algorithm 1 ensures that validation and test sets contain the same tag sets as the training set. In addition to that, it also selects the best possible splitting point based on

FIGURE 7 - The chronological order of the problems' online appearance for the first time

the geometric mean. However algorithm 1 provides only a splitting point for hundreds of thousands of validation and test data points. To reduce the redundant data based on different levels of conditions, we formulate the problem of selecting data points to a linear programming problem (more on this in Section 2.1).

Algorithm 1 Validation and Test Split Creation  
Input: A held-out dataset  $\mathcal{D}_{\mathrm{ho}}$  , a fraction value  $\gamma$  where  $0\leq \gamma \leq$  1, an integer  $N$  indicating number of seeds.   
Output:  $\mathcal{D}_{\mathrm{valid}}$ $\mathcal{D}_{\mathrm{test}}$  spits   
Initialize: count  $= 0$  bestScore  $= \gamma +1$    
while count  $<  N$  do seed  $=$  getSeed() Shuffle  $\mathcal{D}_{\mathrm{ho}}$ $\mathcal{D}_{\mathrm{valid}} = \mathcal{D}_{\mathrm{ho}}[0:|\mathcal{D}_{\mathrm{ho}}|\times \gamma ]$  Tvalid  $=$  set of tags in  $\mathcal{D}_{\mathrm{valid}}$ $\mathcal{D}_{\mathrm{test}} = \mathcal{D}_{\mathrm{ho}}[|\mathcal{D}_{\mathrm{ho}}|\times \gamma :|\mathcal{D}_{\mathrm{ho}}|]$  Ttest  $=$  set of tags in  $\mathcal{D}_{\mathrm{test}}$  if  $\mathcal{T}_{\mathrm{valid}}\neq \mathcal{T}_{\mathrm{test}}$  then continue end if for all  $T$  in  $\mathcal{T}_{\mathrm{valid}}$  do  $\gamma_{T} = \frac{\#_{\mathrm{samples~in~}\mathcal{D}_{\mathrm{valid}}~with~tag~T}}{\#_{\mathrm{samples~in~}\mathcal{D}_{\mathrm{test}}~with~tag~T}}$  end for  $\mu = geoMean(\{\gamma_T\}_{T\in T_{\mathrm{valid}}})$  if  $|\gamma -bestScore| > |\gamma -\mu |$  then bestScore  $= \mu$  save current split  $\{\mathcal{D}_{\mathrm{valid}},\mathcal{D}_{\mathrm{test}}\}$  count  $= count + 1$  end if end while

# D HYPER PARAMETER TUNING FOR CIRCULATION PROBLEM

Let  $M$  be the number of samples we want to select for any set of submissions. We call any  $(m_p, m_t, x_p, x_t)$  a valid tuple if the flow network has a feasible flow for the circulation problem defined in eq. in 2.1. Let  $d = \left\lfloor (\sum_{i=1}^{N} f(s, P_i) - M)^2 / \Delta \right\rfloor$ , representing the squared difference between samples we want and the samples selected for the flow and  $\Delta$  reduces the resolution in which

we look for differences. Here  $d$  defines a boundary from  $M$  where we allow choosing an expected solution with  $m_p, m_t, x_p$ , and  $x_t$ . Finally, the lexicographical ordering  $(-d, m_t, -x_t, -x_p, m_p)$  is used to find the largest element in the collection of valid tuples which always exist if we limit our search space to a finite set. The largest element in this ordering depicts the nearest (close to  $M$ ) selection of samples that maximizes the minimum number of samples per tag  $m_t$ . When there are many solutions with the same  $(-d, m_t)$ , we prioritize reducing the maximum number of samples per tag,  $x_t$ . Similarly, we prioritize  $x_p$  and  $m_p$  as defined in the lexicographical ordering.

# D.1 SEARCH TECHNIQUES

1. It was manually checked that  $(m_p, m_t, x_p, x_t) = (1, 1, 1000, 1000)$  is a valid tuple for any set of submissions that were processed and  $\Delta = 1000$  was chosen.  
2. In Tag classification task (Section E.1) and Code compilation task (Section E.2),  $M$  is 2000, 10000 for any language for validation, test split respectively. For Code translation (Section E.4)  $M$  was 400, 2000 for the same.  
3. Search largest tuple  $(-d_{1},m_{t_{1}}, - x_{t_{1}}, - x_{p_{1}},m_{p_{1}})$  where  $m_{t_1}\in \{1,6,11,\dots ,496\}$ ,  $m_{p_1}\in \{1,2,3,\dots ,19\}$  and  $x_{p_1} = x_{t_1} = 1000$ . Since  $(m_p,m_t,x_p,x_t) = (1,1,1000,1000)$  is a valid solution, hence the set of valid tuples is nonempty. Let  $f_{1}$  be the flow for the flow network defined for  $m_{t_1}, - x_{t_1}, - x_{p_1},m_{p_1}$ . Let  $f_{P_1} = \max_{1\leq i\leq N}f_1(s,P_i),f_{T_1} = \max_{1\leq k\leq K}f_1(T_k,t)$  be the maximum flow through edges from  $s$  to  $P_{i}$ , and same through edges from  $T_{k}$  to  $t$ .  
4. Now again search the largest tuple  $(-d_{2},m_{t_{2}}, - x_{t_{2}}, - x_{p_{2}},m_{p_{2}})$  where  $m_{t_2}\in \{m_{t_1},m_{t_1} + 1,\dots ,m_{t_1} + 49\}$ ,  $x_{t_2}\in \{f_{T_1} - 100,f_{T_1} - 80,\dots ,f_{T_1}\}$ ,  $x_{p_2}\in \{f_{P_1} - 5,f_{P_1} - 4,\dots ,f_{P_1}\}$ ,  $m_{p_2}\in \{m_{p_1},m_{p_1} + 1\}$ . Since  $m_{t_1},f_{T_1},m_{p_1},f_{P_1}$  is included a solution is found in this step too. Define  $f_{P_2},f_{T_2}$  similar to previous step.  
5. Finally search the largest tuple  $(-d_3, m_{t_3}, -x_{t_3}, -x_{p_3}, m_{p_3})$  where  $m_{t_3} = m_{t_2}, x_{t_3} \in \{f_{T_2} - 100, f_{T_2} - 99, \dots, f_{T_2}\}$ ,  $x_{p_3} = x_{p_2}, m_{p_3} = m_{p_2}$ .

While it is not an exhaustive search, we prioritize minimizing  $x_{t} - m_{t}$  over  $x_{p} - m_{p}$ .

# D.2 RESULTS

We compared the performance of data selection using circulation problem technique with randomly selecting equal number of samples for validation and test sets of all languages and measured the skew  $\tilde{\mu}_3$ , and standard deviation  $\sigma$  of the distribution of tags in the selected data. Here lower value of  $|\tilde{\mu}_3|$  means more symmetric distribution. On the other hand, a lower value of  $\sigma$  represents that the number of samples in each tag are closer to the mean.

# E TASKS CONSTRUCTION PROCESS

# E.1 TAG CLASSIFICATION

We formulate the task as a multi-label classification problem in two settings: Code-to-Tag (Code2Tag) and Problem Description-and-Code to Tag (DesCode2Tag). In Code2Tag, given a code  $C$  in any language, the task is to predict the corresponding tag set  $\mathbb{T}$ . In DesCode2Tag, the natural language problem description is also given as input in addition to the code. The performance difference between Code2Tag and DescCode2Tag settings can suggest if the problem description can help models identify the problem tags (i.e., the type of solution needed)..

For these tasks, the split for validation and test is done with a ratio of  $1:5$  (i.e.,  $\gamma = 0.2$ ) using Algorithm 1. To get the final  $\mathcal{D}_{\mathrm{valid}}$  and  $\mathcal{D}_{\mathrm{test}}$  with a feasible number of samples, we use flow network-based data selection approach with the details of hyper-parameter settings presented in Section 2.1.

The distribution of the samples according to the tags is presented in Fig 8.

We further propose a language-specific tag classification task, in which each programming language has its own Code2Tag and DesCode2Tag settings.

TABLE 7 - Comparison of skew and standard deviation of tags using circulation problem technique and random data selection (lower value is better).  

<table><tr><td rowspan="3">Language</td><td colspan="4">Skew, μ3</td><td colspan="4">Std. deviation, σ</td></tr><tr><td colspan="2">Validation</td><td colspan="2">Test</td><td colspan="2">Validation</td><td colspan="2">Test</td></tr><tr><td>Random</td><td>Circ.</td><td>Random</td><td>Circ.</td><td>Random</td><td>Circ.</td><td>Random</td><td>Circ.</td></tr><tr><td colspan="9">Tag Classification</td></tr><tr><td>C</td><td>2.778</td><td>2.499</td><td>2.848</td><td>2.440</td><td>249.161</td><td>213.849</td><td>880.881</td><td>772.549</td></tr><tr><td>C++</td><td>2.405</td><td>1.873</td><td>2.315</td><td>1.655</td><td>233.530</td><td>157.889</td><td>1154.538</td><td>751.023</td></tr><tr><td>Python</td><td>2.731</td><td>2.365</td><td>2.689</td><td>2.173</td><td>265.193</td><td>240.248</td><td>1125.133</td><td>992.904</td></tr><tr><td>Java</td><td>2.652</td><td>1.990</td><td>2.545</td><td>2.050</td><td>258.587</td><td>207.881</td><td>1175.790</td><td>972.703</td></tr><tr><td>C#</td><td>3.066</td><td>2.598</td><td>2.971</td><td>2.506</td><td>314.219</td><td>291.813</td><td>846.426</td><td>760.069</td></tr><tr><td colspan="9">Code Translation</td></tr><tr><td>C</td><td>2.744</td><td>2.455</td><td>2.941</td><td>2.332</td><td>117.298</td><td>99.261</td><td>267.214</td><td>215.881</td></tr><tr><td>C++</td><td>2.424</td><td>2.112</td><td>2.287</td><td>1.565</td><td>131.632</td><td>120.979</td><td>243.100</td><td>150.498</td></tr><tr><td>Python</td><td>2.533</td><td>2.379</td><td>2.635</td><td>2.294</td><td>123.710</td><td>110.076</td><td>271.219</td><td>237.179</td></tr><tr><td>Java</td><td>2.558</td><td>2.208</td><td>2.605</td><td>1.827</td><td>134.314</td><td>114.840</td><td>259.510</td><td>193.211</td></tr><tr><td>C#</td><td>3.147</td><td>2.532</td><td>2.943</td><td>2.395</td><td>103.838</td><td>96.747</td><td>250.049</td><td>220.615</td></tr><tr><td>PHP</td><td>2.506</td><td>2.744</td><td>2.520</td><td>2.730</td><td>59.321</td><td>59.877</td><td>270.582</td><td>278.530</td></tr><tr><td>Rust</td><td>2.520</td><td>2.393</td><td>2.534</td><td>2.311</td><td>59.269</td><td>60.253</td><td>269.352</td><td>264.507</td></tr><tr><td>Go</td><td>2.807</td><td>2.359</td><td>2.676</td><td>2.424</td><td>72.415</td><td>66.666</td><td>266.565</td><td>254.986</td></tr><tr><td>Javascript</td><td>2.611</td><td>2.611</td><td>2.473</td><td>2.473</td><td>64.090</td><td>64.090</td><td>246.483</td><td>246.483</td></tr><tr><td>Ruby</td><td>2.875</td><td>2.686</td><td>2.968</td><td>2.762</td><td>74.153</td><td>70.760</td><td>280.000</td><td>271.539</td></tr><tr><td>Kotlin</td><td>2.865</td><td>2.576</td><td>3.108</td><td>2.534</td><td>59.765</td><td>56.114</td><td>266.430</td><td>257.155</td></tr></table>

TABLE 8 - Size of the datasets for each task and the evaluation metrics. For Program Synthesis train data {problem description, solution} comes from 7514 problems of 11-17 languages where the input for validation and test data is only natural language text (problem description) independent of programming languages. For all other tasks, validation and test samples are reported for a total number of languages.  

<table><tr><td>Task Type</td><td>Task</td><td>|Lang|</td><td>|Train|</td><td>|Validation|</td><td>|Test|</td><td>Metric</td></tr><tr><td rowspan="2">Classification</td><td>Tag Classification</td><td>11</td><td>5,494,008</td><td>18,696</td><td>74,733</td><td>macro-f1</td></tr><tr><td>Code Compilation</td><td>11</td><td>19,915,150</td><td>6,394</td><td>30,388</td><td>accuracy</td></tr><tr><td rowspan="3">Generative</td><td>Program Synthesis</td><td>11</td><td>5,538,841</td><td>106</td><td>952</td><td>pass@k</td></tr><tr><td>Code Translation</td><td>11</td><td>5,538,841</td><td>7,034</td><td>20,356</td><td>pass@k</td></tr><tr><td>Automatic Program Repair</td><td>11</td><td>4,672,070</td><td>5,068</td><td>17,699</td><td>pass@k</td></tr><tr><td rowspan="2">Retrieval</td><td>Code-Code Retrieval</td><td>17</td><td>45,270</td><td>2,335</td><td>9,508</td><td rowspan="2">Acc@k</td></tr><tr><td>NL-Code Retrieval</td><td>17</td><td>55,924</td><td>2,780</td><td>11,157</td></tr></table>

# E.2 CODE COMPILEATION

Given a code  $C$  in a language  $L$  and its compiler or interpreter version  $B$ , the code compilation task is to decide whether the code compiles or not. The validation and test splits are created using a modified version of Algorithm 1 that balances the partition based on the compilation outcome of the code instead of the tags of the problem that the code belongs to. We use a ratio  $\gamma$  of  $1:5$ . Then a simplified version of the circulation problem is used to prevent too many codes coming from a single problem, and also to ensure a balanced output distribution. The details of hyper-parameter settings of

FIGURE 8 - Tag distribution in XCODEEVAL. In XCODEEVAL, often multiple tags are assigned to the same problem as there are usually many different ways to solve a problem or it may require a combination of different approaches.

the circulation problem technique are presented in Section 2.1. In the flow network construction, tags  $\{T_k\} = \{\text{true}, \text{false}\}$  as true if the code compiles or not. Furthermore true and false examples are present in equal numbers in both validation and test dataset.

We propose three generative tasks which require a global understanding of programming languages. For the evaluation of generative tasks, we follow execution-based evaluation instead of lexical similarity. All the generative tasks are evaluated using ExecEval execution engine. We provide complete unit tests for all problems in the validation and test dataset which also satisfy the conditions of the input-output description of the problem.

# E.3 PROGRAM SYNTHESIS

Given a problem described in natural language, program synthesis task is to write a program that solves the problem. We can express each sample in the dataset as a tuple  $(C,P,l,L)$ , where  $C$  denotes a solution code written in a programming language  $L$  for the problem  $P$ , and  $l$  denotes the compiler/interpreter version of the code. All code samples in our dataset are unique and marked as a correct solution (PASSED outcome) to the problem. The validation and test splits are created from the heldout problems using Algorithm 1 with a ratio  $(\gamma)$  of  $1:9$ . The generated code is judged based on executions on the unit tests.

# E.4 CODE TRANSLATION

Each sample in the code translation data can be expressed as a tuple  $(\mathcal{C}, P, l, L)$ , where  $\mathcal{C}$  denotes a set of solution codes in a programming language  $L$  for the problem  $P$ , and  $l$  denotes the compiler/interpreter version of the code. All codes in set  $\mathcal{C}$  are unique and guaranteed to be marked as a correct (PASSED outcome) solution to the problem by the compiler/interpreter.

The validation and test splits are created from the held-out problems using Algorithm 1 with a ratio  $(\gamma)$  of  $1:5$ , and employ the data selection method with flow network (Sec. 2.1) to have a practical evaluation setup while ensuring a balanced distribution over problems and tags. Figure 9 shows the distribution of the machine translation tasks. Since code translation considers all possible directions of translation between languages, in addition, to train, validation, and test split, we also provide a small validation split.

FIGURE 9 - Distribution of samples across all problems in the train, validation, test splits for all languages in the machine translation task.

# E.5 AUTOMATIC PROGRAM REPAIR (APR)

We consider APR as a task to synthesize a fix for a detected program bug. We create a bug-fix pair by matching a buggy code (1-5 execution outcome in Sec. 2.2) with a PASSED solution. Given a bug-specific defect, the objective of this task is to generate a correct fix that passes all the unit tests.

Let  $\mathbb{C} = \{C_1, \ldots, C_m\}$  be the set of programs submitted by a participant in a chronological order in order to solve a specific problem  $P$ . Some of these submissions can be 'buggy', while some can be PASSED. We create the 'bug-fix' pairs from  $\mathbb{C}$  as follows.

1. We iterate over  $\mathbb{C}$  and mark the PASSED ones as 'fixed'. Let  $C_j^*$  is one such case.  
2. For each buggy submission that was made before  $C_j^*$ , we measure its lexical similarity with  $C_j^*$  and select the one (say  $C_k$  where  $k < j$ ) with the highest similarity score to pair it with  $C_j^*$  and form a bug-fix pair  $(C_k, C_j^*)$ . We use difflib<sup>11</sup> to measure the similarity.  
3. With each bug-fix pair  $(C_k, C_j^*)$ , we also include the corresponding problem description  $P$  and execution outcome  $V_k$  (Section 2.2) of  $C_k$ .  
4. The tuple  $(C_k, C_j^*, P, V_k)$  represents a sample in our APR task.

We repeat this process for each participant and problem to create the final APR dataset. As reported in Table 8, it comprises more than 5M practical bug-fix pairs and supports 11 programming languages. For data selection in APR, we considered execution outcome (section 2.2) as tags in the network flow construction (section 2.1).

Due to the large input specification of the APR task, sometimes the input sequence length becomes too large. However, we have not compromised the benchmarks by selecting only small sequence length samples but rather keep them as challenging tasks for the language models. Figure 10 shows the length distribution of validation and test input sequence.

# E.6 CODE RETRIEVAL

Code retrieval tasks typically aim to measure the mere semantic relatedness between a natural language (NL) query and a programming language (PL) code. However, a code that is relevant, can still be buggy and thus be misleading (see an example in Figure 11). In view of this, we propose two new and more challenging retrieval tasks in our benchmark, which require a deeper understanding of the NL query and code. In particular, we propose NL-Code and Code-Code retrieval tasks that

FIGURE 10 - Distribution of sequence length of tokenized (bigscience/bloom) samples in the validation and test splits for all languages in the APR task (Section E.5). Each sample contains a buggy code with its problem description.

involve identifying a correct code from a large pool of candidates containing similar codes. In both tasks, for each programming language, we aggregate all the submitted codes and their test cases to create a retrieval corpus and a testbed for evaluating their correctness against test cases. Figure 9 gives a detailed statistics of our retrieval tasks. The datasets for the subtasks and the evaluation schema are discussed below.

FIGURE 11 – A code retrieval example. The candidate code on the left has a bug highlighted in red and that on the right has a fix highlighted in green. Both our proposed NL-Code and Code-Code retrieval tasks ensure differentiating between them and pose a more challenging task that aims to comprehend both semantic and logical relatedness.

NL-Code Retrieval This task involves matching an NL problem description to the most relevant and correct code from a pool of candidates. An example of an NL description and its corresponding codes are shown in Figure 1. To gather data for this task, we only use instances where the NL description is valid and there is at least one correct solution code (i.e., with execution outcome PASSED). For an NL problem description, we consider all the correct solutions as positive examples and all the wrong (or buggy) solutions as negative examples.

Code-Code Retrieval Given an input code (as a query), this task involves finding similar and logically equivalent code (i.e., passes the same set of test cases) from a collection of candidates. We ensure that the query code solves a specific problem (i.e., a correct solution without any detected bugs) and evaluate whether the retrieved candidate also solves the same problem or not. To collect data for this task, we only consider the programming problems which have at least two correct code solutions that pass all the corresponding test cases (i.e., with execution outcome PASSED).

TABLE 9 - Retrieval subtasks statistics. |Sizel denotes the number of instances. For each train/dev instance, we provide multiple positive and negative examples, and |Pos| and |Neg| refer to the total number of positive and negative annotations.  

<table><tr><td rowspan="2">Lang</td><td rowspan="2">Subtask</td><td colspan="3">Train</td><td colspan="3">Dev</td><td rowspan="2">Test |Size|</td><td rowspan="2">Retrieval Corpus |Size|</td></tr><tr><td>|Size|</td><td>|Pos|</td><td>|Neg|</td><td>|Size|</td><td>|Pos|</td><td>|Neg|</td></tr><tr><td rowspan="2">C</td><td>NL-Code</td><td>5,196</td><td>149,000</td><td>146,852</td><td>209</td><td>11,282</td><td>11,293</td><td>853</td><td rowspan="2">787,516</td></tr><tr><td>Code-Code</td><td>4,391</td><td>122,758</td><td>145,849</td><td>193</td><td>9,162</td><td>11,282</td><td>798</td></tr><tr><td rowspan="2">C#</td><td>NL-Code</td><td>4,878</td><td>75,386</td><td>55,579</td><td>207</td><td>6,574</td><td>4,757</td><td>828</td><td rowspan="2">251,147</td></tr><tr><td>Code-Code</td><td>4,397</td><td>69,016</td><td>54,886</td><td>194</td><td>5,854</td><td>4,742</td><td>785</td></tr><tr><td rowspan="2">C++</td><td>NL-Code</td><td>6,181</td><td>612,647</td><td>608,088</td><td>269</td><td>25,752</td><td>25,516</td><td>1,098</td><td rowspan="2">18,212,508</td></tr><tr><td>Code-Code</td><td>6,181</td><td>554,465</td><td>608,088</td><td>269</td><td>23,503</td><td>25,516</td><td>1,098</td></tr><tr><td rowspan="2">D</td><td>NL-Code</td><td>3,359</td><td>7,624</td><td>3,655</td><td>133</td><td>351</td><td>142</td><td>521</td><td rowspan="2">15,984</td></tr><tr><td>Code-Code</td><td>1,968</td><td>4,265</td><td>2,722</td><td>80</td><td>218</td><td>119</td><td>293</td></tr><tr><td rowspan="2">Go</td><td>NL-Code</td><td>3,764</td><td>25,656</td><td>18,957</td><td>165</td><td>1,466</td><td>750</td><td>662</td><td rowspan="2">68,237</td></tr><tr><td>Code-Code</td><td>3,090</td><td>21,787</td><td>18,079</td><td>148</td><td>1,242</td><td>727</td><td>563</td></tr><tr><td rowspan="2">Haskell</td><td>NL-Code</td><td>3,173</td><td>15,138</td><td>7,084</td><td>173</td><td>2,172</td><td>936</td><td>687</td><td rowspan="2">44,682</td></tr><tr><td>Code-Code</td><td>2,305</td><td>11,863</td><td>6,373</td><td>160</td><td>1,871</td><td>922</td><td>596</td></tr><tr><td rowspan="2">Java</td><td>NL-Code</td><td>5,930</td><td>393,891</td><td>375,416</td><td>250</td><td>17,623</td><td>16,008</td><td>1,021</td><td rowspan="2">2,523,044</td></tr><tr><td>Code-Code</td><td>5,792</td><td>320,738</td><td>375,176</td><td>245</td><td>14,022</td><td>15,981</td><td>991</td></tr><tr><td rowspan="2">Javascript</td><td>NL-Code</td><td>2,609</td><td>15,605</td><td>13,706</td><td>134</td><td>1,322</td><td>1,345</td><td>534</td><td rowspan="2">56,917</td></tr><tr><td>Code-Code</td><td>1,986</td><td>12,821</td><td>12,678</td><td>116</td><td>1,144</td><td>1,306</td><td>436</td></tr><tr><td rowspan="2">Kotlin</td><td>NL-Code</td><td>4,017</td><td>46,487</td><td>25,600</td><td>158</td><td>1,859</td><td>1,036</td><td>654</td><td rowspan="2">121,569</td></tr><tr><td>Code-Code</td><td>3,237</td><td>39,813</td><td>24,948</td><td>127</td><td>1,600</td><td>1,009</td><td>518</td></tr><tr><td rowspan="2">Ocaml</td><td>NL-Code</td><td>1,424</td><td>2,327</td><td>1,382</td><td>97</td><td>219</td><td>114</td><td>381</td><td rowspan="2">7,012</td></tr><tr><td>Code-Code</td><td>485</td><td>903</td><td>746</td><td>50</td><td>122</td><td>82</td><td>170</td></tr><tr><td rowspan="2">PHP</td><td>NL-Code</td><td>1,965</td><td>6,301</td><td>8,870</td><td>136</td><td>896</td><td>834</td><td>547</td><td rowspan="2">29,179</td></tr><tr><td>Code-Code</td><td>1,180</td><td>4,303</td><td>6,689</td><td>99</td><td>723</td><td>745</td><td>389</td></tr><tr><td rowspan="2">Pascal</td><td>NL-Code</td><td>4,432</td><td>113,222</td><td>105,127</td><td>216</td><td>10,113</td><td>8,568</td><td>853</td><td rowspan="2">494,473</td></tr><tr><td>Code-Code</td><td>3,949</td><td>97,179</td><td>104,320</td><td>208</td><td>8,496</td><td>8,564</td><td>816</td></tr><tr><td rowspan="2">Perl</td><td>NL-Code</td><td>1,276</td><td>3,903</td><td>1,957</td><td>102</td><td>559</td><td>338</td><td>412</td><td rowspan="2">11,035</td></tr><tr><td>Code-Code</td><td>678</td><td>2,627</td><td>1,531</td><td>64</td><td>457</td><td>305</td><td>309</td></tr><tr><td rowspan="2">Python</td><td>NL-Code</td><td>4,930</td><td>317,013</td><td>284,975</td><td>213</td><td>17,131</td><td>15,194</td><td>859</td><td rowspan="2">2,290,854</td></tr><tr><td>Code-Code</td><td>4,736</td><td>266,459</td><td>284,657</td><td>210</td><td>14,144</td><td>15,192</td><td>837</td></tr><tr><td rowspan="2">Ruby</td><td>NL-Code</td><td>2,349</td><td>15,230</td><td>7,278</td><td>157</td><td>2,371</td><td>866</td><td>649</td><td rowspan="2">44,934</td></tr><tr><td>Code-Code</td><td>1,742</td><td>12,714</td><td>6,683</td><td>145</td><td>2,113</td><td>854</td><td>569</td></tr><tr><td rowspan="2">Rust</td><td>NL-Code</td><td>3,860</td><td>30,673</td><td>14,923</td><td>137</td><td>742</td><td>303</td><td>551</td><td rowspan="2">59,829</td></tr><tr><td>Code-Code</td><td>3,062</td><td>26,779</td><td>14,290</td><td>104</td><td>605</td><td>288</td><td>428</td></tr><tr><td rowspan="2">Scala</td><td>NL-Code</td><td>2,555</td><td>7,858</td><td>5,210</td><td>144</td><td>867</td><td>459</td><td>591</td><td rowspan="2">24,780</td></tr><tr><td>Code-Code</td><td>1,527</td><td>5,268</td><td>4,078</td><td>123</td><td>723</td><td>442</td><td>448</td></tr></table>

From each of these problems, we randomly choose one correct solution as a (PL code) query and pair it with the other correct solutions as positive examples and the corresponding wrong solutions (i.e., with execution outcome WRONG ANSWER) as negative examples.

Retrieval Corpus Metadata and Evaluation Protocol We preserve the problem specifications and execution outcomes (e.g., PASSED, WRONG ANSWER) for each candidate code in our retrieval database. For both the NL-code and code-code retrieval tasks, we use this information to determine the correctness of a retrieved code, checking if that solves the same programming problem as the input query by passing all its unit tests or not.

Evaluation Metrics We evaluate the retrieval performance in terms of retrieval accuracy@k: computed as the proportion of queries for which a correct code retrieved within top-k.

Our retrieval benchmark has 17 programming languages and our training dataset is the largest that provides annotations of similar codes that are found logically equivalent or correct based on the passing of test cases. For evaluation purposes (i.e., for test sets), we release the input problem description (in NL-Code) or the input code (in Code-Code) only and keep all other metadata confidential. Covered programming languages and their data statistics in both tasks are summarized in Table 9.

Retrieval Evaluation Figure 12 reports one retrieval task (code-code) performance. As anticipated, the retrieval capability for the same language pair (a.k.a., monolingual retrieval) of our baseline model performances are relatively stronger and we observe performance degradation when performing cross-lingual retrieval between different languages. However, surprisingly, mono-lingual retrieval accuracies for popular languages like  $C$ ,  $C++$ ,  $C#$ , Python, and Java are lower than others such as Go, Haskell, Javascript, Kotlin, Ruby, Scala etc., possibly due to their large retrieval corpus size and presence of more hard negative candidates (very similar to the correct code). Furthermore it is suspected that the lack of enough resource on  $D$  programming language in both The Stack (Kocetkov et al., 2022) and xCODEEval is the primary reason for its poor scores.

FIGURE 12 -  $17 \times 17$  matrix of top-100 accuracy scores of StarEncoder finetuned on retrieval Code-Code dataset. Here a cell  $(x, y)$  denotes the top-100 accuracy score for code queries from language  $x$  and the retrieval corpus of language  $y$ . The average mono lingual retrieval accuracy is 84.19, and average cross lingual score is 56.93.

```json
{ "C": "GNU C11", "C#" : "Mono C#" , "C++": "GNU C++17", "Go": "Go", "Java": "Java 17", "Javascript": "Node.js", "Kotlin": "Kotlin 1.4", "PHP": "PHP", "Python": "PyPy 3", "Ruby": "Ruby 3", "Rust": "Rust 2018", }
```

FIGURE 13 - List of ExecEval compiler versions used to evaluate the generated codes.

# F EVALUATION METRICS

Tag Classification : Since it is a multi-class multi-label classification problem, we use  $f1$  score with macro averaging over the classes (in this case the tags) to measure the performance as macro averaging is class population size independent. This is done by first calculating the  $f1$  score for each class (tag)  $T \in \mathcal{T}$  (the set of all tags) with the following formula Taha and Hanbury (2015)

$$
\mathrm {f l} _ {T} = \frac {2 * \text {P r e c i s i o n} _ {T} * \text {R e c a l l} _ {T}}{\text {P r e c i s i o n} _ {T} + \text {R e c a l l} _ {T}} = \frac {2 * \mathrm {T P} _ {T}}{2 * \mathrm {T P} _ {T} + \mathrm {F P} _ {T} + \mathrm {F N} _ {T}}
$$

And then the macro average is calculated as the mean of  $\mathrm{f1}_T$  for all  $T\in \mathcal{T}$  Opitz and Burst (2019).

$$
\mathrm {f l} _ {\text {m a c r o}} = \frac {1}{| \mathcal {T} |} \sum_ {T \in \mathcal {T}} \mathrm {f l} _ {T}.
$$

Code Compilation : Since it is a binary classification problem, we use accuracy which is defined as the proportion of correct prediction among all predictions Metz (1978). That is

$$
\text {A c c u r a c y} = \frac {\mathrm {T P} + \mathrm {T N}}{\mathrm {T P} + \mathrm {T N} + \mathrm {F P} + \mathrm {F N}}.
$$

Generative Tasks : The generative tasks in XCODEEval(i.e. Automatic Program Repair, Code Translation, Program Synthesis) are all evaluated using pass@k used in Chen et al. (2021).

Code Retrieval : The Code-Code, and NL-Code retrieval tasks in XCODEEVALis evaluated using top- $k$  accuracy (Thakur et al., 2021).

# G IMPLEMENTATION DETAILS

Classification tasks : OpenAI chat completion API with gpt-3.5-turbo-0301 model was used at temperature 0.325 and  $n = 1$ . List prompts for Code2Tag, DescCode2Tag, Code Compilation as figure or inline styling, with example api response. Then evaluate each through corresponding metric as mentioned in Appendix F.

Generative tasks : OpenAI chat completion API with gpt-3.5-turbo-0301 model was used at temperature np.linspace(0,2,20) and  $n = 1$  for Program Synthesis. Then upon identifying best temperature at 0.325, another batch of codes were generated at temperature 0.325 and  $n = 20$ . For  $APR$ , Code Translation temperature of 0.325 and  $n = 10$  was used. The generated codes were executed with ExecEval with default parameters (follow appendix H for a list of different parameters and their default values) to determine its functional correctness and then evaluated using pass@k. Figure 13 shows the compiler versions used to execute the generated codes.

Retrieval tasks : We finetuned a DPR  $^{12}$  model with starencoder for both query and corpus encoder. Both NL-Code, and Code-Code were trained with maximum sequence length of 1024, and effective batch size 48 for 37 epochs. The model is trained with a multilingual manner. For Code-Code we used xCODEEval as it is, and for NL-Code we made the following template: 'Description: {{description}}' Input specification: {{input_spec}}' Output specification: {{output_spec}}' for the query string. For evaluation we used corpus provided by xCODEEval to generate the dense vectors and then perform queries with test split for both Code-Code, and NL-Code. Finally the top- $k$  accuracies were measured.

# H EXECEL DETAILS

TABLE 10 - Supported languages and their compiler/interpreter versions of our dataset in ExecEval.  

<table><tr><td>Language</td><td>Versions</td></tr><tr><td>Ruby</td><td>Ruby 3, Ruby</td></tr><tr><td>Javascript</td><td>Node.js, JavaScript</td></tr><tr><td>Go</td><td>Go 1.19</td></tr><tr><td>C++</td><td>GNU C++17, GNU C++17 (64), GNU C++20 (64), GNU C++11, Clang++17 Diagnostics, GNU C++, GNU C++14, GNU C++17 Diagnostics, Clang++20 Diagnostics, MS C++, GNU C++0x, MS C++ 2017</td></tr><tr><td>C</td><td>GNU C11, GNU C</td></tr><tr><td>Java</td><td>Java 6, Java 7, Java 17, Java 11, Java 8</td></tr><tr><td>Python</td><td>PyPy 3, PyPy 3-64, Python 3 + labs, Python 2, PyPy 2, Python 3</td></tr><tr><td>C#</td><td>MS C#, C# 8, Mono C#, .NET Core C#, C# 10</td></tr><tr><td>PHP</td><td>PHP 8.1</td></tr><tr><td>Rust</td><td>Rust, Rust 2021</td></tr><tr><td>Kotlin</td><td>Kotlin, Kotlin 1.4, Kotlin 1.5, Kotlin 1.7, Kotlin 1.6</td></tr></table>

ExecEval is an automated code execution and evaluation engine distributed through docker for security and portability. It supports 44 compiler versions for 11 programming languages as shown in table 10. It exposed NUM_WORKERS CLI argument to spawn multiple workers that can execute the codes. It is highly scalable in the sense of adding support for more languages or one can just change NUM_WORKERS to execute more codes in parallel. At the top level of ExecEval, there is a HTTP server that exposes 2 API endpoints /api/execute_code, /api/all_runtimes. Figure 15 shows a simple usage of execute_code API. By default the execution of a code is stopped when the code doesn't pass a unit test as pass@k depends on whether all the unit tests passed or not. This can be disabled by adding "stop_at_first_fail": false', in which case all unit tests for a given code will be executed irrespective of the outcomes for other unit tests. Figure 6 is generated with disabling 'stop_at_first_fail'. It is worth noting that, disabling this setting can increase the evaluation time significantly (e.g. in table 3 for program synthesis (N) where 23,320 codes were executed the difference was of approximately 12 minutes and 2 hours 12 minutes where ExecEval was running with 61 workers).

Security Measures ExecEval uses prlimit<sup>13</sup>, and seccomp<sup>14</sup> to limit system resources allocated for any instance of code executed through the API endpoint in addition to using unique unprivileged users for each worker spawned with NUM_WORKERS. Table 11 shows the default values provided to prlimit, furthermore nofile, and nproc are customized for each of the supported languages. The seccomp is used to block socket system call, which disables network access (this is default). One can enable network access by adding "block_network": false' in the request body as shown in fig. 15. Similarly, adding a 'limits' object in the request body allows one to change the

```json
{ "compile_cmd": "node", "compile_flags": "--check", "execute_cmd": "node", "execute_flags": "", "has_sanitizer": false, "is_compiled": true, "runtime_name": "JavaScript", "timelimit_factor": 3 }
```

FIGURE 14 - An example runtime object the response from /api/all_runtimes contains list of such objects.

```jsonl
{ "data": [ { "exec_outcome": "PASSED"  $\rightarrow$  1   
{ "input": "1 1", "output": [ "2"] ], "result": "2" }， { "exec_outcome": "PASSED"  $\leftrightarrow$  1   
{ "input": "1 10", "output": ["11"]} "result": "11" } ]
```

FIGURE 15 - On left: An example request body for /api/execute_code The Python code takes 2 numbers as input and prints their sum. On right: The response by ExecEval in response to the request shown in left.

TABLE 11 - Default resource limits values for prlimit used by ExecEval. The comment column shows the variable names as defined in sys/resource.h with some additional information.  

<table><tr><td>Resource</td><td>Value</td><td>Comment</td></tr><tr><td>core</td><td>0</td><td>RLIMIT_CORE</td></tr><tr><td>data</td><td>-1</td><td>RLIMIT_DATA</td></tr><tr><td>FSIZE</td><td>0</td><td>RLIMITFSIZE</td></tr><tr><td>sigpending</td><td>0</td><td>RLIMIT_sigPENDING</td></tr><tr><td>rss</td><td>-1</td><td>RLIMIT_RSS</td></tr><tr><td>nofile</td><td>4</td><td>RLIMIT_NOFILE</td></tr><tr><td>msgqueue</td><td>0</td><td>RLIMIT_MSGQUEUE</td></tr><tr><td>rtprio</td><td>0</td><td>RLIMIT_RTPRIO</td></tr><tr><td>stack</td><td>-1</td><td>RLIMITSTACK</td></tr><tr><td>cpu</td><td>2</td><td>RLIMIT_CPU, CPU time, in seconds</td></tr><tr><td>nproc</td><td>1</td><td>RLIMIT_NPROC</td></tr><tr><td>as</td><td>2 × 10243</td><td>RLIMIT_AS set to 2GB by default</td></tr><tr><td>locks</td><td>0</td><td>RLIMIT_LOCKS</td></tr></table>

limits for executing an individual code. $^{15}$  The execution of code via an unprivileged user disables the read, write, or execute permissions of any sensitive files. Figure 16, 17, and 18 shows an example of a fork bomb written in C, a network request in Python, and an escalated access in Python which are all blocked by ExecEval, respectively.

```txt
include<stdio.h>   
#include <sys/types.h>   
int main()   
{ while(1) fork(); return 0;   
}   
{"data":[ "exec_outcome":"<  $\leftrightarrow$  TIME_LIMIT_EXCeded $\leftrightarrow$  ", "input": "", "output": [ ], "result": null }   
]
```

FIGURE 16 - Left: An fork bomb written in C. Right: ExecEval ran the code with allowing only 1 process and thus the infinite loop resulted in TIME_LIMIT_EXCEEDED.

# I DEFINITION OF DATA ATTRIBUTES

All the raw data can be downloaded from the huggingface  $^{16}$ . For each of the tasks we have two data files that are required for multiple tasks.

1. problem descriptions.jsonl  
2. unittest_db.json

You can find these two files in the root directory of the main branch of huggingface dataset repository. To avoid data redundancy we didn't include these data with the relevant tasks, rather we add a unique id srcuid to retrieve these data. We include a data loader using datasets package that defines the tasks.

We provide the definition for each of the data attributes of xCODEEval in the following sections.

# I.1 PROBLEM DESCRIPTION (PROBLEM DESCRIPTION)

The problem descriptions are in the problemDescriptions.json file. This data source is linked to the proposed tasks by matching the srcuid column for each sample in the relevant tasks. The columns copied from the problemDescriptions.json file are prefixed with prob_desc_.

1. description: Problem description in textual format, math operations are written in latex.  
2. input_from: How the program should take the unit test.  
3. output_to: Where the program should output the result of the unit test.  
4. time_limit: Time limit to solve the problem.  
5. memory_limit: Memory limit to solve the problem.  
6. input_spec: How and in what order the input will be given to the program? It also includes the date range, types, and sizes.  
7. output_spec: How the outputs should be printed. Most of the time the unit test results are matched with an exact string match or floating point comparison with a precision boundary.  
8. sample_entries: A sample input for the code that is expected to solve the problem described in description.  
9. sample_outputs: The expected output for the sample_input that is expected to solve the problem described in description.  
10. notes: Explanation of sample_entries & sample_outputs.  
11. tags: The problem categories.  
12. srcuid: The unique id of the problem. This ID is referred to in the task data samples instead of putting all this information.  
13. difficulty: How difficult is it to solve the problem for a human (annotated by an expert human).  
14. created_at: The Unix timestamp when the problem was released. Use datetime lib in Python to parse it to a human-readable format.

# I.2 UNIT TESTS (HIDDEN_UNIT_TEST)

The unit tests needed for execution based evaluation are in the unittest_db.json file. This data source is linked to the proposed tasks by matching the srcuid column for each sample in the relevant tasks. The columns copied from the unittest_db.json file are under the attribute hidden_unit_test.

1. unittest_db.json dict keys i.e., db884d679d9cfb1dc4bc511f83beedda are the srcuid from problem descriptions.jsonl.  
2. input: Input of the unit test.  
3. output: List of expected outputs for the unit test.

# I.3 TAG CLASSIFICATION (TAG_CLASSIFICATION)

Given a source_code the objective is to classify the code into multi-label tags (label:tags).

1. lang: Runtime/compiler version of the source_code.  
2. source_code: A program.  
3. tags: List of potential algorithmic techniques required to write the program.  
4. lang_cluster: A generic programming language name the value of lang belongs to.  
5. code.uid: A unique ID for the sample. It is not important for model training. If you find any issue with the sample, you can report it to us by mentioning the code.uid.  
6. srcuid: A specific identifier that shows which problem the code is associated with. This identifier is important for the training of the model. The problem referred to by the srcuid provides a natural description of the problem that the code successfully solved. Refer to Structure of problem descriptions.jsonl.  
7. Difficulty: Difficulty rating of the problem indicated by srcuid. The higher the harder.

# I.4 CODE COMPILATION (CODE_COMPILATION)

Given a source_code the objective is to classify if the code compiles or not (label:compilation_error).

1. lang: Runtime/Compiler version of the source_code.  
2. source_code: A program.  
3. lang_cluster: A generic programming language name the value of lang belongs to.  
4. compilation_error: True/False, Indicates if the code generates a compilation error or not.  
5. code.uid: A unique ID for the sample. It is not important for model training. If you find any issue with the sample, you can report it to us by mentioning the code.uid.  
6. src_uid: A specific identifier that shows which problem the code is associated with. This identifier is important for the training of the model. The problem referred to by the src_uid provides a natural description of the problem that the code successfully solved. Refer to Structure of problem descriptions.jsonl.  
7. Difficulty: Difficulty rating of the problem indicated by srcuid. The higher the harder.  
8. file_name: Name of the source JSON file from where data is loaded.

# I.5 AUTOMATIC PROGRAM REPAIR (APR)

Given a bug_source_code the objective is to generate a fixed version of the code that passes all the unit tests. Use fix_source_code for training.

1. similarity_score: A similarity score between bug_source_code and fix_source_code given by difflib.  
2. equal_cnt: A metric comparing bug_source_code and fix_source_code. Recommended by difflib.  
3. replace_cnt: A metric comparing bug_source_code and fix_source_code. Recommended by difflib.  
4. delete_cnt: A metric comparing bug_source_code and fix_source_code. Recommended by difflib.  
5. insert_cnt: A metric comparing bug_source_code and fix_source_code. Recommended by difflib.  
6. fixOps_cnt: A metric comparing bug_source_code and fix_source_code. Recommended by difflib.  
7. bug_source_code: Buggy code.  
8. fix_source_code: A potential fix of the buggy code that passed all the unit tests.

9. lang: Runtime/Compiler version of the source_code.  
10. fix_code_UID: A unique ID for the fix code. It is not important for model training. If you find any issue with the sample, you can report it to us mentioning the fix_code_UID.  
11. bug_code_UID: A unique ID for the buggy code. It is not important for model training. If you find any issue with the sample, you can report it to us mentioning the bug_code_UID.  
12. srcuid: A specific identifier that shows which problem the code is associated with. This identifier is important for the training of the model. The problem referred to by the srcuid provides a natural description of the problem that the code successfully solved. Refer to Structure of problem descriptions.jsonl.  
13. apr_id: A unique ID for the apr sample. It is not important for model training. If you find any issue with the sample, you can report it to us mentioning the apr_id.  
14. difficulty: Difficulty rating of the problem indicated by srcuid. The higher the harder.  
15. tags: List of potential algorithmic techniques required to write the program.  
16. bug_exec_outcome: A pre-run execution outcome of bug_source_code. Follow Section 2.2 to know the potential list of outcomes. The exec_outcome flags in the training data comes from a pre-run environment from the source website and they are not verified in ExecEval. However, training data doesn't include unit-test to avoid potential hacks and confusion. We provide unit test for only validation and test data.  
17. fix_exec_outcome: A pre-run execution outcome of fix_source_code. Follow Section 2.2 to know the potential list of outcomes. The exec_outcome flags in the training data comes from a pre-run environment from the source website and they are not verified in ExecEval. However, training data doesn't include unit-test to avoid potential hacks and confusion. We provide unit test for only validation and test data.  
18. potential Dominant_fix_op: A potential fix op recommended by difflib.  
19. lang_cluster: A generic programming language name the value of lang belongs to.  
20. prob_desc_description: Problem description in textual format, math operations are written in latex.  
21. prob_desc_input_from: How the program should take the unit test.  
22. prob_desc_output_to: Where the program should output the result of the unit test.  
23. prob_desc_time_limit: Time limit to solve the problem.  
24. prob_desc_memory_limit: Memory limit to solve the problem.  
25. prob_desc_input_spec: How and in what order the input will be given to the program? It also includes the date range, types, and sizes.  
26. prob_desc_output_spec: How the outputs should be printed. Most of the time the unit test results are matched with an exact string match or floating point comparison with a precision boundary.  
27. prob_desc_sample_entries: A sample input for the code that is expected to solve the problem described in description.  
28. prob_desc_sample_outputs: The expected output for the sample_input that is expected to solve the problem described in description.  
29. prob_desc_notes: Explanation of sample_entries & sample_outputs.  
30. prob_desc_created_at: The Unix timestamp when the problem was released. Use datetime lib in Python to parse it to a human-readable format.  
31. file_name: Name of the source jsonl file from where data is loaded.  
32. hidden_unitTests: a list of unit tests returned as string. use json.dumps(hidden_unitTests) to load the data.

# I.6 CODE TRANSLATION (CODE_TRANSLATION)

Given a source code (source_code) in lang_cluster, generate a code in target programming language.

1. lang: Runtime/Compiler version of the source_code.  
2. source_code: A program.

3. code.uid: A unique ID for the sample. It is not important for model training. If you find any issue with the sample, you can report it to us by mentioning the code.uid.  
4. srcuid: A specific identifier that shows which problem the code is associated with. This identifier is important for the training of the model. The problem referred to by the srcuid provides a natural description of the problem that the code successfully solved. Refer to Structure of problem descriptions.jsonl  
5. difficulty: Difficulty rating of the problem indicated by srcuid. The higher the harder.  
6. exec_outcome: Execution outcome status. Follow Section 2.2 to know the potential list of outcomes. The exec_outcome flags in the training data comes from a pre-run environment from the source website and they are not verified in ExecEval. However, training data doesn't include unit-test to avoid potential hacks and confusion. We provide unit test for only validation and test data  
7. lang_cluster: A generic programming language name the value of lang belongs to.  
8. prob_desc_description: Problem description in textual format, math operations are written in latex.  
9. prob_desc_input_from: How the program should take the unit test.  
10. prob_desc_output_to: Where the program should output the result of the unit test.  
11. prob_desc_time_limit: Time limit to solve the problem.  
12. prob_desc_memory_limit: Memory limit to solve the problem.  
13. prob_desc_input_spec: How and in what order the input will be given to the program? It also includes the date range, types, and sizes.  
14. prob_desc_output_spec: How the outputs should be printed. Most of the time the unit test results are matched with an exact string match or floating point comparison with a precision boundary.  
15. prob_desc_sample_entries: A sample input for the code that is expected to solve the problem described in description.  
16. prob_desc_sample_outputs: The expected output for the sample_input that is expected to solve the problem described in description.  
17. prob_desc_notes: Explanation of sample_entries & sample_outputs.  
18. prob_desc_created_at: The Unix timestamp when the problem was released. Use datetime lib in Python to parse it to a human-readable format.  
19. file_name: Name of the source jsonl file from where data is loaded.  
20. hidden_unitTests: a list of unit tests returned as string. use json.dumps(hidden_unitTests) to load the data.

# I.7 PROGRAM SYNTHESIS (PROGRAM_SYNTHESIS)

Given a srcuid read problem description from problem descriptions.json and generate a solution for problem description.

1. lang: Runtime/Compiler version of the source_code.  
2. source_code: A program.  
3. code.uid: A unique ID for the sample. It is not important for model training. If you find any issue with the sample, you can report it to us by mentioning the code.uid.  
4. srcuid: A specific identifier that shows which problem the code is associated with. This identifier is important for the training of the model. The problem referred to by the srcuid provides a natural description of the problem that the code successfully solved. Refer to Structure of problem descriptions.jsonl.  
5. difficulty: Difficulty rating of the problem indicated by srcuid. The higher the harder.  
6. exec_outcome: Execution outcome status. Follow Section 2.2 to know the potential list of outcomes. The exec_outcome flags in the training data comes from a pre-run environment. However, training data doesn't include unit-test to avoid potential hacks. We provide unit tests for only dev and test data.

7. lang_cluster: A generic programming language name the value of lang belongs to.  
8. prob_desc_description: Problem description in textual format, math operations are written in latex.  
9. prob_desc_input_from: How the program should take the unit test.  
10. prob_desc_output_to: Where the program should output the result of the unit test.  
11. prob_desc_time_limit: Time limit to solve the problem.  
12. prob_desc_memory_limit: Memory limit to solve the problem.  
13. prob_desc_input_spec: How and in what order the input will be given to the program? It also includes the date range, types, and sizes.  
14. prob_desc_output_spec: How the outputs should be printed. Most of the time the unit test results are matched with an exact string match or floating point comparison with a precision boundary.  
15. prob_desc_sample_entries: A sample input for the code that is expected to solve the problem described in description.  
16. prob_desc_sample_outputs: The expected output for the sample_input that is expected to solve the problem described in description.  
17. prob_desc_notes: Explanation of sample_entries & sample_outputs.  
18. prob_desc_created_at: The Unix timestamp when the problem was released. Use datetime lib in Python to parse it to a human-readable format.  
19. file_name: Name of the source jsonl file from where data is loaded.  
20. hidden_unit/tests: a list of unit tests returned as a string. use json.dumps(hidden_unit/tests) to load the data.

# I.8 RETRIEVAL CORPUS (RETRIEVAL_CORPUS)

Use the retrieval Corpus to perform query for retrieval_n1_code (appendix I.9) and retrieval_code_code (appendix I.10).

1. idx: An integer index to identify the code. It is unique within the codes of each language.  
2. source_code: A program.  
3. file_name: Name of the source json file from where data is loaded.

# I.9 RETRIEVAL NL-CODE (RETRIEVAL_NL_CODE)

Given a NL (problem description) retrieve similar source code from retrieval Corpus (appendix I.8).

1. nl : Problem description in textual format, math operations are written in latex. Given as input query.  
2. positive_code : list of positive codes for nl.  
3. negative_code : list of negative codes for nl.  
4. srcuid: A specific identifier that shows which problem the code is associated with. This identifier is important for the training of the model. The problem referred to by the srcuid provides a natural description of the problem that the code successfully solved. Refer to Structure of problem descriptions.jsonl.  
5. file_name: Name of the source json file from where data is loaded.

# I.10 RETRIEVAL CODE-CODE (RETRIEVAL_CODE_CODE)

Given a source_code, retrieve similar source code from retrieval Corpus (appendix I.8).

1. positive_code : list of positive codes for nl.  
2. negative_code : list of negative codes for nl.  
3. srcuid: A specific identifier that shows which problem the code is associated with. This identifier is important for the training of the model. The problem referred to by the srcuid provides a natural description of the problem that the code successfully solved. Refer to Structure of problem descriptions.jsonl.

4. source_code: A source code given as input query.  
5. file_name: Name of the source jsonl file from where data is loaded.

# J DATASHEETS FOR DATASETS

We follow the questionnaires from Gebru et al. (2021) as the datasheet for XCODEEval.

# J.1 MOTIVATION

For what purpose was the dataset created? XCODEEval dataset was specifically created to address three main aspects: (i) Reasoning, (ii) Multilinguality in terms of programming languages, and (iii) Executability of the programming languages. These aspects were thoroughly discussed in Section 1 of the main paper, providing detailed insights into the motivation behind the dataset creation.

Who created the dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)? XCODEEVAL is an output of a passion project driven by a group of researchers from (i) Islamic University of Technology (ii) Nanyang Technological University (iii) Bosch Research.

Who funded the creation of the dataset? Nanyang Technological University provided the necessary computing resources for the project. None of the project members received any remuneration for their contributions.

# J.2 COMPOSITION

What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)? Please follow the Section I for details.

How many instances are there in total (of each type, if appropriate)? Please follow the Table 8, 2, and 9 for the details statistics of the dataset.

Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set? Dataset contains all possible instances.

What data does each instance consist of? Please follow the Section I for details.

Is there a label or target associated with each instance? Please follow the Section I for details.

Is any information missing from individual instances? For a few problem description, difficulty is assigned as None due to data unavailability.

Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)? Please follow the Section I for details.

Are there recommended data splits (e.g., training, development/validation, testing)? We explicitly defined the training, validation and test split for XCODEVAL. Please follow the Section 2.1 for more details.

Are there any errors, sources of noise, or redundancies in the dataset? To the best of our knowledge there are not errors, sources of noise or redundancies in XCODEVAL.

Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets) The dataset is self-contained.

Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor- patient confidentiality, data that includes the content of individuals' non-public communications)? The dataset is collected from open sourced sources. There are no confidentiality or non-public entity in the dataset.

Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? To the best of our knowledge there are no offensive, insulting, threatening content in the dataset.

Does the dataset identify any subpopulations (e.g., by age, gender)? No.

Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset? There are no attributes in the dataset that allow to identify individuals.

Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals race or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)? There are no attributes in the dataset that allow this.

# J.3 COLLECTION PROCESS

How was the data associated with each instance acquired? Following Li et al. (2022), the data was collected from codeforces.com and then associated with different tasks. Please follow Section E for more details.

If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)? Dataset wasn't sampled from a large dataset. We proposed the dataset for the first time.

Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)? Dataset was collected by the author of this paper.

Over what timeframe was the data collected? The data was downloaded in between Feb, 2022 to January, 2023.

Were any ethical review processes conducted (e.g., by an institutional review board)? No.

Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)? The data is downloaded by an author. No third parties or other sources are involved.

Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted? Potential impact of the dataset is discussed at section 5 and section 6.

# J.4 PREPROCESSING/CLEANING/LABELING

Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)? We de-anonymized the data and remove data with sensitive information (i.e., email, large infographic, toxic keywords). The labels come as a metadata from the sources.

Was the "raw" data saved in addition to the preprocessed/cleaned/ labeled data (e.g., to support unanticipated future uses)? No.

Is the software that was used to preprocess/clean/label the data available? No software was used for labeling the data.

Has the dataset been used for any tasks already? Yes. We evaluated ChatGPT and trained StarEncoder using the dataset.

Is there a repository that links to any or all papers or systems that use the dataset Yes, https://github.com/ntunlp/xCodeEval.

What (other) tasks could the dataset be used for? We proposed 7 different tasks for XCODEEVAL. Please follow table 8 for details.

Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses? No.

# J.5 DISTRIBUTION

Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created? Please follow the Licensing section in https://github.com/ntunlp/xCodeEval for details.

When will the dataset be distributed? The data is already distributed via huggingface  $^{18}$ .

Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)? I Please follow the Licensing section in https : //github.com/ntunlp/xCodeEval for details.

Have any third parties imposed IP-based or other restrictions on the data associated with the instances? No.

Do any export controls or other regulatory restrictions apply to the dataset or to individual instances? No.

# J.6 MAINTENANCE

Who will be supporting/hosting/maintaining the dataset? Huggingface is hosting the dataset. The authors are maintaining the dataset. Nanyang Technological University is supporting the dataset.

How can the owner/curator/manager of the dataset be contacted (e.g., email address)? Email.

Is there an erratum? None at this point. The dataset is hosted through git lfs. The future erratum can be tracked easily.

Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances)?

To the best of our knowledge there are no errors in the dataset. The authors do not intend to add new instances at this point. But the authors remain open to remove/correct instances given that any labeling errors found.

If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were the individuals in question told that their data would be retained for a fixed period of time and then deleted)? The dataset doesn't relate to people.

Will older versions of the dataset continue to be supported/hosted/maintained? Yes. The older version should be accessed via git LFS.

If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so? Since the dataset is fixed, there is currently no way to contribute to it. Please note that any extensions or augmentations of the dataset are subject to the same license as this dataset.

# K DISCUSSION ON THE POSSIBILITY OF DATA LEAKAGE AND CONTAMINATION

Although we completely separate our validation and test data, LLMs might have possible data leakage from pretraining. We find that even identifying data leakage (test data exists or not in the preretraining corpora) is challenging using conventional data search methods due to search cost & complexity (e.g., exact match or token overlapping methods) while hashing based searches suffer from not having properly segmented text. For leakage-free evaluation, we approach employs "knowledge cut-off" which show that the data contamination significantly impacts the model performance and it needs to be interpreted with proper analyses. We plan to evaluate on seperate human written testset in future.

# L THE DATASET NUTRITION LABEL

We follow the framework proposed by Holland et al. (2018). Table 12 gives an overview of dataset facts. The variable description can be found in appendix I. We discuss about provenance in appendix J.3 and appendix J.6.

# M DATA CARD

Data card for the XCODEVAL is distributed via huggingface platform. [Link]

TABLE 12 - Dataset facts for XCODEEval . It covers the metadata related to the whole dataset.  

<table><tr><td colspan="2">Metadata</td></tr><tr><td>Filename</td><td>File names for each of the tasks can be found here .</td></tr><tr><td>Format</td><td>jsonl, json, arrow dataset loader.</td></tr><tr><td>Url</td><td>https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval</td></tr><tr><td>Domain</td><td>Programming Language, Competitive Programming</td></tr><tr><td>Keywords</td><td>programming-language, code, program-synthesis, automatic-code-repair, code-retrieval, code-translation, code-classification, execution, benchmark, multilingual, multitask, unit-test</td></tr><tr><td>Type</td><td>columnar</td></tr><tr><td>Rows</td><td>Follow table 2, table 8, and table 9</td></tr><tr><td>Columns</td><td>Follow appendix I</td></tr><tr><td>Missing</td><td>0%</td></tr><tr><td>License</td><td>CC BY-NC 4.0</td></tr><tr><td>Released</td><td>MARCH 2023</td></tr><tr><td>Range</td><td>From Feb 19, 2010 to Nov 21, 2022</td></tr><tr><td>Description</td><td>We introduce xCodeEval, the largest executable multilingual multitask benchmark to date consisting of 25 M document-level coding examples from about 7.5 K unique problems covering up to 17 programming languages with execution-level parallelism. It features a total of seven tasks involving code understanding, generation, translation and retrieval, and it employs an execution-based evaluation. We develop a test-case based multilingual code execution engine, ExecEval that supports all the programming languages in xCodeEval. We also propose a novel data splitting and a data selection schema for balancing data distributions over multiple attributes based on geometric mean and graph-theoretic principle.</td></tr></table>

```python
{
    "data": [ 
        "exec_outcome": "< 
            RUNTIME_ERROR", 
            "input": "", 
            "output": [ 
                ],
            ]
        ],
    "result": "Traceback (< 
        most recent call < 
            last): \n File ") 
            /usr/lib/python3. 
            11/urllib/request < 
                .py", line 1348, 
                in do_open\n 
                h.request (req. 
                get_method(), req 
                .selector, req. 
                data, headers, \n 
                File "/usr/lib/ 
                python3.11/http/ 
                client.py", line 
                1282, in request 
                \n self. 
                _send_request (< 
                    method, url, body 
                    , headers, 
                    encode_chunked) \n 
                    **Truncated** 
line 941, in connect\n 
                self.sock = 
                self. 
                _create_Connection< 
                (\n 
                ) 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    >
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
    > 
        */
        else:  
            print(f'Request_failed')  
            with.status;  
            code:{response}  
            .strip()  
print(f'Request_failed')  
            with.status;  
            code:{response}  
            .strip()  
else:
```

FIGURE 17 - Left: A python code performing a network request. Right: ExecEval responded with RUNTIME_ERROR as the socket system call is blocked.

FIGURE 18 - Left: A python code performing a subprocess call to run 'ps -ef'. Right: ExecEval responded with RUNTIME_ERROR as nofile (table 11) is limiting the execution of such codes.  
```txt
{
    "data": [
        "exec_outcome": "<>
            RUNTIME_ERROR",
            "input": ],
            "output": [
                ""
            ]
            ],
    "result": "Traceback (<)
            → most recent call <>
            → last): \n File \
            )
            → /codestore/6cd9b< 
            → 5215a524abab3712< 
            → bc897de2be5/test.< 
            → py", line 5, in < 
            → <module>\n < 
            → process = < 
            → subprocess.Popen(< 
            → command, stdout=< 
            → subprocess.PIPE, 
            → stderr=subprocess< 
            → .PIPE)\n < 
            → 
            → >/usr/lib/< 
            → File \
            ) = self.< 
            → subprocess.py", < 
            → line 890, in < 
            → __init__\n < 
            → errread, errwrite< 
            → ) = self.< 
            → _get_handle(< 
            → stdin, stdout, < 
            → stderr)\n < 
            → >/usr/lib/< 
            → /python3.11/< 
            → subprocess.py", < 
            → line 1664, in < 
            → _get Handles\n < 
            → c2pread, < 
            → c2pwrite = os.< 
            → pipe()\n < 
            → >/usr/lib/< 
            → nOSError: [Errno < 
            → 24] Too many open< 
            → files\n"
```

# Footnotes:

Page 0: *Equal Contribution †Work done when the author was on leave from Nanyang Technological University 1. https://github.com/ntunlp/xCodeEval 2. https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval 3. https://github.com/ntunlp/ExecEval 
Page 3: 4.  $\mathbb{Z}_{+}$  denotes the set of non-negative integers. 
Page 7: 5. Evaluation done at temperature 0.32, generating 10 samples per problem. 
Page 17: 6. https://sites.research.google/datacardsplaybook/ 7. https://github.com/ntunlp/ExecEval 8. https://github.com/ntunlp/xCodeEval 9. https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval 
Page 19: 10. https://eslint.org 
Page 24: 11. https://docs.python.org/3/library/difflib.html 
Page 29: 12. https://github.com/facebookresearch/DPR 13. https://man7.org/linux/man-pages/man1/prlimit.1.html 14. https://man7.org/linux/man-pages/man2/seccomp.2.html 
Page 31: 15. For more details follow: https://github.com/ntunlp/ExecEval. 
Page 39: 18. https://huggingface.co/datasets/NTU-NLP-sg/xCodeEval 
