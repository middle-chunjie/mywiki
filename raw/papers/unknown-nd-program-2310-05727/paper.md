# THE PROGRAM TESTING ABILITY OF LARGE LANGUAGE MODELS FOR CODE

Weimin Xiong $^{1,2}$ , Yiwen Guo $^{3*}$ , Hao Chen $^{4}$

$^{1}$ Tencent Security Big Data Lab,  $^{2}$ Peking University,  $^{3}$ Independent Researcher,  $^{4}$ UC Davis weiminxiong@tencent.com, guoyiwen89@gmail.com, chen@ucdavis.edu

# ABSTRACT

Recent development of large language models (LLMs) for code like CodeX and CodeT5+ demonstrates tremendous promise in achieving code intelligence. Their ability of synthesizing code that completes a program for performing a pre-defined task has been intensively tested and verified on benchmark datasets including HumanEval and MBPP. Yet, evaluation of these LLMs from more perspectives (than just program synthesis) is also anticipated, considering their broad scope of applications in software engineering. In this paper, we explore the ability of LLMs for testing programs/code. By performing thorough analyses of recent LLMs for code in program testing, we show a series of intriguing properties of these models and demonstrate how program testing ability of LLMs can be improved. Following recent work which utilizes generated test cases to enhance program synthesis, we further leverage our findings in improving the quality of the synthesized programs and show  $+11.77\%$  and  $+4.22\%$  higher code pass rates on HumanEval+ comparing with the GPT-3.5-turbo baseline and the recent state-of-the-art, respectively. Our code is available at https://github.com/WeiminXiong/TestingLLM.

# 1 INTRODUCTION

The community has witnessed a surge in the development of large language models (LLMs), which have achieved incredible ability in understanding and generating not only texts but also code. LLMs for code (CodeX (Chen et al., 2021), StarCoder (Li et al., 2023b), CodeT5+ (Wang et al., 2023b), etc) have been widely adopted to a variety of applications to achieve code intelligence. However, current evaluation of these LLMs mostly focuses on program completion/synthesis, despite the models can also be utilized in other applications. As the field continues to advance, evaluation of these models from more perspectives is anticipated, which could facilitate deeper understanding of the LLMs.

As the core of analyzing the behavior of code, the ability of generating proper test cases is of great desire to software engineering. Although embryonic development of using deep models in testing has been shown (Tufano et al., 2020; 2022), with the remarkable progress in LLMs, it is unclear how far have such abilities of AI been advanced when these powerful models are equipped. In this paper, we, for the first time, analyze the ability of recent LLMs in testing programs/code. Our analyses are performed based on 164 problems from HumanEval+ (Chen et al., 2021) and 427 sanitized problems from MBPP (Austin et al., 2021). We consider 4 test-case generation settings (i.e., self-generated, all-generated, oracle, and placeholder in Figure 1) and test a collection of 11 competitive LLMs for code (including 4 LLMs that have around 1 billion parameters and 7 substantially larger LLMs). We conducted a variety of experiments, from which many intriguing takeaway messages are delivered.

Several very recent papers (Shi et al., 2022; Li et al., 2023a; Chen et al., 2023) have shown that appropriate usage of even generated test cases can improve the quality of program synthesise, in a spirit that the synthesized programs that could pass a large number of test cases are more likely to be correct. Nevertheless, the quality of the generated test cases largely impacts the performance of such methods. Due to the lack of systematic evaluation of the testing ability of LLMs for code, it is unclear how to craft test cases that could be potentially more helpful to program synthesis and, more broadly, code intelligence. The studies in this paper aim to shed light on this. We will demonstrate

that, substantially improved program synthesis performance can be obtained by utilizing takeaway messages in our studies. Specifically, on GPT-3.5-turbo, we can achieve  $+11.77\%$  higher code pass rate on HumanEval+, in comparison with the GPT-3.5-turbo baseline. When compared with a very recent state-of-the-art called CodeT, our solution achieves  $+4.22\%$  higher code pass rate.

# 2 EVALUATION METRICS

To make the evaluation more reliable and comprehensive, it is crucial to first design some suitable metrics, like BLEU (Papineni et al., 2002), ROUGE (Lin, 2004), and the pass rate (Chen et al., 2021) for evaluating machine translation, text summarization, and program synthesis, respectively. In this section, we specify two main evaluation metrics to evaluate the program testing ability of LLMs, from the perspective of correctness and diversity.

Pass rate In software engineering, we expect test cases to represent some desired "ground-truth" functionality of the tested program/code. In practice, such "ground-truth" functionality can be described in the header comments of a function (i.e., docstrings of the function) and tested using the oracle implementation, as in HumanEval (Chen et al., 2021) and MBPP Austin et al. (2021). The oracle program/code should be able to pass the test, if a generated test case is correct. Therefore, we leverage the pass rate as a measure to evaluate the correctness of the generated test cases. For a fair comparison, we instruct each model to generate three test cases in the prompt, and, when a model generates more than three test cases, we select the first three for evaluation. Assuming that there are in total  $M$  programming problems in an experimental dataset and, for each problem, we have  $N$  program/code implementations to be generated test cases for. Each model has only one chance to generate these test cases for each program/code. Then, we calculate the pass rate as:

$$
P = \frac {1}{M N} \sum_ {i = 1} ^ {M} \sum_ {j = 1} ^ {N} \frac {p _ {i j}}{n _ {i j}}, \tag {1}
$$

where  $n_{ij}$  is the number of test cases in  $\mathcal{Q}_{ij}$  which includes no more than three test cases generated for the  $j$ -th program/code implementation of the  $i$ -th problem by the evaluated LLM at once, i.e.,  $\mathcal{Q}_{ij} = \{(x_{ijk}, y_{ijk})\}_k$ , and  $p_{ij}$  is the number of test cases (in  $\mathcal{Q}_{ij}$ ) that do not fail the oracle.

The pass rate defined in Eq. (1) measures correctness of the generated test cases. However, as can be seen in Figure 1, the model can generate duplicate test cases that are less helpful, even though they are correct. To avoid such an evaluation bias, we further advocate dedduplication in the set of test cases that are considered as correct, which leads to computation of a deduplicated pass rate defined as  $P' = \frac{1}{MN} \sum \sum p_{ij}' / n_{ij}'$ , where we use ' to denote the numbers of unique test cases.

Coverage rate In addition to the above pass rates, we further consider coverage rate as a more fine-grained metric for evaluating the diversity of the generated test cases. According to its definition, converge rate computes the degree to which the code is executed, given a test case. Since, for each program/code, we keep no more than three test cases at once, we calculate how much percentage of the control structure is covered given these test cases. Similar to Eq. (1), we evaluate the performance of testing all programs/code over all  $M \times N$  times of generation, i.e., we calculate

$$
C = \frac {1}{M N} \sum_ {i = 1} ^ {M} \sum_ {j = 1} ^ {N} c _ {i j}. \tag {2}
$$

We utilize the pytest library to evaluate branch coverage for all the three test cases for each code and aggregate the results for all programs/code and all problems. Apparently, a higher  $C$  indicates better testing ability of an LLM, since we expect all parts of the programs/code to be executed to find our all potential bugs, with the set of test cases generated by this LLM.

# 3 LARGE LANGUAGE MODELS FOR CODE

In this section, we outline the evaluated models. We adopt some "small" models whose numbers of parameters are around 1B (to be more specific, from 770M to 1.3B in our choices) and some larger models that achieve state-of-the-art performance in the task of program synthesis.

(a) Self-generated  
```python
def cycpattern_check(a, b):
    "You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word."
    for i in range(len(b)):
        if b in a:
            return True
            b = b[1:] + b[0]
        return False
# Check the correctness of this function with three test cases
assert cycpattern_check("abcd", "cdab") == True
assert cycpattern_check("hello", "lohel") == True
assert cycpattern_check("abcd", "cdab") == True
```

(b) Positioner  
```python
def cycpattern_check(a, b):
    "You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word."
    pass
# Check the correctness of this function with three test cases
    assert cycpattern_check("rotation", "ion") == True
    assert cycpattern_check("rotation", "onr") == True
    assert cycpattern_check("rotation", "noi") == False
```

(c) Oracle  
```python
def cycpattern_check(a, b):
    "You are given 2 words. You need to return True if the second word or any of its rotations is a substring in the first word."
    l = len(b)
    pat = b + b
    for i in range(len(a) - 1 + 1):
        for j in range(1 + 1):
            if a[i:i+1] == pat[j:j+1]:
                return True
            return False
# Check the correctness of this function with three test cases
assert cycpattern_check("abcde", "deabc") == True
assert cycpattern_check("abcdef", "defabc") == True
assert cycpattern_check("12345", "45123") == False
```

(d) All-generated  
```python
def cycpattern_check(a, b):
    "You are given 2 words. You need to return True if the second word or any of its rotations is a. If you want to use a different word
    c = a
    b = b[::-1]
    for i in b:
        if i in c:
            return True
        else
            return False
    return False
    Set of code generated by several different LLMs
# Check the correctness of this function with three test cases
assert cycpattern_check("a", "b") == True
assert cycpattern_check("a", "def") == False
assert cycpattern_check("abcde", "cde") == True
```

Figure 1: Generating test cases for (a) self-generated code, (b) placeholder, (c), oracle code, and (d) all-generated code.

For the small models, we use InCoder (1.3B) (Fried et al., 2023), CodeGen2 (1B) (Nijkamp et al., 2023a), CodeT5+ (770M) (Wang et al., 2023b), and SantaCoder (1.1B) (Allal et al., 2023). InCoder is a unified generative model that can perform program/code synthesis as well as code editing, and it combines the strengths of causal language modeling and masked language modeling. The CodeGen2 model was trained on a deduplicated subset of the Stack v1.1 dataset (Kocetkov et al., 2023), and its training is formatted with a mixture of objectives for causal language modeling and span corruption. CodeT5+ is an encoder-decoder model trained on several pre-training tasks including span denoising and two variants of causal language modeling. SantaCoder was trained on the Python, Java, and JavaScript code in the Stack dataset. The pass rate (Chen et al., 2021) of programs generated by these models is compared in Table 1. When evaluating the (program) pass rate, we let the model generate 200 code implementations for each problem, and we set the temperature to 0.2, 0.6, and 0.8 for calculating pass@1, pass@10, and pass@100, respectively.

As for larger models that achieve state-of-the-art program synthesis performance, we use CodeGen2 (16B) (Nijkamp et al., 2023a), CodeGen-Multi (16B) Nijkamp et al. (2023b), CodeGen-Mono (16B) Nijkamp et al. (2023b), StarCoder (15B) (Li et al., 2023b), WizardCoder (15B) (Luo et al., 2023), CodeGeeX2 (6B) (Zheng et al., 2023), and GPT-3.5-turbo. CodeGen-Multi and CodeGen-Mono are two large models from the first version of CodeGen. CodeGen-Multi was first trained on the pile dataset (Gao et al., 2020) and then trained on a subset of the publicly available BigQuery dataset which contains code written in C, C++, Go, Java, JavaScript, and Python. Based on the 16B CodeGen-Multi model, CodeGen-Mono (16B) was obtained by further tuning on a set of Python code collected from GitHub. Given a base model that was pre-trained on 1 trillion tokens from the Stack dataset, the 15B StarCoder model was obtained by training it on 35B tokens of Python code. WizardCoder further empowers StarCoder with instruction tuning, following a similar instruction evolution strategy as in WizardLM (Xu et al., 2023). CodeGeeX2, the second generation of a multilingual generative model for code, is implemented based on the ChatGLM2 architecture and trained on more code data. GPT-3.5-turbo is a very capable commercial LLM developed by OpenAI and we accessed it in August, 2023. For these large LLMs, we tested pass@1 of all models except GPT-3.5-turbo (whose result can be directly collected from Liu et al. (2023a)'s paper). By sorting their pass@1 from high to low, they are ranked as: GPT-3.5-turbo  $(61.7\%)$ , WizardCoder  $(46.23\%, 15\mathrm{B})$ , CodeGeeX2  $(29.97\%, 6\mathrm{B})$ , StarCoder  $(27.9\%, 15\mathrm{B})$ , CodeGen-Mono  $(26.15\%, 16\mathrm{B})$ , CodeGen2  $(19.33\%, 16\mathrm{B})$ , CodeGen-Multi  $(15.35\%, 16\mathrm{B})$ . The ranks on the MBPP dataset are similar.

<table><tr><td>Model</td><td>Size</td><td>Pass@1</td><td>Pass@10</td><td>Pass@100</td></tr><tr><td>InCoder</td><td>1.3B</td><td>6.99%/14.06%</td><td>14.20%/34.98%</td><td>23.76%/55.34%</td></tr><tr><td>CodeGen2</td><td>1B</td><td>9.19%/17.50%</td><td>16.06%/36.86%</td><td>25.90%/59.32%</td></tr><tr><td>CodeT5+</td><td>770M</td><td>12.95%/28.02%</td><td>25.09%/47.69%</td><td>37.56%/65.26%</td></tr><tr><td>SantaCoder</td><td>1.1B</td><td>15.21%/29.42%</td><td>26.01%/51.30%</td><td>43.80%/69.10%</td></tr></table>

Table 1: Program synthesis performance of the small LLMs (whose number of parameters is around 1 billion) evaluated on HumanEval+ / MBPP (sanitized).

# 4 CODE TO BE TESTED

For evaluating the testing ability of LLMs, we need an oracle to express the ground-truth functionality of the tested code. Fortunately, current datasets for evaluating program synthesis performance often provide such oracles (see HuamnEval (Chen et al., 2021) and MBPP (Austin et al., 2021)). In our experiments, we utilize an amended version of HumanEval called HumanEval+ (Liu et al., 2023a), together with MBPP (the sanitized version). These datasets are established to evaluate basic Python programming performance of LLMs, and they contain 164 and 427 problems, respectively.

# 4.1 IMPERFECT CODE IMPLEMENTATIONS

In order to simulate real-world scenarios where the tested code is often buggy, we first adopt synthesized programs/code as the programs/code to be tested, considering that the synthesis of even state-of-the-art LLMs is still imperfect. We evaluate the performance of each LLM in testing code that was generated by itself (which is denoted as "Self-generated") and code in a set consisting of program completion results of several different LLMs (which is denoted by "All-generated"). That said, the compared LLMs take different code implementations when generating test cases for each programming problem in the self-generated setting. Whereas, in the all-generated setting, the same program/code implementations are given to different LLMs for generating test cases for comparison. In practice, we apply InCoder (1.3B), CodeGen2 (1B), CodeT5+ (770M), and SantaCoder (1.1B) to construct the all-generated program/code set, while, in the self-generated setting, each LLM first synthesize code and complete a program to fulfill the requirement of each programming problem, and the LLM then generates test cases with the synthesized programs/code in its prompts. The temperature for all LLMs is uniformly set to 0.2 for synthesizing the programs/code in both settings. We obtain 100 program/code completions for each problem and we prompt each LLM to generate 3 test cases for every program/code implementation in the self-generated setting, and we sampled 100 implementations from the synthesis results of InCoder (1.3B), CodeGen2 (1B), CodeT5+ (770M), and SantaCoder (1.1B) to form the all-generated code set, i.e., we have  $N = 100$  for these settings.

We follow the same way of generating code as introduced in the papers of these LLMs. For model without instruction tuning, like InCoder and CodeT5+, we synthesize programs/code using the default prompt given by each programming problem in the test dataset, while, for models that have adopted instruction tuning, e.g., WizardCoder, we use the recommended prompt in their papers.

# 4.2 OPTIMAL CODE IMPLEMENTATIONS

As a reference, we also report the performance of generating accurate and diverse test cases when the written code is perfectly correct, which is achieved by adopting the oracle as the programs/code to be tested (and such a setting is denoted by "Oracle"). Since Liu et al. (2023a) have reported that some oracle code in the HumanEval dataset can be incorrect, we adopt the amended oracle set in HumanEval+ in this setting. We further used the revised oracle code implementations instead of the original ones in evaluating the pass rate (i.e.,  $P'$ ) of the generated test cases. Considering that the public datasets often only provide one oracle implementation for each problem, and to keep the uncertainty of evaluation results consistent, we copy the oracle implementation by  $100 \times$  and we prompt to generate 3 test cases for each of these copies. It can be regarded as letting  $N = 100$ , just like in the previous settings in Section 4.1.

# 4.3 NO IMPLEMENTATION

In certain scenarios, we require test cases before the function/program has been fully implemented, hence we also evaluate in a setting where the main body of a tested function/program is merely a placeholder, as depicted in Figure 1(b). This scenario often occurs when the main code has not yet been implemented for a function/program or the test engineer does not want to introduce implementation bias to the LLM when generating test cases for a function/program. We denote such a setting as "Placeholder" in this paper. We also let  $N = 100$ , as in the oracle setting.

# 5 TEST CASE GENERATION

In this section, we introduce how test cases can be generated, when the implementation of a function/program is given as described in Section 4. In this paper, a desired test case is a pair of input and its expected output for the function/program defined in the context. As an example, Figure 1 demonstrates some test cases for the programming problem of checking whether the two words satisfy a specific rotation pattern. To generate test cases, we use the LLMs introduced in Section 3.

We wrote extra prompts to instruct the LLMs to generate three test cases for each given code which include docstrings that describe the purpose of this function, as depicted in Figure 1. Our instruction commands the LLMs (1) to "check the correctness of this function with three test" and (2) to start writing test code with an "assert" statement and the tested function, which specifies the format of the test cases as input-output pairs that can be parsed. For instance, given the example in Figure 1, the extra prompt should be "# Check the correctness of this function with three test cases \n assert cycpattern_check".

We then concatenate the extra prompt with the code and feed the concatenation into each LLM, for extracting test cases from the model output. The LLM will try to complete the given input by generating one or more "assert" statement(s), and we split the generation results into sub-strings, with "assert" as the separator. Each sub-string is then considered as a test statement, and we only take the first three statements if there exist more than three statements, as has been introduced in Section 2. Such a split can be considered as an effective post-processing operation which largely improves the quality of the generated test code, considering that some non-sense code pieces may be generated in the output of the LLMs. When using HumanEval+ and MBPP, we try removing test cases in the docstrings of the function, if there exist any, just to get rid of the broad hints from the docstrings (Chen et al., 2023). The temperature for generating test cases is kept as 0.2.

Once obtained, the generated test cases are then compiled, and evaluated for their correctness and diversity to report the pass rate  $P'$  and the coverage rate  $C$ . When calculating, for each problem and every set of completions generated, we create a temporary folder.

# 6 MAIN RESULTS FOR TEST CASE GENERATION

The experiment results of small and large LLMs on HumanEval+ can be found in Table 2 and Table 3, respectively. Table 4 shows the results on MBPP. There are several takeaways from these tables.

- First, the test cases generated by LLMs can show a descent pass rate, and this pass rate is even higher than the code pass rate on HumanEval+, which holds for both large and small LLMs. Such a result is consistent with intuitions from previous work which rejects code that cannot pass the generated tests to improve the quality of program synthesis.

<table><tr><td>Model</td><td>Size</td><td>Oracle</td><td>Self-generated</td><td>All-generated</td><td>Positioner</td></tr><tr><td>InCoder</td><td>1.3B</td><td>21.31% (61.43%)</td><td>23.37% (59.36%)</td><td>22.72% (61.10%)</td><td>25.19% (62.75%)</td></tr><tr><td>CodeGen2</td><td>1B</td><td>31.63% (71.55%)</td><td>30.62% (69.38%)</td><td>30.93% (69.70%)</td><td>30.69% (69.00%)</td></tr><tr><td>CodeT5+</td><td>770M</td><td>35.43% (71.45%)</td><td>32.34% (70.45%)</td><td>31.49% (69.75%)</td><td>32.67% (70.67%)</td></tr><tr><td>SantaCoder</td><td>1.1B</td><td>30.97% (71.46%)</td><td>30.43% (70.81%)</td><td>30.13% (70.55%)</td><td>30.78% (71.24% s)</td></tr></table>

Table 2: The pass rates (and coverage rate) of the test cases generated on HumanEval+ in different settings for LLMs with around 1 billion parameters.

<table><tr><td>Model</td><td>Size</td><td>Oracle</td><td>Self-generated</td><td>All-generated</td><td>Position</td></tr><tr><td>CodeGen-Multi</td><td>16B</td><td>43.88% (67.91%)</td><td>41.85% (69.30%)</td><td>40.38% (66.97%)</td><td>39.74% (68.28%)</td></tr><tr><td>CodeGen2</td><td>16B</td><td>46.34% (73.07%)</td><td>45.44% (73.17%)</td><td>42.00% (72.45%)</td><td>42.69% (72.86%)</td></tr><tr><td>CodeGen-Mono</td><td>16B</td><td>49.03% (74.82%)</td><td>45.73% (73.74%)</td><td>43.91% (73.66%)</td><td>44.92% (73.63%)</td></tr><tr><td>StarCoder</td><td>15B</td><td>55.07% (76.02%)</td><td>52.52% (72.45%)</td><td>48.20% (72.30%)</td><td>50.58% (74.52%)</td></tr><tr><td>CodeGeeX2</td><td>6B</td><td>57.03% (74.42%)</td><td>53.16% (73.55%)</td><td>49.28% (70.32%)</td><td>51.78% (73.08%)</td></tr><tr><td>WizardCoder</td><td>15B</td><td>53.89% (77.87%)</td><td>55.47% (76.07%)</td><td>48.02% (75.27%)</td><td>49.89% (75.12%)</td></tr><tr><td>GPT-3.5-turbo</td><td>-</td><td>71.03% (77.85%)</td><td>72.45% (77.24%)</td><td>59.24% (74.99%)</td><td>66.28% (74.03%)</td></tr></table>

Table 3: The pass rates (and coverage rate) of the test cases generated on HumanEval+ in different settings for LLMs whose parameters are obviously more than 1 billion.

Figure 2: The correlation between code past rate and test pass rate in the "Oracle" setting.

Figure 3: How the correctness of the test cases changes with their order when being generated.

- Second, the correctness of the generated test cases is positively correlated with the LLM's ability of generating code (see Figure 2, where each red cross represents the performance of a model), which means an LLM showing the state-of-the-art program synthesis performance is possibly also the state-of-the-art LLM for program testing. As shown in Tables 2 and 3, GPT-3.5-turbo, which synthesizes programs/code with the highest correctness, provides test cases with the highest pass rate  $(71.03\%)$  on HumanEval+. For an LLM, the more accurate it is capable of synthesizing programs/code on a dataset, the more powerful testing ability will probably be exhibited on the same dataset. There also exist a few exceptions, e.g., SantaCoder (1.1B) outperforms CodeT5+ (770M) and CodeGen2 (1B) in generating code, but it shows inferior performance in program testing on HumanEval+. By carefully examining the test cases yielded by SantaCoder on HumanEval+, we found that it tends to generate more complex and longer test cases than CodeT5+ for several problems on HumanEval+, which are often more desirable in program testing. This is also why the SantaCoder test cases show higher coverage rates in Table 2. To be concrete, in Problem 131 in HumanEval+, where the program is required to return the product of all digits with an odd position in a positive integer  $n$  (which is the input), the test input provided by CodeT5+ tends to be small for this problem, e.g.,  $n = 2$ , while the SantaCoder test cases tend to have more digits (e.g.,  $n = 12358$ ), which is helpful in digging out hidden bugs. Yet, generating longer and more complex test cases is more challenging, and the correctness can be lower.

- Third, as can be seen in Tables 3 and 4, generating test cases using large LLMs with their self-generated code (in the prompts) often leads to a higher level of correctness, compared with the placeholder results. This observation is in fact unsurprising, considering that generating code first and test case afterwards resembles the chain-of-thought prompting (Wei et al., 2022) (if adopting the placeholder is regarded as a plain prompting), which is beneficial to reasoning. Moreover, the self-generated performance of an LLM sometimes even outperforms its testing performance with an oracle, and we ascribe this to: 1) randomness in the style of the oracles which are few in number and/or 2) less distribution shift between self-generated code in prompt and the training code, for some powerful LLMs.

<table><tr><td>Model</td><td>Size</td><td>Oracle</td><td>Self-generated</td><td>All-generated</td><td>Placeholder</td></tr><tr><td>InCoder</td><td>1.3B</td><td>21.56% (46.81%)</td><td>17.98% (46.11%)</td><td>19.53% (46.45%)</td><td>22.58% (46.72%)</td></tr><tr><td>CodeGen2</td><td>1B</td><td>25.61% (54.26%)</td><td>21.85% (53.09%)</td><td>23.15% (50.43%)</td><td>22.81% (52.11%)</td></tr><tr><td>CodeT5+</td><td>770M</td><td>29.02% (56.86%)</td><td>24.44% (52.31%)</td><td>24.84% (53.20%)</td><td>25.59% (55.81%)</td></tr><tr><td>SantaCoder</td><td>1.1B</td><td>32.37% (55.68%)</td><td>26.40% (52.38%)</td><td>26.20% (52.83%)</td><td>26.53% (53.86%)</td></tr><tr><td>CodeGen-Multi</td><td>16B</td><td>41.32% (60.63%)</td><td>35.96% (59.03%)</td><td>34.17%, (58.09%)</td><td>34.84% (58.92%)</td></tr><tr><td>CodeGen2</td><td>16B</td><td>45.30% (62.15%)</td><td>38.67% (60.16%)</td><td>36.77% (58.59%)</td><td>37.27% (59.16%)</td></tr><tr><td>CodeGen-Mono</td><td>16B</td><td>50.24% (64.39%)</td><td>43.94% (62.94%)</td><td>39.55% (61.99%)</td><td>42.41% (62.31%)</td></tr><tr><td>StarCoder</td><td>15B</td><td>54.84% (65.10%)</td><td>46.77% (63.60%)</td><td>42.80% (61.95%)</td><td>45.35% (62.66%)</td></tr><tr><td>CodeGeeX2</td><td>6B</td><td>52.45% (64.64%)</td><td>44.52% (63.72%)</td><td>41.72% (60.48%)</td><td>43.86%, (63.51%)</td></tr><tr><td>WizardCoder</td><td>15B</td><td>57.85% (66.68%)</td><td>46.56% (64.86%)</td><td>41.62% (60.72%)</td><td>47.45% (64.54%)</td></tr><tr><td>GPT-3.5-turbo</td><td>-</td><td>74.30% (66.19%)</td><td>66.14% (65.30%)</td><td>49.56% (62.95%)</td><td>63.34% (64.72%)</td></tr></table>

Table 4: The pass rates (and coverage rate) of the test cases generated on MBPP.

- Fourth, with only a few exception, test cases obtained using the oracle code exhibit slightly higher code coverage, while the coverage rate achieved in the other settings (i.e., the self-generated, all-generated, and the placeholder settings) is often slightly lower.

The above four takeaway messages can all be inferred from Tables 2, 3, and 4. In addition to all these results, we conduct more experiments to achieve the following takeaway messages.

- Fifth, by analyzing the relationship between the quality of code in prompts and the correctness of test, we found that correct code implementation in the prompt often leads to higher quality of test code generation than the case when some incorrect code is given. We conducted an experiments where we first select programming problems in HumanEval+, where the code pass rate of an LLM is neither  $0\%$  or  $100\%$ . Then we separate self-generated programs/code of the model into two groups, with one group only contains programs/code that are considered as correct and the other only contains incorrect programs/code. In Table 5, we compare the performance of using these two sorts of code in the prompt, for generating test cases using the same LLM. Apparently, the quality of test cases obtained with correct programs/code is obviously higher. We further evaluate the overall testing performance of LLMs with only correct self-generated programs/code, if there exists any, in their prompts. Unlike in Table 5 where we do not take problems that can be  $100\%$  or  $0\%$  solved, we take all given problems in this evaluation, except, for every problem, we eliminate all incorrect self-generated programs/code if there exist at least one correct implementation synthesized by the evaluated LLM. By doing so, we can observe substantially improved program testing ability on HumanEval+ (i.e.,  $74.95\%$  for GPT-3.5-turbo,  $56.87\%$  for WizardCoder,  $54.33\%$  for CodeGeeX2, and  $53.24\%$  for StarCoder), comparing with the original self-generated results in Table 3. The same on MBPP. Recall that, in our third takeaway message, we have mentioned that test cases obtained with self-generated programs/code sometimes even outperform those yielded in the oracle setting on HumanEval+, maybe partly due to less distribution shift between the synthesized programs/code and the training code, the above results further confirm that, if we can improve the correctness of synthesized programs/code while keeping proper styles, then more powerful testing ability can further be achieved.

- Sixth, by conducting an additional experiment, we further compare the quality of test cases collected from different positions in the generation results. For every set of the three generated test cases, we analyze the relationship between their correctness and the order when they are generated. The results are illustrated in Figure 3. As can be seen in the figure, the first generated test case often shows the best correctness and the latterly generated ones are more incorrect. This may be due to the fact that the model tends to first generate content with a high level of confidence (which is also more likely to be correct).

# 7 IMPROVING PROGRAM SYNTHESIS USING THE GENERATED TEST CASES

High quality test cases are not only desired in program analyses, but also helpful to program synthesis. Previous methods have successfully used generated test cases to improve the performance of LLMs in synthesizing programs/code. For instance, Li et al. (2023a) designed a special prompt

<table><tr><td>Model</td><td>Size</td><td>w/ correct code</td><td>w/ incorrect code</td><td>#Problem</td></tr><tr><td>InCoder</td><td>1.3B</td><td>28.55%</td><td>27.39%</td><td>27</td></tr><tr><td>CodeGen2</td><td>1B</td><td>27.25%</td><td>25.74%</td><td>11</td></tr><tr><td>CodeT5+</td><td>770M</td><td>40.19%</td><td>36.78%</td><td>27</td></tr><tr><td>SantaCoder</td><td>1.1B</td><td>37.45%</td><td>34.08%</td><td>24</td></tr><tr><td>CodeGen-Multi</td><td>16B</td><td>55.49%</td><td>50.06%</td><td>32</td></tr><tr><td>CodeGen2</td><td>16B</td><td>43.56%</td><td>39.31%</td><td>29</td></tr><tr><td>CodeGen-Mono</td><td>16B</td><td>45.18%</td><td>42.86%</td><td>56</td></tr><tr><td>StarCoder</td><td>15B</td><td>58.16%</td><td>57.08%</td><td>68</td></tr><tr><td>CodeGeeX2</td><td>6B</td><td>52.84%</td><td>48.63%</td><td>51</td></tr><tr><td>WizardCoder</td><td>15B</td><td>48.02%</td><td>45.12%</td><td>54</td></tr><tr><td>GPT-3.5-turbo</td><td>-</td><td>75.39%</td><td>68.52%</td><td>126</td></tr></table>

Table 5: With the correct (self-generated) code, the LLMs show stronger ability of generating correct test cases on HumanEval+ (evaluated only on those problems that can neither be  $0\%$  solved nor  $100\%$  solved), than in the case where incorrect self-generated code is given in the prompts. Since most LLMs cannot generate any correct code for many hard problems while they often generate incorrect code even for easy problems, the number of tested problems in this experiment increases with the power of the tested LLM, as shown in the rightmost column.

which involves the test cases as an preliminary, if they are available, for generating programs/code. Shi et al. (2022) introduced a Bayes risk decoding mechanism, which executes generated code from a candidate set on a small number of test inputs and selects by marginalizing over implementations that share the same outputs when given these test inputs. It utilizes consistency between the output of code that is correctly implemented. One step further, Chen et al. (2023) proposed CodeT, which leverages the LLM to obtain test cases first and tests all synthesized programs/code with these test cases by performing a dual execution agreement, which considers both the agreement between the execution output and the test output and the consistency between the output of correct program implementations, to obtain state-of-the-arts. We encourage interested reader to read the original paper.

In the previous section, we have obtained results about many intriguing properties of the program testing performance of LLMs for code. In this section, we would like to drive the readers to think whether it is possible to utilize these results to improve the program synthesis performance, considering that the test cases (hand-crafted and given or automatically generated in particular) are widely and successfully used in program synthesis. We shall demonstrate that, by utilizing takeaway messages in Section 6, the program synthesis performance of previous methods can be improved significantly. Taking CodeT as an example of the previous state-of-the-art, the method uses a placeholder to generate test cases and treats all the test cases as equally correct as a prior. However, as discussed in our third takeaway message, using self-generated code helps to achieve more powerful ability in generating correct test cases. Moreover, if multiple test cases are provided in a single run of generation given an LLM, the correctness of the test cases decreases with their generation order, as shown in our fifth point. Hence, to obtain superior program synthesis performance, we introduce two simple modifications to it: 1) we employ the "self-generated" setting instead of the "placeholder" setting for generating test cases, which means we synthesize programs first and then generate test cases for each program in the prompt, 2) we assign different weights to the generated test cases based on their order in each generation result.

We test the effectiveness of using 1) the prompt which involves self-generated (SG) code and 2) the rank-based weighted (RW) test cases, in improving program synthesis performance on HumanEval+. The details of our implementation are introduced as follows. Following Chen et al. (2023), we used a temperature of 0.8 to generate code and self-generated test cases. Each test case is weighted by  $p^{i - 1}$  with  $i$  being its order in the model output, and we let  $p = 0.8$ .

Table 6 shows the results. We compare CodeT with CodeT+SG, CodeT+RW, and CodeT+SG+RW. For CodeT, we follow their official implementation and generate  $100 \times 5$  test cases for each problem. For fair comparison, we ensure that our solutions with SR and/or RW generate the same numbers of program implementations and test cases as CodeT does. Hence, for each problem in HumanEval+, we synthesize a program together with its 5 test cases for 100 times when SR and/or RW are incorporated, i.e., we have  $i \in \{1,2,3,4,5\}$ . It can be seen from the table that both SG and WR improves the program synthesis performance considerably on most LLMs, except for Incoder, CodeGen2-1B,

<table><tr><td>Model</td><td>Size</td><td>Baseline</td><td>CodeT</td><td>+ SG</td><td>+ RW</td><td>+ SG &amp; RW</td></tr><tr><td>InCoder</td><td>1.3B</td><td>6.99%</td><td>9.85%</td><td>9.45%</td><td>10.26%</td><td>9.98%</td></tr><tr><td>CodeGen2</td><td>1B</td><td>9.19%</td><td>15.15%</td><td>14.89%</td><td>15.67%</td><td>15.35%</td></tr><tr><td>CodeT5+</td><td>770M</td><td>12.95%</td><td>16.57%</td><td>16.28%</td><td>17.19%</td><td>16.98%</td></tr><tr><td>SantaCoder</td><td>1.1B</td><td>15.21%</td><td>18.43%</td><td>18.17%</td><td>18.75%</td><td>18.63%</td></tr><tr><td>CodeGen-Multi</td><td>16B</td><td>15.35%</td><td>24.50%</td><td>25.71%</td><td>25.72%</td><td>26.95%</td></tr><tr><td>CodeGen2</td><td>16B</td><td>19.33%</td><td>27.56%</td><td>28.51%</td><td>28.43%</td><td>29.63%</td></tr><tr><td>CodeGen-Mono</td><td>16B</td><td>26.15%</td><td>35.63%</td><td>36.69%</td><td>36.63%</td><td>37.95%</td></tr><tr><td>StarCoder</td><td>15B</td><td>27.90%</td><td>40.46%</td><td>41.21%</td><td>42.12%</td><td>43.15%</td></tr><tr><td>CodeGeeX2</td><td>6B</td><td>29.97%</td><td>44.16%</td><td>45.23%</td><td>44.92%</td><td>46.32%</td></tr><tr><td>WizardCoder</td><td>15B</td><td>46.23%</td><td>58.41%</td><td>60.13%</td><td>59.60%</td><td>61.45%</td></tr><tr><td>GPT-3.5-turbo</td><td>-</td><td>61.70%</td><td>69.25%</td><td>72.45%</td><td>70.75%</td><td>73.47%</td></tr></table>

Table 6: Program synthesis performance (Pass@1) of LLMs can be significantly improved by using our takeaway messages in Section 6. The experiment is on HumanEval+.

CodeT5+, and SantaCoder for which the test cases generated in the placeholder setting show similar or even higher correctness than in the self-generated setting and SG fails with them. For some LLMs, SG is more powerful, while, on the other models including SantaCoder and StarCoder, RW is more powerful. By combining SG and RW, the program synthesis performance of most powerful LLMs in Table 6 improves, comparing to only using one of the two. On GPT-3.5-turbo and WizardCoder, which are the best two models in synthesizing programs on HumanEval+, we achieve +4.22% and +3.04% performance gains for CodeT, respectively, with SG & RW.

We believe there exist other ways of using our takeaway messages for improving the program synthesis performance, and we would like to encourage future work to explore more in this direction.

# 8 RELATED WORK

Test case generation via program analysis. Generating reasonable test cases for analyzing programs is a long standing problem in the software engineering community. Various program analysis techniques, e.g., fuzzing, have been developed for achieving this goal. AFL++ (Fioraldi et al., 2020) is the most popular tool which incorporate many techniques in this category. A major weakness of these techniques is understandability of the generated test cases.

Test case generation via deep learning. The invention of transformer (Vaswani et al., 2017) and self-supervised pre-training (Devlin et al., 2018; Lewis et al., 2019; Raffel et al., 2020; Radford et al., 2018) have brought a breakthrough to programming language processing. After being trained in a self-supervised manner on a large and diverse code corpus, LLMs have demonstrated remarkable abilities in understanding and synthesizing programs. We have also witnessed the adaptation of pretrained LLMs (e.g., ChatGPT) to fuzz testing (Xia et al., 2023) very recently. Nevertheless, there still lack and requirie in-depth analyses and intensive comparisons of different LLMs in program testing. In particular, powerful LLMs emerge continuously. For instance, the recent WizardCoder (Luo et al., 2023) exhibits an oblivious program synthesis superiority over other open-source LLMs. In our study, we focus on the analyses and comparison of the LLMs in writing test code and generating test cases.

Evaluation of Large Language Model. Recently, large language models (LLMs) has incited substantial interest in both academia and industry. In order to evaluate the capabilities of large language models, a variety of effort have been devoted from the perspectives of natural/programming language processing accuracy, robustness, ethics, biases, and trustworthiness, etc. For instance, PromptBench (Zhu et al., 2023) demonstrates that current LLMs are sensitive to adversarial prompts, and careful prompt engineering is necessary for achieving descent performance with them. Another example, DecodingTrust (Wang et al., 2023a), offers a multifaceted exploration of trustworthiness of the GPT models, especially GPT-3.5 and GPT-4. The evaluation expands beyond the typical trustworthiness concerns to include several new critical aspects. Agentbench (Liu et al., 2023b) evaluates LLM as agents on challenging tasks in interactive environments. Their experimental results show that, while top commercial LLMs present a strong ability of acting as agents in complex environments, there is a significant disparity in performance between them and their open-source competitors.

# 9 CONCLUSION

In this paper, we have performed thorough analyses of recent LLMs (mostly LLMs for code) in testing programs/code. Through comprehensive experiments with 11 LLMs on programming benchmark datasets including HumanEval+ and MBPP (the sanitized version), we have uncovered a range of intriguing characteristics of these LLMs for program/code testing. We have illustrated how the program testing capabilities of these LLMs can be enhanced in comparing intensive empirical results in four different settings. Based on our findings, we are also capable of improving the performance of state-of-the-art LLMs in synthesizing programs/code with test cases of higher quality. As a preliminary research work, we believe our paper can provide new research insights and spark new ideas in program/code synthesis, test-case generation, and LLM understanding, and we look forward to future exploration in this direction in future work.

# REFERENCES

Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki, Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan Dey, et al. Santacoder: don't reach for the stars! arXiv preprint arXiv:2301.03988, 2023.  
Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma, Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry, Quoc Le, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.  
Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin, Jian-Guang Lou, and Weizhu Chen. Codet: Code generation with generated tests. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=ktrw68Cmu9c.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805, 2018.  
Andrea Fioraldi, Dominik Maier, Heiko Eißfeldt, and Marc Heuse.  $\{\mathrm{AFL}++\}$ : Combining incremental steps of fuzzing research. In 14th USENIX Workshop on Offensive Technologies (WOOT 20), 2020.  
Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Scott Yih, Luke Zettlemoyer, and Mike Lewis. Incoder: A generative model for code infilling and synthesis. In The Eleventh International Conference on Learning Representations, 2023. URL https://openreview.net/forum?id=hQwb-lbM6EL.  
Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al. The pile: An 800gb dataset of diverse text for language modeling. arXiv preprint arXiv:2101.00027, 2020.  
Denis Kocetkov, Raymond Li, Loubna Ben allal, Jia LI, Chenghao Mou, Yacine Jernite, Margaret Mitchell, Carlos Muñoz Ferrandis, Sean Hughes, Thomas Wolf, Dzmitry Bahdanau, Leandro Von Werra, and Harm de Vries. The stack: 3 TB of permissively licensed source code. Transactions on Machine Learning Research, 2023. ISSN 2835-8856. URL https://openreview.net/forum?id=pxpbTdUEpD.  
Mike Lewis, Yinhan Liu, Naman Goyal, Marjan Ghazvininejad, Abdelrahman Mohamed, Omer Levy, Ves Stoyanov, and Luke Zettlemoyer. Bart: Denoising sequence-to-sequence pretraining for natural language generation, translation, and comprehension. arXiv preprint arXiv:1910.13461, 2019.  
Jia Li, Yunfei Zhao, Yongmin Li, Ge Li, and Zhi Jin. Towards enhancing in-context learning for code generation. arXiv preprint arXiv:2303.17780, 2023a.

Raymond Li, Loubna Ben Allal, Yangtian Zi, Niklas Muennighoff, Denis Kocetkov, Chenghao Mou, Marc Marone, Christopher Akiki, Jia Li, Jenny Chim, et al. Starcoder: may the source be with you! arXiv preprint arXiv:2305.06161, 2023b.  
Chin-Yew Lin. Rouge: A package for automatic evaluation of summaries. In Text summarization branches out, pp. 74-81, 2004.  
Jiawei Liu, Chunqiu Steven Xia, Yuyao Wang, and Lingming Zhang. Is your code generated by chatgpt really correct? rigorous evaluation of large language models for code generation. arXiv preprint arXiv:2305.01210, 2023a.  
Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. Agentbench: Evaluating llms as agents, 2023b.  
Ziyang Luo, Can Xu, Pu Zhao, Qingfeng Sun, Xiubo Geng, Wenxiang Hu, Chongyang Tao, Jing Ma, Qingwei Lin, and Daxin Jiang. Wizardcoder: Empowering code large language models with evol-instruct. arXiv preprint arXiv:2306.08568, 2023.  
Erik Nijkamp, Hiroaki Hayashi, Caiming Xiong, Silvio Savarese, and Yingbo Zhou. Codegen2: Lessons for training llms on programming and natural languages. arXiv preprint arXiv:2305.02309, 2023a.  
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. Codegen: An open large language model for code with multi-turn program synthesis, 2023b.  
Kishore Papineni, Salim Roukos, Todd Ward, and Wei-Jing Zhu. Bleu: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311-318, 2002.  
Alec Radford, Karthik Narasimhan, Tim Salimans, Ilya Sutskever, et al. Improving language understanding by generative pre-training. 2018.  
Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. The Journal of Machine Learning Research, 21(1):5485-5551, 2020.  
Freda Shi, Daniel Fried, Marjan Ghazvininejad, Luke Zettlemoyer, and Sida I Wang. Natural language to code translation with execution. arXiv preprint arXiv:2204.11454, 2022.  
Michele Tufano, Dawn Drain, Alexey Svyatkovskiy, Shao Kun Deng, and Neel Sundaresan. Unit test case generation with transformers and focal context. arXiv preprint arXiv:2009.05617, 2020.  
Michele Tufano, Shao Kun Deng, Neel Sundaresan, and Alexey Svyatkovskiy. Methods2test: A dataset of focal methods mapped to test cases. In Proceedings of the 19th International Conference on Mining Software Repositories, pp. 299-303, 2022.  
Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.  
Boxin Wang, Weixin Chen, Hengzhi Pei, Chulin Xie, Mintong Kang, Chenhui Zhang, Chejian Xu, Zidi Xiong, Ritik Dutta, Ryan Schaeffer, Sang T. Truong, Simran Arora, Mantas Mazeika, Dan Hendrycks, Zinan Lin, Yu Cheng, Sanmi Koyejo, Dawn Song, and Bo Li. Decodingtrust: A comprehensive assessment of trustworthiness in gpt models, 2023a.  
Yue Wang, Hung Le, Akhilesh Deepak Gotmare, Nghi DQ Bui, Junnan Li, and Steven CH Hoi. Codet5+: Open code large language models for code understanding and generation. arXiv preprint arXiv:2305.07922, 2023b.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al. Chain-of-thought prompting elicits reasoning in large language models. Advances in Neural Information Processing Systems, 35:24824-24837, 2022.  
Chunqiu Steven Xia, Matteo Paltenghi, Jia Le Tian, Michael Pradel, and Lingming Zhang. Universal fuzzing via large language models. arXiv preprint arXiv:2308.04748, 2023.  
Can Xu, Qingfeng Sun, Kai Zheng, Xiubo Geng, Pu Zhao, Jiazhan Feng, Chongyang Tao, and Daxin Jiang. Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244, 2023.  
Qinkai Zheng, Xiao Xia, Xu Zou, Yuxiao Dong, Shan Wang, Yufei Xue, Zihan Wang, Lei Shen, Andi Wang, Yang Li, et al. Codegeex: A pre-trained model for code generation with multilingual evaluations on humaneval-x. arXiv preprint arXiv:2303.17568, 2023.  
Kaijie Zhu, Jindong Wang, Jiaheng Zhou, Zichen Wang, Hao Chen, Yidong Wang, Linyi Yang, Wei Ye, Neil Zhenqiang Gong, Yue Zhang, and Xing Xie. Promptbench: Towards evaluating the robustness of large language models on adversarial prompts, 2023.

# A APPENDIX

# A.1 FURTHER ANALYSIS OF EXPERIMENTAL RESULTS

In this part, we provide further analysis of the experimental results in Section 6.

With regard to the situation where the test case quality generated by SantaCoder is lower than that generated by CodeT5+ on the HumanEval+ dataset, we have explained that this is probably because SantaCoder tends to generate longer and more complex test cases. Here we further demonstrate that SantaCoder is capable to generate more accuracy output when given the same testing input as that of CodeT5+.s. To show this, we first extract the input part of the test cases (which includes testing inputs paired with their corresponding outputs) generated by CodeT5+ in the oracle setting. We then let SantaCoder to generate testing outputs given these inputs, and assessed the accuracy of such test cases. The results show that, given these testing inputs already, SantaCoder and CodeT5+ obtain an correctness of  $41.67\%$  and  $40.34\%$ , respectively, showing that SantaCoder is indeed stronger, if the same testing input is given and it does not have the chance to yield more complex testing inputs.

# Footnotes:

Page 0: *Corresponding author 
Page 1: 1 https://pytest.org 
