# Unsupervised Evaluation of Code LLMs with Round-Trip Correctness

Miltiadis Allamanis  $^{*1}$  Sheena Panthaplackel  $^{*1}$  Pengcheng Yin  $^{*1}$

# Abstract

To evaluate large language models of code, research has relied on a few small manually curated benchmarks, such as HumanEval and MBPP, which represent a narrow part of the real-world software domains. In this work, we introduce round-trip correctness (RTC) as an alternative evaluation method. RTC allows Code LLM evaluation on a broader spectrum of real-world software domains without the need for costly human curation. RTC rests on the idea that we can ask a model to make a prediction (e.g., describe some code using natural language), feed that prediction back (e.g., synthesize code from the predicted description), and check if this round-trip leads to code that is semantically equivalent to the original input. We show how to employ RTC to evaluate code synthesis and editing. We find that RTC strongly correlates with model performance on existing narrow-domain code synthesis benchmarks while allowing us to expand to a much broader set of domains and tasks which was not previously possible without costly human annotations.

# 1. Introduction

While large language models (LLMs) have shown exceptional abilities in a wide range of tasks, their evaluation remains costly, commonly requiring laborious human-curated datasets. This is particularly true for code capabilities of LLMs that commonly require highly-skilled programmers to create evaluation benchmarks. Existing benchmarks, such as HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), ARCADE (Yin et al., 2022), and DS-1000 (Lai et al., 2023) have been developed by asking human annotators to provide natural language, code, tests, and sometimes the target problems themselves.

At the same time, established human-annotated evaluation

*Equal contribution 'Google DeepMind. Correspondence to: Miltiadis Allamanis <mallamanis@google.com>.

Proceedings of the  $41^{st}$  International Conference on Machine Learning, Vienna, Austria. PMLR 235, 2024. Copyright 2024 by the author(s).

benchmarks focus on narrow domains of code: HumanEval and MBPP evaluate introductory-level, standalone programming tasks; ARCADE and DS-1000 focus on simple data science tasks using few popular open-source libraries (e.g., numpy, pandas). These benchmarks are "necessary" but not "sufficient" for good performance for Code LLM users that work on different development environments which feature a much broader spectrum of domains, programming libraries and frameworks. However, manually creating new benchmarks with expanded scope is costly and impractical given the general-purpose nature of code and fast-paced software evolution.

Towards ameliorating these limitations, we use the concept of round-trip correctness (RTC) which allows us to perform unsupervised evaluation over certain LLM capabilities, complementing existing human-annotated evaluations. RTC relies on the idea that we can use an LLM to perform both an action (e.g., "Describe these lines of code") and its inverse ("Implement the code given this description"). See Fig. 1 for an example. We can then judge whether the round-trip has retained the semantics of the input. This can be achieved via computing various discrete (e.g., exact match) or continuous metrics (e.g., BLEU) or involve execution-based oracles (e.g., unit tests) which commonly already exist and require no additional human involvement. Our contributions are:

- We propose an unsupervised method for evaluating LLMs via round-trip correctness (RTC) (Sec. 2) and instantiate it for code synthesis and editing (Sec. 3).  
- We show that RTC strongly correlates with existing metrics on narrow-domain benchmarks (HumanEval and ARCADE) measuring the same LLM capability within that narrow domain (Sec. 4.1).  
- We show that RTC allows us to measure an LLM's performance over a wide-range of real-life software domains — without human-provided annotations — and complements existing narrow-domain benchmarks (Sec. 4.2).  
- We demonstrate that RTC is a general metric that can be used for other tasks, like code editing, for which there are no well-established benchmarks (Sec. 3.2).

Our code can be found at https://github.com/google-deepmind/icml2024-roundtrip-correctness.

Figure 1. Round-trip correctness (RTC) for Code Synthesis: An LLM is asked to describe the highlighted code (left) within the context of the file. Subsequently, it is asked to implement the relevant code within the code context given the description it previously generated (right).

# 2. Round-Trip Correctness

Background We draw inspiration from a software testing technique known as property-based testing (Fink & Bishop, 1997). It allows defining properties that must hold between inputs and outputs of a program (e.g., all items in the input list must also appear in the output list). Round-trip correctness is one such property (e.g., compressing and subsequently decompressing data must yield the original data). Property-based testing allows software developers to expand their tests beyond few human-curated examples. In this work, we re-purpose this concept for LLM evaluation.

RTC for Model Evaluation Consider two forms of data  $\mathbb{X}$  and  $\mathbb{Y}$ , such as source code and natural language descriptions of code (Fig. 1) and two (probabilistic) models whose task is to "translate" from one form of data to the other, i.e., a forward model  $M: \mathbb{X} \to \mathbb{Y}$  and a backward model  $M^{-1}: \mathbb{Y} \to \mathbb{X}$ . These models could be a single LLM prompted differently.

The central idea for unsupervised evaluation is the concept of round-trip correctness (RTC). Intuitively, for a "good" forward and backward model we expect  $\hat{x} = M^{-1}(M(x))$  to be semantically equivalent to  $x$ . For example, as we discuss in Sec. 3, we can describe code with natural language in the forward pass and then generate back the code from the sampled natural language descriptions in the backward pass (Fig. 1). To compute RTC we need some function  $\sin(x, \hat{x})$  that estimates the semantic equivalence between the original  $x$  and each predicted sample  $\hat{x}$ . Such functions may include discrete or continuous metrics such as exact match, CodeBLEU (Ren et al., 2020), or CodeBERTScore (Zhou et al., 2023), and execution-based semantic equivalence oracles, such as unit test execution.

We can then measure round-trip correctness as the ability of

$M$  and  $M^{-1}$  to accurately perform the round-trip from an  $x \in X$  to a  $y = M(x) \in \mathbb{Y}$  and back to  $\hat{x} = M^{-1}(y) \in X$ ,

$$
\operatorname {R T C} _ {\operatorname {s i m}} (x) \triangleq E _ {y \sim M (x)} \left[ E _ {\hat {x} \sim M ^ {- 1} (y)} [ \operatorname {s i m} (\hat {x}, x) ] \right], \tag {1}
$$

where  $\mathrm{sim}(\cdot)$  estimates the semantic equivalence of  $x$  and  $\hat{x}$ . Since we cannot compute the expectations exactly, we draw small number of  $N_{f}$  forward and  $N_{b}$  backward samples and approximate RTC as

$$
\mathsf {R T C} _ {\mathrm {s i m}} (x) \approx \frac {1}{N _ {f} N _ {b}} \sum_ {y \sim M (x)} \sum_ {\hat {x} \sim M ^ {- 1} (y)} \mathrm {s i m} (\hat {x}, x).
$$

Measuring the forward lift In certain situations, it is possible for the backward model  $M^{-1}$  to perform its task without any input from the forward model or with very low-quality forward model samples. For instance, in text-to-code synthesis (Fig. 1), the code may be obvious within the code context, without requiring any explicit natural language instruction. To measure this, we employ the notion of the forward lift. Namely, we replace the forward sample with an uninformative/generic utterance  $\epsilon$  and measure it as

$$
L _ {M} ^ {\mathrm {s i m}} (x) = \mathsf {R T C} _ {\mathrm {s i m}} (x) - E _ {\hat {x} \sim M ^ {- 1} (\epsilon)} [ \mathrm {s i m} (\hat {x}, x) ].
$$

$L_{M}^{\mathrm{sim}}$  can also be approximated by sampling. Thus,  $L_{M}^{\mathrm{sim}}$  acts as a measure of the additional information encoded in the forward samples  $\{y\}$  that is available to the backward model  $M^{-1}$  and can serve as a weak measure of a model's ability to perform the forward task. A negative  $L_{M}^{\mathrm{sim}}$  may imply that the forward model  $M$  makes "confusing" predictions, distracting the backward model from performing its task. A lift larger than zero implies that  $M$  provides helpful predictions that contribute towards the backward generation. Finally a  $L_{M}^{\mathrm{sim}}$  close to zero may imply that the forward model  $M$  yields uninformative predictions or the input samples  $x$  are easy for the backward model and  $M$  cannot possibly provide any additional information.

Limitations While RTC allows us to evaluate Code LLMs without human annotations, it is not without limitations. First, the quality of RTC as a measure depends on that of the similarity function  $\mathrm{sim}(\cdot)$ . A weak measure of semantic similarity may yield arbitrary results. Second, the measurement of the performance of the forward and backward tasks is coupled: if  $M$  is unable to provide plausible samples, we cannot expect to measure the ability of the backward model  $M^{-1}$ . This may be a problem if we care for only one of the forward or backward tasks.

Finally, RTC assumes "reasonably" trained and instruction-tuned LLMs. In an adversarial setting, a forward model  $M$  can ignore the instruction and recite its input  $x$ . Then, the backward model  $M^{-1}$  can copy the output of the forward model  $M$  achieving perfect RTC. While this is unlikely

to happen for models that employ common (pre)training methods, if we train/fine-tune models with the objective of Eq. 1 such a behavior can arise naturally.

# 3. RTC for Code

While RTC is general, it is well-suited for code where automatically judging the semantic similarity (via  $\mathrm{sim}(\cdot)$ ) is easier than natural language, commonly through proxies such as unit tests. In this section, we discuss two code-specific evaluations for two common code application: code synthesis and code editing. A summary is shown in Table 1.

# 3.1. Round-trip Code Synthesis (SYNTHESISRTC)

Code synthesis with LLMs is one of the most studied tasks. Evaluating on this task commonly requires a human-annotated dataset of natural language descriptions, function signatures (or full implementations), and unit tests. Instead, for RTC we construct SYNTHESISRTC as an in-context code synthesis task (Fig. 1) that does not require input natural language descriptions. Given a coherent region of code within a source code file, we ask a forward LLM  $(M)$  to describe the code region concisely with natural language. This yields a natural language utterance  $y$ . Then, we remove the code region and replace it with a TODO comment containing the natural language description generated by  $M$ . Finally, we ask  $M^{-1}$  to synthesize code implementing the TODO.

For a model to be round-trip correct, the synthesized code must be semantically equivalent to the original. In practice, proving semantic equivalence is hard, and we employ unit tests as a proxy and measure  $\mathsf{RTC}_{\mathrm{pass}}$ . Note that unit tests are commonly included in well-developed code, and so they are often readily available without any additional human effort. If they do not exist, automatic test generation methods (Nebut et al., 2006) can be employed, but it is also possible to consider other weaker similarity functions like exact match or BLEU, as done in prior work. Finally, to measure the lift for SYNTHESISRTC we can use an uninformative TODO comment (e.g.,  $\epsilon =$  "TODO: Implement.") and check the correctness with  $\sin(\cdot)$ . The lift  $L_{M}^{\mathrm{pass}}$  can be attributed to the additional information provided by the forward model  $M$ .

# 3.2. Round-trip Code Editing (EDITINGRTC)

In addition to SYNTHESISRTC, which evaluates an LLM's ability to generate new code from scratch, we also explore RTC as an evaluation metric for code editing. Editing is a common real-world software engineering scenario in which code is modified (e.g., bug fixing, refactoring, implementing new or updated functionality). Performing this task requires complex reasoning, which entails identifying parts of the existing code that need to be changed, determining how

they should be changed, and retaining all other parts of the code (Jimenez et al., 2023). However, this is not a common evaluation task for LLMs, and there are no well-established metrics or benchmarks.

Our approach closely follows Sec. 3.1. Namely, as the input to the forward model  $M$  we provide an edit, represented as an old code snippet followed by the new version (with the edit applied). We ask the model to describe the edit concisely using natural language, which results in a predicted edit description  $y$ . Finally, we provide the old code snippet (before the edit) and the generated edit description to the backward model  $M^{-1}$ , and we ask it to generate the new code, i.e., apply the edit described in  $y$ .

Rather than relying on human-provided natural language labels to perform supervised evaluation, we leverage RTC, in which we compute the similarity between the original edit that is provided as input to  $M$  and the edit predicted by  $M^{-1}$ . Similar to Li et al. (2022b), we use exact match as the similarity metric  $\mathrm{sim}(\cdot)$ . To approximately measure the quality of the generated edit descriptions we compute the lift for a baseline task in which we provide  $M^{-1}$  an uninformative edit description ( $\epsilon = \text{"Edit"}$ ).

# 4. Evaluation

Experimental Setup Unless stated otherwise, to compute RTC we draw 3 forward samples and one backward sample per forward sample. We use temperature of 0.8 for the forward model (to allow for disparate forward samples) and 0.1 for the backward samples (to generate high-probability code generations). We use three-shot prompting with identical few-shot prompts for all models. Given the time constraints and compute limitations, we selected these hyperparameters and have not explored any variations. Finally, we limit the length of the forward samples to 128 characters. In this way, models are forced to "compress" their understanding in a succinct natural language sentence and more verbose models cannot gain an advantage over less verbose ones.

We note that different forward and backward models could be used. However, there are confounders in doing so that we wanted to avoid: different forward models may generate natural language descriptions that are confusing to different backward models. Such a communication chasm would misrepresent the capabilities of these models. For instance, consider two models  $M_{1}$  and  $M_{2}$  that are both good at code synthesis based on natural language instructions. However, suppose  $M_{1}$  performs poorly in code-to-natural language generation whereas  $M_{2}$  is very good. Now, if we use  $M_{1}$  to create forward samples for the SYNTHESISRTC task, we may end up with low-quality natural language descriptions of code. If we then provide these low-quality descriptions

Table 1. Summary of Code RTC tasks discussed in this work.  

<table><tr><td></td><td>SYNTHESISRTC</td><td>EDITINGRTC</td></tr><tr><td>Input X</td><td>Source Code Region (one or more statements)</td><td>Code Edit</td></tr><tr><td>Intermediate Y</td><td>Natural Language Description of Code Region</td><td>Natural Language Description of Edit</td></tr><tr><td>Forward Task Context</td><td>Source Code around Target Region</td><td>Source Code around Edit</td></tr><tr><td>Backwards Task Context</td><td>Source Code around Target Region</td><td>Original Version Code before Edit</td></tr><tr><td>Uninformative utterance ε</td><td>Code Completion/Infilling of Target Region</td><td>-</td></tr><tr><td>sim(·)</td><td>All Unit Test Pass (0/1)</td><td>Exact Match</td></tr></table>

as input to  $M_2$  to generate backward samples, we may also end up with low-quality code samples. This would lead us to falsely conclude that  $M_2$  performs poorly on code synthesis. Due to these, we aim to minimize any extrinsic factors that may affect the result and use the same model for both the forward and backward samples to compute RTC. At the same time, using the same model for both the forward and backward tasks can also enable RTC to measure a model's skill at generating both code as well as natural language targets. This would help distinguish more "well-rounded" LLMs that excel at both code and description generation from models that can only handle a single task (e.g., models trained on code-heavy data may only predict code, see Muennighoff et al. (2023)).

# 4.1. Does RTC correlate with existing metrics on narrow-domain benchmarks?

We make the hypothesis that RTC, when applied to existing benchmarks, is highly correlated with widely accepted metrics, such as  $\text{pass} @ k$  (Chen et al., 2021) which in turn is recognized as a metric strongly correlating with how LLM users perceive its performance.

To test this hypothesis we employ HumanEval (Chen et al., 2021) and ARCADE (Yin et al., 2022) that represent two common domains: general-purpose coding to solve algorithmic problems and multi-turn data science programming in Jupyter notebooks. HumanEval is a function-level code generation dataset where each problem is a Python function with a natural language description in its docstring. We re-purpose each problem by removing the docstring and use the ground-truth function solution as input to the forward model. LLMs are subsequently asked to describe the body of the ground-truth function (forward) and then implement it based on the generated description. For ARCADE, each original problem comes with a natural language question (e.g., "What's the GDP growth rate of that country?") and a ground-truth code solution along with any notebook context (e.g., the question and solution for a prior turn "Which country received the highest aid?"). To compute RTC, we use the notebook context and the code solution of a turn as input to generate question descriptions  $y$  in the forward pass. The backward model is subsequently asked to implement

Table 2.  $\mathsf{RTC}_{\mathsf{pass}}$  vs standard pass@1 metric and  $L_{M}^{\mathrm{pass}}$  across models. Results sorted by  $\mathsf{RTC}_{\mathsf{pass}}$ . DSC stands for DeepSeekCoder.  

<table><tr><td></td><td>pass@1</td><td>\( RTC_{pass} \)</td><td>\( L_M^{pass} \)</td></tr><tr><td colspan="4">HumanEval (Chen et al., 2021)</td></tr><tr><td>PaLM 2-S</td><td>19.5%</td><td>8.3%</td><td>-0.2%</td></tr><tr><td>PaLM 2-S+</td><td>29.3%</td><td>10.6%</td><td>3.3%</td></tr><tr><td>PaLM 2-S*</td><td>37.6%</td><td>18.3%</td><td>9.8%</td></tr><tr><td>Gemini Nano 2</td><td>33.4%</td><td>18.9%</td><td>3.7%</td></tr><tr><td>StarCoder2 15B</td><td>46.3%</td><td>31.7%</td><td>20.7%</td></tr><tr><td>Gemini v1 Pro</td><td>63.4%</td><td>34.8%</td><td>19.6%</td></tr><tr><td>DSC33B-IT</td><td>75.6%</td><td>40.2%</td><td>30.4%</td></tr><tr><td colspan="4">ARCADE (Yin et al., 2022)</td></tr><tr><td>PaLM 2-S</td><td>5.5%</td><td>2.7%</td><td>-</td></tr><tr><td>PaLM 2-S+</td><td>8.2%</td><td>3.5%</td><td>-</td></tr><tr><td>PaLM 2-S*</td><td>15.3%</td><td>6.5%</td><td>-</td></tr><tr><td>Gemini Nano 2</td><td>14.4%</td><td>7.7%</td><td>-</td></tr><tr><td>Gemini v1 Pro</td><td>18.3%</td><td>11.1%</td><td>-</td></tr><tr><td>StarCoder2 15B</td><td>25.4%</td><td>12.1%</td><td>-</td></tr><tr><td>DSC33B-IT</td><td>24.8%</td><td>15.1%</td><td>-</td></tr></table>

the code using the generated description  $y$  and the notebook context. The few-shot prompts of Yin et al. (2022) are used.

Table 2 show the scores achieved by two classes of models PaLM 2 (Anil et al., 2023) and Gemini (Team Gemini et al., 2023) along with two open models DeepSeekCoder-33B-Instruct (Guo et al., 2024) (abbreviated as as DSC33B-IT) and StarCoder2 (Lozhkov et al., 2024). The HumanEval pass@1 scores for Gemini, StarCoder2, and DeepSeekCoder models were obtained from the base scores of the independent EvalPlus Leaderboard. While not surprising, we see that there is a strong correlation between the "standard" pass@1 and  $\mathsf{RTC}_{\mathrm{pass}}$  for both benchmarks. The Pearson correlation is  $r = 0.96$  for HumanEval and  $r = 0.96$  for ARCADE and the Spearman correlation is  $\rho = 0.90$  and  $\rho = 0.81$  respectively.

These results suggest that RTC strongly correlates with existing metrics on benchmarks that have been curated through costly human annotations. Thanks to the strong correlation, we can conclude that RTC is a valid metric that reflects the real-world performance of LLMs and thus can complement existing human-annotated benchmarks. It should be noted

that these correlations are not perfect. This is to be expected as  $\mathsf{RTC}_{\mathrm{pass}}$  couples the code-to-text and the text-to-code synthesis capabilities whereas pass@1 only measures the text-to-code synthesis capability of each model.

One potential caveat could be that RTC's correlation with pass@1 is only due to the different sizes of the models, as it is generally true that larger models overperform smaller ones. However, all the PaLM 2 models used have the same number of parameters and RTC and pass@1 still correlates strongly across these models. This suggests that RTC correlates with the coding abilities controlling for model size rather than some other capability that coincidentally improves with model size.

Note that the absolute  $\mathsf{RTC}_{\mathrm{pass}}$  is worse than the pass@1 for HumanEval in absolute terms. This difference can be explained by two factors: (a) the standard HumanEval prompt includes input-output examples which the forward model is not prompted to generate. This makes the backward code generation task harder. Indeed, recently Liu et al. (2024) confirmed that removing those input-output examples from the original HumanEval prompts leads to significant drop in pass@1 (Liu et al., 2024, Table 3). (b) The forward description generation task also introduces some noise.

Evaluating Code-to-Description Table 2 shows the lift  $L_{M}^{\mathrm{pass}}$  for HumanEval. For SYNTHESISRTC the uninformative backward task is code completion within the code context but no (natural language) information about the implementation or its intention. For HumanEval, Table 2 shows that all models — except from PaLM 2-S — show a lift, but better models offer a larger lift than small ones. For ARCADE, the baseline task (code completion without any natural language instruction) is not meaningful given the nature of the benchmark as the baseline backward pass@1 would be always zero and hence we do not compute it.

Sensitivity of RTC Similar to pass@k metrics LLM sampling temperature, number of samples per example, and dataset size affect the statistical robustness of the RTC reported. In this work, we chose to use low temperature for the backward model and a mildly high temperature for the forward model. The low temperature of the backward pass renders the  $RTC_{pass}$  relatively stable: we performed the HumanEval experiments (Table 2) 10 times and measure the standard deviation to be  $\sigma = 1.11\%$  which is small and shows that the results shown are fairly stable. Changing  $N_f$ ,  $N_b$  and the forward/backward sampling temperatures beyond the ones discussed changes the results. For example, we observe that increasing the temperature of the backward model requires a significant increase in  $N_b$  to reduce variance.

# 4.2. Do LLMs perform similarly across domains?

Previously, we showed that RTC strongly correlates with pass@1 on limited-domain, human-annotated benchmarks. It is unclear whether these narrow-domain benchmarks fully reflect an LLM's capabilities across all code domains. To investigate this question, we collect a set of 77 of permissively licensed open-source Python projects that have a test suite which we can execute and all the tests pass. We then sample ranges within the code by sampling one or more consecutive sequential statements from the project's code whose size is between 32 and 384 characters long and are covered by the test suite. Specifically, each statement corresponds to a node on the concrete syntax tree (CST) of the code. The probability of a sequence of statements being sampled is proportional to the number of characters they contain and inversely proportional to the number of other candidate nodes in the CST that also contain those characters. Files containing unit tests are excluded. We also filter out any ranges that when deleted have no effect to the results of the test suite. This ensures that the selected code has at least some observable effect to the expected outputs. This last step is crucial and filters about  $40\%$  of the sampled ranges, although this statistic varies across projects.

Finally, we randomly sample 100 ranges per-project. If for a project we cannot collect 80 or more samples, we discard the entire project. The resulting dataset is made of 5,961 samples from 58 open source Python projects representing varying domains. To avoid giving advantage to models with longer contexts, we provide as context the few lines before and after each selected code range such that no more than 1024 characters of context are included and each line is either shown entirely or not at all. Alternative implementations, e.g., including the entire file, more files, or even an entire repository as in Ding et al. (2024) are also reasonable and we expect they will become increasingly viable in the future. The used projects are shown in Appendix A.

Subsequently, we compute  $\mathsf{RTC}_{\mathrm{pass}}$  as in Sec. 4.1. Fig. 2 shows a plot of the average  $\mathsf{RTC}_{\mathrm{pass}}$  for each project showing a wide variability on both Gemini Nano 2 and Gemini Pro's performance. This is not unexpected: HumanEval and ARCADE only test for two specific, fairly narrow domains that do not reflect the full diversity of domains in real-life software. From Fig. 2 we see that for some projects the performance is really good compared to the rest. In particular, the TheAlgorithms project that contains educational implementations of popular computer science algorithms achieves very good  $\mathsf{RTC}_{\mathrm{pass}}$ . In contrast, jedi, a static analysis library for Python has the lowest  $\mathsf{RTC}_{\mathrm{pass}}$ . There is also a wide variability among the  $\mathsf{RTC}_{\mathrm{pass}}$  of the other projects showing the diverse performance of LLMs across different domains. While the  $\mathsf{RTC}_{\mathrm{pass}}$  of the two models are correlated ( $r = 0.75$ , Spearman's  $\rho = 0.76$ ), this correlation is

Figure 2. Round-trip correctness (RTC) for Code Synthesis across 58 open-source projects of diverse domains for Gemini Pro and Nano 2: RTCpass varies widely across projects/domains, something that common code synthesis benchmarks fail to capture.

not perfect, showing that different LLMs can have varying performance characteristics in different domains.

These results suggest that existing narrow-domain benchmarks do not capture the LLM's capabilities across multiple domains. Therefore for Code LLMs that need to support a wide range of coding domains additional benchmarks are needed. While manually curating and annotating such benchmarks is possible, unsupervised evaluation through round-trip correctness offers a reasonable alternative.

Lift across domains The lift  $L_{M}^{\mathrm{pass}}$  shows a similar variation across different projects (shown in Appendix B in Fig. 5). We note that the average lift for Gemini Nano 2 is  $L_{M} = 7.0\%$  and for the Gemini Pro  $L_{M} = 21.5\%$  which is higher compared to HumanEval (Table 2). This may be because HumanEval's function signatures used as context (e.g., def is_palindrome(string: str) -> bool) in the backward model are somewhat informative and the forward samples may not be able to provide additional information. This further suggests that HumanEval may be a relatively simple benchmark compared to the one in this section.

Qualitative Analysis Fig. 3 shows a cherry-picked example from an open-source project that we have selected as a good representative of common error modes. The target code range (yellow box) performs a relatively simple operation that maintains the invariant on the counts dictionary. The first two natural language samples explain this in a plausible way. However, the first sample misses the crucial information about the outer if condition, which is well-captured by the second one. Overall, qualitatively we often see some important part of the logic not being captured in the natural language description. Additionally, the first sample takes a very formulaic and unnatural way of describing the code giving emphasis on the low-level op-

Table 3. Evaluating EDITINGRTC with a random sample of 1K examples from the CodeReviewer test set of Li et al. (2022b).  

<table><tr><td></td><td>Gemini Nano 2</td><td>Gemini Pro</td></tr><tr><td colspan="3">Unsupervised</td></tr><tr><td>\( RTC_{ExactMatch} \)(%)</td><td>5.2</td><td>12.9</td></tr><tr><td>\( L^{ExactMatch}_{M} \)(%)</td><td>4.8</td><td>12.4</td></tr><tr><td>\( RTC_{BLEU} \)(%)</td><td>72.3</td><td>73.7</td></tr><tr><td>\( RTC_{ROUGE} \)(%)</td><td>81.4</td><td>82.3</td></tr><tr><td colspan="3">Supervised</td></tr><tr><td>Edit → NL (BLEU)</td><td>0.5</td><td>1.0</td></tr><tr><td>NL → Edit (Exact Match)</td><td>11.7</td><td>19.4</td></tr></table>

erations rather than the underlying logic. This seems like a common error mode for the forward model and happens more often on shorter snippet. The third sample seems like a hallucination and is just wrong.

All backward samples are reasonable implementations of their respective forward sample. However, note that the second sample has the ambiguous phrase " discard the oldest element" which can reasonably imply the use of popleft() instead of just peeking at the leftmost element of window. Despite this difference and thanks to the implementation semantics of deque the second sample is semantically equivalent to the original code when placed within the code context, hence achieving an  $\mathsf{RTC}_{\mathrm{pass}} = 1 / 3$ .

# 4.3. Evaluating beyond Code Synthesis: Code Editing

To evaluate code editing, we use the CodeReviewer test set (Li et al., 2022b), which is a parallel corpus of GitHub Pull Request (PR) review comments paired with their edited code, including both the old and new versions (i.e., before and after the edit for resolving the review comment). We randomly sample 1K examples from the original test set.

# Input Code and Target Region

```python
def unique_in_window(iterable, n, key=None):
    '''Yield the items from *iterable* that haven't been seen recently. *n* is the size of the lookback window.''
    if n <= 0:
        raise ValueError('nmust_be_greater Than_0')
    window = deque(maxlen=n)
    counts = defaultdict(int)
    use_key = key is not None
```

```python
for item in iterable:  
    if len(window) == n:  
        to_discard = window[0]  
        if counts[to_discard] == 1:  
            del counts[to_discard]  
        else:  
            counts[to_discard] -= 1  
    k = key(item) if use_key else item  
    if k not in counts:  
        yield item  
    counts[k] += 1  
    window.append(k)
```

# Forward samples

1. 'to_discard' takes the first value in 'window'. If the count for that value in 'counts' is '1', we remove the value, otherwise we decrement the count.  
2. if the window is at capacity, discard the oldest element, and update counts so that this element is only considered if it is seen again.  
3. 'to_discard' equals 'window[(len(window) - 1) % len(window)]'

# Backward samples

```python
Generation for forward sample #1  
to_discard = window.popleft()  
if counts[to_discard] == 1:  
    del counts[to_discard]  
else:  
    counts[to_discard] -= 1
```

```python
Generation for forward sample #2  
if len(window) == n:  
    k = window.popleft()  
    counts[k] -= 1  
    if counts[k] == 0:  
        del counts[k]
```

```txt
Generation for forward sample #3
to_discard = window[(lenwindow) - 1) % len(window)]
```

Figure 3. SYNTHESISRTC example from more-itertools from Gemini Pro. Code slightly reformatted/abbreviated for space.

The resulting set contains samples from multiple programming languages (Java  $17\%$ , Go  $23\%$ , Python  $21\%$ , C++  $9\%$ , PHP  $7\%$ , JavaScript  $7\%$ , Ruby  $5\%$ , C#  $5\%$ , C  $4\%$ ). We use 3-shot prompting (with 3 examples from the CodeReviewer validation set) and conduct experiments with Gemini models (Team Gemini et al., 2023). We draw 3 forward (temperature=1.0) and 1 backward sample (temperature=0.0).

Results are shown in Table 3. Overall, we observe trends consistent to those of SYNTHESISRTC (Table 2):  $\mathsf{RTC}_{\mathsf{ExactMatch}}$  is higher for Gemini Pro, suggesting that Gemini Pro has superior capabilities in terms of generating natural language edit descriptions (forward task) and editing code based on natural language descriptions of edits

(backward task). Additionally, the forward lift  $(L_M^{\mathrm{ExactMatch}})$  is much higher for Gemini Pro, once again demonstrating the quality of the descriptions generated by  $M$ .

Note that  $\mathsf{RTC}_{\mathsf{ExactMatch}}$  and  $L_{M}^{\mathsf{ExactMatch}}$  are computed in a completely unsupervised manner, without relying on any ground truth labels for edit descriptions or edited code snippets. We consider this to be particularly useful for benchmarking tasks related to code editing, as it is extremely difficult to collect a high-quality labeled test set. In fact, the CodeReviewer dataset was extracted automatically, using rough heuristics to align natural language edit descriptions with corresponding code edits. Consequently, it is a noisy test set, with some inconsistencies between descriptions and edits. If we take the PR comment as the ground truth edit description, we can perform standard evaluation on edit description generation (i.e., given the old and new versions of the code snippet, sample an edit description with temperature=0.0) using standard text generation metrics like BLEU (Papineni et al., 2002). As shown in Table 3, these scores are very low for the two models and uninformative. Since in editing tasks much of the input is often preserved in the target output, a simple baseline which retains the input unchanged fares well with standard metrics like BLEU, ROUGE, and even embedding-based metrics like BERTScore (Zhang et al., 2019). For this reason, we consider exact match to be a more reliable metric, but in Table 3 we have included the other metrics for completeness.

One of the main reasons for this is that traditional supervised metrics for evaluating text generation rely on lexical overlap, against reference texts. Here, there is only one reference (i.e., the PR comment) and the lexical overlap can be fairly limited. For instance, the PR comment in Fig. 4 appears to be irrelevant to the edit and is thus of low quality. On the other hand, the three edit descriptions sampled from  $M$  are all arguably consistent with the edit. In fact, when they are fed into  $M^{-1}$ , they lead to a predicted edit which exactly matches the original, resulting to an  $\mathsf{RTC}_{\mathrm{ExactMatch}} = 100.0\%$ . However, the average BLEU score w.r.t. the PR comment is extremely low, at 1.837, which incorrectly suggests that the model fails to generate a high-quality edit description for this example.

Furthermore, we can perform supervised evaluation on code edit generation. That is, given the old version of the code snippet and the PR comment, we sample one edit with temperature  $= 0.0$  (greedy decoding). We can then compute exact match performance with respect to the ground truth code edit snippet in the CodeReviewer test set, as shown in Table 3. However, when the PR comment is misaligned with the labeled code edit, this evaluation will mischaracterize the code editing capabilities of an LLM. For the example in Fig. 4, the edit predicted based on the PR comment actually appears to be consistent with the comment. However, the

labeled ground truth edit is inconsistent with the PR comment, and so the fact that the predicted edit does not match the ground truth edit will falsely suggest that the model fails to generate a high-quality edit for this example. This demonstrates that even when weakly labeled test sets are available, noisy labels and other drawbacks of supervised metrics often make the evaluation unreliable. In such cases, we believe that RTC provides a more reliable evaluation. We show more EDITINGRTC examples in Appendix C.

# 5. Related Work

Supervised Code Evaluation Metrics Accuracy, or how often model-generated code exactly matches reference code, is a metric for evaluating code generation (Agashe et al., 2019; Li et al., 2022b; Yin & Neubig, 2017; Chakraborty & Ray, 2021). However, it is overly conservative and significantly underestimates the actual performance, as it is possible to generate semantically correct code without exactly matching a reference (e.g., different variable names, ordering of statements, or logic). Agashe et al. (2019); Li et al. (2022b); Yin & Neubig (2017); Wei et al. (2019) among others adopted BLEU (Papineni et al., 2002). However, Ren et al. (2020) showed that BLEU is not well-suited for evaluating code correctness since it fails to capture the syntax, semantics, and functionality of code. CodeBLEU (Ren et al., 2020) addresses this by augmenting BLEU with syntactic and data flow information. However, CodeBLEU requires the model-generated code to follow the same structure of the reference code, which is still overly conservative penalizing correct code that follows different ordering of statements or logic while still partly relying on BLEU's token overlap score. More recently, Zhou et al. (2023) proposed CodeBERTScore which measures the similarity of the model-generated code and the reference code based on the dot-product similarity of their contextualized vector representations from a pretrained LLM. However, CodeBERTScore does not explicitly evaluate semantic similarity.

To capture code's functional correctness, research has moved towards test-based oracles: generated code is executed against predefined test cases, and if the generated code passes these test cases, it is considered correct. This does not require the model-generated code to match the naming, structure, or logic of a reference, but checks if aspects of the functionality are correct. It should be noted that unit tests are partial indicators of functional correctness and cannot guarantee semantic equivalence which in the general case is undecidable.

Self-Consistency Self-consistency is often used to describe how consistent a model's generated samples are with one another. If a model generates consistent predictions multiple times for the same input, the model is likely more confident that those predictions are correct. Based on this

intuition, Wang et al. (2022) designed a decoding strategy using self-consistency to identify the most likely correct answer from a set of samples, which they showed improved the model's ability to do chain-of-thought reasoning. While self-consistency can be used as an uncertainty estimator, it is not always well-suited to evaluate accuracy, as a model can be consistently wrong. In contrast, RTC can be a more reliable metric since it requires functional correctness which is a stronger signal than consistency.

RTC also relates to IdentityChain of Min et al. (2023) who propose to measure the self-consistency of a Code LLM via multiple efforts to make a round-trip. In contrast to RTC, IdentityChain still requires an annotated human corpus and Min et al. (2023) argue that IdentityChain measures a distinct quality from conventional accuracy. Instead, we have shown that RTC is strongly correlated with conventional accuracy for a given benchmark, does not require human annotated examples, and covers multiple tasks.

Back Translation At a high-level, the forward and backward samples in RTC resemble back translation – a data augmentation technique for machine translation (Sennrich et al., 2016; Edunov et al., 2018; Sugiyama & Yoshinaga, 2019). Back translation commonly does not enforce semantic equivalence and may result in noisy data, when back translation is used to perform data augmentation at scale, it can still be very useful for training since models are robust to some level of noise. Instead our focus is model evaluation.

Code Synthesis Benchmarks The most common benchmarks for code synthesis include HumanEval (Chen et al., 2021), MBPP (Austin et al., 2021), APPS (Hendrycks et al., 2021), and CodeContests (Li et al., 2022a). These benchmarks have a curated set of input (natural language) specifications and test cases for each example, and this is nontrivial to expand. In contrast to these benchmarks, RTC can fairly easily be expanded to include new domains (Sec. 4.2).

Finally, the HumanEvalExplain task of HumanEvalPack (Muennighoff et al., 2023) is a special case of SYNTHESISRTC without forward/backward sampling for HumanEval. In contrast to that benchmark, we acknowledge that tasks like HumanEvalExplain evaluate both synthesis and code description, make them more robust through sampling, and introduce new tasks. Furthermore, we show that SYNTHESISRTC tasks do actually correlate with widely accepted metrics and thus are worth measuring.

Faithfulness Atanasova et al. (2023) proposed evaluating the faithfulness of natural language explanations based on a model's ability to generate the same output if the explanation is included in the input. The consistency between the outputs are evaluated, since a faithful explanation should guide the model in generating the same output. However, the output does not necessarily have to be correct, and the

```txt
Describe concisely and accurately with natural language the differences between the old and new code shown below.   
[old] def remote_shutdown(self): udp_sock  $=$  socket(socket(AF_INET,socket.SOCK_DGRAM) udp_sock.sendto(b"Shutdown","localhost",self.management_port)) self.log.debug("Shutdown command sent") time.sleep(10) udp_sock.sendto(b"StopTestNow","localhost",self.management_port)) self.log.debug("StopTestNow command sent")   
[new] def remote_shutdown(self): udp_sock  $=$  socket(socket(AF_INET,socket.SOCK_DGRAM) udp_sock.sendto(b"Shutdown","127.0.0.1",self.management_port)) self.log.debug("Shutdown command sent") time.sleep(10) udp_sock.sendto(b"StopTestNow","127.0.0.1",self.management_port)) self.log.debug("StopTestNow command sent") [edit description]   
Description Edit  $\rightarrow$  NL (BLEU) NL  $\rightarrow$  Edit (Exact Match)   
PR Comment: After first command, check if JMeter has finished - X   
[ new ] def remote_shutdown(self): udp_sock  $=$  socket(socket(AF_INET,socket.SOCK_DGRAM) udp_sock.sendto(b"Shutdown","localhost",self.management_port)) self.log.debug("Shutdown command sent") time.sleep(10) udp_sock.sendto(b"StopTestNow","localhost",self.management_port)) self.log.debug("StopTestNow command sent") If self.is_jmeter_finished(): self.log.debug("JMeter has finished") else: self.log.debug("JMeter has not finished")   
Baseline: Edit. - X   
[ new ] def remote_shutdown(self): udp_sock  $=$  socket(socket(AF_INET,socket.SOCK_DGRAM) udp_sock.sendto(b"Shutdown","localhost",self.management_port)) self.log.debug("Shutdown command sent") time.sleep(10) udp_sock.sendto(b"StopTestNow","localhost",self.management_port)) self.log.debug("StopTestNow command sent")   
EDITINGRTC Forward Samples: Replace localhost with 127.0.0.1 to avoid potential conflicts 1.314 1.837 100.0   
on a dual-stacked machine. Use the constant of 127.0.0.1 instead of "localhost" 2.098 2.098 2.098 2.098
```

Figure 4. A CodeReviewer example with Gemini Pro predictions: 3 sampled descriptions in the forward pass as well as their corresponding predicted edits in the backward pass. We additionally include predictions from the backward pass when the provided description is instead the PR comment or baseline description. Examples have minor edits/re-format due to space constraints.

explanation does not have to be descriptive. The main requirement is that the explanation does not include any information that might lead the model to generate a different output. In contrast, RTC evaluates the correctness between the input and the prediction from the backward pass, and necessitates good predictions in both directions.

# 6. Discussion & Conclusions

In this work we used the concept of round-trip correctness for model evaluation and found that RTC strongly correlates with performance on narrow-domain human-curated benchmarks, measuring a similar quality of a model's performance. RTC allows us to complement existing narrow-domain human-annotated benchmarks and measure

an LLM's performance on a much wider spectrum of domains. RTC allows us to expand our evaluations into new tasks such as code and edit description, and code editing without requiring human annotations.

We hope that this work motivates the community towards expanding the code evaluation benchmarks beyond narrow-domain ones. RTC seems to strike a balance between correlating with existing metrics and allowing to expand to new domains and tasks. Thus we recommend to complement existing benchmarks with RTC using a strong  $\mathrm{sim}(\cdot)$  to expand the breadth of the evaluated domains and tasks. We note that RTC should be complemented with good qualitative understanding of the LLM's error modes and suggestions rather than used blindly as a metric to be maximized.

# Impact Statement

This paper presents work whose goal is to advance the evaluation of LLMs. Having accurate LLM evaluations can help produce more useful models and reduce the (environmental) cost of their development. The deployment of LLMs has many potential societal consequences, none which we feel must be specifically highlighted here. However, it should be noted that optimizing for any single (evaluation) metric without considering the broader setting — including any ethical concerns — in which a machine learning model will be deployed should be avoided: No single metric, including those presented in this work, can capture the wide range of effects — positive or negative — that the deployment of a model may have and hence we encourage users of RTC to always take a holistic approach when evaluating their models.

# Acknowledgements

We would like to thank Nate Kushner, Charles Sutton, and Danny Tarlow for useful discussions. We would also like to thank the anonymous reviewers for their constructive feedback.

# References

Agashe, R., Iyer, S., and Zettlemoyer, L. JuICe: A large scale distantly supervised dataset for open domain context-based code generation. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pp. 5436-5446, 2019.  
Anil, R., Dai, A. M., First, O., Johnson, M., Lepikhin, D., Passos, A., Shakeri, S., Taropa, E., Bailey, P., Chen, Z., et al. PaLM 2 technical report. arXiv preprint arXiv:2305.10403, 2023.  
Atanasova, P., Camburu, O.-M., Lioma, C., Lukasiewicz, T., Simonsen, J. G., and Augenstein, I. Faithfulness tests for natural language explanations. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers), pp. 283-294. Association for Computational Linguistics, July 2023.  
Austin, J., Odena, A., Nye, M., Bosma, M., Michalewski, H., Dohan, D., Jiang, E., Cai, C., Terry, M., Le, Q., et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.  
Chakraborty, S. and Ray, B. On multi-modal learning of editing source code. In 2021 36th IEEE/ACM International Conference on Automated Software Engineering (ASE), pp. 443-455. IEEE, 2021.

Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.  
Ding, Y., Wang, Z., Ahmad, W., Ding, H., Tan, M., Jain, N., Ramanathan, M. K., Nallapati, R., Bhatia, P., Roth, D., et al. CrossCodeEval: A diverse and multilingual benchmark for cross-file code completion. Advances in Neural Information Processing Systems, 36, 2024.  
Edunov, S., Ott, M., Auli, M., and Grangier, D. Understanding back-translation at scale. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pp. 489-500, 2018.  
Fink, G. and Bishop, M. Property-based testing: a new approach to testing for assurance. ACM SIGSOFT Software Engineering Notes, 22(4):74-80, 1997.  
Guo, D., Zhu, Q., Yang, D., Xie, Z., Dong, K., Zhang, W., Chen, G., Bi, X., Wu, Y., Li, Y., et al. DeepSeekCoder: When the large language model meets programming - the rise of code intelligence. arXiv preprint arXiv:2401.14196, 2024.  
Hendrycks, D., Basart, S., Kadavath, S., Mazeika, M., Arora, A., Guo, E., Burns, C., Puranik, S., He, H., Song, D., et al. Measuring coding challenge competence with apps. arXiv preprint arXiv:2105.09938, 2021.  
Jimenez, C. E., Yang, J., Wettig, A., Yao, S., Pei, K., Press, O., and Narasimhan, K. SWE-bench: Can language models resolve real-world GitHub issues? arXiv preprint arXiv:2310.06770, 2023.  
Lai, Y., Li, C., Wang, Y., Zhang, T., Zhong, R., Zettlemoyer, L., Yih, W.-t., Fried, D., Wang, S., and Yu, T. DS-1000: A natural and reliable benchmark for data science code generation. In International Conference on Machine Learning, pp. 18319-18345. PMLR, 2023.  
Li, Y., Choi, D., Chung, J., Kushman, N., Schrittwieser, J., Leblond, R., Eccles, T., Keeling, J., Gimeno, F., Dal Lago, A., et al. Competition-level code generation with alphabet. Science, 378(6624):1092-1097, 2022a.  
Li, Z., Lu, S., Guo, D., Duan, N., Jannu, S., Jenks, G., Majumder, D., Green, J., Svyatkovskiy, A., Fu, S., et al. Automating code review activities by large-scale pretraining. In Proceedings of the 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, pp. 1035-1047, 2022b.  
Liu, C., Zhang, S. D., and Jabbarvand, R. CodeMind: A framework to challenge large language models for code reasoning. arXiv preprint arXiv:2402.09664, 2024.

Lozhkov, A., Li, R., Allal, L. B., Cassano, F., Lamy-Poirier, J., Tazi, N., Tang, A., Pykhtar, D., Liu, J., Wei, Y., Liu, T., Tian, M., Kocetkov, D., Zucker, A., Belkada, Y., Wang, Z., Liu, Q., Abulkhanov, D., Paul, I., Li, Z., Li, W.-D., Risdal, M., Li, J., Zhu, J., Zhuo, T. Y., Zheltonozhskii, E., Dade, N. O. O., Yu, W., Krauß, L., Jain, N., Su, Y., He, X., Dey, M., Abati, E., Chai, Y., Muennighoff, N., Tang, X., Oblokulov, M., Akiki, C., Marone, M., Mou, C., Mishra, M., Gu, A., Hui, B., Dao, T., Zebaze, A., Dehaene, O., Patry, N., Xu, C., McAuley, J., Hu, H., Scholak, T., Paquet, S., Robinson, J., Anderson, C. J., Chapados, N., Patwary, M., Tajbakhsh, N., Jernite, Y., Ferrandis, C. M., Zhang, L., Hughes, S., Wolf, T., Guha, A., von Werra, L., and de Vries, H. StarCoder 2 and The Stack v2: The next generation, 2024.  
Min, M. J., Ding, Y., Buratti, L., Pujar, S., Kaiser, G., Jana, S., and Ray, B. Beyond accuracy: Evaluating self-consistency of code large language models with IdentityChain. arXiv preprint arXiv:2310.14053, 2023.  
Muennighoff, N., Liu, Q., Zebaze, A., Zheng, Q., Hui, B., Zhuo, T. Y., Singh, S., Tang, X., Von Werra, L., and Long-pre, S. OctoPack: Instruction tuning code large language models. arXiv preprint arXiv:2308.07124, 2023.  
Nebut, C., Fleurey, F., Le Traon, Y., and Jezequel, J.-M. Automatic test generation: A use case driven approach. IEEE Transactions on Software Engineering, 32(3):140-155, 2006.  
Papineni, K., Roukos, S., Ward, T., and Zhu, W.-J. BLEU: a method for automatic evaluation of machine translation. In Proceedings of the 40th annual meeting of the Association for Computational Linguistics, pp. 311-318, 2002.  
Ren, S., Guo, D., Lu, S., Zhou, L., Liu, S., Tang, D., Sundaresan, N., Zhou, M., Blanco, A., and Ma, S. CodeBLEU: a method for automatic evaluation of code synthesis. arXiv preprint arXiv:2009.10297, 2020.  
Sennrich, R., Haddow, B., and Birch, A. Improving neural machine translation models with monolingual data. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 86-96, 2016.  
Sugiyama, A. and Yoshinaga, N. Data augmentation using back-translation for context-aware neural machine translation. In Proceedings of the fourth workshop on discourse in machine translation (DiscoMT 2019), pp. 35-44, 2019.  
Team Gemini, Anil, R., Borgeaud, S., Wu, Y., Alayrac, J.-B., Yu, J., Soricut, R., Schalkwyk, J., Dai, A. M., Hauth, A., et al. Gemini: a family of highly capable multimodal models. arXiv preprint arXiv:2312.11805, 2023.

Wang, X., Wei, J., Schuurmans, D., Le, Q. V., Chi, E. H., Narang, S., Chowdhery, A., and Zhou, D. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, 2022.  
Wei, B., Li, G., Xia, X., Fu, Z., and Jin, Z. Code generation as a dual task of code summarization. Advances in neural information processing systems, 32, 2019.  
Yin, P. and Neubig, G. A syntactic neural model for general-purpose code generation. In Proceedings of the 55th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 440-450, 2017.  
Yin, P., Li, W.-D., Xiao, K., Rao, A., Wen, Y., Shi, K., Howland, J., Bailey, P., Catasta, M., Michalewski, H., et al. Natural language to code generation in interactive data science notebooks. arXiv preprint arXiv:2212.09248, 2022.  
Zhang, T., Kishore, V., Wu, F., Weinberger, K. Q., and Artzi, Y. BERTScore: Evaluating text generation with BERT. arXiv preprint arXiv:1904.09675, 2019.  
Zhou, S., Alon, U., Agarwal, S., and Neubig, G. CodeBERTScore: Evaluating code generation with pretrained models of code. arXiv preprint arXiv:2302.05527, 2023.

# A. Open-Source Projects for Sec. 4.2

<table><tr><td>github.com/AliRn76/panther</td><td>Web framework</td></tr><tr><td>github.com/JoshData/python-mail-validator</td><td>Email validation library</td></tr><tr><td>github.com/Ousret/charset_normalizer</td><td>Encoding detector</td></tr><tr><td>github.com/PantherPy/pantherdb</td><td>Database</td></tr><tr><td>github.com/SimonGreenhill/treemaker</td><td>Tree formatting</td></tr><tr><td>github.com/Textualize/rich</td><td>Rich text formatting</td></tr><tr><td>github.com/TheAlgorithms/Python</td><td>Educational algorithm implementations</td></tr><tr><td>github.com/agronholm/typeguard</td><td>Runtime type checking</td></tr><tr><td>github.com/aio-libs/async-lru</td><td>Cache for Python&#x27;s asyncio</td></tr><tr><td>github.com/akoumian/datefinder</td><td>Extract dates from text</td></tr><tr><td>github.com/alexmojaki/executing</td><td>Python execution frame inspection</td></tr><tr><td>github.com/andalbrecht/sqlparse</td><td>SQL parser</td></tr><tr><td>github.com/antonagestem/phantom-types</td><td>Runtime type annotations</td></tr><tr><td>github.com/caesar0301/treelib</td><td>Tree data structures</td></tr><tr><td>github.com/casbin/pycasbin</td><td>Authorization library</td></tr><tr><td>github.com/chaimleib/intervaltree</td><td>Interval tree implementation</td></tr><tr><td>github.com/cpburnz/python-pathspec</td><td>File path pattern matching</td></tr><tr><td>github.com/datafolklabs/cement</td><td>Application framework</td></tr><tr><td>github.com/dateutil/dateutil</td><td>Date and time utilities</td></tr><tr><td>github.com/davidhalter/jedi</td><td>Autocompletion and refactoring library</td></tr><tr><td>github.com/dgasmith/opt_einsum</td><td>Optimizing einsum functions</td></tr><tr><td>github.com/eigenein/protobuf</td><td>Python ProtoBuf implementation</td></tr><tr><td>github.com/ekzhu/dataschema</td><td>Probabilistic Data Structures</td></tr><tr><td>github.com/fabiocaccamo/python-benedict</td><td>Dictionary implementation</td></tr><tr><td>github.com/facelessuser/sousieve</td><td>CSS Selector</td></tr><tr><td>github.com/facelessuser/wcmatch</td><td>Wilcard File Name matching library</td></tr><tr><td>github.com/foutaise/texttable</td><td>ASCII table creation</td></tr><tr><td>github.com/graphq1python/graphq1-core</td><td>GraphQL port</td></tr><tr><td>github.com/hgrecco/print</td><td>Measurement Unit Library</td></tr><tr><td>github.com/hukkin/tomli</td><td>TOML parser</td></tr><tr><td>github.com/jab/bidict</td><td>Bidirectional map data structure</td></tr><tr><td>github.com/jaraco/jaracocollections</td><td>Collection data structures</td></tr><tr><td>github.com/jaraco/path</td><td>File system path manipulation</td></tr><tr><td>github.com/jd/tenacity</td><td>Retrying library</td></tr><tr><td>github.com/jsh9/pydoclint</td><td>Docsting linter</td></tr><tr><td>github.com/kjd/idna</td><td>Internationalized Domain Names library</td></tr><tr><td>github.com/lemon24/reader</td><td>Feed reader library</td></tr><tr><td>github.com/lidatong/dataclasses.json</td><td>Dataclass serialization library</td></tr><tr><td>github.com/magmaxp/python-readchar</td><td>Library to read characters and key strokes</td></tr><tr><td>github.com/mahmoud/boltons</td><td>Generic Python utilities</td></tr><tr><td>github.com/marcusbuffett.Command-line-chess</td><td>Chess in the CLI</td></tr><tr><td>github.com/martinblech/xmltodict</td><td>XML library</td></tr><tr><td>github.com/mbr/ascitree</td><td>Print trees in ASCII</td></tr><tr><td>github.com/mckinsey/vizro</td><td>Data visualization</td></tr><tr><td>github.com/mgedmin/objectgraph</td><td>Inspect object graphs</td></tr><tr><td>github.com.microsoft/lsprotocol</td><td>Code generator for LSP</td></tr><tr><td>github.com/mkdocsstrings/griffe</td><td>API Documentation</td></tr><tr><td>github.com/mkorpela/overrides</td><td>Override decorator</td></tr><tr><td>github.com/more-itertools/more-itertools</td><td>Iterator utilities</td></tr><tr><td>github.com/mozilla/bleach</td><td>HTML sanitization</td></tr><tr><td>github.com/msiemens/tinydb</td><td>Document database</td></tr><tr><td>github.com/openlawlibrary/python-docx</td><td>DocX library</td></tr><tr><td>github.com/pallets.click</td><td>CLI interface</td></tr><tr><td>github.com/pallets/flask</td><td>Web app framework</td></tr><tr><td>github.com/pavdmyt/yaspin</td><td>Terminal spinner</td></tr><tr><td>github.com/pydantic/pydantic</td><td>Data validation library</td></tr><tr><td>github.com/pydata/patsy</td><td>Statistical Model Description</td></tr><tr><td>github.com/pydata/xarray</td><td>n-D arrays</td></tr><tr><td>github.com/pygments/pygments</td><td>Code highlighting</td></tr><tr><td>github.com/pyparsing/pyParsing</td><td>PEG parsers</td></tr><tr><td>github.com/python-jschemes/jschemeschema</td><td>JSON Schema library</td></tr><tr><td>github.com/pytoollz/toolz</td><td>Functional utilities</td></tr><tr><td>github.com/pypupio/safety</td><td>Vulnerability Detection</td></tr><tr><td>github.com/serge-sans-paillebeniget</td><td>Static Python analysis</td></tr><tr><td>github.com/simonw/datasette</td><td>Tool for data exploration and publishing</td></tr><tr><td>github.com/sqids/sqids-python</td><td>Short Unique Ids library</td></tr><tr><td>github.com/sybronstuel/python-rsa</td><td>RSA implementation</td></tr><tr><td>github.com/tartley/colorama</td><td>Colored terminal text</td></tr><tr><td>github.com/tiangolo/fastapi</td><td>Web framework</td></tr><tr><td>github.com/tqdm/tqdm</td><td>Progress bar library</td></tr><tr><td>gitlab.com/ericvsmith/toposort</td><td>Topological sorting library</td></tr><tr><td>launchpad.net/code/beautifulsoup</td><td>HTML and XML parsing</td></tr></table>

# B. Lift on Diverse Projects

Fig. 5 show a plot of the lift  $L_{M}$  for each project. Again, we see a wide variability of the lift across different projects.

# C. EDITINGRTC Examples

In this section, we provide several qualitative EDITINGRTC examples. In Example #1, all three edit descriptions sampled in the forward pass lead to edits which exactly match the original in the backward pass, consequently achieving a perfect RTC

Figure 5. Lift

score of 100.0. This is possible because the underlying model is able to generate high-quality descriptions that capture the essence of the edit in the forward pass and also precisely perform the described edit in the backward pass. This demonstrates how we can reliably evaluate a model without a labeled evaluation set.

Nonetheless, in our setting, we can consider the PR comments in the CodeReviewer test set as ground truth edit descriptions. For this particular example, we find that leveraging the PR comment in the backward pass (in place of the sampled descriptions) does lead to the model predicting the matching edit. With a supervised evaluation setup, we can assess the quality of the sampled edit descriptions by comparing to the ground truth edit description with a standard text generation metric like BLEU. Here, we find that irrespective of whether the BLEU score is high (e.g., Sample #3) or low (e.g., Sample #2), the descriptions sufficiently describe the edit in such a way that the model can re-derive the edit using these descriptions in the backward pass. While the average description BLEU is still relatively high in Example #1, we find that it is often much lower even when the predicted descriptions accurately reflect the edit, such as in Example #2.

The previous examples pertain to instances in which the RTC score is high. We now consider cases in which the RTC score is low. In Example #3, none of the sampled descriptions accurately reflect the edit, and consequently, they do not lead to the matching edit in the backward pass. While the underlying model is able to generate an edit which is consistent with the description for Sample #3, the descriptions for Sample #1 and Sample #2 are too confusing for the model such that it simply copies the old code without performing any edits. Despite the fact that these descriptions are inaccurate, the BLEU score (with respect to the PR comment) is relatively high, especially for Sample #1, again highlighting another instance in which RTC serves as a more reliable evaluation metric.

In Example #4, we have an example in which the RTC score is 33.333. Of the three predicted descriptions, only one accurately describes the edit (Sample #1), which is also the only one which leads to a matching edit in the backward pass. The predicted edit for the description in Sample #2 is partially consistent with the incorrect description as it does remove schedule from wp_cron_schedule_http_detection. For Sample #3, the model is likely confused by the reference to wp_http_detection in the predicted description, since it does not exist in the code, and consequently it simply copies the old code without performing any edits. Interestingly, the predicted descriptions in Sample #1 and Sample #3 achieve similar BLEU scores with respect to the PR comment, despite the fact that the first one is accurate while the third one is not. On the other hand, the first description leads to a matching edit during the backward pass while the third one does not.

Example #1

<table><tr><td>Describe concisely and accurately with natural language the differences between the old and new code shown below. [old] 
} 
cmd := []string{&quot;go&quot;, &quot;build&quot;} 
if ctx.Config.BuildFLAGS != &quot;&quot; { 
  cmd = append ctx, strings.Splitctx.Config BUILDFLAGS, &quot;&quot;, ..., } 
} 
cmd = append ctx, &quot;-ldflags=&quot;+ldflags, &quot;-o&quot;, output, ctx.Config.Build.Main) 
if err := run(goos, goarch, cmd); err != nil { 
[new] 
} 
cmd := []string{&quot;go&quot;, &quot;build&quot;} 
if ctx.Config BUILD Flags != &quot;&quot; { 
  cmd = append ctx, strings[, Fieldsctx, Config, Build, Flags], ... ) 
} 
cmd = append ctx, &quot;-ldflags=&quot;+ldflags, &quot;-o&quot;, output, ctx.ConfigBuild.Main) 
if err := run(goos, goarch, cmd); err != nil { 
[edit description] 
Description 
Desc BLEU (→) 
Edit Exact Match (←) 
PR Comment: Is &#x27;strings Fields(+) and keep the code consistent with above. 
15.917 
√ 
#2: Please use correct strings,+ fields call. 
7.171 
√ 
#3: This should use a Go-style append of &#x27;strings Fields(+) to cmd. 
50.868 
√ 
Avg Desc BLEU 
RTCExactMatch 
24.652 
100.0</td></tr></table>

# Example #2

```txt
Describe concisely and accurately with natural language the differences between the old and new code shown below.   
[old]   
1   
// check if files were uploaded through the manifest   
if dirManifest.Size()  $= = 0$  { return swarm.ZeroAddress, fmt.Error("no files added from tar")   
}   
[ new ]   
1   
// check if files were uploaded through the manifest   
if dirManifest.Length()  $= = 0$  { return swarm.ZeroAddress, fmt.Error("no files added from tar")   
}   
[edit description]
```

<table><tr><td>Description</td><td>Desc BLEU(→)</td><td>Edit Exact Match(←)</td></tr><tr><td>PR Comment: I think &#x27;Length&#x27; is more legible here as &#x27;Size&#x27; might infer actual size in bytes</td><td>-</td><td>✓</td></tr><tr><td>Baseline: Edit.</td><td>-</td><td>X</td></tr><tr><td>EDITINGRTC Forward Samples:</td><td></td><td></td></tr><tr><td>#1: please use the length (Length() method instead of the size() method.</td><td>2.416</td><td>✓</td></tr><tr><td>#2: In this case, &quot;Size&quot; is not a good thing to read.</td><td>2.390</td><td>✓</td></tr><tr><td>&quot;Length&quot; is better.</td><td></td><td></td></tr><tr><td>#3: Please use Length() instead of Size() as this code will work more correctly across different go versions.</td><td>2.443</td><td>✓</td></tr><tr><td></td><td>Avg Desc BLEU</td><td>\( RTC_{ExactMatch} \)</td></tr><tr><td></td><td>2.416</td><td>100.0</td></tr></table>

Example #3  
```python
Describe concisely and accurately with natural language the differences between the old and new code shown below. [old] kms/master_key_id = long_uid() sse_specification = {"Enabled": True, "SSTYPE": "KMS", "KMSMasterKeyId": kms/master_key_id} kms/master_key_arn = "arn:aws:ks:%s:key/%s" % (aws_stack.get_local_region(), TEST AwsACCOUNT_ID, kms/master_key_id, result = dynamodb.create_table(TableName=table_name, [new] kms/master_key_id = long_uid() sse_specification = {"Enabled": True, "SSTYPE": "KMS", "KMSMasterKeyId": kms/master_key_id} kms/master_key_arn = "aws.stack.kms.key_arn(kms/master_key_id) result = dynamodb.create_table(TableName=table_name, [edit description] Description & Edit Desc BLEU  $(\rightarrow)$  Edit Exact Match  $(\leftarrow)$  PR Comment: Same as above - we can use 'aws_stack.kms_key_arn(..)' here. [new] kms/master_key_id = long_uid() sse_specification = {"Enabled": True, "SSTYPE": "KMS", "KMSMasterKeyId": kms/master_key_id} kms/master_key_arn = "arn:aws:ks:%s:%s:key/%s" % (aws_stack.get_local_region(), TEST AwsACCOUNT_ID, kms/master_key_id, result = dynamodb.create_table(TableName=table_name, Baseline: Edit. [new] kms/master_key_id = long_uid() sse_specification = {"Enabled": True, "SSTYPE": "KMS", "KMSMasterKeyId": kms/master_key_id} kms/master_key_arn = "arn:aws:ks:%s:%s:key/%s" % (aws_stack.get_local_region(), TEST AwsACCOUNT_ID, kms/master_key_id, ) result = dynamodb.create_table(TableName=table_name, EDITINGRTC Sample #1: Please rename the aws_stack function. 'kms_key_arn' is the convention for methods 20.829 [new] kms/master_key_id = long_uid() sse_specification = {"Enabled": True, "SSTYPE": "KMS", "KMSMasterKeyId": kms/master_key_id} kms/master_key_arn = "arn:aws:ks:%s:%s:key/%s" % (aws_stack.get_local_region(), TEST AwsACCOUNT_ID, kms/master_key_id, ) result = dynamodb.create_table(TableName=table_name, EDITINGRTc Sample #2: Avoid calling into other modules for simple formatting operations. The kms key_arn operation in aws_stack is a one-liner, so no need to extract it as its own function. Similarly, avoid calling AWS APIs on the module level if you need to pass parameters. [new] kms/master_key_id = long_uid() sse_specification = {"Enabled": True, "SSTYPE": "KMS", "KMSMasterKeyId": kms/master_key_id} kms/master_key_arn = "arn:aws:ks:%s:%s:key/%s" % (aws_stack.get_local_region(), TEST AwsACCOUNT_ID, kms/master_key_id, ) result = dynamodb.create_table(TableName=table_name,
```

EDITINGRTC Sample #3: Remove "get_local_region()" call from here.

This should be using the region from the AWS stack.

5.066

```python
[ new ]
kms/master_key_id = long.uid()
sse_specification = {"Enabled": True, "SSTYPE": "KMS",
"KMSMasterKeyId": kms/master_key_id}
kms/master_key_arn = "arn:aws:kms:%s:%s:key/%s" % (
    aws.stack.get_region()
    TESTAWSACCOUNT_ID,
    kms/master_key_id,
) 
result = dynamodb.create_table(
    TableName=table_name,
    kms/master_key_id,
)
```

Avg Desc BLEU RTCExactMatch

11.746 0.000

# Example #4

```txt
Describe concisely and accurately with natural language the differences between the old and new code shown below.   
[old]   
}   
// HTTPS detection.   
add_action('init','wp_cron_schedule HTTPS_detector');   
add_action('wp_http锨','wp_update_urlps锨errors');   
add_filter('cron_request'，'wp_cron_conditionally_prevent_sslyverify'，9999）;   
[ new ]   
}   
// HTTPS detection.   
add_action('init','wp_schedule HTTPS_detector');   
add_action('wp_http锨','wp_update_urlps锨errors');   
add_filter('cron_request'，'wp_cron_conditionally_prevent_sslyverify'，9999）;   
[edit description]
```

<table><tr><td>Description &amp; Edit</td><td>Desc BLEU(→)</td><td>Edit Exact Match (←)</td></tr><tr><td>PR Comment: Would this be better as &#x27;admin_init&#x27; instead?</td><td></td><td></td></tr><tr><td>This will reduce frontend DB writes.</td><td>-</td><td></td></tr><tr><td>“‘suggestion add_action(‘admin_init’, ‘wp_cron_schedule_http_detection’)”;”</td><td></td><td></td></tr><tr><td>[ new]</td><td></td><td></td></tr><tr><td>// HTTPS detection.</td><td></td><td>X</td></tr><tr><td>add_action( &#x27;admin_init&#x27;, &#x27;wp_cron_schedule_http_detection&#x27;);</td><td></td><td></td></tr><tr><td>add_action( &#x27;wp_http_detection&#x27;, &#x27;wp_update_http_detectionErrors&#x27;);</td><td></td><td></td></tr><tr><td>add_filter( &#x27;cron_request&#x27;, &#x27;wp_cron_conditionally Prevent_SSLverify&#x27;, 9999);</td><td></td><td></td></tr><tr><td>Baseline: Edit.</td><td>-</td><td></td></tr><tr><td>[ new]</td><td></td><td></td></tr><tr><td>// HTTPS detection.</td><td></td><td>X</td></tr><tr><td>add_action( &#x27;init&#x27;, &#x27;wp_cron\C_schedule_http_detection&#x27;);</td><td></td><td></td></tr><tr><td>add_action( &#x27;wp_http_detection&#x27;, &#x27;wp_update_http_detectionErrors&#x27;);</td><td></td><td></td></tr><tr><td>add_filter( &#x27;cron_request&#x27;, &#x27;wp_cron_conditionally Prevent_SSLverify&#x27;, 9999);</td><td></td><td></td></tr><tr><td>EDITINGRTC Sample #1: The filter/hook name used to schedule events changed from &#x27;wp_cron_schedule_ to just &#x27;wp_schedule_&#x27;.</td><td>11.667</td><td></td></tr><tr><td>Remove the &#x27;wp_cron_prefix&#x27; here.</td><td></td><td></td></tr><tr><td>[ new]</td><td></td><td></td></tr><tr><td>// HTTPS detection.</td><td></td><td>✓</td></tr><tr><td>add_action( &#x27;init&#x27;, &#x27;wp_schedule_http_detection&#x27;);</td><td></td><td></td></tr><tr><td>add_action( &#x27;wp_http_detection&#x27;, &#x27;wp_update_http_detectionErrors&#x27;);</td><td></td><td></td></tr><tr><td>add_filter( &#x27;cron_request&#x27;, &#x27;wp_cron_conditionally Prevent_SSLverify&#x27;, 9999);</td><td></td><td></td></tr><tr><td>EDITINGRTC Sample #2: Please replace &quot;schedule&quot; with &quot;cron&quot;; this causes confusion to those who are unfamiliar with Wordpress jargon.</td><td>0.757</td><td></td></tr><tr><td>[ new]</td><td></td><td></td></tr><tr><td>// HTTPS detection.</td><td></td><td>X</td></tr><tr><td>add_action( &#x27;init&#x27;, &#x27;wp_cron_http_detection&#x27;);</td><td></td><td></td></tr><tr><td>add_action( &#x27;wp_http_detection&#x27;, &#x27;wp_update_http_detectionErrors&#x27;);</td><td></td><td></td></tr><tr><td>add_filter( &#x27;cron_request&#x27;, &#x27;wp_cron_conditionally Prevent_SSLverify&#x27;, 9999);</td><td></td><td></td></tr></table>

```txt
EDITINGRTC Sample #3: Please rename the actions and filter used in this commit from 'wp_http_detection' to 'wp_cron_conditionally_prevent_SSLverify' 11.345 to 'wp_http_detection'.
```

Figure 6. Relationship between output length and  $\mathsf{RTC}_{\mathrm{ExactMatch}}$  for EDITINGRTC.

```txt
[ new ]  
}  
// HTTPS detection.  
add_action('init','wp_cron{}_scheduleHttps_detector');  
add_action('wpHttps_detector','wp_updateHttps_detectorErrors');  
add_filter('cron_request','wp_cronconditionally_prevent sslverify',9999);  
Avg Desc BLEU RTCExactMatch  
7.923 33.333
```

# D. Length Statistics

For EDITINGRTC, we find that the outputs which achieve a non-zero  $\mathsf{RTC}_{\mathsf{ExactMatch}}$  score are generally shorter than the outputs that achieve an  $\mathsf{RTC}_{\mathsf{ExactMatch}}$  score of zero (Fig. 6). For instance, with Gemini Nano 2, the average number of characters for outputs achieving non-zero scores is 302.6 while it is 454.2 for outputs which have  $\mathsf{RTC}_{\mathsf{ExactMatch}}$  scores of zero. This is intuitive since it becomes more difficult to exactly match the input as the length increases.

# Footnotes:

Page 2: If unit tests were available we would have preferred them. 
