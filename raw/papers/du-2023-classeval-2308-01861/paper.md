ClassEval: A Manually-Crafted Benchmark  for Evaluating LLMs on Class-level Code Generation
=============================================================================================

Xueying Du Mingwei Liu Kaixin Wang Hanlin Wang Junwei Liu  
Yixuan Chen Jiayi Feng Chaofeng Sha Xin Peng Yiling LouFudan University Shanghai, China[xueyingdu21, kxwang23, wanghanlin23@m.fudan.edu.cn](mailto:xueyingdu21,%20kxwang23,%20wanghanlin23@m.fudan.edu.cn)[jwliu22, 23212010005, 23210240148@m.fudan.edu.cn](mailto:jwliu22,%2023212010005,%2023210240148@m.fudan.edu.cn)[liumingwei, cfsha, pengxin, yilinglou@fudan.edu.cn](mailto:liumingwei,%20cfsha,%20pengxin,%20yilinglou@fudan.edu.cn)

###### Abstract.

Recently, many large language models (LLMs) have been proposed, showing advanced proficiency in code generation. Meanwhile, many efforts have been dedicated to evaluating LLMs on code generation benchmarks such as HumanEval. Although being very helpful for comparing different LLMs, existing evaluation focuses on a simple code generation scenario (i.e., function-level or statement-level code generation), which mainly asks LLMs to generate one single code unit (e.g., a function or a statement) for the given natural language description. Such evaluation focuses on generating independent and often small-scale code units, thus leaving it unclear how LLMs perform on generating more complicated code.

To fill this knowledge gap, we make the first attempt to evaluate LLMs in a more challenging code generation scenario, i.e., class-level code generation. We first manually construct the first class-level code generation benchmark ClassEval of 100 class-level Python code generation tasks with approximately 500 person-hours. Based on the new benchmark ClassEval, we then perform the first study of 11 state-of-the-art LLMs on class-level code generation. Based on our results, we have the following main findings. First, we find that all existing LLMs show much worse performance on class-level code generation compared to on standalone method-level code generation benchmarks like HumanEval; and the method-level coding ability cannot equivalently reflect the class-level coding ability among LLMs. Second, we find that GPT-4 and GPT-3.5 still exhibit dominate superior than other LLMs on class-level code generation, and the second-tier models includes Instruct-StarCoder, Instruct-CodeGen, and WizardCoder with very similar performance. Third, we find that generating the entire class all at once (i.e., holistic generation strategy) is the best generation strategy only for GPT-4 and GPT-3.5, while method-by-method generation (i.e., incremental and compositional) is better strategies for the other models with limited ability of understanding long instructions and utilizing the middle information. Lastly, we find the limited model ability of generating method-dependent code and discuss the frequent error types in generated classes. Our benchmark is available at https://github.com/FudanSELab/ClassEval

Class-level Code Generation, Large Language Model, Benchmark

1. Introduction
----------------

Code generation techniques automatically generate code snippets for the given natural language description, which can be leveraged to improve development productivity and have been extensively studied in literature*([vikram2023large,](#bib.bib1 "") ; [10172763,](#bib.bib2 "") ; [kang2023explainable,](#bib.bib3 "") )*. The recent advance in large language models (LLMs) has brought significant advancements in the code generation domain.
To date, researchers have proposed various LLMs*([openai2023gpt4,](#bib.bib4 "") ; [bian2023chatgpt,](#bib.bib5 "") ; [zheng2023vicuna,](#bib.bib6 "") ; [du2022chatglm,](#bib.bib7 "") ; [luo2023wizardcoder,](#bib.bib8 "") ; [li2023starcoder,](#bib.bib9 "") ; [intructcodegen,](#bib.bib10 "") ; [zheng2023codegeex,](#bib.bib11 "") ; [xu2022polycoder,](#bib.bib12 "") ; [fried2022incoder,](#bib.bib13 "") ; [allal2023santacoder,](#bib.bib14 "") )* (such as GPT-4*([openai2023gpt4,](#bib.bib4 "") )*, WizardCoder*([luo2023wizardcoder,](#bib.bib8 "") )*, and Instruct-CodeGen*([intructcodegen,](#bib.bib10 "") )*) by training large models with over billions of parameters on massive general or code-specific corpora and instructions.

To fully understand the code generation capability of emerging LLMs, many efforts have been dedicated to evaluating LLMs on automatically or manually constructed code generation benchmarks. To date, many code generation benchmarks have been proposed, such as HumanEval*([chen2021huamneval,](#bib.bib15 "") )* and MBPP*([austin2021mbpp,](#bib.bib16 "") )*. Although being very helpful for people to understand and compare the performance of different LLMs, existing evaluation actually focuses on a rather simple code generation scenario, i.e., function-level or statement-level code generation. They mainly ask LLMs to generate one single code unit (e.g., a function or a statement) for the given natural language descriptions in a standalone way, which inherently have two limitations in evaluating LLMs in code generation. First, such evaluation tends to focus on generating code of short length, e.g., each task in the most widely-used benchmark HumanEval only involves generating code of 11.5 lines and 24.4 tokens on average.
Such a number of generated tokens is far within the maximum number of tokens in recent LLMs (e.g., 2,048 for WizardCoder*([luo2023wizardcoder,](#bib.bib8 "") )*). Therefore, it remains unclear about the further potential of LLMs in generating long code snippets. Second, such evaluation mainly focuses on generating one single code unit, e.g., one function or one statement. However, as shown in previous work*([yu2023codereval,](#bib.bib17 "") )*, only 30% of methods are independent to other code contexts in the open-source projects. Therefore, it remains unclear how LLMs perform in generating a compound code unit of multiple methods111As we currently focus on Python, we distinguish concepts “method” and “function”: a method is associated to an object and requires an object instance to be invoked, while a function is an independent code block that can be called from anywhere. which are dependent to each other (e.g., invoking each other or accessing a same variable).

Benchmark ClassEval. To fill this knowledge gap, this work makes the first attempt to evaluate LLMs in a more challenging code generation scenario, i.e., class-level code generation. In particular, we evaluate the model capability of generating a class of multiple interdependent methods for the given natural language description. As no existing benchmarks cover class-level code generation tasks, we manually construct the first class-level code generation benchmark ClassEval in a rigorous and time-intensive way, which takes approximately 500 person-hours to construct 100 class-level Python code generation tasks. Overall, ClassEval covers a wide range of topics in practical software development (e.g., management systems and game development). Each task is constructed with a test suite of high testing sufficiency (e.g., 98.2% and 99.7% branch-level or statement-level coverage) so as to facilitate reliable correctness checking of the generated code; furthermore, each task is designed to generate a class of multiple methods with diverse dependencies (e.g., field, method, and library dependencies).

Empirical study. Based on the new benchmark ClassEval, we then perform the first study to evaluate LLMs on class-level code generation. In particular, our experiments include 11 state-of-the-art LLMs, which are diverse in model sizes, foundation models, sources, or domains. For each studied LLM, we explore its performance in generating class-level code with three different generation strategies, i.e., holistic generation (generating the entire class all at once), incremental generation and compositional generation (generating the class method by method). For each generated code snippet, we measure its correctness with the widely-used metric Pass@k *([chen2021pass_at_k,](#bib.bib18 "") )*. In addition, we also investigate the model ability of generating dependent code and analyze bad cases of incorrect classes.

Main findings and implications. Based on our results, we have the following main findings. First, we find that all existing LLMs show much worse performance on class-level code generation compared to on standalone method-level code generation benchmarks like HumanEval; and the method-level coding ability cannot equivalently reflect the class-level coding ability among LLMs. Second, we find that GPT-4 and GPT-3.5 still exhibit dominate superior than other LLMs on class-level code generation, and the second-tier models includes Instruct-StarCoder, Instruct-CodeGen, and WizardCoder with very similar performance. Third, we find that generating the entire class all at once (i.e., holistic generation strategy) is the best generation strategy only for GPT-4 and GPT-3.5, while step-by-step generation (i.e., incremental and compositional) is better strategies for the other models with limited ability of understanding long instructions and utilizing the middle information. Lastly, we find the limited model ability of generating method-dependent code and discuss the frequent error types in generated classes.

In summary, this paper makes the following contributions:

* •

    The first benchmark ClassEval for class-level code generation, which is manually constructed with 500 person-hours and publicly available on*([classeval,](#bib.bib19 "") )*;

* •

    The first study to evaluate 11 representative LLMs on class-level code generation with three different generation strategies;

* •

    Findings and implications on analyzing the model capability and future directions for LLMs on class-level code generation.

2. Background
--------------

We first introduce the recent LLMs for code generation in Section[2.1](#S2.SS1 "2.1. Large Language Models for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") and then motivate our study by revisiting existing code generation benchmarks in Section[2.2](#S2.SS2 "2.2. Existing Benchmarks for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation").

### 2.1. Large Language Models for Code Generation

Code generation is a task focusing on generating code snippets for the given natural language description, which has been extensively studied in recent literature*([vikram2023large,](#bib.bib1 "") ; [10172763,](#bib.bib2 "") ; [kang2023explainable,](#bib.bib3 "") )*. General LLMs (e.g., GPT-4*([openai2023gpt4,](#bib.bib4 "") )* and ChatGLM*([du2022chatglm,](#bib.bib7 "") )*), which are large models with more than billions of parameters trained on general textual/code corpora and instructions, demonstrate remarkable capabilities not only in general NLP tasks*([chang2023survey,](#bib.bib20 "") )* but also promising performance in code generation. For example, GPT-4 achieves the highest pass rate on HumanEval benchmark*([luo2023wizardcoder,](#bib.bib8 "") )*. Therefore, there has recently been an increasing trend to evaluate the code generation capacity even for general LLMs*([chen2021huamneval,](#bib.bib15 "") ; [shen2023pangucoder2,](#bib.bib21 "") )*. Code LLMs, which are large models mainly trained with massive code-specific corpora and instructions, often have better capability than general LLMs in code generation tasks*([luo2023wizardcoder,](#bib.bib8 "") ; [christopoulou2022pangucoder,](#bib.bib22 "") ; [zan-etal-2023-large,](#bib.bib23 "") )*.
Existing code LLMs are designed with different training objectives. For example, some are using next-token prediction, while some code LLMs (e.g., InCoder*([fried2022incoder,](#bib.bib13 "") )* and StarCoder*([li2023starcoder,](#bib.bib9 "") )*) are trained with “filling-in-the middle” (FIM) capability, i.e., infilling the missing portion based on the context. To date, a large number of code LLMs have been proposed, such as WizardCoder*([luo2023wizardcoder,](#bib.bib8 "") )*, Instruct-StarCoder*([intructstarcoder,](#bib.bib24 "") )*, and Instruct-CodeGen*([intructcodegen,](#bib.bib10 "") )*.

*Table 1. Existing Benchmarks for Code Generation*

BenchmarkTimeLanguageManual/AutomatedSourceGranularity#Tasks#Tests#LOC#TokensInput InformationConcode*([iyer2018concode,](#bib.bib25 "") )*2018JavaAutomatedGithubFunction-level2,000--26.3NLCoNaLA*([yin2018conala,](#bib.bib26 "") )*2018PythonAutomatedStack OverflowStatement-level500-14.6NLAPPS*([hendrycks2021apps,](#bib.bib27 "") )*2021PythonAutomatedContest SitesCompetitive5,00013.221.458NL + Example Inputs/OutputsHumanEval*([chen2021huamneval,](#bib.bib15 "") )*2021PythonManual-Function-level1647.711.524.4NL + Function Signature + Example Inputs/OutputsMBPP*([austin2021mbpp,](#bib.bib16 "") )*2021PythonManual-Function-level9743.06.824.2NLmath-qa*([austin2021mbpp,](#bib.bib16 "") )*2021PythonManualMath Study SitesStatement-level2,985-7.624.6NLMulti-HumanEval*([athiwaratkun2023multilingual,](#bib.bib28 "") )*2022MultilingualManual-Function-level1647.711.524.4NL + Function Signature + Example Inputs/OutputsMBXP*([athiwaratkun2023multilingual,](#bib.bib28 "") )*2022MultilingualManual-Function-level9743.06.824.2NLmulti-math-qa*([athiwaratkun2023multilingual,](#bib.bib28 "") )*2022MultilingualManualMath Study SitesStatement-level2,985-7.624.6NLCodeContests*([li2022codecontest,](#bib.bib29 "") )*2022Python, C++AutomatedContest SitesCompetitive165203.759.8184.8NL + Example Inputs/OutputsDS-1000*([lai2023ds1000,](#bib.bib30 "") )*2022PythonAutomatedStack OverflowStatement-level1,0001.63.812.8NLHumanEval+*([liu2023humanevalplus,](#bib.bib31 "") )*2023PythonManual-Function-level164774.811.524.4NL + Function Signature + Example Inputs/OutputsCoderEval*([yu2023codereval,](#bib.bib17 "") )*2023Python, JavaAutomatedGithubFunction-level230-30108.2NL + Function SignatureClassEval2023PythonManual-Class-level10033.145.7123.7Class Skeleton

### 2.2. Existing Benchmarks for Code Generation

Code generation benchmarks typically include various coding tasks where a natural language description serves as input, and the corresponding code serves as the ground truth output. Evaluation metrics such as passing rate (Pass@k *([chen2021pass_at_k,](#bib.bib18 "") )*) are commonly used to assess the correctness of the generated code.

To date, many code generation benchmarks have been constructed via automated or manual manners. In this study, we revisit widely-used code generation benchmarks from the three following sources: (i) Top-10 popular datasets with the highest download volumes from Huggingface code generation datasets*([codegeneration,](#bib.bib32 "") )*, (ii) benchmarks associated with recent LLM papers (released between June 2021 and June 2023), and (iii) enhanced benchmarks such as HumanEval+*([liu2023humanevalplus,](#bib.bib31 "") )* and Multi-HumanEval*([athiwaratkun2023multilingual,](#bib.bib28 "") )*.
Table[1](#S2.T1 "Table 1 ‣ 2.1. Large Language Models for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") provides an overview of the 13 distinct benchmarks collected from the three sources, including their release time, construction method (i.e., manually written or automatically collected from public code corpus or competitions), benchmark size (#Tasks), target code granularity, target code language, code scale (#LOC: average lines of code, #Tokens: average number of tokens), average number of test cases per task (#Tests), and detailed input information. We also present our constructed benchmark ClassEval in the last row for comparison.

Based on Table[1](#S2.T1 "Table 1 ‣ 2.1. Large Language Models for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), we find that existing benchmarks actually shape a rather simple code generation scenario, which mainly evaluate the capability of LLMs in generating one single code unit (a function or a statement) in a rather standalone way. In particular, existing benchmarks typically focus on function-level or statement-level code generation tasks (Column “Granularity”) and rarely include additional code contexts in the input (Column “Input Information”), which assumes that the code to be generated is an independent unit and thus leads to two limitations in evaluating LLMs.

First, existing benchmarks mainly focus on short code generation tasks, like generating one function or one statement. These tasks typically involve limited number of lines (e.g., 1 to 30) and tokens (e.g., 4.6 to 108.2), which may not fully explore the capabilities of recent LLMs that can handle much longer sequences, such as WizardCoder with 2,048 tokens. Thus, the potential of LLMs in generating longer code snippets remains unclear.
Second, existing benchmarks mainly focus on generating independent code units without considering other code contexts. For instance, as shown in Figure[1](#S2.F1 "Figure 1 ‣ 2.2. Existing Benchmarks for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), benchmarks like MBPP and HumanEval only provide limited information as input, such as natural language descriptions or function signatures with example inputs and outputs. However, in real-world scenarios, methods often depend on each other or share variables.
Previous work*([yu2023codereval,](#bib.bib17 "") )* indicates that only 30% of methods in open-source projects are independent of other code contexts.
Therefore, it remains unclear how LLMs perform in generating a compound code unit of multiple methods which are dependent to each other (e.g., invoking each other or accessing a same variable).

<img src='x1.png' alt='Refer to caption' title='' width='415' height='196' />

*Figure 1. Examples in Existing Benchmarks*

Our Motivation. Existing benchmarks cannot facilitate the model evaluation on more complicated code generation tasks, such as generating longer and compound code units of multiple interdependent methods. To address this gap, we manually construct the first class-level code generation benchmark ClassEval and perform the first study to evaluate LLMs on class-level code generation tasks, which ask LLMs to generate a class of multiple interdependent methods based on a given natural language description.

3. New benchmark ClassEval
---------------------------

In this section, we introduce our new benchmark ClassEval. We present the benchmark format (Section[3.1](#S3.SS1 "3.1. Benchmark Format ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")), the construction procedure (Section[3.2](#S3.SS2 "3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")), and the benchmark characteristics (Section[3.3](#S3.SS3 "3.3. Benchmark Characteristics ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")).

### 3.1. Benchmark Format

Each coding task in ClassEval comprises an input description for the target class (i.e., the class to be generated), a test suite for verifying the correctness of the generated code, and a canonical solution that acts as a reference implementation of the target class.

<img src='x2.png' alt='Refer to caption' title='' width='406' height='475' />

*Figure 2. An Example of Class Skeleton in ClassEval*

Typically, LLMs generate code snippets based on input descriptions and the correctness is verified with the provided test suite.
The generated code must conform to a consistent interface (e.g., the types of input parameters and return values) specified in the test suite for valid execution.
For example, the benchmark HumanEval specifies the signature of the target function (Figure[1](#S2.F1 "Figure 1 ‣ 2.2. Existing Benchmarks for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")) to ensure that the generated bodies are validly checked by the given test suite.
To achieve this, we define a class skeleton format for the input descriptions in our coding tasks. The class skeleton serves as a structured blueprint for the target class, containing both class-level information (import statements, class name, class description, and class constructor) and method-level information (method signature, functional description, parameter/return descriptions, and example input/outputs).
The detailed definitions of elements in the class skeleton are in Table[2](#S3.T2 "Table 2 ‣ 3.1. Benchmark Format ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"). Column “Mand.” indicates whether the element is mandatory in the class skeleton.
Method-level elements are all adopted from existing benchmarks like HumanEval.
Figure[2](#S3.F2 "Figure 2 ‣ 3.1. Benchmark Format ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") further illustrates an example of a class skeleton, with different components highlighted in various colors.
The class skeleton, inspired by contract programming*([contract,](#bib.bib33 "") )*, serves as formal and precise specifications for code generation by outlining expected behaviors, pre-conditions, and post-conditions. LLMs generate class-level code that aligns with the given test suite based on the class skeleton.

*Table 2. Elements Defined in Class Skeleton*

ElementsMand.DefinitionClassLevelInfo.Class Name✓The name of the target classClass Description✓The description of the overall functionality of the target classImport Statements✕Indicating the external libraries or modules necessary for implementing the target classClass Constructor✕The initial method automatically invoked to initialize the attributes once the class is instantiatedMethodContractDesignMethod Signature✓Defining the target method name, input parameters, and return typeFunctional Description✓Natural language descriptions on the functionality of each methodParameter/Return Description✕Textual descriptions on expected inputs (e.g., parameter types) and outputs (e.g., return values) for each methodExample Input/Output✕Concrete examples of input values and corresponding output values on executing the target method

### 3.2. Benchmark Construction Procedure

Figure[3](#S3.F3 "Figure 3 ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") illustrates the procedure of constructing ClassEval.
We follow four steps to create ClassEval: (i) select suitable coding tasks using different strategies (Section[3.2.1](#S3.SS2.SSS1 "3.2.1. Task Selection ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")); (ii) construct class skeletons based on the principles of contract programming*([contract,](#bib.bib33 "") )* and test-driven development*([bhat2006testdriven,](#bib.bib34 "") )* (Section[3.2.2](#S3.SS2.SSS2 "3.2.2. Class Skeleton Construction ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")); (iii) create the test suite for each class skeleton (Section[3.2.3](#S3.SS2.SSS3 "3.2.3. Test Construction ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")); and (iv) write the canonical solution for each coding task (Section[3.2.4](#S3.SS2.SSS4 "3.2.4. Canonical Solution Construction ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")). The constructed class skeletons, test suites, and canonical solutions form our class-level code generation benchmark ClassEval.

<img src='x3.png' alt='Refer to caption' title='' width='401' height='128' />

*Figure 3. Overview of ClassEval Construction Process*

To avoid the coding tasks being seen by LLMs during their training, our benchmark is constructed completely manually, so as to mitigate potential data leakages from existing code sources. Our manual construction involves a time-intensive process with approximately 500 person-hours on constructing 100 class-level coding tasks. Due to the significant manual efforts required, we currently stop the benchmark scale to this size.
Moreover, following the trend of most existing benchmarks*([chen2021huamneval,](#bib.bib15 "") ; [austin2021mbpp,](#bib.bib16 "") )*, our benchmark primarily focuses on Python given its prevalence*([srinath2017python,](#bib.bib35 "") )*.

#### 3.2.1. Task Selection

In this step, we design class-level coding tasks (i.e., a unique class description for each task as defined in Table[2](#S3.T2 "Table 2 ‣ 3.1. Benchmark Format ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")) for our benchmark.

Inclusion Sources. We design our coding tasks to cover diverse and real-world development topics, based on the following three sources.
(i) Revisiting Existing Benchmarks. We refer to well-established benchmarks like HumanEval and MBPP (Table[1](#S2.T1 "Table 1 ‣ 2.1. Large Language Models for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")) to include prevalent and common topics.
(ii) Exploring PyPI Topics. We manually explore the Python Package Index (PyPI)*([pypi,](#bib.bib36 "") )*, which hosts a vast repository of Python software packages and provides a diverse range of potential task topics.
(iii) Brainstorming.
All authors (with 2-8 years of Python development experience) actively participate in brainstorming to generate potential coding tasks beyond the ones collected above.

Exclusion Criteria. Our benchmark focuses on coding tasks that can be implemented within one single class. Therefore, we exclude tasks that have complicated dependencies on the execution environment, including those related to (i) Network Programming, (ii) Graphical User Interface (GUI) Design, (iii) Data Visualization, (iv) System Programming, and (v) Concurrent Programming. These tasks often require interactions with other classes or cannot be easily verified with assertion statements in unit tests.

In this way, we obtain a list of 100 diverse class-level coding tasks, covering a wide spectrum of topics, such as Game Development, File Handling, and Management Systems. Table[3](#S3.T3 "Table 3 ‣ 3.2.1. Task Selection ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") presents the topic distribution of our tasks.

*Table 3. Topic Type Definitions in ClassEval*

TopicDescriptionExamples#TasksManagement SystemsOperational functionalities in common software management systems projectsStudent Registration System, Movie Booking System27Data FormattingProcessing data according to specific rules or patternsText-to-number Conversion, URL Format Validation26Mathematical OperationsAlgorithms for mathematical and statistical problemsBasic Arithmetic Operations, Area Calculation16Game DevelopmentAlgorithms for game functionalities, including mechanics and state managementMinesweeper Game, Gomoku Game10File HandlingCommon file operations including reading, writing, and simple processing data in filesCSV File Processor, JSON File Processor9Database OperationsImplementation of common database operationsLibrary Database Operation, SQL Query Generator7Natural Language ProcessingTechniques for processing and analyzing text dataStop Word Removal, Longest Word Identification5

#### 3.2.2. Class Skeleton Construction

During this step, we manually construct the class skeleton for each coding task, involving 5 participants with an average of 3 years of Python development experience.
Each skeleton is initially assigned to two participants, one responsible for writing the class skeleton and the other for double-checking it. In case of disagreements, a third participant facilitates discussions to reach a consensus on the class skeleton. The entire process adheres to the following design principles.

Principle 1 (dependency): Each class skeleton should contain methods with diverse dependencies, i.e., the methods are dependent to other code contexts within the class.
Previous work*([yu2023codereval,](#bib.bib17 "") )* has shown that the majority of methods (over 70%) are dependent on other code contexts in the project.
Unlike previous benchmarks (e.g., HumanEval and MBPP shown in Figure[1](#S2.F1 "Figure 1 ‣ 2.2. Existing Benchmarks for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation")) that focus on standalone function-level code generation,
our class-level benchmark aims to capture the real-world scenario where methods often have dependencies with other code contexts.
To distinguish our benchmark from function-level ones, we deliberately avoid tasks that generate a class with independent methods, which would essentially be a collection of individual method-level coding tasks. Instead, class skeletons in our benchmark includes methods with diverse dependencies, including (i) Library Dependency, where methods rely on external libraries; (ii) Field Dependency, where methods depend on class instance variables (fields); (iii) Method Dependency, where methods rely on other methods within the same class; and (iv) Standalone, where methods function independently without dependencies on fields, methods, or external libraries.

Principle 2 (class constructor): The class constructor (if has) in each class skeleton should define the class fields and their default values. The constructor also includes natural language descriptions of the class fields to provide a clear understanding of their meanings. Importantly, the constructor does not make calls to other methods within the class to preserve the independence and self-contained nature of the class initialization process.

Principle 3 (method functionality): We avoid including complex functionalities like
closing database connections, which are not easily testable and verifiable. Additionally, we enhance code reusability and maintainability by breaking down common and repetitive functionalities into separate methods. This principle fosters potential interdependencies between methods, simulating a more interconnected and practical coding scenario.

Principle 4 (method parameter): The method parameters are limited to primitive data types, avoiding object-level parameters or loosely defined arguments like **kwargs. This principle not only enhances clarity in method invocation but also facilitates testing, making it easier to create unit tests and verify the functionality of individual methods in isolation.

Principle 5 (method return value): Methods should include return values whenever possible for testing. For indicating success or failure, they use Boolean return types for standardization instead of custom strings. Additionally, method designs may encompass evaluative conditions for input parameters and include exception handling mechanisms. Detailed specifications of exception types, message content, and triggering circumstances are provided to ensure comprehensive testing and validation of exception handling.

Each constructed class skeleton would contain mandatory elements (i.e., the class description, the class name, the method signature, and the functional description) and optional elements (i.e., import statements, class constructor, parameter/return descriptions and the example input/output).

#### 3.2.3. Test Construction

In this step, we manually construct a test suite for each coding task based on its class skeleton. The participants who were responsible for creating the class skeleton now take on the task of writing the corresponding test suite. Similarly, one participant focuses on writing the unit test cases, while the other ensures the quality and correctness of the test cases.

The methods in each class skeleton are designed to have multiple dependent relationships, as mentioned in Principle 1 in Section[3.2.2](#S3.SS2.SSS2 "3.2.2. Class Skeleton Construction ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"). Therefore, participants are required to construct test cases at two levels: method-level tests and class-level tests, so as to fully test the correctness of the implemented methods when they are invoked individually or together. Method-level tests primarily check the correctness of each method under test by independently invoking it without invoking any other methods in the class. On the other hand, class-level tests mainly check the correctness of multiple methods under test by invoking them sequentially together. Method-level tests ensure that the correctness of each method under test is individually checked without being impacted by the incorrect implementation of other methods, while class-level tests evaluate the overall correctness of the class by considering its interactions. Figure[4](#S3.F4 "Figure 4 ‣ 3.2.3. Test Construction ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") provides two examples of both method-level and class-level test cases constructed for the class skeleton in Figure[2](#S3.F2 "Figure 2 ‣ 3.1. Benchmark Format ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"). Additionally, we include examples of test cases from existing benchmarks HumanEval and MBPP to highlight the differences. The function-level tests in existing benchmarks are comparable to the method-level tests in our benchmark, but the major difference is that function-level tests in existing benchmarks only check the return values of the function under test while our method-level tests further check the fields of the class. As shown in Figure [4](#S3.F4 "Figure 4 ‣ 3.2.3. Test Construction ‣ 3.2. Benchmark Construction Procedure ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), when testing the purchase_item method, the method-level test in ClassEval not only verifies the return value but also evaluates the operations performed on the inventory field. Moreover, existing benchmarks lack class-level tests since they primarily focus on single-function generation.

We then introduce the main principles of constructing method-level tests and class-level tests, respectively. For method-level tests, participants are asked to create at least five test cases to cover diverse scenarios of each method under test. For class-level tests, participants are required to construct test cases with different combinations of methods under test, ensuring that each method is invoked at least once in the class-level tests. To simplify test construction, participants are required to use the existing unittest framework*([unittest,](#bib.bib37 "") )*, which provides diverse assertion APIs and a set of Test Fixtures (e.g., setUp and tearDown methods) for preparation and cleanup tasks before and after test execution. Additionally, all constructed test cases are limited to a five-second running time to prevent potential infinite loops in the generated code.

<img src='x4.png' alt='Refer to caption' title='' width='922' height='171' />

*Figure 4. Test Cases in Existing Benchmarks and ClassEval*

#### 3.2.4. Canonical Solution Construction

In this step, we manually write the canonical solution for each coding task based on its constructed class skeleton and test cases. Four participants (each with 2 - 4 years of Python development experience) who were not involved in constructing the class skeletons and test cases are engaged in this step. Each coding task is assigned to two participants, with one responsible for writing the canonical solution and the other for double-checking it. Participants are required to execute the solutions with test cases to identify and fix any bugs.

### 3.3. Benchmark Characteristics

In this way, we manually build a new benchmark ClassEval of 100 class-level coding tasks.
The detailed characteristics are as follows.

Scale. ClassEval consists of 100 classes and 412 methods. To facilitate a direct comparison with other code generation benchmarks, we include the statistical data of ClassEval in Table[1](#S2.T1 "Table 1 ‣ 2.1. Large Language Models for Code Generation ‣ 2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"). The results reveal large differences in lines of code for ClassEval (45.7) compared to the two most widely used handwritten benchmarks, HumanEval and MBPP, with multipliers of 4.0 and 6.7 respectively. Additionally, we perform additional statistics on the average number of tokens in the entire docstring information (class skeleton) in ClassEval (259.3), surpassing HumanEval (67.7) and MBPP (14.5) by a factor of 3.8 and 17.9 respectively. These results demonstrate that the class-level code generation task in ClassEval presents higher complexities, involving longer code generation, as well as more detailed and sophisticated docstring information.

*Table 4. Test Coverage and Test Cases Statistics*

| Benchmark | Statement | Branch | #Tests/M | #Tests/C |
| --- | --- | --- | --- | --- |
| HumanEval | 98.8% | 83.2% | 7.7 | - |
| MBPP | 98.6% | 76.4% | 3.0 | - |
| ClassEval | 99.7% | 98.2% | 8.0 | 33.1 |

Test Sufficiency. Table[4](#S3.T4 "Table 4 ‣ 3.3. Benchmark Characteristics ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") provides coverage statistics for the test cases in our benchmark compared to HumanEval and MBPP. We collect the statement-level and branch-level coverage of the test cases on the canonical solution code using the Python toolkit coverage *([coverage,](#bib.bib38 "") )*. Additionally, we provide the average number of method-level tests (#Tests/M) and average class-level tests (#Tests/C). As shown in Table[4](#S3.T4 "Table 4 ‣ 3.3. Benchmark Characteristics ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), the test cases in ClassEval achieve significantly higher statement-level and branch-level coverage (both over 98%) compared to HumanEval and MBPP. This indicates more extensive code checking for the generated solutions in our benchmark, which is supported by the fact that ClassEval also includes a larger number of method-level and class-level tests on average.

Dependency. ClassEval focuses on class-level code generation tasks, distinguishing it from previous benchmarks. Table[5](#S3.T5 "Table 5 ‣ 3.3. Benchmark Characteristics ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") shows the distribution of dependency levels within methods across ClassEval and previous benchmarks, as explained in Section[3.1](#S3.SS1 "3.1. Benchmark Format ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"). Notably, Library, Field, and Method dependencies are not mutually exclusive, and some methods may have a combination of Field and Method dependencies. We classify methods with either Field or Method dependencies as class-level dependent methods, totaling 314 (76.2%) within ClassEval.
This inclusion makes ClassEval a comprehensive benchmark, suitable for evaluating LLMs that must account for intricate class-level interactions and contextual dependencies.

*Table 5. Comparative Distribution of Dependency Levels*

| Dependency | MBPP | HumanEval | ClassEval |
| --- | --- | --- | --- |
| Standalone | 974 (100%) | 157 (95.8%) | 58 (14.1%) |
| Library | - | 7 (4.2%) | 89 (21.7%) |
| Field | - | - | 269 (65.5%) |
| Method | - | - | 107 (26.0%) |

Overall, in comparison to previous manually-crafted code generation benchmarks, ClassEval contains complicated class-level coding tasks involving larger-scale code snippets, diverse dependencies, sufficient test cases, and a wider range of topics from practical software development.

4. Empirical Study
-------------------

Using ClassEval, we conduct the first study to evaluate existing LLMs on class-level code generation by answering the following research questions.

* •

    RQ1 (Overall Correctness): how do LLMs perform on class-level code generation?

* •

    RQ2 (Generation Strategies): how do different generation strategies perform for LLMs on class-level code generation?

* •

    RQ3 (Dependency Generation): how do LLMs perform on generating code dependent to other contexts during class-level code generation?

* •

    RQ4 (Bad Case Analysis): what are the common errors during class-level code generation?

*Table 6. Studied LLMs*

ModelBase ModelTimeSizeIFFIMCodeLLMInstruct-CodeGen*([intructcodegen,](#bib.bib10 "") )*CodeGen-multi*([nijkamp2022condegen,](#bib.bib39 "") )*2022.316B✓✓WizardCoder*([luo2023wizardcoder,](#bib.bib8 "") )*StarCoder*([li2023starcoder,](#bib.bib9 "") )*2023.615B✓✓Instruct-StarCoder*([intructstarcoder,](#bib.bib24 "") )*StarCoder*([li2023starcoder,](#bib.bib9 "") )*2023.515B✓✓CodeGeeX*([zheng2023codegeex,](#bib.bib11 "") )*-2023.313B✕✕InCoder*([fried2022incoder,](#bib.bib13 "") )*Dense*([dense,](#bib.bib40 "") )*2022.46B✕✓PolyCoder*([xu2022polycoder,](#bib.bib12 "") )*GPT-2*([radford2019gpt2,](#bib.bib41 "") )*2022.22.7B✕✕SantaCoder*([allal2023santacoder,](#bib.bib14 "") )*GPT-2*([radford2019gpt2,](#bib.bib41 "") )*2023.11.1B✕✓GeneralLLMVicuna*([zheng2023vicuna,](#bib.bib6 "") )*LLaMA*([touvron2023llama,](#bib.bib42 "") )*2023.37B✓✓ChatGLM*([du2022chatglm,](#bib.bib7 "") )*GLM*([zeng2022glm130b,](#bib.bib43 "") )*2022.36B✓✓GPT-3.5*([openai2023gpt4,](#bib.bib4 "") )*-2022.11-✓✓GPT-4*([openai2023gpt4,](#bib.bib4 "") )*-2023.3-✓✓

### 4.1. Studied LLMs

We select the state-of-the-art LLMs that have been widely studied in recent code generation work*([luo2023wizardcoder,](#bib.bib8 "") ; [liu2023humanevalplus,](#bib.bib31 "") )*. In particular, we focus on recent models released since 2022, and we exclude the small models (with less than 1B parameters) due to their limited efficacy or the large models (with more than 20B parameters) due to our resource limits. Table[6](#S4.T6 "Table 6 ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") presents the 11 LLMs studied in our experiments with their releasing time (Column “Time”), model sizes (Column “Size”), and base models. In addition, we also summarize the training characteristics of the studied models, including whether the model has been trained to possess the ability of “filling-in-the-middle” (FIM) and whether it possesses the instruction-following (IF) ability via instruction tuning. Both FIM and IF capabilities are essential for the class-level code generation tasks.
As shown in Table[6](#S4.T6 "Table 6 ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), our study includes a wide scope of LLMs that are diverse in multiple dimensions, such as (i) being both closed-source and open-source, (ii) utilizing different base models, (iii) covering a range of model sizes from 1B to 16B, (iv) being trained by both general or code-specific instructions, and (v) exhibiting different FIM and IF capabilities.

### 4.2. Studied Generation Strategies

Given a class-level code generation task, we study the performance of each model with three different generation strategies as follows:

* •

    Holistic Generation: the model is asked to generate the entire class all at once with the class skeleton as inputs.

* •

    Incremental Generation: the model is asked to generate the class in a method-by-method manner. Each iteration is based on the method bodies that have been generated in previous iterations. The iterative process repeats until all methods in the class are generated.

* •

    Compositional Generation: the model is asked to generate the class in a method-by-method manner. Each iteration is independent, without considering the other generated methods. All the generated methods are assembled to form the class lastly.

The holistic generation strategy evaluates the model ability of handling long and complicated coding tasks all at once, while the incremental and compositional generation strategies focus on step-by-step class completion. The incremental strategy simulates progressive software development, where developers incrementally implement current methods based on existing ones. In constrast, the compositional strategy simulates real-world programming scenarios, where developers implement current methods based on other available method signatures.
The compositional generation strategy is not influenced by the hints (if the implemented methods are correct) or the misleading information (if the implemented methods are incorrect) since it does not use other method implementation as input.
Notably, both incremental and compositional generation strategies differ from standalone function-level code generation tasks in existing benchmarks like HumanEval, since our inputs include the class-level context such as the class constructor and other method signatures in the class skeleton.

### 4.3. Prompt Design

We then describe how we prompt LLMs to solve each class-level code generation task in ClassEval with each generation strategy.

LLMs with IF ability. Following the common practice of prompting LLMs with IF ability like WizardCoder*([luo2023wizardcoder,](#bib.bib8 "") )*, we set their prompts of two parts: (i) a system prompt as the beginning sentence to initialize the model, and followed by (ii) a task instruction to describe the goal of the task. Each generation strategy is set with its specific task instruction, i.e., Instruction-H for holistic generation, Instruction-I for incremental generation, and Instruction-C for a compositional generation. The prompt template is as follows, and each element is previously defined in Table[2](#S3.T2 "Table 2 ‣ 3.1. Benchmark Format ‣ 3. New benchmark ClassEval ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation").

System Prompt: Provided below is an instruction detailing a task. Compose a response that aptly fulfills the request.

Instruction-H: Please complete the class ${Class Name} in the subsequent code. ${Class Skeleton}

Instruction-I: Please complete the method ${Method Name} within the following class ${Class Name}. ${Class-level Info} ${Generated Methods with Contract Designs} ${Target Method Contract Design}

Instruction-C: Please complete the method ${Method Name} within the following class ${Class Name}. ${Class-level Info} ${Other Method Signatures} ${Target Method Contract Design}

LLMs without IF ability. The prompt of these models is the code context without any instruction: (i) for holistic generation, the prompt is just the class skeleton; (ii) for incremental generation, the prompt in each iteration includes the class-level information, generated methods, and the target method contract design; (iii) for compositional generation, the prompt for each method includes the class-level information, other method signatures, and the target method contract design.

### 4.4. Metrics

For correctness evaluation, we use the widely-used Pass@k *([chen2021pass_at_k,](#bib.bib18 "") )* metric, which calculates the percentage of problems solved based on $k$ code samples generated for each task:

| (1) |  | $\textbf{Pass@k}\=\underset{\text{Problems}}{\mathbb{E}}\left[1-{\binom{n-c}{k}}/{\binom{n}{k}}\right]$ |  |
| --- | --- | --- | --- |

In Eq.[1](#S4.E1 "In 4.4. Metrics ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), $n$ represents the total number of samples, $c$ denotes the number of correct samples, and $k$ stands for $k$ in $pass@k$. In particular, we calculate both class-level Pass@k and method-level Pass@k in class-level code generation tasks: class-level Pass@k considers code samples at the class granularity and method-level Pass@k consider code samples at the method granularity. A class-level code sample is deemed correct if it passes all the method-level and class-level test cases; and a method-level sample is deemed correct if it passes all the method-level test cases. In order to maintain an acceptable cost and response time in practical settings, we set $n$ to five. To address the challenge of high sampling variance, we employ an unbiased estimator in line with previous work*([chen2021huamneval,](#bib.bib15 "") )*.

In addition to code correctness, we further measure the model capability of generating code that is dependent to the contexts (i.e., invoking the other methods declared in the class or assessing the fields in the class). Such capability is essential in class-level code generation. To this end, we design the metric DEP, which calculates the percentage of dependencies generated per method compared to the actual number of dependencies in the canonical solution method. In particular, we consider method dependencies DEP(M) and field dependencies DEP(F):

| (2) |  | $\footnotesize\operatorname{\textbf{DEP}}(M)\=\frac{\sum_{i\=1}^{n}{G_{i}(M)}}{\sum_{i\=1}^{n}{S_{i}(M)}}$ |  |
| --- | --- | --- | --- |

| (3) |  | $\footnotesize\operatorname{\textbf{DEP}}(F)\=\frac{\sum_{i\=1}^{n}{G_{i}(F)}}{\sum_{i\=1}^{n}{S_{i}(F)}}$ |  |
| --- | --- | --- | --- |

$G_{i}(M/F)$ is the number of generated method/field dependencies in the $i^{th}$ method, and $S_{i}(M/F)$ is the number of actual method/field dependencies in the $i^{th}$ method of the canonical solution.

For each generation strategy, we employ nucleus sampling to generate 5 samples and calculate Pass@k metrics with $k\={{1,3,5}}$. In addition, we also use the greedy sampling strategy to generate one single greedy sample and calculate Pass@1 and DEP metrics. More sampling details are in Section[4.5](#S4.SS5 "4.5. Implementation Details ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation").

### 4.5. Implementation Details

We use the OpenAI API interface, specifically the “gpt-4” and “gpt-3.5-turbo” model interface*([api,](#bib.bib44 "") )*, in July 2023. For open-source LLMs, we directly obtain and run their released versions from their official repositories based on the documentation. The maximum window length is set to 2,048 tokens for all LLMs, determined by the smallest maximum window length among the studied LLMs.

In line with recent work*([yu2023codereval,](#bib.bib17 "") )*, we consider two sampling methods for code generation: (i) nucleus sampling*([DBLP:conf/iclr/nsample,](#bib.bib45 "") )*, where five solution code samples are randomly generated for each task with a temperature of 0.2*([chen2021huamneval,](#bib.bib15 "") )* and default top_p, and (ii) greedy sampling*([DBLP:journals/tsp/gsample,](#bib.bib46 "") )*, where only one single solution code sample is generated for each task using greedy decoding, i.e., setting the “do_sample” hyperparameter to false (temperature of 0). During each iteration in incremental and compositional generation, we obtain the Top-1 generated result for each method.
Our experiments are run on a computational infrastructure comprising eight A800-80G GPUs.

*Table 7. Pass@k with Nucleus Sampling on ClassEval*

ModelClass-levelMethod-levelPass@1Pass@3Pass@5Pass@1Pass@3Pass@5GPT-437.6%41.3%42.0%62.8%67.4%68.5%GPT-3.529.6%34.9%36.0%50.4%59.0%61.1%WizardCoder12.2%20.0%23.0%35.2%47.1%51.1%Instruct-StarCoder10.2%12.7%14.0%23.1%26.5%27.7%SantaCoder8.6%9.9%10.0%27.7%33.0%34.9%Instruct-CodeGen8.2%12.3%13.0%24.9%34.3%37.1%CodeGeeX7.2%9.4%10.0%21.2%27.1%29.5%InCoder6.2%7.6%8.0%21.1%26.5%29.1%Vicuna3.0%3.6%4.0%11.0%15.8%18.4%ChatGLM1.4%2.6%3.0%8.2%11.2%12.4%PolyCoder1.4%2.2%3.0%13.2%17.5%19.6%

5. Results
-----------

### 5.1. RQ1: Overall Correctness

Figure[5](#S5.F5 "Figure 5 ‣ 5.1. RQ1: Overall Correctness ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") shows the class-level and method-level Pass@1 with greedy sampling of studied LLMs on ClassEval and HumanEval. Due to space limits, we only present the best class-level Pass@1 (and corresponding method-level Pass@1) for each model among the three generation strategies. A detailed comparison among three generation strategies is discussed in Section[5.3](#S5.SS3 "5.3. RQ3: Dependency Generation ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"). Method-level Pass@1 results on HumanEval are directly adopted from the latest work*([luo2023wizardcoder,](#bib.bib8 "") )*, and ChatGLM results on HumanEval are absent from existing evaluation. Table[7](#S4.T7 "Table 7 ‣ 4.5. Implementation Details ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") presents the class-level and method-level Pass@k with nucleus sampling on ClassEval. Similarly, due to space limits, we only present results for the generation strategy with the highest class-level Pass@1. Based on Figure[5](#S5.F5 "Figure 5 ‣ 5.1. RQ1: Overall Correctness ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") and Table[7](#S4.T7 "Table 7 ‣ 4.5. Implementation Details ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), we have the following observations.

<img src='x5.png' alt='Refer to caption' title='' width='346' height='212' />

*Figure 5. Pass@1 (greedy) on ClassEval and HumanEval*

Class-level code generation v.s. Method-level code generation. Based on Figure[5](#S5.F5 "Figure 5 ‣ 5.1. RQ1: Overall Correctness ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), we observe a significant decrease in correctness for all studied models on our class-level benchmark ClassEval compared to the existing method-level benchmark HumanEval.
In particular, the best-performing models GPT-4 and GPT-3.5 achieve 85.4%/68.9% correctness on method-level tasks in HumanEval, but only 37.0%/27.0% correctness on class-level tasks in ClassEval. Similar trends can be observed on other models, e.g., WizardCoder correctly generates 59.8% methods on HumanEval, but only 11.0% correct classes in our benchmark.
Despite the inherent challenges of generating a class with multiple methods, the observed decrease in correctness on our benchmark ClassEval is not solely due to the larger number of methods to generate. The code generated by all models also shows lower method-level correctness on ClassEval compared to HumanEval.
For instance, the method-level Pass@1 of GPT-4 and GPT-3.5 drops from 85.4%/68.9% (on HumanEval) to 62.5%/52.5% (on ClassEval). This drop could be attributed to the complexity of generating code that depends on other context, which is known to be more challenging than generating standalone code. This finding is consistent with recent work*([yu2023codereval,](#bib.bib17 "") )*. In summary, our results show that existing LLMs still have limited performance in solving complicated coding tasks, such as class-level code generation.

In addition, we observe that the model performance in the standalone method-level code generation tasks does not necessarily reflect their capability of class-level code generation. For example, while WizardCoder and Instruct-StarCoder exhibit much higher method-level Pass@1 (59.8.4% and 34.1%) compared to SantaCoder (14.6%) on HumanEval, all three model exhibit similar performance on class-level code generation tasks in ClassEval (around 10% - 11% Pass@1). This indicates that the method-level coding ability cannot equivalently represent the class-level coding ability among LLMs, further confirming the necessity of building a class-level code generation benchmark.

Finding 1: Existing LLMs demonstrate substantially lower performance on class-level code generation tasks compared to standalone method-level code generation tasks. Additionally, the method-level coding ability cannot equivalently represent the class-level coding ability among LLMs. These findings strongly confirm the motivation and necessity of constructing class-level code generation benchmarks.

Comparison among models. As shown in Figure[5](#S5.F5 "Figure 5 ‣ 5.1. RQ1: Overall Correctness ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") and Table[7](#S4.T7 "Table 7 ‣ 4.5. Implementation Details ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), GPT series (GPT-4 and GPT-3.5) substantially outperform all the other models on solving class-level coding tasks with both greedy sampling and nucleus sampling. For example, in Table[7](#S4.T7 "Table 7 ‣ 4.5. Implementation Details ‣ 4. Empirical Study ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), they outperform the third-ranked model WizardCoder by 25.4% and 17.4% in class-level Pass@1 with nucleus sampling. Such results indicate the relatively stable dominance of GPT models when generalized to solve more challenging class-level coding tasks.

The second-ranked tier includes larger code models like Instruct-StarCoder, Instruct-CodeGen, and WizardCoder, achieving similar Pass@1 with greedy sampling ranging from 10.0% - 11.1%. Notably, while these models show significant performance differences on method-level coding tasks in HumanEval (WizardCoder outperforms Instruct-CodeGen by 27.5% on HumanEval), they perform similarly on class-level coding tasks. Smaller models (e.g., PolyCoder) or general models (e.g., ChatGLM) often exhibit worse performance, as expected due to the importance of model size and instruction datasets for generalization. The only exception is SantaCoder, which achieves comparable performance to larger code models (Instruct-StarCoder, WizardCoder, and Instruct-CodeGen) with a much smaller model size.

Finding 2: On class-level code generation, GPT-4/GPT-3.5 still exhibits dominate superior than other LLMs; Instruct-StarCoder, Instruct-CodeGen, and WizardCoder perform similarly as the second tier; small or general models often perform the worse, except SantaCoder, which achieves comparable performance to larger models but with much less parameters.

<img src='x6.png' alt='Refer to caption' title='' width='106' height='79' />

*(a) Class-level Pass@5*

<img src='x7.png' alt='Refer to caption' title='' width='106' height='79' />

*(b) Method-level Pass@5*

*Figure 6. Pass@5 of Three Generation Strategies*

### 5.2. RQ2: Generation Strategies

Figure[6](#S5.F6 "Figure 6 ‣ 5.1. RQ1: Overall Correctness ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") compares the class-level Pass@5 and method-level Pass@5 of three different generation strategies (i.e., holistic, incremental, and compositional generation). Based on the figure, overall, we find that the best generation strategy varies among different LLMs.

Holistic strategy v.s. others. On one hand, holistic generation is the best generation strategy only for the two models GPT-4 and GPT-3.5, which achieves much higher class-level Pass@5 than the other two strategies (i.e., the improvements range from 6% to 9% for GPT-4 and 4% to 14% for GPT-3.5). In addition, even for the method-level correctness, holistic generation still outperforms generating method in an incremental or compositional way (i.e., 1.4% - 9.0% improvement in method-level Pass@5). On the other hand, the trends are different for the other models, which actually perform much better when generating the class method by method, namely with the incremental or compositional strategies. For example, in terms of the class-level correctness, CodeGeeX and SantaCoder generate 9% and 7% more correct classes with the incremental strategy compared to the holistic generation strategy. The main reason is that these models are able to generate much more correct methods (i.e., 27.9% and 19.2% higher method-level Pass@5) when generating each method in separate iterations
compared to generating all methods at once. Therefore, these models have higher chance to generate more correct classes if they are able to generate more correct methods with the incremental or compositional strategy.

One potential reason for the observation above might be that most models (except GPT ones), exhibit rather limited capability of utilizing long input contexts, thus finding it more challenging to fully understand the code generation tasks given the entire class skeleton. As revealed by the recent work*([DBLP:journals/corr/abs-2307-03172,](#bib.bib47 "") )*, LLMs often become substantially less effective with the increasing length of inputs; and in particular they tend to make better usage of the information located at the beginning or end of the inputs than that in the middle of inputs. Therefore, most existing LLMs perform better in generating a class method by method, since the task inputs are with the more atomic focus in such an incremental or compositional generation scenario; for models like GPT-3.5 and GPT-4 with a better understanding of long instructions, feeding the class-level context all at once is actually beneficial for them to fully capture and utilize the constraints between each method, leading to better class-level code correctness.

Incremental strategy v.s. compositional strategy. As for the two method-by-method strategies (i.e., incremental and compositional strategies), we find the studied models actually have different preference on them. In particular, compared to the compositional generation manner, the additional inputs (the method body generated in previous iterations) in the incremental strategy are helpful for some models such as Instruct-CodeGen, InCoder, CodeGeeX, and SantaCoder. In contrast, the previously-generated method bodies can negatively affect the performance of models like Instruct-StarCoder and WizardCoder, resulting in a lower class-level correctness in incremental generation. In addition to the limited capability of handling long inputs mentioned above, another potential reason for the model’s preference on a rather individual generation manner might be that the compositional generation aligns better with simple and atomic task instructions during instruction tuning.

Finding 3: Generating the entire class all at once (i.e., holistic strategy) is the best generation strategy only for GPT-4 and GPT-3.5. For the other models, method-by-method generation (i.e., incremental and compositional) works better. Such a disparity might stem from their limited capability of understanding the long instructions and utilizing the middle information.

### 5.3. RQ3: Dependency Generation

Method dependency v.s. Field dependency. Figure[7](#S5.F7 "Figure 7 ‣ 5.3. RQ3: Dependency Generation ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") presents the average field dependencies DEP(F) and the method dependencies DEP(M) of each model with the nucleus sampling. For space limits, we only present the best results among three generation strategies. Based on Figure[7](#S5.F7 "Figure 7 ‣ 5.3. RQ3: Dependency Generation ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), we can find that all models exhibit a much higher success rate in generating code dependent to fields than generating code dependent to other methods (i.e., higher DEP(F) than DEP(M) on all the models). In other words, it might be much easier for models to generate field-accessing code than method-invoking code. In addition, among all the models, GPT models still show consistent superior in generating dependent code, e.g., GPT-4 substantially outperform other LLMs by at least 12.6%/6.3% improvement in DEP(F)/DEP(M).

<img src='x8.png' alt='Refer to caption' title='' width='300' height='225' />

*Figure 7. DEP(F) and DEP(M) in Nucleus Sampling*

Given our observation above that it is more challenging to generate method dependency, we further investigate how each model performs at correctly generating code that invokes different number of other methods. Figure [8](#S5.F8 "Figure 8 ‣ 5.3. RQ3: Dependency Generation ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") is a stacked-bar plot that show the ratio of correctly-generated methods to all methods with the given number (i.e., 0, 1, 2) of method dependencies (based on the canonical solution). Based on the figure, we can find that all the models perform best when generating methods that do not invoke any other method declared in the class (the blue bar in the figure). In addition, we find that no obvious difference when most models generate code invoking one other method (the green bar) or invoking two other methods (the yellow bar). In particular, for all the models, the average ratio of correctly-generated code that invokes one or two method(s) is 27.7% and 27.6% respectively.

<img src='x9.png' alt='Refer to caption' title='' width='322' height='236' />

*Figure 8. Distribution of correctly-generated methods in increasing method dependencies*

Finding 4: It is easier for all the models to generate field-accessing code than method-invoking code. Additionally, they are better at generating standalone methods that do no invoke any other method.

### 5.4. RQ4: Bad Case Analysis

We further analyze the incorrectly-generated classes. To this end, we automatically parse the error logs generated during interpretation and execution, and present the error distribution of all models in Figure[10](#S5.F10 "Figure 10 ‣ 5.4. RQ4: Bad Case Analysis ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"). In particular, we find that most incorrect code encounters AttributeError and TypeError, indicating the limited model ability of understanding and satisfying syntactic or semantic constraints in the code context. Additionally, a few cases encounter KeyError due to erroneous operations on the dictionary variable. Figure[10](#S5.F10 "Figure 10 ‣ 5.4. RQ4: Bad Case Analysis ‣ 5. Results ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation") shows such an example from GPT-3.5, resulting from a misinterpretation of the field dependency. Specifically, the model erroneously accesses the first element of the field BMI_std list, which is a dictionary with the key “male”. Attempting to access the key self.sex as “female” within this dictionary triggers a KeyError. This case indicate one of the challenges that LLMs might encounter in handling inherent class-level dependencies.

<img src='x10.png' alt='Refer to caption' title='' width='461' height='388' />

*Figure 9. Error Distribution*

<img src='x11.png' alt='Refer to caption' title='' width='461' height='431' />

*Figure 10. KeyError Code Example*

Finding 5: The classes generated by LLMs suffer from AttributeError and TypeError most frequently. In addition, the models might encounter difficulties in understanding the dependent contexts in the class.

6. Threats to Validity
-----------------------

Threats in benchmark construction. One potential threat is the data leakage between our benchmark and model training data, thus we manually construct the benchmark ClassEval. We also involve multiple participants to mitigate the subjectiveness and mistakes in manual participation. Another threat lies in the limited size and programming languages in our current benchmark, which cannot guarantee the generalizability of our findings, and we plan to continually extend our benchmark in the future. Threats in empirical study. To avoid buggy model implementation, we adopt the public versions following official guidelines of each model. Another threat lies in the prompts used in our experiments, which might impact our findings. To avoid underestimating studied models, we perform a pilot study on a small set of prompt candidates and select the one with the best performance on three separate class-level coding tasks.
We also report the results with greedy decoding, which is deterministic, so as to mitigate the randomness in model responses.

7. Related Work
----------------

Since we have discussed most relevant work on LLMs and existing code generation benchmarks in Section[2](#S2 "2. Background ‣ ClassEval: A Manually-Crafted Benchmark for Evaluating LLMs on Class-level Code Generation"), we mainly introduce related work on LLM evaluation in this section.
Multi-faceted evaluation for LLMs is crucial for understanding the model capabilities given the black-box nature of LLMs. To date, the evaluation for LLMs has covered a wide range*([chang2023survey,](#bib.bib20 "") )*, encompassing not only traditional NLP tasks (e.g., sentiment analysis*([bang2023multitask,](#bib.bib48 "") )*, question answering*([bai2023benchmarking,](#bib.bib49 "") )*, and reasoning*([bian2023chatgpt,](#bib.bib5 "") )*) but also some specific downstream domains (e.g., medicine*([chervenak2023promise,](#bib.bib50 "") )*, agent*([huang2023language,](#bib.bib51 "") )*, and recommendation system*([fan2023recommender,](#bib.bib52 "") )*). Specifically in software engineering domain, current evaluation focuses primarily on code generation tasks*([chen2021huamneval,](#bib.bib15 "") ; [austin2021mbpp,](#bib.bib16 "") ; [li2023enabling,](#bib.bib53 "") ; [liu2023humanevalplus,](#bib.bib31 "") )*.
Many code LLMs (e.g., Codex*([chen2021huamneval,](#bib.bib15 "") )* and PanGu-Coder2*([shen2023pangucoder2,](#bib.bib21 "") )*) are released along with its rigorous evaluation on HumanEval to demonstrate their capabilities on code generation.
While these previous efforts do not take scenarios beyond function-level code generation into account, our work fills this gap by manually constructing the first class-level code generation benchmark for evaluating LLM on more complicated and practical software development tasks.

8. Conclusion
--------------

This work makes the first attempt to evaluate LLMs on class-level code generation. We first manually construct the first class-level code generation benchmark ClassEval and perform the first study of 11 state-of-the-art LLMs on class-level code generation. We find that all LLMs perform much worse on class-level code generation compared to the method-level. While GPT models still dominate other LLMs on class-level code generation, the ranking of model performance on method-level code generation no longer holds in the class-level code generation. Besides, most models (except GPT models) perform better when generating the class method by method; and they have the limited ability of generating dependent code.

References
----------

* (1)V. Vikram, C. Lemieux, and R. Padhye, “Can large language models write good
property-based tests?” 2023.
* (2)S. Kang, J. Yoon, and S. Yoo, “Large language models are few-shot testers:
Exploring llm-based general bug reproduction,” in *2023 IEEE/ACM 45th
International Conference on Software Engineering (ICSE)*, 2023, pp.
2312–2323.
* (3)S. Kang, B. Chen, S. Yoo, and J.-G. Lou, “Explainable automated debugging via
large language model-driven scientific debugging,” 2023.
* (4)OpenAI, “GPT-4 technical report,” *CoRR*, vol. abs/2303.08774, 2023.
[Online]. Available: [https://doi.org/10.48550/arXiv.2303.08774](https://doi.org/10.48550/arXiv.2303.08774 "")
* (5)N. Bian, X. Han, L. Sun, H. Lin, Y. Lu, and B. He, “Chatgpt is a knowledgeable
but inexperienced solver: An investigation of commonsense problem in large
language models,” *CoRR*, vol. abs/2303.16421, 2023. [Online].
Available: [https://doi.org/10.48550/arXiv.2303.16421](https://doi.org/10.48550/arXiv.2303.16421 "")
* (6)L. Zheng, W. Chiang, Y. Sheng, S. Zhuang, Z. Wu, Y. Zhuang, Z. Lin, Z. Li,
D. Li, E. P. Xing, H. Zhang, J. E. Gonzalez, and I. Stoica, “Judging
llm-as-a-judge with mt-bench and chatbot arena,” *CoRR*, vol.
abs/2306.05685, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2306.05685](https://doi.org/10.48550/arXiv.2306.05685 "")
* (7)Z. Du, Y. Qian, X. Liu, M. Ding, J. Qiu, Z. Yang, and J. Tang, “GLM: general
language model pretraining with autoregressive blank infilling,” in*Proceedings of the 60th Annual Meeting of the Association for
Computational Linguistics (Volume 1: Long Papers), ACL 2022, Dublin,
Ireland, May 22-27, 2022*, S. Muresan, P. Nakov, and A. Villavicencio,
Eds. Association for Computational
Linguistics, 2022, pp. 320–335. [Online]. Available:[https://doi.org/10.18653/v1/2022.acl-long.26](https://doi.org/10.18653/v1/2022.acl-long.26 "")
* (8)Z. Luo, C. Xu, P. Zhao, Q. Sun, X. Geng, W. Hu, C. Tao, J. Ma, Q. Lin, and
D. Jiang, “Wizardcoder: Empowering code large language models with
evol-instruct,” *CoRR*, vol. abs/2306.08568, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2306.08568](https://doi.org/10.48550/arXiv.2306.08568 "")
* (9)R. Li, L. B. Allal, Y. Zi, N. Muennighoff, D. Kocetkov, C. Mou, M. Marone,
C. Akiki, J. Li, J. Chim, Q. Liu, E. Zheltonozhskii, T. Y. Zhuo, T. Wang,
O. Dehaene, M. Davaadorj, J. Lamy-Poirier, J. Monteiro, O. Shliazhko,
N. Gontier, N. Meade, A. Zebaze, M. Yee, L. K. Umapathi, J. Zhu, B. Lipkin,
M. Oblokulov, Z. Wang, R. M. V, J. Stillerman, S. S. Patel, D. Abulkhanov,
M. Zocca, M. Dey, Z. Zhang, N. Moustafa-Fahmy, U. Bhattacharyya, W. Yu,
S. Singh, S. Luccioni, P. Villegas, M. Kunakov, F. Zhdanov, M. Romero,
T. Lee, N. Timor, J. Ding, C. Schlesinger, H. Schoelkopf, J. Ebert, T. Dao,
M. Mishra, A. Gu, J. Robinson, C. J. Anderson, B. Dolan-Gavitt,
D. Contractor, S. Reddy, D. Fried, D. Bahdanau, Y. Jernite, C. M. Ferrandis,
S. Hughes, T. Wolf, A. Guha, L. von Werra, and H. de Vries, “Starcoder: may
the source be with you!” *CoRR*, vol. abs/2305.06161, 2023. [Online].
Available: [https://doi.org/10.48550/arXiv.2305.06161](https://doi.org/10.48550/arXiv.2305.06161 "")
* (10)(2023) Instruct-codegen. [Online]. Available:[https://huggingface.co/sahil2801/instruct-codegen-16B](https://huggingface.co/sahil2801/instruct-codegen-16B "")
* (11)Q. Zheng, X. Xia, X. Zou, Y. Dong, S. Wang, Y. Xue, Z. Wang, L. Shen, A. Wang,
Y. Li, T. Su, Z. Yang, and J. Tang, “Codegeex: A pre-trained model for
code generation with multilingual evaluations on humaneval-x,” *CoRR*,
vol. abs/2303.17568, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2303.17568](https://doi.org/10.48550/arXiv.2303.17568 "")
* (12)F. F. Xu, U. Alon, G. Neubig, and V. J. Hellendoorn, “A systematic evaluation
of large language models of code,” in *MAPS@PLDI 2022: 6th ACM
SIGPLAN International Symposium on Machine Programming, San Diego, CA, USA,
13 June 2022*, S. Chaudhuri and C. Sutton, Eds. ACM, 2022, pp. 1–10. [Online]. Available:[https://doi.org/10.1145/3520312.3534862](https://doi.org/10.1145/3520312.3534862 "")
* (13)D. Fried, A. Aghajanyan, J. Lin, S. Wang, E. Wallace, F. Shi, R. Zhong, S. Yih,
L. Zettlemoyer, and M. Lewis, “Incoder: A generative model for code
infilling and synthesis,” in *The Eleventh International Conference on
Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. [Online]. Available:[https://openreview.net/pdf?id\=hQwb-lbM6EL](https://openreview.net/pdf?id=hQwb-lbM6EL "")
* (14)L. B. Allal, R. Li, D. Kocetkov, C. Mou, C. Akiki, C. M. Ferrandis,
N. Muennighoff, M. Mishra, A. Gu, M. Dey, L. K. Umapathi, C. J. Anderson,
Y. Zi, J. Lamy-Poirier, H. Schoelkopf, S. Troshin, D. Abulkhanov,
M. Romero, M. Lappert, F. D. Toni, B. G. del Río, Q. Liu, S. Bose,
U. Bhattacharyya, T. Y. Zhuo, I. Yu, P. Villegas, M. Zocca, S. Mangrulkar,
D. Lansky, H. Nguyen, D. Contractor, L. Villa, J. Li, D. Bahdanau,
Y. Jernite, S. Hughes, D. Fried, A. Guha, H. de Vries, and L. von Werra,
“Santacoder: don’t reach for the stars!” *CoRR*, vol. abs/2301.03988,
2023. [Online]. Available: [https://doi.org/10.48550/arXiv.2301.03988](https://doi.org/10.48550/arXiv.2301.03988 "")
* (15)M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan,
H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger,
M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder,
M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P.
Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss,
W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji,
S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra,
E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer,
P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and
W. Zaremba, “Evaluating large language models trained on code,”*CoRR*, vol. abs/2107.03374, 2021. [Online]. Available:[https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374 "")
* (16)J. Austin, A. Odena, M. I. Nye, M. Bosma, H. Michalewski, D. Dohan, E. Jiang,
C. J. Cai, M. Terry, Q. V. Le, and C. Sutton, “Program synthesis with large
language models,” *CoRR*, vol. abs/2108.07732, 2021. [Online].
Available: [https://arxiv.org/abs/2108.07732](https://arxiv.org/abs/2108.07732 "")
* (17)H. Yu, B. Shen, D. Ran, J. Zhang, Q. Zhang, Y. Ma, G. Liang, Y. Li, T. Xie, and
Q. Wang, “Codereval: A benchmark of pragmatic code generation with
generative pre-trained models,” *CoRR*, vol. abs/2302.00288, 2023.
[Online]. Available: [https://doi.org/10.48550/arXiv.2302.00288](https://doi.org/10.48550/arXiv.2302.00288 "")
* (18)M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. de Oliveira Pinto, J. Kaplan,
H. Edwards, Y. Burda, N. Joseph, G. Brockman, A. Ray, R. Puri, G. Krueger,
M. Petrov, H. Khlaaf, G. Sastry, P. Mishkin, B. Chan, S. Gray, N. Ryder,
M. Pavlov, A. Power, L. Kaiser, M. Bavarian, C. Winter, P. Tillet, F. P.
Such, D. Cummings, M. Plappert, F. Chantzis, E. Barnes, A. Herbert-Voss,
W. H. Guss, A. Nichol, A. Paino, N. Tezak, J. Tang, I. Babuschkin, S. Balaji,
S. Jain, W. Saunders, C. Hesse, A. N. Carr, J. Leike, J. Achiam, V. Misra,
E. Morikawa, A. Radford, M. Knight, M. Brundage, M. Murati, K. Mayer,
P. Welinder, B. McGrew, D. Amodei, S. McCandlish, I. Sutskever, and
W. Zaremba, “Evaluating large language models trained on code,”*CoRR*, vol. abs/2107.03374, 2021. [Online]. Available:[https://arxiv.org/abs/2107.03374](https://arxiv.org/abs/2107.03374 "")
* (19)Classeval github. [Online]. Available:<https://github.com/FudanSELab/ClassEval>
* (20)Y. Chang, X. Wang, J. Wang, Y. Wu, K. Zhu, H. Chen, L. Yang, X. Yi, C. Wang,
Y. Wang, W. Ye, Y. Zhang, Y. Chang, P. S. Yu, Q. Yang, and X. Xie, “A survey
on evaluation of large language models,” *CoRR*, vol. abs/2307.03109,
2023. [Online]. Available: [https://doi.org/10.48550/arXiv.2307.03109](https://doi.org/10.48550/arXiv.2307.03109 "")
* (21)B. Shen, J. Zhang, T. Chen, D. Zan, B. Geng, A. Fu, M. Zeng, A. Yu, J. Ji,
J. Zhao, Y. Guo, and Q. Wang, “Pangu-coder2: Boosting large language models
for code with ranking feedback,” 2023.
* (22)F. Christopoulou, G. Lampouras, M. Gritta, G. Zhang, Y. Guo, Z. Li, Q. Zhang,
M. Xiao, B. Shen, L. Li, H. Yu, L. Yan, P. Zhou, X. Wang, Y. Ma,
I. Iacobacci, Y. Wang, G. Liang, J. Wei, X. Jiang, Q. Wang, and Q. Liu,
“Pangu-coder: Program synthesis with function-level language modeling,”*CoRR*, vol. abs/2207.11280, 2022. [Online]. Available:[https://doi.org/10.48550/arXiv.2207.11280](https://doi.org/10.48550/arXiv.2207.11280 "")
* (23)D. Zan, B. Chen, F. Zhang, D. Lu, B. Wu, B. Guan, W. Yongji, and J.-G. Lou,
“Large language models meet NL2Code: A survey,” in *Proceedings of
the 61st Annual Meeting of the Association for Computational Linguistics
(Volume 1: Long Papers)*. Toronto,
Canada: Association for Computational Linguistics, Jul. 2023, pp. 7443–7464.
[Online]. Available: [https://aclanthology.org/2023.acl-long.411](https://aclanthology.org/2023.acl-long.411 "")
* (24)(2023) Instruct-starcoder. [Online]. Available:[https://huggingface.co/GeorgiaTechResearchInstitute/starcoder-gpteacher-code-instruct](https://huggingface.co/GeorgiaTechResearchInstitute/starcoder-gpteacher-code-instruct "")
* (25)S. Iyer, I. Konstas, A. Cheung, and L. Zettlemoyer, “Mapping language to code
in programmatic context,” in *Proceedings of the 2018 Conference on
Empirical Methods in Natural Language Processing, Brussels, Belgium, October
31 - November 4, 2018*, E. Riloff, D. Chiang, J. Hockenmaier, and J. Tsujii,
Eds. Association for Computational
Linguistics, 2018, pp. 1643–1652. [Online]. Available:[https://doi.org/10.18653/v1/d18-1192](https://doi.org/10.18653/v1/d18-1192 "")
* (26)P. Yin, B. Deng, E. Chen, B. Vasilescu, and G. Neubig, “Learning to mine
aligned code and natural language pairs from stack overflow,” in*Proceedings of the 15th International Conference on Mining Software
Repositories, MSR 2018, Gothenburg, Sweden, May 28-29, 2018*, A. Zaidman,
Y. Kamei, and E. Hill, Eds. ACM,
2018, pp. 476–486. [Online]. Available:[https://doi.org/10.1145/3196398.3196408](https://doi.org/10.1145/3196398.3196408 "")
* (27)D. Hendrycks, S. Basart, S. Kadavath, M. Mazeika, A. Arora, E. Guo, C. Burns,
S. Puranik, H. He, D. Song, and J. Steinhardt, “Measuring coding challenge
competence with APPS,” in *Proceedings of the Neural Information
Processing Systems Track on Datasets and Benchmarks 1, NeurIPS Datasets and
Benchmarks 2021, December 2021, virtual*, J. Vanschoren and S. Yeung, Eds.,
2021. [Online]. Available:[https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html](https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/c24cd76e1ce41366a4bbe8a49b02a028-Abstract-round2.html "")
* (28)B. Athiwaratkun, S. K. Gouda, Z. Wang, X. Li, Y. Tian, M. Tan, W. U. Ahmad,
S. Wang, Q. Sun, M. Shang, S. K. Gonugondla, H. Ding, V. Kumar, N. Fulton,
A. Farahani, S. Jain, R. Giaquinto, H. Qian, M. K. Ramanathan, and
R. Nallapati, “Multi-lingual evaluation of code generation models,” in*The Eleventh International Conference on Learning Representations,
ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. [Online]. Available:[https://openreview.net/pdf?id\=Bo7eeXm6An8](https://openreview.net/pdf?id=Bo7eeXm6An8 "")
* (29)Y. Li, D. H. Choi, J. Chung, N. Kushman, J. Schrittwieser, R. Leblond,
T. Eccles, J. Keeling, F. Gimeno, A. D. Lago, T. Hubert, P. Choy,
C. de Masson d’Autume, I. Babuschkin, X. Chen, P. Huang, J. Welbl, S. Gowal,
A. Cherepanov, J. Molloy, D. J. Mankowitz, E. S. Robson, P. Kohli,
N. de Freitas, K. Kavukcuoglu, and O. Vinyals, “Competition-level code
generation with alphacode,” *CoRR*, vol. abs/2203.07814, 2022.
[Online]. Available: [https://doi.org/10.48550/arXiv.2203.07814](https://doi.org/10.48550/arXiv.2203.07814 "")
* (30)Y. Lai, C. Li, Y. Wang, T. Zhang, R. Zhong, L. Zettlemoyer, S. W. Yih,
D. Fried, S. I. Wang, and T. Yu, “DS-1000: A natural and reliable
benchmark for data science code generation,” *CoRR*, vol.
abs/2211.11501, 2022. [Online]. Available:[https://doi.org/10.48550/arXiv.2211.11501](https://doi.org/10.48550/arXiv.2211.11501 "")
* (31)J. Liu, C. S. Xia, Y. Wang, and L. Zhang, “Is your code generated by chatgpt
really correct? rigorous evaluation of large language models for code
generation,” *CoRR*, vol. abs/2305.01210, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2305.01210](https://doi.org/10.48550/arXiv.2305.01210 "")
* (32)Code generation datasets in huggingface. [Online]. Available:[https://hf.co/datasets?other\=code-generation](https://hf.co/datasets?other=code-generation "")
* (33)B. Meyer, “Applying ”design by contract”,” *Computer*, vol. 25, no. 10,
pp. 40–51, 1992. [Online]. Available: [https://doi.org/10.1109/2.161279](https://doi.org/10.1109/2.161279 "")
* (34)T. Bhat and N. Nagappan, “Evaluating the efficacy of test-driven development:
industrial case studies,” in *2006 International Symposium on Empirical
Software Engineering (ISESE 2006), September 21-22, 2006, Rio de Janeiro,
Brazil*, G. H. Travassos, J. C. Maldonado, and C. Wohlin, Eds. ACM, 2006, pp. 356–363. [Online]. Available:[https://doi.org/10.1145/1159733.1159787](https://doi.org/10.1145/1159733.1159787 "")
* (35)K. Srinath, “Python–the fastest growing programming language,”*International Research Journal of Engineering and Technology*, vol. 4,
no. 12, pp. 354–357, 2017.
* (36)Pypi. [Online]. Available: <https://pypi.org/search>
* (37)Unittest framework. [Online]. Available: <https://pypi.org/project/unitest>
* (38)Coverage library. [Online]. Available: <https://pypi.org/project/coverage>
* (39)E. Nijkamp, B. Pang, H. Hayashi, L. Tu, H. Wang, Y. Zhou, S. Savarese, and
C. Xiong, “A conversational paradigm for program synthesis,” *CoRR*,
vol. abs/2203.13474, 2022. [Online]. Available:[https://doi.org/10.48550/arXiv.2203.13474](https://doi.org/10.48550/arXiv.2203.13474 "")
* (40)(2021) Dense-6.7b. [Online]. Available:[https://huggingface.co/KoboldAI/fairseq-dense-6.7B-Shinen](https://huggingface.co/KoboldAI/fairseq-dense-6.7B-Shinen "")
* (41)A. Radford, J. Wu, R. Child, D. Luan, D. Amodei, I. Sutskever *et al.*,
“Language models are unsupervised multitask learners,” *OpenAI blog*,
vol. 1, no. 8, p. 9, 2019.
* (42)H. Touvron, T. Lavril, G. Izacard, X. Martinet, M. Lachaux, T. Lacroix,
B. Rozière, N. Goyal, E. Hambro, F. Azhar, A. Rodriguez, A. Joulin,
E. Grave, and G. Lample, “Llama: Open and efficient foundation language
models,” *CoRR*, vol. abs/2302.13971, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2302.13971](https://doi.org/10.48550/arXiv.2302.13971 "")
* (43)A. Zeng, X. Liu, Z. Du, Z. Wang, H. Lai, M. Ding, Z. Yang, Y. Xu, W. Zheng,
X. Xia, W. L. Tam, Z. Ma, Y. Xue, J. Zhai, W. Chen, Z. Liu, P. Zhang,
Y. Dong, and J. Tang, “GLM-130B: an open bilingual pre-trained model,” in*The Eleventh International Conference on Learning Representations,
ICLR 2023, Kigali, Rwanda, May 1-5, 2023*. OpenReview.net, 2023. [Online]. Available:[https://openreview.net/pdf?id\=-Aw0rrrPUF](https://openreview.net/pdf?id=-Aw0rrrPUF "")
* (44)Openai api interface. [Online]. Available:[https://platform.openai.com/docs/api-reference](https://platform.openai.com/docs/api-reference "")
* (45)A. Holtzman, J. Buys, L. Du, M. Forbes, and Y. Choi, “The curious case of
neural text degeneration,” in *8th International Conference on Learning
Representations, ICLR 2020, Addis Ababa, Ethiopia, April 26-30,
2020*. OpenReview.net, 2020. [Online].
Available: [https://openreview.net/forum?id\=rygGQyrFvH](https://openreview.net/forum?id=rygGQyrFvH "")
* (46)S. Chen, R. Varma, A. Sandryhaila, and J. Kovacevic, “Discrete signal
processing on graphs: Sampling theory,” *IEEE Trans. Signal
Process.*, vol. 63, no. 24, pp. 6510–6523, 2015. [Online]. Available:[https://doi.org/10.1109/TSP.2015.2469645](https://doi.org/10.1109/TSP.2015.2469645 "")
* (47)N. F. Liu, K. Lin, J. Hewitt, A. Paranjape, M. Bevilacqua, F. Petroni, and
P. Liang, “Lost in the middle: How language models use long contexts,”*CoRR*, vol. abs/2307.03172, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2307.03172](https://doi.org/10.48550/arXiv.2307.03172 "")
* (48)Y. Bang, S. Cahyawijaya, N. Lee, W. Dai, D. Su, B. Wilie, H. Lovenia, Z. Ji,
T. Yu, W. Chung, Q. V. Do, Y. Xu, and P. Fung, “A multitask, multilingual,
multimodal evaluation of chatgpt on reasoning, hallucination, and
interactivity,” *CoRR*, vol. abs/2302.04023, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2302.04023](https://doi.org/10.48550/arXiv.2302.04023 "")
* (49)Y. Bai, J. Ying, Y. Cao, X. Lv, Y. He, X. Wang, J. Yu, K. Zeng, Y. Xiao,
H. Lyu, J. Zhang, J. Li, and L. Hou, “Benchmarking foundation models with
language-model-as-an-examiner,” *CoRR*, vol. abs/2306.04181, 2023.
[Online]. Available: [https://doi.org/10.48550/arXiv.2306.04181](https://doi.org/10.48550/arXiv.2306.04181 "")
* (50)J. Chervenak, H. Lieman, M. Blanco-Breindel, and S. Jindal, “The promise and
peril of using a large language model to obtain clinical information: Chatgpt
performs strongly as a fertility counseling tool with limitations,”*Fertility and Sterility*, 2023.
* (51)S. Huang, L. Dong, W. Wang, Y. Hao, S. Singhal, S. Ma, T. Lv, L. Cui, O. K.
Mohammed, B. Patra, Q. Liu, K. Aggarwal, Z. Chi, J. Bjorck, V. Chaudhary,
S. Som, X. Song, and F. Wei, “Language is not all you need: Aligning
perception with language models,” *CoRR*, vol. abs/2302.14045, 2023.
[Online]. Available: [https://doi.org/10.48550/arXiv.2302.14045](https://doi.org/10.48550/arXiv.2302.14045 "")
* (52)W. Fan, Z. Zhao, J. Li, Y. Liu, X. Mei, Y. Wang, J. Tang, and Q. Li,
“Recommender systems in the era of large language models (llms),”*CoRR*, vol. abs/2307.02046, 2023. [Online]. Available:[https://doi.org/10.48550/arXiv.2307.02046](https://doi.org/10.48550/arXiv.2307.02046 "")
* (53)J. Li, G. Li, Y. Li, and Z. Jin, “Enabling programming thinking in large
language models toward code generation,” *CoRR*, vol. abs/2305.06599,
2023. [Online]. Available: [https://doi.org/10.48550/arXiv.2305.06599](https://doi.org/10.48550/arXiv.2305.06599 "")
