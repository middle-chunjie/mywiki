Code Search based on Context-aware Code Translation
====================================================

Weisong Sun[weisongsun@smail.nju.edu.cn](mailto:weisongsun@smail.nju.edu.cn)State Key Laboratory for Novel Software TechnologyNanjing UniversityChina,Chunrong Fang[fangchunrong@nju.edu.cn](mailto:fangchunrong@nju.edu.cn)State Key Laboratory for Novel Software TechnologyNanjing UniversityChina,Yuchen Chen[yuc.chen@outlook.com](mailto:yuc.chen@outlook.com)State Key Laboratory for Novel Software TechnologyNanjing UniversityChina,Guanhong Tao[taog@purdue.edu](mailto:taog@purdue.edu)Purdue UniversityWest LafayetteIndianaUSA,Tingxu Han[hantingxv@163.com](mailto:hantingxv@163.com)School of Information ManagementNanjing UniversityChinaandQuanjun Zhang[quanjun.zhang@smail.nju.edu.cn](mailto:quanjun.zhang@smail.nju.edu.cn)State Key Laboratory for Novel Software TechnologyNanjing UniversityChina

(2022)

###### Abstract.

Code search is a widely used technique by developers during software development.
It provides semantically similar implementations from a large code corpus to developers based on their queries.
Existing techniques leverage deep learning models to construct embedding representations for code snippets and queries, respectively.
Features such as abstract syntactic trees, control flow graphs, etc., are commonly employed for representing the semantics of code snippets.
However, the same structure of these features does not necessarily denote the same semantics of code snippets, and vice versa.
In addition, these techniques utilize multiple different word mapping functions that map query words/code tokens to embedding representations.
This causes diverged embeddings of the same word/token in queries and code snippets.
We propose a novel context-aware code translation technique that translates code snippets into natural language descriptions (called translations).
The code translation is conducted on machine instructions, where the context information is collected by simulating the execution of instructions.
We further design a shared word mapping function using one single vocabulary for generating embeddings for both translations and queries.
We evaluate the effectiveness of our technique, called TranCS, on the CodeSearchNet corpus with 1,000 queries.
Experimental results show that TranCS significantly outperforms state-of-the-art techniques by 49.31% to 66.50% in terms of $\mathtt{MRR}$ (mean reciprocal rank).

code search, deep learning, code translation

††journalyear: 2022††copyright: acmcopyright††conference: 44th International Conference on Software Engineering; May 21–29, 2022; Pittsburgh, PA, USA††booktitle: 44th International Conference on Software Engineering (ICSE ’22), May 21–29, 2022, Pittsburgh, P A, USA††price: 15.00††doi: 10.1145/3510003.3510140††isbn: 978-1-4503-9221-1/22/05††ccs: Software and its engineering Search-based software engineering

1. Introduction
----------------

Software development is usually a repetitive task, where same or similar implementations exist in established projects or online forums.
Developers tend to search for those high-quality implementations for reference or reuse, so as to enhance the productivity and quality of their development*(Brandt et al., [2009](#bib.bib6 ""), [2010](#bib.bib5 ""); Gharehyazie
et al., [2017](#bib.bib17 ""))*.
Existing studies*(Brandt et al., [2009](#bib.bib6 ""); Shuai
et al., [2020](#bib.bib57 ""))* show that developers often spend 19% of their time on finding reusable code examples during software development.
Code search (CS) is an active research field*(of Software Engineering Work Practices, [1997](#bib.bib50 ""); McMillan et al., [2012](#bib.bib42 ""); Sadowski
et al., [2015](#bib.bib55 ""); Xia
et al., [2017](#bib.bib64 ""); Gu et al., [2018](#bib.bib19 ""); Cambronero et al., [2019](#bib.bib8 ""); Wan
et al., [2019](#bib.bib63 ""); Shuai
et al., [2020](#bib.bib57 ""); Xu
et al., [2021](#bib.bib65 ""); Zeng et al., [2021](#bib.bib68 ""))*, which aims at designing advanced techniques to support code retrieval services.
Given a query by the developer, CS retrieves code snippets that are related to the query from a large-scale code corpus, such as GitHub*(GitHub, [2008](#bib.bib18 ""))* and Stack Overflow*(Inc;, [2008](#bib.bib27 ""))*.
Figure[1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Code Search based on Context-aware Code Translation") shows an example.
The query “how to calculate the factorial of a number” in Figure[1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Code Search based on Context-aware Code Translation")(a) is provided by the developer, which is usually a short natural language sentence describing the functionality of the desired code snippet*(Liu
et al., [2020](#bib.bib36 ""))*.
The method/function*(Shuai
et al., [2020](#bib.bib57 ""); Wan
et al., [2019](#bib.bib63 ""); Nie
et al., [2016](#bib.bib49 ""); Keivanloo
et al., [2014](#bib.bib28 ""))* in Figure[1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Code Search based on Context-aware Code Translation")(b) is a possible code snippet that satisfies the developer’s requirement.

<img src='x1.png' alt='Refer to caption' title='' width='415' height='172' />

*Figure 1. An Example of Query and Code Snippet*

Existing CS techniques can be categorized into traditional methods that use keyword matching between queries and code snippets such as information retrieval-based code search*(Poshyvanyk et al., [2006](#bib.bib52 ""); Brandt et al., [2010](#bib.bib5 ""); McMillan et al., [2011](#bib.bib41 ""); Keivanloo
et al., [2014](#bib.bib28 ""); Sadowski
et al., [2015](#bib.bib55 ""))* and query reformulation-based code search*(Hill
et al., [2009](#bib.bib23 ""); Lemos
et al., [2014](#bib.bib32 ""); Lv
et al., [2015](#bib.bib38 ""); Lu
et al., [2015](#bib.bib37 ""); Nie
et al., [2016](#bib.bib49 ""); Rahman and Roy, [2018](#bib.bib53 ""))*, and deep learning methods that encode queries and code snippets into embedding representations capturing semantic information.
Traditional methods simply treat queries and code snippets as plain texts, and retrieve query-related code snippets by only looking at matched keywords.
They fail to capture the semantics of both query sentences and code snippets.
Deep learning (DL) methods transform input queries and code snippets into embedding representations.
Specifically, for a given query, all the words in the query sentence are first represented as word embeddings and then fed to a DL model to produce a query embedding*(Gu et al., [2018](#bib.bib19 ""); Shuai
et al., [2020](#bib.bib57 ""))*.
For a code snippet, multiple aspects are extracted as features, such as tokens, abstract syntactic trees (ASTs), and control flow graphs (CFGs).
These features are transformed into corresponding embeddings and processed by another DL model to produce a code embedding*(Gu et al., [2018](#bib.bib19 ""); Shuai
et al., [2020](#bib.bib57 ""); Fang
et al., [2020](#bib.bib13 ""); Xu
et al., [2021](#bib.bib65 ""); Zeng et al., [2021](#bib.bib68 ""))*.
The code search task is hence to find similar pairs between query embeddings and code embeddings.
While DL methods surpass traditional methods in capturing the semantics of queries and code snippets, their performances are still limited due to the insufficiency of encoding semantics and the embedding discrepancy between queries and code snippets.
Existing techniques miss either data dependencies among code statements like MMAN *(Wan
et al., [2019](#bib.bib63 ""))* or control dependencies such as DeepCS*(Gu et al., [2018](#bib.bib19 ""))*, CARLCS-CNN *(Shuai
et al., [2020](#bib.bib57 ""))*, and TabCS *(Xu
et al., [2021](#bib.bib65 ""))*.
Furthermore, the embedding representations of code snippets are largely different from those of query sentences written in natural language, causing semantic mismatch during the code search task.
For example, MMAN*(Wan
et al., [2019](#bib.bib63 ""))* uses different word mapping functions (that map a word or token to an embedding representation) to encode queries, and tokens, ASTs, and CFGs in code snippets.
For the widely used word length in both queries and code snippets, the embedding representations are different in those word mapping functions, leading to poor code search performance as we will discuss in Section[3](#S3 "3. Motivation ‣ Code Search based on Context-aware Code Translation") and experimentally show in Section[5.2.2](#S5.SS2.SSS2 "5.2.2. RQ2: Contribution of Key Components ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation").

We propose a novel context-aware code translation technique that translates code snippets into natural language descriptions (called translations).
Such a translation can bridge the representation discrepancy between code snippets (in programming languages) and queries (in natural language).
Specifically, we utilize a standard program compiler and a disassembler to generate the instruction sequence of a code snippet. However, the context information such as local variables, data dependency, etc., are missed from the instruction sequence.
We hence simulates the execution of instructions to collect those desired contexts.
A set of pre-defined translation rules are then used to translate the instruction sequence and contexts into translations.
Such a code translation is context-aware.
The translations of code snippets are similar to those descriptions in queries, in which they share a range of words.
We hence design a shared word mapping mechanism using one single vocabulary for generating embeddings for both translations and queries, substantially reducing the semantic discrepancy and improving the overall performance (see results in Section[5.2.2](#S5.SS2.SSS2 "5.2.2. RQ2: Contribution of Key Components ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation")).

In summary, we make the following contributions.

* •

    We propose a context-aware code translation technique that transforms code snippets into natural language descriptions with preserved semantics.

* •

    We introduce a shared word mapping mechanism, which bridges the discrepancy of embedding representations from code snippets and queries.

* •

    We implement a code search prototype called TranCS. We evaluate it on the CodeSearchNet corpus*(Husain et al., [2019](#bib.bib26 ""))* with 1,000 queries. Experimental results show that TranCS improves the top-1 hit rate of code search by 67.16% to 102.90% compared to state-of-the-art techniques. In addition, TranCS achieves $\mathtt{MRR}$ of 0.651, outperforming DeepCS*(Gu et al., [2018](#bib.bib19 ""))* and MMAN*(Wan
    et al., [2019](#bib.bib63 ""))* by 66.50% and 49.31%, respectively. The source code of TranCS and all the data used in this paper are released and can be downloaded from the website*(Sun and Chen., [2022](#bib.bib59 ""))*.

2. Background
--------------

### 2.1. Machine Instruction

Since the context-aware code translation technique we propose is performed at the machine instruction level, we first introduce the background about machine instructions.

A program runs by executing a sequence of machine instructions*(Ding
et al., [2014](#bib.bib12 ""))*. A machine instruction consists of an opcode specifying the operation to be performed, followed by zero or more operands embodying values to be operated upon*(Moskovitch et al., [2008](#bib.bib46 ""); Lindholm et al., [2021a](#bib.bib34 ""))*. For example, in Java Virtual Machine, $\mathtt{istore\_2}$ is a machine instruction where $\mathtt{istore}$ is an opcode whose operation is “store $\mathtt{int}$ into local variable”, and $\mathtt{2}$ is an operand that represents the index of the local variable. Machine instructions have been widely used in software engineering activities, such as malware detection*(Bilar, [2007](#bib.bib4 ""); Moskovitch et al., [2008](#bib.bib46 ""); Ding
et al., [2014](#bib.bib12 ""))*, API recommendation*(Nguyen
et al., [2016](#bib.bib48 ""))*, code clone detection*(Tufano et al., [2018](#bib.bib61 ""))*, program repair*(Ghanbari
et al., [2019](#bib.bib16 ""))*, and binary code search*(Xue
et al., [2019](#bib.bib66 ""))*. Machine instructions are generated by disassembling the binary files, such as the $\mathtt{.class}$ file in Java. Therefore, it is also called bytecode*(Nguyen
et al., [2016](#bib.bib48 ""); Su et al., [2016](#bib.bib58 ""); Ghanbari
et al., [2019](#bib.bib16 ""))* or bytecode mnemonic opcode*(Tufano et al., [2018](#bib.bib61 ""))* in some of the works mentioned above. For ease of understanding, the terminology “instruction” is used uniformly in this paper.

### 2.2. Deep Learning-based Code Search

<img src='x2.png' alt='Refer to caption' title='' width='461' height='106' />

*Figure 2. A General Framework of DL-based CS techniques*

As shown in Figure [2](#S2.F2 "Figure 2 ‣ 2.2. Deep Learning-based Code Search ‣ 2. Background ‣ Code Search based on Context-aware Code Translation"), we can observe that deep learning (DL)-based CS techniques usually consist of three components, a query encoder, a code encoder, and a similarity measurement component. The query encoder is an embedding network that can encode the query $q$ given by the developer into a $d$-dimensional embedding representation $\bm{e}^{q}\in\mathbb{R}^{d}$. To train such a query encoder, existing DL-based CS techniques have tried various neural network architectures, such as RNN*(Gu et al., [2018](#bib.bib19 ""))*, LSTM*(Wan
et al., [2019](#bib.bib63 ""))*, and CNN*(Xu
et al., [2021](#bib.bib65 ""))*. In DL-based CS studies, it is a common practice to use code comments as queries during the training phase of the encoder*(Gu et al., [2018](#bib.bib19 ""); Wan
et al., [2019](#bib.bib63 ""); Shuai
et al., [2020](#bib.bib57 ""))*. Code comments are natural language descriptions used to explain what the code snippets want to do*(Hu
et al., [2018](#bib.bib25 ""))*. For example, the first line of Figure[3](#S3.F3 "Figure 3 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation")(a) is a comment for the code snippet $s_{a}$. Therefore, we do not strictly distinguish the meaning of the two terms comment and query, and use the term comment during encoder training, and query at other times. The code encoder is also an embedding network that can encode $n$ code snippets in the code corpus $S$ into corresponding embedding representations $\bm{E}^{S}\in\mathbb{R}^{n\times d}$. In existing DL-based CS techniques, the code encoder is usually much more complicated than the query encoder. For example, the code encoder of MMAN*(Wan
et al., [2019](#bib.bib63 ""))* consists of three sub-encoders that are built on the LSTM*(Hochreiter and
Schmidhuber, [1997](#bib.bib24 ""))*, Tree-LSTM*(Tai
et al., [2015](#bib.bib60 ""))*, and GGNN*(Li
et al., [2016](#bib.bib33 ""))* architectures with the goal of encoding different features of the code snippet, e.g., tokens, ASTs, and CFGs. The similarity measurement component is used to measure the cosine similarity between $\bm{e}^{q}$ and each $\bm{e}^{s}\in\bm{E}^{s}$. The target of DL-based CS techniques is to rank all code snippets in $S$ by the cosine similarity*(Gu et al., [2018](#bib.bib19 ""))*. The higher the similarity, the higher relevance of the code snippet to the given query.

3. Motivation
--------------

In this section, we study the limitations of commonly used representations of code snippets as well as the representation discrepancy between code snippets and comments in existing works*(Gu et al., [2018](#bib.bib19 ""); Wan
et al., [2019](#bib.bib63 ""))*.

<img src='x3.png' alt='Refer to caption' title='' width='461' height='202' />

*Figure 3. Code Snippets*

<img src='x4.png' alt='Refer to caption' title='' width='461' height='191' />

*Figure 4. Abstract Syntactic Trees*

Figure[3](#S3.F3 "Figure 3 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation") shows two code snippets for calculating the sum of a given int array. Figure[3](#S3.F3 "Figure 3 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation")(a) uses a for statement to loop over all the elements in the array (line 5) and add their values to variable sum (line 6). Figure[3](#S3.F3 "Figure 3 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation")(b) employs a while statement for the same task (lines 5-8). Semantically, the two code snippets have the exact same meaning. In Figure[4](#S3.F4 "Figure 4 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation"), we show the abstract syntax trees (ASTs) for the above two code snippets $s_{a}$ (left figure) and $s_{b}$ (right figure), respectively. Observe that the sub-trees circled in dotted lines are different for the two code snippets. Such representations cause the inconsistency of code semantics, leading to inferior results in code search as we will show in Section[5.2.1](#S5.SS2.SSS1 "5.2.1. RQ1: Effectiveness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation"). Control flow graph (CFG) is also commonly used for representing code snippets. Figure[5](#S3.F5 "Figure 5 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation") depicts the CFGs for the two code snippets $s_{a}$ and $s_{1}$ (see Figure[1](#S1.F1 "Figure 1 ‣ 1. Introduction ‣ Code Search based on Context-aware Code Translation")(b) in Section[1](#S1 "1. Introduction ‣ Code Search based on Context-aware Code Translation")). The task of $s_{1}$ is to calculate the factorial of a given number, while $s_{a}$ is to calculate the sum of a given array. The two code snippets have completely different goals. However, the CFGs shown in Figure[5](#S3.F5 "Figure 5 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation") have the same graph structure, which cannot differentiate the semantic difference between the two code snippets. This example delineates the insufficiency of utilizing CFGs for representing code semantics. Our experimental results in Section[5.2.1](#S5.SS2.SSS1 "5.2.1. RQ1: Effectiveness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation") show that a state-of-the-art technique MMAN*(Wan
et al., [2019](#bib.bib63 ""))* leveraging ASTs and CFGs has a limited performance.

<img src='x5.png' alt='Refer to caption' title='' width='461' height='218' />

*Figure 5. Control Flow Graphs*

Existing techniques leverage deep learning models (i.e., the encoders introduced in Section[2.2](#S2.SS2 "2.2. Deep Learning-based Code Search ‣ 2. Background ‣ Code Search based on Context-aware Code Translation")) for code search, where code snippets and comments need to be transformed into numerical forms in order to train those models and produce desired outputs. A common way is to build vocabularies for code snippets and comments, and construct corresponding numerical representations (e.g., word embeddings). A word mapping function is a dictionary with the key of a token in code snippets or a word in comments (from vocabularies) and the value of a fixed-length real-valued vector. DeepCS*(Gu et al., [2018](#bib.bib19 ""))* builds four mapping functions for method names (MN), API sequences (APIs), tokens, and comments, separately. MMAN*(Wan
et al., [2019](#bib.bib63 ""))* utilizes four different mapping functions for tokens, ASTs, CFGs, and comments, respectively. The embeddings in these mapping functions are randomly initialized and learned during the training process of the encoder. Such a learning procedure introduces discrepant embedding representations for a same key (e.g., a code token). For instance, ASTs are composed of code tokens, which share a portion of same keys with the token vocabulary. Token names can also appear in comments. For example, more than 50% of keys appear in both code snippets and comments vocabularies used by DeepCS and MMAN. Inconsistent embeddings for same words/tokens can lead to unsuitable matches between code snippets and comments, causing poor performance of code search (see Section[5.2.2](#S5.SS2.SSS2 "5.2.2. RQ2: Contribution of Key Components ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation")).

<img src='x6.png' alt='Refer to caption' title='' width='461' height='351' />

*Figure 6. Code Translations of $s_{a}$ and $s_{b}$*

Our solution. We propose a novel code search technique, called TranCS, that better preserves the semantics of code snippets and bridges the discrepancy between code snippets and comments. Different from existing techniques that leverage ASTs and CFGs, we directly translate code snippets into natural language sentences. Specifically, we utilize a standard program compiler and a disassembler to generate the instruction sequence of a code snippet. Such a sequence, however, lacks the context information such as local variables, data dependency, etc. We propose to simulate the execution of instructions to collect those desired contexts. A set of pre-defined translation rules are then used to translate the instruction sequence and contexts into natural language sentences. Details can be found in Section[4](#S4 "4. Methodology ‣ Code Search based on Context-aware Code Translation"). Figure[6](#S3.F6 "Figure 6 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation") showcases the translations of the two code snippets $s_{a}$ and $s_{b}$ by TranCS. The different colors denote different variable names used in $s_{a}$ (blue) and $s_{b}$ (red). The numbers/words in bold (e.g., value and 22) denote the data and control dependencies among instructions. Observe that the translations of $s_{a}$ and $s_{b}$ are the same except for local variable names. The overall semantics described by the sentences in Figure[6](#S3.F6 "Figure 6 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation") are the same. The translations are similar to those descriptions in comments, in which they share a range of words. We hence design a shared word mapping function using one single vocabulary for generating embeddings for both code snippets and comments, substantially reducing the semantic discrepancy and improving the overall performance (see results in Section[5.2.2](#S5.SS2.SSS2 "5.2.2. RQ2: Contribution of Key Components ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation")).

4. Methodology
---------------

<img src='x7.png' alt='Refer to caption' title='' width='461' height='223' />

*Figure 7. Framework of TranCS*

### 4.1. Overview

Figure[7](#S4.F7 "Figure 7 ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation") illustrates the overview of our TranCS. The top part shows the training procedure of TranCS and the bottom part shows the usage of TranCS for a given query. During the training procedure of TranCS, two types of input data are leveraged: comments and code snippets. The comments in Figure[7](#S4.F7 "Figure 7 ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation") are natural language descriptions that appear above the code snippet (e.g., Javadoc comments), not in the code body. These comments are input to TranCS in pairs with the corresponding code snippets to train CEncoder and TEncoder. For comments, TranCS transforms them into vector representations $\bm{V}^{C}$ using a shared word mapping function. For code snippets, they are different from natural language expressions such as comments. In this paper, we aim to build a homogeneous representation between comments and code snippets, which can better capture the shared semantic information of these two types. Specifically, we propose a context-aware code translation, which translates code snippets into natural language descriptions as shown in the dotted box (details are discussed in Section [4.2](#S4.SS2 "4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")). The natural language descriptions translated from code snippets are also transformed into vector representations $\bm{V}^{T}$ using the same shared word mapping function. TranCS leverages the two vector representations $\bm{V}^{C}$ and $\bm{V}^{T}$ for building two encoders (i.e., CEncoder and TEncoder) that generate embeddings with preserved semantics for both comments and code snippets. CEncoder takes in the comment vector representations $\bm{V}^{C}$ and produces concise embedding representations $\bm{e}^{C}$ that preserves semantic information from the comments. TEncoder generates embedding representations $\bm{e}^{T}$ for code snippets. Details of training these two encoders are elaborated in Section [4.3](#S4.SS3 "4.3. Model Training ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation"). When TranCS is deployed for usage, it takes in a query from the developer and passes it to CEncoder, which produces an embedding $\bm{e}^{q}$ for the query. TranCS then compares the query embedding $\bm{e}^{q}$ with those code embeddings $\bm{e}^{T}$ from the training set. A top-k selection method is leveraged for providing code snippets to the developer, which are semantically similar to the query.

### 4.2. Context-aware Code Translation

The goal of context-aware code translation is to translate code snippets into natural language descriptions according to the pre-defined translation rules. As shown in the dotted box of Figure [7](#S4.F7 "Figure 7 ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation"), this phase consists of two steps. In step ➀, given code snippets, TranCS utilizes a standard compiler and disassembler to generate their instruction sequences. In step ➁, TranCS applies the pre-defined translation rules to translate the instruction sequences into natural language descriptions. We discuss the two steps in detail in the following sections.

#### 4.2.1. Instruction Generation

In this step, TranCS takes in code snippets and produces their instruction sequences. In practice, for a given code snippet, TranCS first utilizes a standard program compiler and disassembler to generate the disassembly representation of the code snippet. For example, TranCS integrates $\mathtt{javac}$ version 1.8.0_144 (a compiler) and $\mathtt{javap}$ version 1.8.0_144 (a disassembler) to generate the disassembly representations for code snippets written in the Java programming language. For the code snippets that can not be compiled, the main reason is due to the lack of class/method definitions around them. We use JCoffee*(Gupta
et al., [2020](#bib.bib21 ""))* to make them compilable by adding class/method definitions around them to complement the missing pieces. Then, TranCS parses the disassembly representation and extracts the instruction sequence. For example, Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a) shows an instruction sequence, which is generated by inputting the code snippet shown in Figure [3](#S3.F3 "Figure 3 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation")(a) into TranCS.

<img src='x8.png' alt='Refer to caption' title='' width='461' height='351' />

*Figure 8. An Example of Instruction Sequence and Translation Rules. [pc] and [pv] indicate filling in a constant and variable, respectively. [ps] indicates filling a value popped from the operand stack, while [pi] indicates filling in an instruction index.*

In addition to the instruction sequence, TranCS also extracts the local variable table from the disassembly representation, which will be used in the subsequent instruction translation process. For example, Listing [1](#LST1 "Listing 1 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation") shows an example of a local variable table (LocalVariableTable) that presents the local variables involved in the code snippet in detail, and is generated along with the instruction sequence in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a). Details about the usage of local variables are introduced in Section [4.2.2](#S4.SS2.SSS2 "4.2.2. Instruction Translation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation").

[⬇](data:text/plain;base64,ICBMb2NhbFZhcmlhYmxlVGFibGU6CiAgICBTdGFydCAgTGVuZ3RoICBTbG90ICBOYW1lICAgU2lnbmF0dXJlCiAgICAgICAgMCAgICAgIDI0ICAgICAwICB0aGlzICAgTENhbEFycmF5U3VtOwogICAgICAgIDAgICAgICAyNCAgICAgMSBhcnJheSAgIFtJCiAgICAgICAgMiAgICAgIDIyICAgICAyICAgc3VtICAgSQogICAgICAgIDQgICAgICAyMCAgICAgMyAgICAgaSAgIEk=)

1LocalVariableTable:

2StartLengthSlotNameSignature

30240thisLCalArraySum;

40241array[I

52222sumI

64203iI

*Listing 1: An Example of Local Variable Table*

#### 4.2.2. Instruction Translation

In this step, TranCS takes in instruction sequences and produces their natural language descriptions. In this section, we first introduce the translation rules used in TranCS, then introduce the instruction context, and finally present how TranCS implements context-aware instruction translation.

Translation Rules (TR). TR used in TranCS is manually constructed based on the instruction specification. In practice, to construct TR, we collected all operations and descriptions of instructions from the machine instruction specification, such as Java Virtual Machine Specification*(Lindholm et al., [2021a](#bib.bib34 ""))*.
An operation is a short natural language description of an instruction. For example, the instruction $\mathtt{istore}$’s operation is:

“store $\mathtt{int}$ into local variable.”

From this operation, we can know the behavior of $\mathtt{istore}$ is to store an $\mathtt{int}$ value into a local variable. A description is a long natural language description of an instruction, which details the interaction of the instruction on the local variables and operand stack. For example, $\mathtt{istore}$’s description is:

“The *index* is an unsigned byte that must be an index into the local variable array of the current frame. The *value* on the top of the operand stack must be of type $\mathtt{int}$. It is popped from the operand stack, and the value of the local variable at *index* is set to value.”

From this description, we can know that $\mathtt{istore}$ first pops an $\mathtt{int}$ value from the operand stack and then stores the value into the *index*-th position of the local variable array. If we only use the operation as the translation of the instruction, the translation will be inaccurate due to the loss of some important context. If we only use the description as the translation of instructions, the translation will be too long. However, research in the field of natural language processing (NLP) reminds us that capturing the semantics of long texts is more difficult than short texts *(Bengio
et al., [1994](#bib.bib3 ""); Vaswani et al., [2017](#bib.bib62 ""))*. Based on the above, we strive to make the instruction translation short and relatively accurate. Therefore, we use the operation as the basis, combing the context specified in the description, to manually collate a translation for each instruction. Such a translation delicately balances shortness and accuracy. For example, the translation we collate for the instruction $\mathtt{istore}$ as follows:

“store $\mathtt{int}$ [ps] into local variable [pv].”

where [ps] and [pv] denote placeholders that specifies the position where the context will be filled, and details about instruction context are discussed in Section Context-aware Instruction Translation. For example, Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(b) shows the result of TranCS using TR to translate the instruction sequence in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a).

Instruction Context. The context of an instruction consists of constants, local variables, and data and control dependencies with other instructions. Constants and local variables are directly determined by operands. As shown in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a), an opcode is followed by zero or more operands. An operand can be a constant, or an index of a local variable, or an index of an instruction. For example, in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a), the operand 0 following the opcode $\mathtt{iconst}$ represents a constant, while the operand 2 following the opcode $\mathtt{istore}$ represents the index of the local variable $sum$ shown in Listing [1](#LST1 "Listing 1 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation"); Control dependencies between instructions are explicitly passed through the indices of the instruction. The indices are also directly specified by operands. For example, the operand 22 following the opcode $\mathtt{if\_icmpge}$ represents the index of the instruction $\mathtt{iload\_2}$ at line 22 in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a). Data dependencies between instructions are implicitly passed through the operand stack. As described in Section Translation Rules (TR), with the guidance of the description, we can know how each instruction interacts with the operand stack, such as popping or pushing data. If the instruction $i_{a}$ pops (i.e., uses) the data that is pushed onto the operand stack by the instruction $i_{b}$, then we say that $i_{a}$ is data dependent on $i_{b}$. For example, Figure [9](#S4.F9 "Figure 9 ‣ 4.2.2. Instruction Translation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a) shows the changes of the operand stack as the opcode sequence in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation") interacts with the operand stack. The values in the operand stack are the carriers that reflect data dependencies between instructions. Figure [9](#S4.F9 "Figure 9 ‣ 4.2.2. Instruction Translation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(b) shows the data and control dependencies between the instructions in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a). In this figure, nodes represent instructions; the labels of nodes are instructions’ indices; the solid and dashed edges represent data and control dependencies, respectively.

<img src='x9.png' alt='Refer to caption' title='' width='461' height='243' />

*Figure 9. An Example of the Changes of the Operand Stack and Instruction Dependency Graph*

Context-aware Instruction Translation. The basic idea of context-aware instruction translation is to simulate the execution of instructions by statically traversing the instruction sequence from top to down. In the traversal process, we collect the context of each instruction, which will be used to update the TR-based translations of the current or other related instructions.

In the actual execution of instructions, a frame is created when the corresponding code snippet is invoked *(Lindholm et al., [2021b](#bib.bib35 ""))*. A frame contains a local variable array and a last-in-first-out stack (i.e., operand stack). The sizes of the local variable array and the operand stack are determined at compile-time. The local variable array stores all local variables used in the instructions. For example, the local variables shown in Listing [1](#LST1 "Listing 1 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation") are used in the instruction sequence shown in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a). The indices of the local variable array corresponds to that in LocalVariableTable shown in Listing [1](#LST1 "Listing 1 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation"), where the ‘Slot’ column presents indices of the local variables. The names and indices of local variables are determined at compile-time, but their values are dynamically updated with the execution of the instructions. The values in the operand stack are also dynamically updated with the execution of the instructions. As mentioned earlier, the context of an instruction includes constants, local variables, data and control dependencies. Among them, constants, local variables and control dependencies are closely related to instructions’ operands. They can be easily determined by the operands (for constants) or by retrieving the instruction sequence (for control dependencies) using the index specified by the operand. However, determining the values of local variables is a challenging task because they are dynamically updated with the execution of the instruction. Analogously, the determination of data dependencies is a challenging task because they are implicitly passed through the operand stack. The values in operand stack are also dynamically updated with the execution of the instruction. Therefore, we need to know in advance how the instruction interacts with the local variable array (e.g., setting value) or the operand stack (e.g., popping or pushing data). In practice, we obtain such information from the description of each instruction. The description of each instruction has been introduced when we introduced the translation rules earlier. With the guidance of the description, we divide the instructions into the following four categories according to whether they interact with the local variable array or the operand stack.

Category 1, expressed as $\mathbb{I}^{S}$.
In $\mathbb{I}^{S}$, the instruction only interacts with the operand stack.
$\mathbb{I}^{S}$ can be subdivided into the following three types:

$\mathbb{I}^{PU}$.:
:   In this type, the interaction is to push the operand onto the operand stack.

$\mathbb{I}^{PO}$.:
:   In this type, the interaction is to pop values from the operand stack.

$\mathbb{I}^{POU}$.:
:   In this type, the interaction is composed of popping values from the operand stack, performing the operation, and pushing the result of the operation to the operand stack.

Category 2, $\mathbb{I}^{V}$.
In $\mathbb{I}^{V}$, the instruction only interacts with the local variable array.
The interaction is to load the value from the local variable array, or store the new value into it. This type of instruction does not interact with the operand stack. For example, the instruction $\mathtt{iinc\;3,1}$ only interacts with the local variable specified by the first operand, not with the operand stack.

Category 3, $\mathbb{I}^{SV}$.
In $\mathbb{I}^{SV}$, the instruction interacts with the operand stack as well as the local variable array. For example, the instruction $\mathtt{istore\_2}$ first loads the integer value from the operand stack, and then stores the value into a local variable.

Category 4, $\mathbb{I}^{O}$.
In $\mathbb{I}^{O}$, the instruction neither interacts with the operand stack nor with the local variable array, such as the instruction $\mathtt{goto}$ and $\mathtt{nop}$. Table [1](#S4.T1 "Table 1 ‣ 4.2.2. Instruction Translation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation") shows the categories of instructions.

*Table 1. The Category of Instructions*

.Category$\mathbb{I}^{S}$$\mathbb{I}^{V}$$\mathbb{I}^{SV}$$\mathbb{I}^{O}$Type$\mathbb{I}^{PU}$$\mathbb{I}^{PO}$$\mathbb{I}^{POU}$Instructionsaconst_null,anewarray,iconst, fconst,bipush,dconst_¡d¿,fconst_¡f¿,iconst_¡i¿,jsr, jsr_w,lconst_¡l¿,ldc, ldc_w,ldc2_w, new,sipushareturn, if_icmpge,ireturn, athrow,dreturn, freturn,if_acmp¡cond¿,if_icmp¡cond¿,if¡cond¿, ifnonnull,ifnull, invokedynamic,invokeinterface,invokespecial, invokestatic,invokevirtual, ireturn,ishl, ishr, lookupswitch,lreturn, monitorexit, pop,pop2, putfield,putstatic, tableswitchaaload, arraylength, baload,caload, d2f, d2i, d2l, dadd,daload, dcmp¡op¿, ddiv, dmul,dneg, drem, dsub, dup, dup_x1,dup_x2, dup2, dup2_x1, dup2_x2,f2d, f2i, f2l, fadd, faload, fcmp¡op¿,fdiv, fmul, fneg, frem, fsub, getfield,getstatic, i2b, i2c, i2d, i2f, i2l, i2s,iadd, iaload, iand, idiv, imul, ineg,instanceof, ior, irem, isub, iushr,ixor, l2d, l2f, l2i, ladd, laload, land,lcmp, ldiv, lmul, lneg, lor, lrem,lshl, lshr, lsub, lushr, multianewarray,lxor, newarray, saload, swapiinc,wideaastore, aload,aload_¡n¿, astoreastore_¡n¿,bastore, castore, dastore,dload, dload_¡n¿,dstore, dstore_¡n¿,fastore, fload,fload_¡n¿, fstore,fstore_¡n¿, iastore,iload, iload_¡n¿,istore, istore_¡n¿,lastore, lload,lload_¡n¿, lstore,lstore_¡n¿, sastoregoto,checkcast,goto_w,nop,ret,return

*Algorithm 1  Context-aware Instruction Translation*

0:An instruction sequence, $I$; Translation Rules, $TR$; A local variable array, $V$;
The depth of the operand stack, $d$.

0:Instruction Translation, $T$;

1:$S\leftarrow$ initialize an empty operand stack with a depth of $d$.

2:foreach$i$ in $I$do

3:$t\leftarrow$ generate the TR-based translation of $i$ based on $TR$;

4:$operands\leftarrow$ extract the operands from $i$;

5:if$i\in\mathbb{I}^{PU}$then

6:$S\leftarrow$ push $operands$ onto $S$;

7:$t\leftarrow$ replace [pc] in $t$ with $operands$;

8:endif

9:if$i\in\mathbb{I}^{PO}$then

10:$values\leftarrow$ pop values from $S$ by $operands$;

11:$t\leftarrow$ replace [ps] in $t$ with $values$;

12:endif

13:if$i\in\mathbb{I}^{POU}$then

14:$values\leftarrow$ pop values from $S$ by $operands$;

15:$t\leftarrow$ replace [ps] in $t$ with $values$;

16:$new\_value\leftarrow$ do operation;

17:$S\leftarrow$ push $new\_value$ onto $S$;

18:endif

19:if$i\in\mathbb{I}^{V}$then

20:$variable\leftarrow$ get variable from $V$ by $operands$;

21:$t\leftarrow$ replace [pv] in $t$ with $variable$;

22:endif

23:if$i\in\mathbb{I}^{SV}$then

24:$values\leftarrow$ pop values from $S$ by $operands$;

25:$t\leftarrow$ replace [ps] in $t$ with $values$;

26:$variable\leftarrow$ get variable from $V$ by $operands$;

27:$t\leftarrow$ replace [pv] in $t$ with $variable$;

28:endif

29:if$t$ contains [pi]then

30:$t\leftarrow$ replace [pi] in $t$ with $operands$;

31:endif

32:$T\leftarrow T\cup{t}$

33:endfor

34:output$T$;

Based on the above classification, TranCS uses Algorithm[1](#alg1 "Algorithm 1 ‣ 4.2.2. Instruction Translation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation") to perform context-aware instruction translation. TranCS takes an instruction sequence ($I$), translation rules ($TR$), a local variable array ($V$), and the depth of the operand stack ($d$) as inputs. $TR$, $V$ and $d$ have been introduced earlier. TranCS first initializes an stack with a depth of $d$ to store intermediate results produced during traversing $I$ (line 1). TranCS then traverses $I$ from top to down (lines 2 – 33).
For each $i\in I$, TranCS first generates its translation $t$ based on $TR$ (line 3).
Then, TranCS extracts the operands from $i$ (line 4).
The operands are used to update $S$ and $t$ in subsequent processes.
TranCS determines $i$’s category according to the pre-defined categories shown in Table [1](#S4.T1 "Table 1 ‣ 4.2.2. Instruction Translation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation").
According to $i$’s category, TranCS uses different processes to update $S$ and $t$ (line 5 – 31).
For example, Figure [9](#S4.F9 "Figure 9 ‣ 4.2.2. Instruction Translation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a) shows an example of the changes of the operand stack when TranCS traverses the instruction sequence shown in Figure [8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a) from top to down.
After traversing all the instructions in $I$, the algorithm finishes and outputs $I$’s translations $T$.
For example, Figure[6](#S3.F6 "Figure 6 ‣ 3. Motivation ‣ Code Search based on Context-aware Code Translation") shows the translation generated by TranCS for the instruction sequence shown in Figure[8](#S4.F8 "Figure 8 ‣ 4.2.1. Instruction Generation ‣ 4.2. Context-aware Code Translation ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(a).

### 4.3. Model Training

The goal of model training is to train two encoders, which will be deployed to support code search service. This phase consists of two steps as shown in Figure[7](#S4.F7 "Figure 7 ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation"). In step ➂, given translations and comments, TranCS transforms them into vector representations $\bm{V}^{C}$ and $\bm{V}^{T}$ using a shared word mapping function. In step ➃, TranCS leverages $\bm{V}^{C}$ and $\bm{V}^{T}$ to train CEncoder and TEncoder.

#### 4.3.1. Shared Word Mapping

In TranCS, both translations and comments are natural language sentences. Sentence embedding is generated based on word embedding*(Mikolov et al., [2013b](#bib.bib44 ""); Palangi et al., [2015](#bib.bib51 ""))*. Word embedding techniques can map words into fixed-length vectors (i.e., embeddings) so that similar words are close to each other in the vector space*(Mikolov
et al., [2013a](#bib.bib43 ""); Mikolov et al., [2013b](#bib.bib44 ""))*.

A word embedding technique can be considered a word mapping function $\psi$, which can map a word $w_{i}$ into a vector representation $\bm{w}_{i}$, i.e., $\bm{w_{i}}\=\psi(w_{i})$. As aforementioned, both translations and comments are natural language sentences, so we design a shared word mapping function. To implement such a $\psi$, we build a shared vocabulary that includes top-$n$ frequently appeared words in translations and comments. We further transform the vector representations of the words into an embedding matrix $E\in\mathbb{R}^{n\times m}$, where $n$ is the size of the vocabulary, $m$ is the dimension of word embedding. The embedding matrix $E\=(\psi(\bm{w}_{1}),...,\psi(\bm{w}_{i}))^{T}$ is initialized randomly and learned in the training process along with the two encoders. Based on this embedding matrix, TranCS can transforms translations and comments into the vector representations $\bm{V}^{C}$ and $\bm{V}^{T}$. A simple way of sentence vector representations is to view it as a bag of words and add up all its word vector representations*(Le and Mikolov, [2014](#bib.bib31 ""))*.

#### 4.3.2. Encoder Training

In this section, we first introduce the architecture of CEncoder and TEncoder, then present how to jointly train the two encoders.

Encoder Architecture. As described in Section[4.3.1](#S4.SS3.SSS1 "4.3.1. Shared Word Mapping ‣ 4.3. Model Training ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation"), in TranCS both translations and comments are natural language sentences. Therefore, we can use the same sequence embedding network to design comment encoder (CEncoder) and translation encoder (TEncoder) instead of designing different embedding networks for them as the previous DL-based CS techniques, such as DeepCS*(Gu et al., [2018](#bib.bib19 ""))* and MMAN*(Wan
et al., [2019](#bib.bib63 ""))*. In practice, TranCS applies the LSTM architecture to design CEncoder and TEncoder. Consider a translation/comment sentence $s\=w_{1},\cdots,w_{N^{s}}$ comprising a sequence of $N^{s}$ words, TranCS first uses the shared word mapping function to produce vector representations $\bm{v}^{s}$. Then, TranCS passes $\bm{v}^{s}$ to the encoder (i.e., CEncoder or TEncoder) that generates embeddings $\bm{e}^{s}$. The hidden state $\bm{h}_{i}^{s}$ of the $i$-th word in $s$ is calculated as follows:

| (1) |  | $\bm{h}_{i}^{s}\=LSTM(\bm{h}_{i-1}^{s},\bm{w}_{i})$ |  |
| --- | --- | --- | --- |

where $\bm{w}_{i}$ represents the vector of the word $w_{i}$ and comes from the embedding matrix $E$.

In addition, TranCS uses attention mechanism proposed by Bahdanau et al.*(Bahdanau
et al., [2015](#bib.bib2 ""))* to alleviate the long-dependency problem in long text sequences*(Bengio
et al., [1994](#bib.bib3 ""))*. The attention weight for each word $w_{i}$ is calculated as follows:

| (2) |  | $\alpha^{s}_{i}\=\frac{exp(f(\bm{h}^{s}_{i})\cdot\bm{u}^{s})}{\sum_{j\=1}^{N^{s}}{exp(f(\bm{h}^{s}_{j})\cdot\bm{u}^{s})}}$ |  |
| --- | --- | --- | --- |

where $f(\cdot)$ denotes a linear layer;
$\bm{u}^{s}$ denotes the context vector which is a high level representation of all words in $s$;
and $\cdot$ denotes the inner project of $\bm{h}^{s}_{i}$ and $\bm{u}^{s}$.
The context vector $\bm{u}^{s}$ is randomly initialized and jointly learned during training.
Then, $s$’s final embedding representation $\bm{e}^{s}$ can be calculated as follows:

| (3) |  | $\bm{e}^{s}\=\sum_{j\=1}^{N^{s}}{\alpha^{s}_{i}\cdot\bm{h}^{s}_{i}}$ |  |
| --- | --- | --- | --- |

Joint Training. Now we present how to jointly train the two encoders (i.e., CEncoder and TEncoder) of TranCS to transform both translations and comments into a unified vector space with a similarity coordination. We follow a widely adopted assumption that if a translation and a comment have similar semantics, their embedding representations should be close to each other*(Gu et al., [2018](#bib.bib19 ""); Wan
et al., [2019](#bib.bib63 ""); Shuai
et al., [2020](#bib.bib57 ""))*. In other words, given a code snippet $s$ whose translation is $t$ and a comment $c$, we want it to predict a high similarity between $t$ and $c$ if $c$ is a correct comment of $s$, and a little similarity otherwise.

In practice, we first translate all code snippets into translations. Then, we construct each training instance as a triple $\langle t,c^{+},c^{-}\rangle$: for each translation $t$ there is a positive comment $c^{+}$ (a ground-truth comment of $s$) and a negative comment $c^{-}$ (an incorrect comment of $s$). The incorrect comment $c^{-}$ is selected randomly from the pool of all correct comments.
When trained on the set of $\langle t,c^{+},c^{-}\rangle$ triples, TranCS predicts the cosine similarities of both $\langle t,c^{+}\rangle$ and $\langle t,c^{-}\rangle$ pairs and minimizes the ranking loss *(Collobert et al., [2011](#bib.bib11 ""); Frome et al., [2013](#bib.bib15 ""))*:

| (4) |  | $\mathcal{L}(\theta)\=\sum_{\langle\bm{t},\bm{c^{+}},\bm{c^{-}}\rangle\in G}{max(0,\beta-cos(\bm{t},\bm{c^{+}})+cos(\bm{t},\bm{c^{-}}))}$ |  |
| --- | --- | --- | --- |

where $\theta$ denotes the model parameters;
$G$ denotes the training dataset;
$\beta$ is a small and fixed margin constraint;
$\bm{t}$, $\bm{c^{+}}$ and $\bm{c^{-}}$ are the embedded vectors of $t$, $c^{+}$ and $c^{-}$, respectively.
Intuitively, the ranking loss encourages the cosine similarity between a translation and its correct comment to go up, and the cosine similarities between a translation and incorrect comments to go down.

### 4.4. Deployment of TranCS

After the two encoders (i.e., CEncoder and TEncoder) are trained, we can deploy TranCS online for code search service. Figure[7](#S4.F7 "Figure 7 ‣ 4. Methodology ‣ Code Search based on Context-aware Code Translation")(2) shows the deployment of TranCS. For a search query $q$ given by the developer, TranCS first uses the shared word mapping function to transform it into vector representation $\bm{v}^{q}$. TranCS further passes $\bm{v}^{q}$ into CEncoder to generate the embedding $\bm{e}^{q}$. Then, TranCS measures the similarity between $\bm{e}^{q}$ and each $\bm{e}^{t}\in\bm{e}^{T}$.
The similarity is calculated as follows:

| (5) |  | $sim(q,t)\=cos(\bm{e}^{q},\bm{e}^{t})\=\frac{\bm{e}^{q}\cdot\bm{e}^{t}}{\left\|\bm{e}^{q}\right\|\left\|\bm{e}^{t}\right\|}$ |  |
| --- | --- | --- | --- |

TranCS ranks all $\bm{T}$ by their similarities with $q$. The higher the similarity, the higher relevance of the code snippet to $q$. Finally, TranCS outputs the code snippets corresponding to the top-$k$ translations to the developer.

5. Evaluation And Analysis
---------------------------

We conduct experiments to answer the following questions:

RQ1.:
:   What is the effectiveness of TranCS when compared with state-of-the-art techniques?

RQ2.:
:   What is the contribution of key components in TranCS, i.e., context-aware code translation and shared word mapping?

RQ3.:
:   What is the robustness of TranCS when varying the query length and code length?

### 5.1. Experimental Setup

#### 5.1.1. Dataset

We evaluate the performance of our TranCS on a corpus of Java code snippets, collected from the public CodeSearchNet corpus*(CodeSearchNet, [2019](#bib.bib10 ""))*. Actually, we have considered the dataset released by baselines (i.e., DeepCS*(Gu et al., [2018](#bib.bib19 ""))* and MMAN*(Wan
et al., [2019](#bib.bib63 ""))*). However, the dataset of DeepCS only contains the cleaned Java code snippets without the raw data, unable to generate the CFG for MMAN. And the dataset of MMAN is not publicly accessible.

We randomly shuffle the dataset and split it into two parts, i.e., 69,324 samples for training and 1,000 samples for testing. It is worth mentioning a difference between our data processing and the one in*(Gu et al., [2018](#bib.bib19 ""))*.
In*(Gu et al., [2018](#bib.bib19 ""))*, the proposed approach is verified on another isolated dataset to avoid the bias. Since the evaluation dataset does not have the ground truth, they manually labelled the searched results.
As possible subjective bias exists in manual evaluation*(Cambronero et al., [2019](#bib.bib8 ""); Wan
et al., [2019](#bib.bib63 ""))*, in this paper, we also adopt the automatic evaluation.
Figure [10](#S5.F10 "Figure 10 ‣ 5.1.1. Dataset ‣ 5.1. Experimental Setup ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation")(a) and (b) show the length distributions of code snippets and comments on the training set. For a code snippet, its length refers to the number of lines of the code snippet. For a comment, its length refers to the number of words in the comment. From Figure[10](#S5.F10 "Figure 10 ‣ 5.1.1. Dataset ‣ 5.1. Experimental Setup ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation")(a), we can observe that the lines of most code snippets are located between 20 to 40. This was also observed in the quote in*(Martin, [2009](#bib.bib39 ""))* “Functions should hardly ever be 20 lines long”. From Figure [10(b)](#S5.F10.sf2 "In Figure 10 ‣ 5.1.1. Dataset ‣ 5.1. Experimental Setup ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation"), it is noticed that almost all comments are less than 20 in length.
This also confirms the challenge of capturing the correlation between short text with its corresponding code snippet.
Figure [10](#S5.F10 "Figure 10 ‣ 5.1.1. Dataset ‣ 5.1. Experimental Setup ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation")(c) and (d) show the length distributions of code snippets and comments on testing data.
We can observe that, despite shuffling randomly, the distributions of data sizes (i.e., lengths) in the two data sets are consistent, so we can conclude that the testing set is representative.

<img src='x10.png' alt='Refer to caption' title='' width='219' height='120' />

*(a) Code Snippets on Training Set*

<img src='x11.png' alt='Refer to caption' title='' width='219' height='119' />

*(b) Comments on Training Set*

<img src='x12.png' alt='Refer to caption' title='' width='219' height='125' />

*(c) Code Snippets on Testing Set*

<img src='x13.png' alt='Refer to caption' title='' width='219' height='124' />

*(d) Comments on Testing Set*

*Figure 10. Length Distributions*

#### 5.1.2. Evaluation Metrics

In the evaluation, we consider the comment of the code snippet as the query, and the code snippet itself as the ground-truth result of code search, which is similar to*(Wan
et al., [2019](#bib.bib63 ""); Shuai
et al., [2020](#bib.bib57 ""); Haldar
et al., [2020](#bib.bib22 ""))* but different from*(Gu et al., [2018](#bib.bib19 ""); Cambronero et al., [2019](#bib.bib8 ""))*.
During the testing time, we treat each comment in the 1,000 testing samples as a query, the code snippet corresponding to the query as the correct result, and the other 999 code snippets as distractor results. We adopt two automatic evaluation metrics that are widely used in code search studies*(Gu et al., [2018](#bib.bib19 ""); Wan
et al., [2019](#bib.bib63 ""); Cambronero et al., [2019](#bib.bib8 ""); Shuai
et al., [2020](#bib.bib57 ""); Haldar
et al., [2020](#bib.bib22 ""))* to measure the performance of TranCS, i.e., success rate at $k$ ($\mathtt{SuccessRate@k}$) and mean reciprocal rank ($\mathtt{MRR}$).

$\mathtt{SuccessRate@k}$ measures the percentage of queries for which the correct result exists in the top $k$ ranked results*(Kilickaya et al., [2017](#bib.bib29 ""); Wan
et al., [2019](#bib.bib63 ""))*, which is computed as follows:

| (6) |  | $\mathtt{SuccessRate@k}\=\frac{1}{|Q|}\sum_{i\=1}^{|Q|}{\delta(FRank_{Q_{i}}\leq k)}$ |  |
| --- | --- | --- | --- |

where $Q$ denotes a set of queries and $|Q|$ is the size of $Q$; $\delta(\cdot)$ denotes a function which returns 1 if the input is true and returns 0 otherwise; $FRank_{Q_{i}}$ refers to the rank position of the correct result for the $i$-th query in $Q$. $\mathtt{SuccessRate@k}$ is important because a better CS technique should allow developers to discover the expected code snippets by inspecting fewer returned results.
The higher the $\mathtt{SuccessRate@k}$ value, the better the code search performance.

$\mathtt{MRR}$ is the average of the reciprocal ranks of results of a set of queries $Q$*(Wan
et al., [2019](#bib.bib63 ""); Shuai
et al., [2020](#bib.bib57 ""))*. The reciprocal rank of a query is the inverse of the rank of the correct result.
$\mathtt{MRR}$ is computed as follows:

| (7) |  | $\mathtt{MRR}\=\frac{1}{|Q|}\sum_{i\=1}^{|Q|}{\frac{1}{FRank_{Q_{i}}}}$ |  |
| --- | --- | --- | --- |

The higher the $\mathtt{MRR}$ value, the better the code search performance.

Meanwhile, as developers prefer to find the expected code snippets with short inspection, we only test $\mathtt{SuccessRate@k}$ and $\mathtt{MRR}$ on the top-10 (that is, the maximum value of $k$ is 10) ranked list following DeepCS*(Gu et al., [2018](#bib.bib19 ""))* and MMAN*(Wan
et al., [2019](#bib.bib63 ""))*.
In other words, when the rank of $Q_{i}$ is out of 10, then $1/FRank_{Q_{i}}$ is set to 0.

#### 5.1.3. Baselines.

In this paper, we compare the following baselines:

* •

    DeepCS *(Gu et al., [2018](#bib.bib19 ""))*.
    DeepCS is one of the representative DL-based CS techniques.
    DeepCS uses two kinds of model architecture to design the code encoder to embed three aspects of the code snippet, i.e., two RNNs for method names and API sequences, and a multi-layer perceptron (MLP) for tokens.
    Its query encoder also uses RNN architecture.

* •

    MMAN *(Wan
    et al., [2019](#bib.bib63 ""))*.
    MMAN is one of the state-of-the-art DL-based CS techniques.
    MMAN uses multiple kinds of model architectures to design the code encoder to embed multiple aspects of the code snippet, i.e., one LSTM for Token, a Tree-LSTM for AST, and a GGNN for CFG.
    Its query encode uses LSTM architecture.

#### 5.1.4. Implementation Details

To train our model, we first shuffle the training data and set the mini-batch size to 32.
The size of the vocabulary is 15,000.
For each batch, the code snippet is padded with a special token $\langle PAD\rangle$ to the maximum length.
We set the word embedding size to 512.
For LSTM unit, we set the hidden size to 512.
The margin $\beta$ is set to 0.6.
We update the parameters via AdamW optimizer*(Kingma and Ba, [2015](#bib.bib30 ""))* with the learning rate 0.0003.
To prevent over-fitting, we use dropout with 0.1.
In TranCS, the comment and the code snippet share the same embedding weights.
All models are implemented using the PyTorch 1.7.1 framework with Python 3.8.
All experiments are conducted on a server equipped with one Nvidia Tesla V100 GPU with 31 GB memory, running on Centos 7.7.
All the models in this paper are trained for 200 epochs, and we select the best model based on the lowest validation loss.

### 5.2. Evaluation Results

In this section, we present and analyze the experimental results to answer the research questions.

#### 5.2.1. RQ1: Effectiveness of TranCS

Table [2](#S5.T2 "Table 2 ‣ 5.2.1. RQ1: Effectiveness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation") shows the overall performance of TranCS and two baselines, measured in terms of $\mathtt{SuccessRate@k}$ and $\mathtt{MRR}$.
The columns $\mathtt{SR@1}$, $\mathtt{SR@5}$ and $\mathtt{SR@10}$ show the results of the average $\mathtt{SuccessRate@k}$ over all queries when $k$ is 1, 5 and 10, respectively.
The column $\mathtt{MRR}$ shows the $\mathtt{MRR}$ values of the three techniques.
From this table, we can observe that for $\mathtt{SR@k}$, the improvements of TranCS to DeepCS are 102.90%, 45.80% and 32.48% when $k$ is 1, 5, and 10, respectively.
The improvements to MMAN are 67.16%, 35.94%, and 25.42%, respectively.
For $\mathtt{MRR}$, the improvements TranCS to DeepCS and MMAN are 66.50% and 49.31%, respectively.
We can draw the conclusion that under all experimental settings, our TranCS consistently achieves higher performance in terms of both two metrics, which indicates better code search performance.

*Table 2. Overall Performance of TranCS and Baselines*

| Tech | $\mathtt{SR@1}$ | $\mathtt{SR@5}$ | $\mathtt{SR@10}$ | $\mathtt{MRR}$ |
| --- | --- | --- | --- | --- |
| DeepCS | 0.276 | 0.524 | 0.622 | 0.391 |
| MMAN | 0.335 | 0.562 | 0.657 | 0.436 |
| TranCS | 0.560 | 0.764 | 0.824 | 0.651 |

The CodeSearchNet corpus also provides 99 realistic natural languages queries and expert annotations for likely results. Each query/result pair was labeled by a human expert, indicating the relevance of the result for the query. We also conduct experiments on 99 queries provided by the CodeSearchNet corpus for the Java programming language. We use the same metric, normalized discounted cumulative gain (NDCG*(Schütze et al., [2008](#bib.bib56 ""))*), to evaluate baselines and TranCS. Our TranCS achieves NDCG of 0.223, outperforming DeepCS (0.138) and MMAN (0.173) by 62% and 30%, respectively.

*Table 3. Contribution of Key Components in TranCS*

| Tech | $\mathtt{SR@1}$ | $\mathtt{SR@5}$ | $\mathtt{SR@10}$ | $\mathtt{MRR}$ |
| --- | --- | --- | --- | --- |
| TokeCS | 0.247 | 0.477 | 0.586 | 0.359 |
| TranCS (CCT) | 0.352 | 0.569 | 0.664 | 0.455 |
| TokeCS (SWM) | 0.264 | 0.483 | 0.592 | 0.370 |
| DeepCS (SWM) | 0.295 | 0.511 | 0.615 | 0.399 |
| TranCS (CCT+SWM) | 0.560 | 0.764 | 0.824 | 0.651 |

#### 5.2.2. RQ2: Contribution of Key Components

We experimentally verified the effectiveness of two key components of TranCS i.e., context-aware code translation (CCT) and shared word mapping (SWM).
In Table [3](#S5.T3 "Table 3 ‣ 5.2.1. RQ1: Effectiveness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation"), TranCS(CCT) and TranCS(CCT+SWM) are two special versions of TranCS, among which the former uses two different word mapping functions to transform instruction translations and comments to vector representations, while the latter uses SWM.
In other words, if it is only CCT, TranCS uses two vocabularies. In the case of CCT+SWM, TranCS uses a shared vocabulary.
Moreover, numerous existing studies *(Sachdev et al., [2018](#bib.bib54 ""); Shuai
et al., [2020](#bib.bib57 ""); Zhu
et al., [2020](#bib.bib69 ""))* including DeepCS *(Gu et al., [2018](#bib.bib19 ""))* and MMAN *(Wan
et al., [2019](#bib.bib63 ""))* have shown that tokens of code snippets play a key role in code search tasks.
Therefore, we assume that this is a scenario where the code snippet is not translated, and we directly pass the tokens of the code snippet into the model to train the code encoder.
The effectiveness of the token-based CS technique (TokeCS) is shown in the second line of Table [3](#S5.T3 "Table 3 ‣ 5.2.1. RQ1: Effectiveness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation").
To demonstrate the effectiveness of SWM, we also tried to apply SWM to TokeCS, DeepCS and MMAN.
To apply SWM to TokeCS, we use a unified word mapping function to transform tokens and comments.
In DeepCS, the author uses four word mapping functions to transform the MN, APIS, Token and comments into vector representations.
To apply SWM to DeepCS, we first merge the four vocabularies into a shared vocabulary by extracting the union of them.
Then, we use a unified word mapping function to transform MN, APIS, Token and comments.
In MMAN, the author not only uses LSTM architecture to embed tokens, but also uses Tree-LSTM and GGNN to embed AST and CFG, while the three architectures cannot share a word mapping function.
Therefore, SWM can not be applied to MMAN.
The effectiveness of Toke(SWM), DeepCS(SWM) are shown in lines 4–5 of Table [3](#S5.T3 "Table 3 ‣ 5.2.1. RQ1: Effectiveness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation").
From the lines 2–3 of Table [3](#S5.T3 "Table 3 ‣ 5.2.1. RQ1: Effectiveness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation"), we can observe that for $\mathtt{SR@k}$, the improvements of TranCS(CCT) to TokeCS are 42.51%, 19.29% and 13.31% when $k$ is 1, 5, and 10, respectively.
For $\mathtt{MRR}$, the improvement to TokeCS is 26.74%.
Therefore, we can conclude that CCT contributes to TranCS.
For $\mathtt{SR@k}$, the improvements of TranCS(CCT+SWM) to TranCS(CCT) are 59.09%, 34.27% and 24.10%.
For $\mathtt{MRR}$, the improvement of TranCS(CCT+SWM) to TranCS(CCT) is 43.08%.
Therefore, we can conclude that SWM contributes to TranCS.
Besides, we can also observe that SWM also has slight improvements to TokeCS and DeepCS.
Therefore, we can draw the conclusion that SWM and CCT, which promote each other, improve the performance of TranCS jointly.

#### 5.2.3. RQ3: Robustness of TranCS

To analyze the robustness of TranCS, we studied two parameters (i.e., code length and comment length) that may have an impact on the embedding representations of translations and comments.
Figure [11](#S5.F11 "Figure 11 ‣ 5.2.3. RQ3: Robustness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation") shows the performance of TranCS based on different evaluation metrics with varying parameters.
From Figure [11](#S5.F11 "Figure 11 ‣ 5.2.3. RQ3: Robustness of TranCS ‣ 5.2. Evaluation Results ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation"), we can observe that in most cases, TranCS maintains a stable performance even though the code snippet length or comment length increases, which can be attributed to context-aware code translation and shared word mapping we proposed.
When the length of the code snippet exceeds 20 (a common range described in Section [5.1.1](#S5.SS1.SSS1 "5.1.1. Dataset ‣ 5.1. Experimental Setup ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation")), the performance of TranCS decreases as the length increases.
It means that when the length of the code snippets or comments exceeds the common range, as the length continues to increase, it will be more difficult to capture their semantics.
Overall, the results verify the robustness of our TranCS.

<img src='x14.png' alt='Refer to caption' title='' width='219' height='131' />

*(a) Varying Code Snippet Lengths*

<img src='x15.png' alt='Refer to caption' title='' width='219' height='131' />

*(b) Varying Comment Lengths*

*Figure 11. Robustness of TranCS*

6. Case Study
--------------

<img src='x16.png' alt='Refer to caption' title='' width='461' height='367' />

*Figure 12. Example of Two Code Snippets Implementing the Same Functionality*

This is a case to study the performance of TranCS in retrieving code with implantation difference. Figure[12](#S6.F12 "Figure 12 ‣ 6. Case Study ‣ Code Search based on Context-aware Code Translation")(b) and (c) show two code snippets that implement the same functionality, i.e., swapping two elements in the list.
The first one ($s_{1}$) implements the functionality from scratch, and the second one ($s_{2}$) directly calls the external API $Collection.swap()$.
We use TranCS to convert the two code snippets into corresponding translations, which are very different, meaning TranCS can effectively differentiate semantically similar code but differs in APIs used. This is because TranCS reserves API information (e.g., name, parameter) when generating code translation. For example, as shown in Figure[13](#S6.F13 "Figure 13 ‣ 6. Case Study ‣ Code Search based on Context-aware Code Translation"), the translations produced by TranCS reserve the information of the API $Collection.swap()$ invoked by $s_{2}$, including the parameters (e.g., list) and the method name swap.

<img src='x17.png' alt='Refer to caption' title='' width='461' height='144' />

*Figure 13. Translations of the Code Snippet $s_{2}$*

7. Threats to Validity
-----------------------

The metrics used in this paper are $\mathtt{SuccessRate@k}$ and $\mathtt{MRR}$ for evaluating the effectiveness of TranCS and existing techniques. These are the same metrics adopted in MMAN*(Wan
et al., [2019](#bib.bib63 ""))*. We do not use another metric $\mathtt{Precision@k}$ that measures the percentage of relevant results in the top $k$ returned results for each query*(Gu et al., [2018](#bib.bib19 ""))*. This is due to the constraint that the relevant results need to be labelled manually, which is empirically less feasible and can introduce human biases. We hence focus on the two metrics $\mathtt{SuccessRate@k}$ and $\mathtt{MRR}$ in the paper.

TranCS is currently only evaluated on Java programs and may require modifications for extending to other programming languages. The core contribution of TranCS is the context-aware code translation technique. To realize the context-aware code translation, TranCS requires a set of translation rules, such as the operations and descriptions of instructions. In order to extend TranCS to other programming languages, corresponding translation rules need to be designed and provided. We plan to evaluate the performance of TranCS on these programming languages in future work.

8. Related Work
----------------

Early CS techniques were based on IR technology, such as*(Poshyvanyk et al., [2006](#bib.bib52 ""); Brandt et al., [2010](#bib.bib5 ""); McMillan et al., [2011](#bib.bib41 ""); Keivanloo
et al., [2014](#bib.bib28 ""); Sadowski
et al., [2015](#bib.bib55 ""))*.
These techniques simply consider queries and code snippets as plain text and then use keyword matching.
To alleviate the problem of keyword mismatch*(Hill
et al., [2009](#bib.bib23 ""); Carpineto and
Romano, [2012](#bib.bib9 ""))* and noisy keywords*(Gu
et al., [2016](#bib.bib20 ""))*, many query reformulation(QR)-based CS techniques*(Hill
et al., [2009](#bib.bib23 ""); Lemos
et al., [2014](#bib.bib32 ""); Lv
et al., [2015](#bib.bib38 ""); Lu
et al., [2015](#bib.bib37 ""); Nie
et al., [2016](#bib.bib49 ""); Rahman and Roy, [2018](#bib.bib53 ""))* have been proposed one after another.
For example, the words from WordNet*(Miller, [1995](#bib.bib45 ""))*, or Stack Overflow*(Nie
et al., [2016](#bib.bib49 ""))* are used to expand user queries.
However, QR-based CS techniques consider each word independently, while ignoring the context of the word.
In addition, both IR-based and QR-based CS techniques only treat the code snippet as plain text, and cannot capture the deep semantics of the code snippet.
To better capture the semantics of queries and code snippets, deep learning (DL)-based CS techniques*(Gu et al., [2018](#bib.bib19 ""); Sachdev et al., [2018](#bib.bib54 ""); Wan
et al., [2019](#bib.bib63 ""); Cambronero et al., [2019](#bib.bib8 ""); Yao
et al., [2019](#bib.bib67 ""); Shuai
et al., [2020](#bib.bib57 ""); Feng et al., [2020](#bib.bib14 ""); Husain et al., [2019](#bib.bib26 ""))* have been proposed one after another.
Gu et al.*(Gu et al., [2018](#bib.bib19 ""))* first apply DL to the code search task.
They first encode both the query and a set of code snippets into corresponding embeddings using MLP or RNN, and then rank the code snippets according to the cosine similarity of embeddings.
Other DL-based CS techniques are similar to DeepCS*(Gu et al., [2018](#bib.bib19 ""))* with only a difference in choosing the embedding architecture.
For example, to capture the semantics of other aspects of the code snippet, MMAN*(Wan
et al., [2019](#bib.bib63 ""))* integrates multiple embedding networks (i.e., LSTM, Tree-LSTM and GGNN) to capture semantics of multiple aspects, such as Token, AST, and CFG.
CodeBERT*(Feng et al., [2020](#bib.bib14 ""))*, CoaCor*(Yao
et al., [2019](#bib.bib67 ""))*, and baselines in CodeSearchNet Challenge*(Husain et al., [2019](#bib.bib26 ""))* only treat the code snippet as plain text (token sequence), which miss richer information such as APIs, AST, and CFG, etc.
TBCNN*(Mou
et al., [2016](#bib.bib47 ""))* is a tree-based convolutional neural network that encodes the AST of the code snippet. Our baseline MMAN has encoded AST using tree-based neural networks and is inferior to our TranCS.
All these works have a similar idea that first transforms both code snippets and queries into embedding representations into a unified embedding space with two encoders, and then measures the cosine similarity of these embedding representations.
However, TranCS differs from previous work in two major dimensions:
1) TranCS first translates the code snippet into semantic-preserving natural language descriptions.
In this case, the generated translations and comments are homogeneous.
2) Based on code translation, TranCS naturally uses a shared word mapping mechanism, which can produce consistent embeddings for the same words, thereby better capturing the shared semantic information of translations and comments.

9. Conclusion
--------------

In this paper, we propose a context-aware code translation technique, which can translate code snippets into natural language descriptions with preserved semantics.
In addition, we propose a shared word mapping mechanism to produce consistent embeddings for the same words/tokens in comments and code snippets, so as to capture the shared semantic information.
On the basis of context-aware code translation and shared word mapping, we implement a novel code search technique TranCS.
We conduct comprehensive experiments to evaluate the effectiveness of TranCS, and experimental results show that TranCS is an effective CS technique and substantially outperforms the state-of-the-art techniques.

In future work, we will further explore the following two dimensions: (1) as shown in Figure[10](#S5.F10 "Figure 10 ‣ 5.1.1. Dataset ‣ 5.1. Experimental Setup ‣ 5. Evaluation And Analysis ‣ Code Search based on Context-aware Code Translation"), statistical results on large-scale data sets show that most code snippets have no more than 20 lines. Within this range, TranCS is robust and stable. Constructing representations of long code snippets is still an open problem, and we leave it to future work.
(2) LSTM encoder is just a component of TranCS, which can be easily replaced with more advanced (including pre-trained) models in*(Bui et al., [2021](#bib.bib7 ""); Mastropaolo
et al., [2021](#bib.bib40 ""))*. We will explore more advanced models in future work.

Acknowledgement
---------------

The authors would like to thank the anonymous reviewers for insightful comments. This work is supported partially by National Natural Science Foundation of China(61690201, 62141215).

References
----------

* (1)
* Bahdanau
et al. (2015)Dzmitry Bahdanau,
Kyunghyun Cho, and Yoshua Bengio.
2015.Neural Machine Translation by Jointly Learning to
Align and Translate. In *Proceedings of the 3th
International Conference on Learning Representations*. San
Diego, CA, USA, 1–15.
* Bengio
et al. (1994)Yoshua Bengio, Patrice Y.
Simard, and Paolo Frasconi.
1994.Learning long-term dependencies with gradient
descent is difficult.*IEEE Transactions on Neural Networks*5, 2 (1994),
157–166.
* Bilar (2007)Daniel Bilar.
2007.Opcodes as predictor for malware.*International Journal of Electronic Security
and Digital Forensics* 1, 2
(2007), 156–168.
* Brandt et al. (2010)Joel Brandt, Mira
Dontcheva, Marcos Weskamp, and Scott R.
Klemmer. 2010.Example-centric programming: integrating web search
into the development environment. In *Proceedings
of the 28th International Conference on Human Factors in Computing Systems*.
ACM, Atlanta, Georgia, USA,
513–522.
* Brandt et al. (2009)Joel Brandt, Philip J.
Guo, Joel Lewenstein, Mira Dontcheva,
and Scott R. Klemmer. 2009.Two studies of opportunistic programming:
interleaving web foraging, learning, and writing code. In*Proceedings of the 27th International Conference on
Human Factors in Computing Systems*. ACM,
Boston, MA, USA, 1589–1598.
* Bui et al. (2021)Nghi D. Q. Bui, Yijun Yu,
and Lingxiao Jiang. 2021.Self-Supervised Contrastive Learning for Code
Retrieval and Summarization via Semantic-Preserving Transformations. In*Proceedings of the 44th International ACM SIGIR
Conference on Research and Development in Information Retrieval*.
ACM, Virtual Event, Canada,
511–521.
* Cambronero et al. (2019)José Cambronero,
Hongyu Li, Seohyun Kim,
Koushik Sen, and Satish Chandra.
2019.When deep learning met code search. In*Proceedings of the 13 Joint Meeting on European
Software Engineering Conference and Symposium on the Foundations of Software
Engineering*. ACM, Tallinn,
Estonia, 964–974.
* Carpineto and
Romano (2012)Claudio Carpineto and
Giovanni Romano. 2012.A Survey of Automatic Query Expansion in
Information Retrieval.*Comput. Surveys* 44,
1 (2012), 1–50.
* CodeSearchNet (2019)CodeSearchNet.
2019.CodeSearchNet Data.site:[https://github.com/github/CodeSearchNet#data](https://github.com/github/CodeSearchNet#data "").Accessed: 2021.
* Collobert et al. (2011)Ronan Collobert, Jason
Weston, Léon Bottou, Michael
Karlen, Koray Kavukcuoglu, and Pavel P.
Kuksa. 2011.Natural Language Processing (Almost) from Scratch.*Journal of Machine Learning Research*12, ARTICLE (2011),
2493–2537.
* Ding
et al. (2014)Yuxin Ding, Wei Dai,
Shengli Yan, and Yumei Zhang.
2014.Control flow-based opcode behavior analysis for
Malware detection.*Computers \& Security* 44
(2014), 65–74.
* Fang
et al. (2020)Chunrong Fang, Zixi Liu,
Yangyang Shi, Jeff Huang, and
Qingkai Shi. 2020.Functional code clone detection with syntax and
semantics fusion learning. In *Proceedings of the
29th International Symposium on Software Testing and Analysis*.
ACM, Virtual Event, USA,
516–527.
* Feng et al. (2020)Zhangyin Feng, Daya Guo,
Duyu Tang, Nan Duan,
Xiaocheng Feng, Ming Gong,
Linjun Shou, Bing Qin,
Ting Liu, Daxin Jiang, and
Ming Zhou. 2020.CodeBERT: A Pre-Trained Model for Programming and
Natural Languages. In *Proceedings of the 25th
Conference on Empirical Methods in Natural Language Processing: Findings*.
Association for Computational Linguistics,
Online Event, 1536–1547.
* Frome et al. (2013)Andrea Frome, Gregory S.
Corrado, Jonathon Shlens, Samy Bengio,
Jeffrey Dean, Marc’Aurelio Ranzato, and
Tomás Mikolov. 2013.DeViSE: A Deep Visual-Semantic Embedding Model.
In *proceedings of the 27th Annual Conference Neural
Information Processing Systems*. Curran Associates
Inc., Lake Tahoe, Nevada, United States,
2121–2129.
* Ghanbari
et al. (2019)Ali Ghanbari, Samuel
Benton, and Lingming Zhang.
2019.Practical program repair via bytecode mutation. In*Proceedings of the 28th International Symposium on
Software Testing and Analysis*. ACM,
Beijing, China, 19–30.
* Gharehyazie
et al. (2017)Mohammad Gharehyazie,
Baishakhi Ray, and Vladimir Filkov.
2017.Some from here, some from there: cross-project code
reuse in GitHub. In *Proceedings of the 14th
International Conference on Mining Software Repositories*.
IEEE Computer Society, Buenos Aires,
Argentina, 291–301.
* GitHub (2008)Inc. GitHub.
2008.GitHub.site: <https://github.com>.Accessed: 2021.
* Gu et al. (2018)Xiaodong Gu, Hongyu
Zhang, and Sunghun Kim.
2018.Deep code search. In*Proceedings of the 40th International Conference on
Software Engineering*. ACM,
Gothenburg, Sweden, 933–944.
* Gu
et al. (2016)Xiaodong Gu, Hongyu
Zhang, Dongmei Zhang, and Sunghun
Kim. 2016.Deep API learning. In*Proceedings of the 24th International Symposium on
Foundations of Software Engineering*. ACM,
Seattle, WA, USA, 631–642.
* Gupta
et al. (2020)Piyush Gupta, Nikita
Mehrotra, and Rahul Purandare.
2020.JCoffee: Using Compiler Feedback to Make Partial
Code Snippets Compilable. In *Proceedings of the
36th International Conference on Software Maintenance and Evolution*.
IEEE, Adelaide, Australia,
810–813.
* Haldar
et al. (2020)Rajarshi Haldar, Lingfei
Wu, Jinjun Xiong, and Julia
Hockenmaier. 2020.A Multi-Perspective Architecture for Semantic Code
Search. In *Proceedings of the 58th Annual Meeting
of the Association for Computational Linguistics*.
Association for Computational Linguistics,
Online, 8563–8568.
* Hill
et al. (2009)Emily Hill, Lori L.
Pollock, and K. Vijay-Shanker.
2009.Automatically capturing source code context of
NL-queries for software maintenance and reuse. In*Proceedings of the 31st International Conference on
Software Engineering*. IEEE,
Vancouver, Canada, 232–242.
* Hochreiter and
Schmidhuber (1997)Sepp Hochreiter and
Jürgen Schmidhuber. 1997.Long Short-Term Memory.*Neural Computation* 9,
8 (1997), 1735–1780.
* Hu
et al. (2018)Xing Hu, Ge Li,
Xin Xia, David Lo, and
Zhi Jin. 2018.Deep code comment generation. In*Proceedings of the 26th International Conference on
Program Comprehension*. ACM,
Gothenburg, Sweden, 200–210.
* Husain et al. (2019)Hamel Husain, Ho-Hsiang
Wu, Tiferet Gazit, Miltiadis Allamanis,
and Marc Brockschmidt. 2019.CodeSearchNet Challenge: Evaluating the State of
Semantic Code Search.*CoRR* abs/1909.09436
(2019), 1–6.
* Inc; (2008)Stack Exchange Inc;.
2008.Stack Overflow.site: <https://stackoverflow.com/>.Accessed: 2021.
* Keivanloo
et al. (2014)Iman Keivanloo, Juergen
Rilling, and Ying Zou. 2014.Spotting working code examples. In*Proceedings of the 36th International Conference on
Software Engineering*. ACM,
Hyderabad, India, 664–675.
* Kilickaya et al. (2017)Mert Kilickaya, Aykut
Erdem, Nazli Ikizler-Cinbis, and
Erkut Erdem. 2017.Re-evaluating Automatic Metrics for Image
Captioning. In *Proceedings of the 15th Conference
of the European Chapter of the Association for Computational Linguistics*.
Association for Computational Linguistics,
Valencia, Spain, 199–209.
* Kingma and Ba (2015)Diederik P. Kingma and
Jimmy Ba. 2015.Adam: A Method for Stochastic Optimization. In*Proceedings of the 3th International Conference on
Learning Representations – Poster*. OpenReview.net,
San Diego, CA, USA, 1–15.
* Le and Mikolov (2014)Quoc V. Le and
Tomás Mikolov. 2014.Distributed Representations of Sentences and
Documents. In *Proceedings of the 31th
International Conference on Machine Learning*.
JMLR.org, Beijing, China,
1188–1196.
* Lemos
et al. (2014)Otávio Augusto Lazzarini Lemos,
Adriano Carvalho de Paula,
Felipe Capodifoglio Zanichelli, and
Cristina Videira Lopes. 2014.Thesaurus-based automatic query expansion for
interface-driven code search. In *Proceedings of
the 11th Working Conference on Mining Software Repositories*.
ACM, Hyderabad, India,
212–221.
* Li
et al. (2016)Yujia Li, Daniel Tarlow,
Marc Brockschmidt, and Richard S.
Zemel. 2016.Gated Graph Sequence Neural Networks. In*Proceedings of the 4th International Conference on
Learning Representations*. OpenReview.net,
San Juan, Puerto Rico, 1–20.
* Lindholm et al. (2021a)Tim Lindholm, Frank
Yellin, Gilad Bracha, Alex Buckley,
and Daniel Smith. 2021a.The Java Virtual Machine Specification.site:<https://docs.oracle.com/javase/specs/jvms/se8/html/index.html>.Accessed: 2021.
* Lindholm et al. (2021b)Tim Lindholm, Frank
Yellin, Gilad Bracha, Alex Buckley,
and Daniel Smith. 2021b.The Java Virtual Machine Specification-Local
Variables.site:[https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-2.html#jvms-2.6](https://docs.oracle.com/javase/specs/jvms/se8/html/jvms-2.html#jvms-2.6 "").Accessed: 2021.
* Liu
et al. (2020)Chao Liu, Xin Xia,
David Lo, Cuiyun Gao,
Xiaohu Yang, and John Grundy.
2020.Opportunities and Challenges in Code Search Tools.*CoRR* abs/2011.02297
(2020), 1–35.
* Lu
et al. (2015)Meili Lu, Xiaobing Sun,
Shaowei Wang, David Lo, and
Yucong Duan. 2015.Query expansion via WordNet for effective code
search. In *Proceedings of the 22nd International
Conference on Software Analysis, Evolution, and Reengineering*.
IEEE Computer Society, Montreal, QC,
Canada, 545–549.
* Lv
et al. (2015)Fei Lv, Hongyu Zhang,
Jian-Guang Lou, Shaowei Wang,
Dongmei Zhang, and Jianjun Zhao.
2015.CodeHow: Effective Code Search Based on API
Understanding and Extended Boolean Model (E). In*Proceedings of the 30th International Conference on
Automated Software Engineering*. IEEE Computer
Society, Lincoln, NE, USA, 260–270.
* Martin (2009)Robert C Martin.
2009.*Clean code: a handbook of agile software
craftsmanship*.Pearson Education.
* Mastropaolo
et al. (2021)Antonio Mastropaolo,
Simone Scalabrino, Nathan Cooper,
David Nader-Palacio, Denys Poshyvanyk,
Rocco Oliveto, and Gabriele Bavota.
2021.Studying the Usage of Text-To-Text Transfer
Transformer to Support Code-Related Tasks. In*Proceedings of the 43rd International Conference on
Software Engineering*. IEEE,
Madrid, Spain, 336–347.
* McMillan et al. (2011)Collin McMillan, Mark
Grechanik, Denys Poshyvanyk, Qing Xie,
and Chen Fu. 2011.Portfolio: finding relevant functions and their
usage. In *Proceedings of the 33rd International
Conference on Software Engineering*. ACM,
Waikiki, Honolulu , HI, USA, 111–120.
* McMillan et al. (2012)Collin McMillan, Negar
Hariri, Denys Poshyvanyk, Jane
Cleland-Huang, and Bamshad Mobasher.
2012.Recommending source code for use in rapid software
prototypes. In *Proceedings of the 34th
International Conference on Software Engineering*.
IEEE Computer Society, Zurich,
Switzerland, 848–858.
* Mikolov
et al. (2013a)Tomás Mikolov, Kai
Chen, Greg Corrado, and Jeffrey Dean.
2013a.Efficient Estimation of Word Representations in
Vector Space. In *Proceedings of the 1st
International Conference on Learning Representations, Workshop Track*.
OpenReview.net, Scottsdale, Arizona,
USA, 1–12.
* Mikolov et al. (2013b)Tomas Mikolov, Ilya
Sutskever, Kai Chen, Gregory S. Corrado,
and Jeffrey Dean. 2013b.Distributed Representations of Words and Phrases
and their Compositionality. In *Proceedings of the
27th Annual Conference on Neural Information Processing Systems*.
Curran Associates Inc., Lake Tahoe,
Nevada, United States, 3111–3119.
* Miller (1995)George A. Miller.
1995.WordNet: A Lexical Database for English.*Commun. ACM* 38,
11 (1995), 39–41.
* Moskovitch et al. (2008)Robert Moskovitch, Clint
Feher, Nir Tzachar, Eugene Berger,
Marina Gitelman, Shlomi Dolev, and
Yuval Elovici. 2008.Unknown Malcode Detection Using OPCODE
Representation. In *Proceedings of the First
European Conference on Intelligence and Security Informatics*.
Springer, Esbjerg, Denmark,
204–215.
* Mou
et al. (2016)Lili Mou, Ge Li,
Lu Zhang, Tao Wang, and
Zhi Jin. 2016.Convolutional Neural Networks over Tree Structures
for Programming Language Processing. In*Proceedings of the 30th Conference on Artificial
Intelligence*. AAAI Press, Phoenix,
Arizona, USA, 1287–1293.
* Nguyen
et al. (2016)Tam The Nguyen, Hung Viet
Pham, Phong Minh Vu, and Tung Thanh
Nguyen. 2016.Learning API usages from bytecode: a statistical
approach. In *Proceedings of the 38th International
Conference on Software Engineering*. ACM,
Austin, TX, USA, 416–427.
* Nie
et al. (2016)Liming Nie, He Jiang,
Zhilei Ren, Zeyi Sun, and
Xiaochen Li. 2016.Query Expansion Based on Crowd Knowledge for Code
Search.*IEEE Transactions on Services Computing*9, 5 (2016),
771–783.
* of Software Engineering Work Practices (1997)An Examination of Software Engineering
Work Practices. 1997.Janice Singer and Timothy C. Lethbridge and Norman
G. Vinson and Nicolas Anquetil. In *Proceedings of
the 7th conference of the Centre for Advanced Studies on Collaborative
Research*. IBM, Toronto, Ontario,
Canada, 174–188.
* Palangi et al. (2015)Hamid Palangi, Li Deng,
Yelong Shen, Jianfeng Gao,
Xiaodong He, Jianshu Chen,
Xinying Song, and Rabab K. Ward.
2015.Deep Sentence Embedding Using the Long Short Term
Memory Network: Analysis and Application to Information Retrieval.*CoRR* abs/1502.06922
(2015), 1–15.
* Poshyvanyk et al. (2006)Denys Poshyvanyk, Maksym
Petrenko, Andrian Marcus, Xinrong Xie,
and Dapeng Liu. 2006.Source Code Exploration with Google. In*Proceedings of the 22nd International Conference on
Software Maintenance*. IEEE Computer Society,
Philadelphia, Pennsylvania, USA,
334–338.
* Rahman and Roy (2018)Mohammad Masudur Rahman and
Chanchal K. Roy. 2018.Effective Reformulation of Query for Code Search
Using Crowdsourced Knowledge and Extra-Large Data Analytics. In*Proceedings of the 34th International Conference on
Software Maintenance and Evolution*. IEEE Computer
Society, Madrid, Spain, 473–484.
* Sachdev et al. (2018)Saksham Sachdev, Hongyu
Li, Sifei Luan, Seohyun Kim,
Koushik Sen, and Satish Chandra.
2018.Retrieval on source code: a neural code search. In*Proceedings of the 2nd International Workshop on
Machine Learning and Programming Languages*. ACM,
Philadelphia, PA, USA, 31–41.
* Sadowski
et al. (2015)Caitlin Sadowski,
Kathryn T. Stolee, and Sebastian G.
Elbaum. 2015.How developers search for code: a case study. In*Proceedings of the 10th Joint Meeting on European
Software Engineering Conference and Symposium on the Foundations of Software
Engineering*. ACM, Bergamo, Italy,
191–201.
* Schütze et al. (2008)Hinrich Schütze,
Christopher D Manning, and Prabhakar
Raghavan. 2008.*Introduction to information retrieval*.
Vol. 39.Cambridge University Press Cambridge.
* Shuai
et al. (2020)Jianhang Shuai, Ling Xu,
Chao Liu, Meng Yan, Xin
Xia, and Yan Lei. 2020.Improving Code Search with Co-Attentive
Representation Learning. In *Proceedings of the
28th International Conference on Program Comprehension*.
ACM, Seoul, Republic of Korea,
196–207.
* Su et al. (2016)Fang-Hsiang Su, Jonathan
Bell, Kenneth Harvey, Simha
Sethumadhavan, Gail E. Kaiser, and Tony
Jebara. 2016.Code relatives: detecting similarly behaving
software. In *Proceedings of the 24th International
Symposium on Foundations of Software Engineering*.
ACM, Seattle, WA, USA,
702–714.
* Sun and Chen. (2022)Weisong Sun and Yuchen
Chen. 2022.Source Code and Dataset of TranCS.site: <https://github.com/wssun/TranCS>.Accessed: 2022.
* Tai
et al. (2015)Kai Sheng Tai, Richard
Socher, and Christopher D. Manning.
2015.Improved Semantic Representations From
Tree-Structured Long Short-Term Memory Networks. In*Proceedings of the 53rd Annual Meeting of the
Association for Computational Linguistics*. The
Association for Computer Linguistics, Beijing, China,
1556–1566.
* Tufano et al. (2018)Michele Tufano, Cody
Watson, Gabriele Bavota, Massimiliano Di
Penta, Martin White, and Denys
Poshyvanyk. 2018.Deep learning similarities from different
representations of source code. In *Proceedings of
the 15th International Conference on Mining Software Repositories*.
ACM, Gothenburg, Sweden,
542–553.
* Vaswani et al. (2017)Ashish Vaswani, Noam
Shazeer, Niki Parmar, Jakob Uszkoreit,
Llion Jones, Aidan N. Gomez,
undefinedukasz Kaiser, and Illia
Polosukhin. 2017.Attention is All You Need. In*Proceedings of the 31st Annual Conference on Neural
Information Processing Systems*. Curran Associates
Inc., Long Beach, CA, USA, 5998–6008.
* Wan
et al. (2019)Yao Wan, Jingdong Shu,
Yulei Sui, Guandong Xu,
Zhou Zhao, Jian Wu, and
Philip S. Yu. 2019.Multi-modal Attention Network Learning for Semantic
Source Code Retrieval. In *Proceedings of the 34th
International Conference on Automated Software Engineering*.
IEEE, San Diego, CA, USA,
13–25.
* Xia
et al. (2017)Xin Xia, Lingfeng Bao,
David Lo, Pavneet Singh Kochhar,
Ahmed E. Hassan, and Zhenchang Xing.
2017.What do developers search for on the web?*Empirical Software Engineering*22, 6 (2017),
3149–3185.
* Xu
et al. (2021)Ling Xu, Huanhuan Yang,
Chao Liu, Jianhang Shuai,
Meng Yan, Yan Lei, and
Zhou Xu. 2021.Two-Stage Attention-Based Model for Code Search
with Textual and Structural Features. In*Proceedings of the 28th International Conference on
Software Analysis, Evolution and Reengineering*.
IEEE, Honolulu, HI, USA,
342–353.
* Xue
et al. (2019)Yinxing Xue, Zhengzi Xu,
Mahinthan Chandramohan, and Yang Liu.
2019.Accurate and Scalable Cross-Architecture Cross-OS
Binary Code Search with Emulation.*IEEE Transactions on Software Engineering*45, 11 (2019),
1125–1149.
* Yao
et al. (2019)Ziyu Yao,
Jayavardhan Reddy Peddamail, and Huan
Sun. 2019.CoaCor: Code Annotation for Code Retrieval with
Reinforcement Learning. In *Proceedings of the 28th
The World Wide Web Conference*. ACM,
San Francisco, CA, USA, 2203–2214.
* Zeng et al. (2021)Chen Zeng, Yue Yu,
Shanshan Li, Xin Xia,
Zhiming Wang, Mingyang Geng,
Bailin Xiao, Wei Dong, and
Xiangke Liao. 2021.deGraphCS: Embedding Variable-based Flow Graph for
Neural Code Search.*CoRR* abs/2103.13020
(2021), 1–21.
* Zhu
et al. (2020)Qihao Zhu, Zeyu Sun,
Xiran Liang, Yingfei Xiong, and
Lu Zhang. 2020.OCoR: An Overlapping-Aware Code Retriever. In*Proceedings of the 35th International Conference on
Automated Software Engineering*. IEEE,
Melbourne, Australia, 883–894.
