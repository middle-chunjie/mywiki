# CoCoMIC: Code Completion By Jointly Modeling In-file and Cross-file Context

Yangruibo Ding $^{1,\ast,\dagger}$  Zijian Wang $^{2,\ast,\ddagger}$  Wasi Uddin Ahmad $^{2,*}$   
Murali Krishna Ramanathan $^{2}$  Ramesh Nallapati $^{2}$  Parminder Bhatia $^{2}$   
Dan Roth $^{2}$  Bing Xiang $^{2}$

<sup>1</sup>Columbia University <sup>2</sup>AWS AI Labs

yrbding@cs.columbia.edu {zijwan,wuahmad}@amazon.com

{mkraman, rnallapa, paramib, drot, bxiang}@amazon.com

# Abstract

While pre-trained language models (LM) for code have achieved great success in code completion, they generate code conditioned only on the contents within the file, i.e., in-file context, but ignore the rich semantics in other files within the same project, i.e., cross-file context, a critical source of information that is especially useful in modern modular software development. Such overlooking constrains code language models' capacity in code completion, leading to unexpected behaviors such as generating hallucinated class member functions or function calls with unexpected arguments. In this work, we develop a cross-file context finder, CCFINDER, that locates and retrieves the most relevant cross-file context. We propose CoCOMIC, a framework that incorporates cross-file context to learn the in-file and cross-file context jointly on top of pretrained code LMs. CoCOMIC successfully improves the existing code LM with a  $33.94\%$  relative increase in exact match and a  $28.69\%$  relative increase in identifier matching for code completion when the cross-file context is provided.

# 1 Introduction

In recent years, language models for source code like Codex (Chen et al., 2021) and CodeGen (Nijkamp et al., 2022) have shown promising performance in code completion tasks and have great potential to improve developer productivity (Barke et al., 2022). These code LMs are typically trained with causal language modeling loss and complete the code conditioning on the previous code tokens in the same file, which we refer to as in-file context.

Modular programming (Parnas, 1972; Parnas et al., 1985; Sullivan et al., 2001) is a software design strategy that divides the complex software functionality into several independent, interchangeable components (e.g., files, classes, and functions), such that each component implements only one aspect of the desired functionality and consequently

Figure 1: CodeGen-2B-mono fails to complete a Python program correctly as in-file context does not provide sufficient information. The model needs to know TagHandler takes an argument rawtags, which could be obtained through the function list-tags of git. Generating the correct code requires the presence of class and function definitions as part of the context, which cannot be derived from the current file alone.

becomes easily reusable and testable. It has already been a well-adapted paradigm in modern software development and maintenance. Developing under the modular programming paradigm requires knowledge from the current file and the whole project, to which we refer as cross-file context. As shown in Figure 1, the cross-file context is critical for code completion: the CodeGen Python model (Nijkamp et al., 2022) with 2 billion parameters fails to generate the correct code since it only considers in-file context and lacks visibility to various crucial references for code completion, e.g., member functions of imported classes and arguments of imported functions.

In this work, we argue that code LMs should generate code conditioned jointly on in-file context and cross-file context. However, there are challenges in developing such models. First, the project defines its individual and complex hierarchy and could be of varied sizes. Thus, given a piece of code, it is critical yet challenging to identify the most relevant and useful cross-file context. Second, we must carefully design a framework for aggregating the

information from the in-file and cross-file context. Naively concatenating code from in-file and cross-file context is not feasible for three reasons. (1) They represent distinct types of contextual information, as the former presents the local dependencies and human intentions (e.g., code comments) for code completion, while the latter compensates for the project-level dependencies that do not exist in the surrounding lines. Thus, the model should not always treat them equally. (2) Unlike third-party packages which are mostly available in the pretraining dataset of LLMs, the same project context are likely to be private to the model, i.e., the model didn't see it during pre-training. This makes code completion that requires same project context very difficult if without the right context at inference time. (3) The model's input length is limited, so concatenating all context as input would exceed its context length capacity.

To address the aforementioned challenges, we build CCFINDER, a cross-file context finder that effectively retrieves the most relevant cross-file context given a code snippet to be completed. Furthermore, we propose CoCoMIC, a novel framework that jointly learns in-file and cross-file context to improve code completion.

Cross-file Context Finder We design and implement CCFINDER, a static code analysis tool, to retrieve the most relevant cross-file context for code completion. CCFINDER parses the project hierarchy and code components to extract project information. CCFINDER further builds a project context graph to represent the details of each component (i.e., entity) and the interactions among them (i.e., relation). When an incomplete program requests completion, the tool will first analyze its import statements and pinpoint the related entities from the built context graph. Then, the tool will retrieve the neighbors of the pinpointed entities from the graph as the cross-file context of the current file.

Jointly Modeling In-file and Cross-file Context We propose CoCoMIC, a novel framework built on top of existing code LMs with joint attention to in-file and retrieved cross-file context. We realize this in two steps: First, the model will compress cross-file context and build its representations. Second, when generating code completion, the model will attend to both the compressed cross-file context and the concrete in-file context.

We evaluate the effectiveness of CCFINDER and CoCoMIC on a code completion dataset we built

from the Python Package Index (PyPI), a repository of open-source Python projects. We show that CCFINDER is able to retrieve  $27.07\%$  more relevant context for code completion than in-file context. Experiments show that CoCOMIC with access to relevant cross-file context improves the backbone pretrained code LM, CodeGen (Nijkamp et al., 2022), by  $33.94\%$  in exact match and  $28.69\%$  identifier matches relatively. Our main contributions are:

1. We present CoCoMIC, a novel framework built on top of code LMs that jointly learns in-file and cross-file context to enhance code completion. To power CoCoMIC, we develop CCFINDER, an effective tool at harvesting cross-file context, a critical yet overlooked resource for code completion in the era of modern software development.  
2. We show that CoCOMIC with cross-file context from CCFINDER significantly outperforms baselines by up to  $+33.94\%$  in exact match. We additionally conduct extensive ablation studies to show the contribution of different components.  
3. We release a diverse and high-quality dataset on statement-level code completion that tests model's ability on making use of cross-file context to facilitate further research.

# 2 Preliminaries

For the convenience of discussion, we define concepts that will be used throughout the paper.

Project Entities Project entities are code components that constitute the skeleton of software projects; developers frequently import and reuse these entities as cross-file context. We focus on four types of entities: file, function, class, and global variable. In particular, file contains the file name and file docstring; class contains the class signature, docstring, and attributes; function contains function signature, docstring, and body; global variable contains the variable name and its value.

Entity Relations Entity relations represent the interactions among project entities. We consider two categories of relations: intra-file and inter-file. Intra-file relations describe the in-file code hierarchies pre-defined by the programming language grammar. For example, a class is at the first level of the hierarchy while its member functions are at the

Stage-1: Project Context Graph Construction  
Figure 2: Overview of CCFINDER. First, CCFINDER builds the project context graph, including the bird's-eye view of the whole project and the code details of each module. Then, given the incomplete program, CCFINDER retrieves a set of the most relevant project entities as cross-file context from the graph.

second level. Inter-file relations define the file-to-file dependencies. Under each category, we further define several types of relations (Appendix A).

Locale We define locale as the entity's relative code location within the software project. For example, the locale of class entities is defined as file_name.class_name. The locale is assigned a unique name according to the specific location of a project entity, so we maintain the one-to-one mapping between each entity and its locale. The locale benefits CoCoMIC in two ways: (1) when we construct cross-file context, the locale efficiently maps the relative path of a code snippet to its project entity CCFINDER builds (§3.2), and (2) it indicates hierarchical relations among project entities and helps model with code completion (§6.4).

In-file & Cross-file Context For an incomplete source file  $S$ , we define two types of context: in-file and cross-file. In-file context represents code snippets included in the current file, i.e., code tokens before the predicting position. Cross-file context  $C$  represents the relevant code information (e.g., classes, functions) from the same project that is out of but imported by the current file. Concretely, cross-file context refers to a collection of relevant project entities that might assist with the missing code prediction but are not in  $S$ .

# 3 Cross-file Context Finder: CCFINDER

Software projects typically have complex structures (Parnas et al., 1985) representing the dependencies among distinct code components. To retrieve the most relevant code snippets given a code, we need a tool with two main characteristics. First, the tool

should be able to navigate the project structure to identify the file and module dependencies. Second, the tool can zoom into the dependencies and extract detailed code components. Off-the-shelf tools do not meet the requisites. For example, module dependency analysis tools $^{2,3}$  can only provide the module interactions while missing the hierarchical details inside each module and cannot directly output the concrete code. Therefore, we develop a new tool, CCFINDER, to aggregate cross-file context.

CCFINDER's overall workflow is shown in Figure 2. It has two main steps: (1) Analyze the program dependencies to build a bird's-eye view of the whole project and parse the source code to extract code details of each module. With these, CCFINDER builds the project context graph: graph nodes represent code components that constitute the project's backbone, and edges indicate the relations among components. (2) Given an incomplete program, the tool retrieves the most relevant cross-file context from the built graph. In this work, we focus on Python as the proof-of-concept to showcase our main arguments. However, CCFINDER's conceptual design is extensible to other languages.

# 3.1 Project Context Graph

CCFINDER parses the project structure and corresponding source files to identify the project entities and entity relations. Then, CCFINDER uses entities and entity relations to build graph nodes and directed edges, respectively. The context graph is built top-down. First, we create a root node for the project and connect it with all file nodes. Sec-

ond, each file node will build its own sub-graph, wrapping code components within the file, and also build connections with other files that it depends on, i.e., it imports code from these files. Third, nodes will link to others within the file-level sub-graph based on the dependencies or scope. For example, a class node will have edges to its member functions.

Formally, CCFINDER builds the multi-relational, directed context graph  $\mathcal{G} = (\mathcal{V},\mathcal{E})$  for the project, where  $\mathcal{V}$  is the set of nodes representing code components, and  $\mathcal{E}\subseteq \mathcal{V}\times \mathcal{R}\times \mathcal{V}$  is the set of edges that indicate the interactions among code components, where  $\mathcal{R}$  is the set of edge types (Appendix A).

Note that the project context graph generated by CCFINDER differs from the traditional program dependence graph (Ferrante et al., 1987) and code property graph (Yamaguchi et al., 2014), which are built to estimate and analyze the program execution behaviors statically. These graphs focus on data flows and control flows, while our graph represents the dependencies of different modules within the project. CCFINDER-generated graph is also different from the code knowledge graph (Abdelaziz et al., 2021, 2022) as the latter combines API usage knowledge such as third-party documentation and StackOverflow questions and answers.

# 3.2 Cross-file Context Retrieval

The project context graph represents the project hierarchies and interactions among code components, so the closer a graph neighbor to a specific node, the more relevant that neighbor is. For example, if the input code imports a class, the most useful information regarding this class, such as its member functions and the global variables it depends on, should be only 1 or 2 hops away. Thus, given an input code, we retrieve a set of relevant nodes from the context graph as the cross-file context.

The workflow is presented in Algorithm 1. First, we extract the import statements from the incomplete code file  $(\mathcal{F})$  that only imports code snippets within the same project (GetLocalImportStmt). We iteratively use each import to identify and locate corresponding nodes in the project context graph (LocateNode). With the direct mapping between the code snippets and their locales ( $\S 2$ ), we can locate the node given its relative path. We use such a node as the root node and retrieve its neighbors within  $k$  hops using the depth-first graph search (DepthFirstSearch).  $k$  is a configurable hyper-parameter in which increasing  $k$

Algorithm 1 Retrieve Cross-file context  
$\mathcal{F}$  : Incomplete code file   
 $\mathcal{G}$  : Project context graph   
 $\mathcal{P}$  : Parsing project information   
 $k$  : Maximum depth of graph search   
1: ctx_nodes  $\leftarrow \emptyset$    
2:  $\mathcal{I}\gets$  GetLocalImportStmt(F)   
3: for stmt  $\in \mathcal{I}$  do   
4: root  $\leftarrow$  LocateNode(G,stmt)   
5:  $\mathcal{N}\gets$  DepthFirstSearch(G,root,k)   
6: for  $n\in \mathcal{N}$  do   
7: if  $n\notin$  ctx_nodes then   
8: ctx_nodes.add(n)   
9: end if   
10: end for   
11: ReorderNode(ctx_nodes,  $\mathcal{P}$  12: end for   
13: return ctx_nodes

will retrieve a broader context. We set  $k = 2$  for all the experiments in this work and empirically justify the choice in §6.2. Finally, once we collect the  $k$ -hop neighbors for all import statements, we re-order the nodes (ReorderNode), ensuring the nodes from the same source file follow the original code order, to maintain the naturalness (Hindle et al., 2012) of human-written code.

# 4 Proposed Framework: CoCoMIC

A high-level overview of the CoCoMIC framework is presented in Figure 3. CoCoMIC uses an autoregressive LM to encode (1) in-file code snippet and (2) retrieved cross-file context, and predicts the next code token conditioning on both.

# 4.1 Input Representation

As shown in Figure 3, the model input includes two parts: source code sample  $S$  and its cross-file context  $\mathcal{C}$ . Specifically, the source code sample  $S$  consists of a sequence of tokens  $x_{1},\ldots,x_{T}$ , where  $x_{t}$  is a code token and  $T$  is the length of  $S$ ; the cross-file context, as introduced in §3, is a list of entities,  $\mathcal{C} = (c_{1},\dots,c_{n})$ , retrieved from the project context graph. Each entity,  $c_{i}$ , is a short piece of code sequence describing the details of that entity, i.e.,  $c_{i} = (\text{locale}_{i},w_{i}^{1},\dots,w_{i}^{m},[\text{SUM}])$ , where  $w_{i}^{j}$  is a code token within the entity,  $\text{locale}_{i}$  is the locale (§2) of  $c_{i}$ , and [SUM] is a special token.

Representing Entity Relations with Locales As introduced in §2, each project entity is paired with a locale that indicates its hierarchical relationship. We explore the benefits of preponding locales to

Figure 3: The CoCoMIC framework. Bottom: Given incomplete code, CoCoMIC leverages CCFINDER to identify the corresponding entities in the project context graph (§3.1) and retrieve their k-hop neighbors as cross-file entities (§3.2). Up: CoCoMIC first generates representations for cross-file entities using the appended [SUM] token (§4.2). Then it completes the current code by jointly attending to in-file and cross-file context (§4.3).

provide entities with such relational hints (§6.4). Specifically, for each cross-file entity, we pretend its locale to its code text as a comment, followed by a new line character: for the example in Figure 3, the retrieved entity def listtags() will be pretended with #git.listtags\n.

Better Entity Representation with [SUM] We append a special token [SUM] to entity descriptions. We expect [SUM] token to learn the summarization of the entity since the causal attention (Radford et al., 2019; Brown et al., 2020) allows it to attend to all the previous tokens describing the entity. When completing code, the model will attend to the representations of the [SUM] tokens for each cross-file entity. We compare it with mean pooling in §6.3 and show that [SUM] works better.

# 4.2 Encoding Cross-file Context

The computational cost of Transformers increases exponentially w.r.t. the input length, so it is impractical to comprehend all the retrieved entities as plain text, as they typically contain thousands of tokens (Appendix C). Also, only a few keywords in an entity (e.g., identifiers) play an important role in assisting code completion. Thus, CoCoMIC encodes each entity into a single token to balance the space limitation and the information need.

$$
h _ {c _ {i}} = f _ {\theta} \left(c _ {i}\right) \in \mathbb {R} ^ {d _ {h}}, H _ {\mathcal {C}} = \left(h _ {c _ {1}},..., h _ {c _ {n}}\right) \in \mathbb {R} ^ {n \times d _ {h}}
$$

Specifically, for each entity  $c_{i}$ , the model  $f_{\theta}$  will encode its code sequence into one representation  $h_{c_i} \in \mathbb{R}^{d_h}$ , where  $d_h$  is the hidden dimension.

Then, CoCoMIC takes the hidden state of the last token, [SUM], as the entity representation. Finally, the model will output a list of entity embeddings,  $H_{\mathcal{C}}$ , representing the retrieved cross-file context.

# 4.3 Modeling In-file and Cross-file Context for Code Completion

After getting representations of cross-file context, CoCoMIC continues to encode the in-file context and train the model to learn both context jointly.

In-file Context CoCoMIC utilizes the causal language model setting to support the code completion task, where each token will consider its former texts as in-file context. Specifically, the in-file context of source code  $S$ , at time step  $t$ , will be  $s_t = (x_1, \dots, x_{t-1})$ . We pass these tokens through the model and get the embeddings of each token to construct the representation of the in-file context.

$$
H _ {\mathcal {S}} (t) = f _ {\theta} \left(s _ {t}\right) = f _ {\theta} \left(x _ {1},..., x _ {t - 1}\right) \in \mathbb {R} ^ {(t - 1) \times d _ {h}}
$$

Joint attention to In-file and Cross-file Context Different layers of a Transformer model have been shown to capture different language components (e.g., lower layers learn language syntax or grammar while upper layers capture language semantics (Jawahar et al., 2019)). We hypothesize that both in-file and cross-file context contribute to forming the understanding of language components. Therefore, we fuse the in-file and cross-file context at each Transformer layer so that generating the next token's hidden state will always depend on both context. At each time step  $t$ , for the  $l$ -th layer, we

first compute the keys and values for cross-file and in-file context, using their  $(l - 1)$ -th hidden states.

$$
K _ {C} = H _ {C} ^ {[ l - 1 ]} \mathbf {W} ^ {K}, V _ {C} = H _ {C} ^ {[ l - 1 ]} \mathbf {W} ^ {V}
$$

$$
K _ {S} (t) = H _ {S} (t) ^ {[ l - 1 ]} \mathbf {W} ^ {K}, V _ {S} (t) = H _ {S} (t) ^ {[ l - 1 ]} \mathbf {W} ^ {V}
$$

Then, we concatenate the keys and values from both context so that, at time step  $t$ , the generating token can jointly attend them.

$$
K (t) = \operatorname {c o n c a t} \left(K _ {\mathcal {C}}, K _ {\mathcal {S}} (t)\right), V (t) = \operatorname {c o n c a t} \left(V _ {\mathcal {C}}, V _ {\mathcal {S}} (t)\right)
$$

$$
Q (t) = f _ {\theta} \left(x _ {t}\right) ^ {[ l - 1 ]} \mathbf {W} ^ {Q}, A t t n (t) = \operatorname {s o f t m a x} \left(\frac {Q (t) K (t) ^ {\top}}{\sqrt {d _ {K}}}\right) V (t)
$$

# 5 Experiment Setup

# 5.1 Data

Our data stem from the Python Package Index.4 We collect permissively-licensed projects and filter out those with too few files ( $\leq 5$  python files) or too memory-consuming to build the project context graph ( $\geq 5\mathrm{k}$  nodes), ending up with 60,891 projects. Then, we divide the dataset into  $80\% /10\% /10\%$  train, validation, and test sets. We notice that popular packages, such as numpy, are used as dependencies by many packages and will cause potential information leakage if numpy is part of the test set. Thus, we only include projects that were not used as dependencies by any training projects in the test set. We create prompts by cutting the source file at the location where completion requires cross-file context. See Appendix C for more details.

Figure 1 shows an example prompt we create: it requires the details of TagHandler and git to complete the code accurately. In this work, we consider statement-level code completion, so the ground truth of the test sample is built accordingly. For the convenience of studying the model's prediction on local APIs (i.e., APIs defined within the project), we further filter out the samples that either can not be parsed by the AST parser or do not include local API calls in the target statement (to be completed). Finally, we ended up with the 6,888 held-out prompts for evaluation.

# 5.2 Implementation Details

Cross-file Context CCFINDER uses tree-sitter<sup>5</sup> to parse source code files. Tree-sitter is a widely used source code parser that generates the abstract syntax tree (AST) given a program. CCFINDER will traverse the AST to extract information as

described in §3. Then, CCFINDER analyzes the import statements on top of import-dep $^6$  to build the project context graph. In this work, we retrieve 2-hop neighbors with at max 128 project entities as cross-file context and each entity contains up to 128 tokens. These thresholds are data-driven to ensure the model input covers most of the relevant cross-file context (more details in Appendix B).

Model The backbone of CoCoMIC is CodeGen (Nijkamp et al., 2022) and we use CodeGen-350M-Mono for all experiments. In all settings, we finetune the model for 5 epochs with max sequence length of 2,048 tokens and learning rate of 5e-5 with  $5\%$  warm-up steps then cosine annealing.

# 5.3 Baselines & Evaluation Metrics

CodeGen We consider two variations of the vanilla CodeGen model with in-file context only: (1) zero-shot, where we directly evaluate the pretrained CodeGen model on our test dataset, and (2) finetuned, where we finetune CodeGen on our dataset first and then evaluate.

CodeGen w/ Cross-file Context We also consider a prompting baseline where we prepend the cross-file context to the input sequence and fine-tune. Similar to the configuration of CoCoMIC, we reserve the first 128 tokens of the input for the code tokens from the cross-file context and use the rest tokens for the in-file context.

Evaluation Metrics We compute exact match (EM) and BLEU-4 (Papineni et al., 2002) to assess the accuracy of the generated code. While code match indicates the overall correctness of code completion, we want to zoom into the cases where cross-file context could most contribute, which is API usage. Therefore, we measure the identifier match to evaluate whether cross-file context improves the model's ability to predict the right APIs. To this end, we extract the identifiers from the model prediction and the ground truth, resulting in two ordered lists of identifiers. Then, we compare them and report the identifier match results in exact match, precision, and recall.

Besides, we compute the perplexity of all the tokens on the test set to study whether adding cross-file context degrades performance when the cross-file context is not explicitly required.

<table><tr><td rowspan="2">Model</td><td rowspan="2">Finetuned</td><td rowspan="2">Cross-file Entities</td><td colspan="2">Code Match</td><td colspan="3">ID Match</td><td rowspan="2">PPL (↓)</td></tr><tr><td>EM</td><td>BLEU-4</td><td>EM</td><td>Prec.</td><td>Rec.</td></tr><tr><td>CodeGen</td><td>X</td><td>X</td><td>14.56</td><td>33.12</td><td>22.91</td><td>47.74</td><td>50.75</td><td>2.88</td></tr><tr><td>+ Finetune</td><td>✓</td><td>X</td><td>15.97</td><td>35.11</td><td>24.29</td><td>50.46</td><td>53.07</td><td>2.87</td></tr><tr><td>+ Cross-file context</td><td>✓</td><td>✓</td><td>17.00</td><td>36.34</td><td>25.80</td><td>48.91</td><td>54.76</td><td>2.77</td></tr><tr><td>CoCoMIC (Ours)</td><td>✓</td><td>✓</td><td>21.39</td><td>41.65</td><td>31.26</td><td>55.45</td><td>57.83</td><td>2.69</td></tr></table>

# 6 Results and Analysis

# 6.1 CoCoMIC Outperforms the Baselines

We present the results in Table 1. CoCoMIC outperforms all baselines on all metrics with a clear margin, demonstrating the effectiveness of our proposed framework. We notice that when the cross-file context is prepended as a plain text prompt, CodeGen outperforms the other two baselines without cross-file context. However, limited by the maximum input length, it can only include a very limited amount of cross-file context, which significantly restricts its capacity. In contrast, CoCoMIC encodes the code sequence of an entity into one single token, enabling the model to incorporate more cross-file context while saving the input length. We present additional ablations for the baseline model in Appendix E.2, and case studies in Appendix F.

Besides, we see no degradation when the cross-file context is not explicitly required. We calculate the perplexity of all tokens in the test samples, regardless of whether they require cross-file context. We see that CoCoMIC achieves the lowest perplexity, indicating cross-file context in CoCoMIC is generally beneficial for code completion.

# 6.2 CCFINDER Retrieves Relevant Cross-file Context

The objective of CCFINDER is to locate and retrieve relevant code context from other source files in the project. Identifiers (e.g., function names and parameters) are presumably one of the most critical API information. Therefore, we study the effectiveness of CCFINDER by assessing whether their retrieved context increases recall of the identifiers that appear in the ground truth.<sup>7</sup>

Table 1: Performance of CoCOMIC compared with baselines. We show that using the text prompt for cross-file entities (row 3) helps marginally compared to the in-file-only baseline (row 2). On the contrary, CoCOMIC with cross-file context (row 4) improves the performance by a large margin (+33.94% Code Match EM and +28.69% ID Match EM) compared to the in-file only baseline. In addition, we show that there is no degradation in perplexity (PPL) when evaluating all the tokens in the test set where the cross-file context is not always required, suggesting that adding cross-file context helps in general. See Appendix F for additional case studies.  

<table><tr><td>Code Context Type</td><td>ID Recall (%)</td></tr><tr><td>In-file context</td><td>75.19</td></tr><tr><td>In-file + Cross-file context</td><td>95.55</td></tr></table>

Table 2: CCFINDER retrieves  ${27.07}\%$  more identifiers when compared to only in-file contexts.  

<table><tr><td rowspan="2">Entities From</td><td colspan="2">Code Match</td><td colspan="3">ID Match</td></tr><tr><td>EM</td><td>BLEU-4</td><td>EM</td><td>Prec.</td><td>Rec.</td></tr><tr><td>Random</td><td>15.68</td><td>35.23</td><td>24.07</td><td>49.75</td><td>52.69</td></tr><tr><td>CCFINDER (1-hop)</td><td>18.47</td><td>38.09</td><td>28.14</td><td>53.20</td><td>55.63</td></tr><tr><td>CCFINDER (2-hop)</td><td>21.39</td><td>41.65</td><td>31.26</td><td>55.45</td><td>57.83</td></tr></table>

Table 3: Entities retrieved from CCFINDER are more useful than random entities, and 2-hop retrieval help achieve better performance.

Table 2 shows that the in-file context covers (recall)  $75.19\%$  identifiers that appear in the ground truth. In comparison, prompts augmented with retrieved cross-file identifiers bring up identifier recall to  $95.55\%$ . This indicates that CCFINDER can retrieve most of the cross-file context that can help LM complete the input code. Note that while CCFINDER increases identifier recall by  $27.07\%$ , Table 1 shows only a  $7.08\%$  improvement in identifier recall. This indicates that building intelligent prompting techniques or training LMs to use cross-file context can lead to better performances. Further, Table 3 shows that random entities from the same project do not provide useful information since they are not necessarily related to the input code, and 2-hop retrieval outperforms 1-hop retrieval. These verify that CCFINDER retrieves relevant cross-file context and thus helps CoCoMIC. Appendix E.1 presents additional analysis.

# 6.3 Entity Representation with [SUM] Token

We append a special token [SUM] to cross-file context to summarize their information (Figure 3).

Now, we study the importance of the [SUM] token for a better representation of cross-file context. As a comparison, we apply the widely-used mean pooling that takes the mean over every cross-file token's embedding as the cross-file representation. We train a CoCoMIC model with mean pooling and keep the rest of the settings the same. The result is in Table 4: our proposed [SUM] token effectively summarizes cross-file context and significantly outperforms the mean pooling strategy.

<table><tr><td rowspan="2">CoCoMIC</td><td colspan="2">Code Match</td><td colspan="3">ID Match</td></tr><tr><td>EM</td><td>BLEU-4</td><td>EM</td><td>Prec.</td><td>Rec.</td></tr><tr><td rowspan="2">Mean pooling [SUM]</td><td>16.78</td><td>36.02</td><td>25.01</td><td>50.50</td><td>52.61</td></tr><tr><td>21.39</td><td>41.65</td><td>31.26</td><td>55.45</td><td>57.83</td></tr></table>

# 6.4 Localese Help Learning Cross-file Context

As introduced in §4.1, we prepend locales as relational hints for better entity representations. We study the effectiveness of such relational signals. As a comparison, we further study multi-task learning that encourages embedding relational information into entity representations.

Multi-task w/ Edge Prediction We use multi-task learning (MTL) to encode cross-file relations. Specifically, we train the model with an auxiliary edge prediction task among cross-file entities. We take representations of two cross-file entities generated by the LM layers and ask the model to predict what edge type connects them.

Results Table 5 presents the results. While MTL achieves  $97.2\%$  accuracy in the auxiliary edge prediction task, it hardly improves CoCoMIC in code completion. Such a gap suggests that even if MTL fulfills the expectation of embedding edge information, this information is not directly useful for code completion. In contrast, adding locales consistently improves CoCoMIC across all metrics. We hypothesize that this is due to locales providing an exact and direct signal as text (e.g., class_name.method_name). Thus the model could use them as short-cut in code completion.

# 7 Related Work

In the last couple of years, a significant effort has been made to pretrain Transformer language models using unlabeled source code (Feng et al., 2020; Ahmad et al., 2021; Wang et al., 2021b; Guo et al.,

Table 4: Representing cross-file context with [SUM] token significantly outperforms mean pooling alternative.  

<table><tr><td rowspan="2">CoCoMIC</td><td colspan="2">Code Match</td><td colspan="2">ID Match</td></tr><tr><td>EM</td><td>BLEU-4</td><td>EM</td><td>Prec. Rec.</td></tr><tr><td>No Relations</td><td>20.27</td><td>40.62</td><td>30.02</td><td>55.44 57.46</td></tr><tr><td>MTL</td><td>20.01</td><td>40.00</td><td>29.53</td><td>55.51 56.68</td></tr><tr><td>Locale</td><td>21.39</td><td>41.65</td><td>31.26</td><td>55.45 57.83</td></tr><tr><td>Locale + MTL</td><td>21.25</td><td>41.44</td><td>31.05</td><td>55.83 58.03</td></tr></table>

Table 5: Locales improve performance while learning cross-file relations with multi-task learning does not provide CoCoMIC more than marginal improvement.

2022; Ding et al., 2022) to facilitate software engineering applications (Husain et al., 2019; Iyer et al., 2018; Tufano et al., 2019; Zhou et al., 2019). Among these efforts, developing code generation models is noteworthy (Chen et al., 2021; Xu et al., 2022; Wang and Komatsuzaki, 2021; Black et al., 2021, 2022; Nijkamp et al., 2022; Fried et al., 2022; Li et al., 2022). Since most of these models are autoregressive language models, they can be directly used in code completion - given a code snippet as a prompt, generate the next  $N$  tokens. Until recently, existing works in the literature use code snippet from the current file (where the user is writing code) to prompt the code generation models. In a concurrent work, Zhou et al. (2022) proposed to retrieve API documentation given a natural language (NL) intent and generate code based on them. Our work has the same spirit as we propose to retrieve cross-file context (user-defined classes, functions from other project files) given a source code. The fundamental difference is that we utilize the import statements for structured retrieval.

While the use of in-file or class context is rigorously studied for software engineering applications in the literature, the use of cross-file context is relatively under-explored in code completion backed by code LMs. Earlier works (Henninger, 1991; Rosson and Carroll, 1996; Michail, 2001; Ye et al., 2000; Ye and Fischer, 2002; Cubranic and Murphy, 2003; Inoue et al., 2003; Hill and Rideout, 2004; Holmes and Murphy, 2005) in software engineering literature focused on developing tools to extract information from software repositories to help developers complete code fragments (e.g., variable, method name or body completion). On the other hand, recent works focus on modeling cross-file information in neural approaches. Wang et al. (2021a) proposed to model intra- and inter-class context for code summarization by extracting the Unified Modeling Language (UML) class diagrams. A recent work (Shrivastava et al.,

2022) proposed a prompt engineering technique that learns a repository-level prompt generator to generate example-specific prompts. A concurrent work (Zhang et al., 2023) proposed an iterative retrieval-generation framework to augment prompt with cross-file context.

# 8 Conclusion

The absence of cross-file context for code language models (LMs) limits their practicality in modern software development. In this work, we propose CoCoMIC, a framework that incorporates both in-file and cross-file context for code completion based on autoregressive code LMs. For this purpose, we build CCFINDER, a static code analysis tool that builds the project context graph, and find the most relevant cross-file context based on import statements. Empirical results show that CCFINDER successfully retrieves  $27.07\%$  more relevant context that are not in the current file, and our best CoCoMIC model achieves  $33.94\%$  relative improvement over the in-file-context-only baseline.

# Limitations

Extension to other languages and third-party packages Our work focuses on Python language, which is widely used and has great availability of open-sourced software projects through PyPI. However, the main concept introduced in our work should be extensible to other languages. In addition, we focus on the project (repo) context in this work, and a potential extension is to incorporate third-party packages and building models to suggest the right third-party libraries to use. We leave these as future work.

Model performances with the absence of cross-file context In this work, we assumed that CoCoMIC could access the other source code files within the project to understand source code dependencies and utilize them accordingly to generate the target code completion. However, CoCoMIC may not access the code files in many cases, e.g., users do not want an AI code LM to read their private or sensitive project APIs. Therefore, it is valid to ask - how CoCoMIC performs when the cross-file context is absent. We evaluate CoCoMIC without access to cross-file context and compare with finetuned CodeGen model (second row in Table 1). The results show that CoCoMIC performs  $5 - 7\%$  lower (relative performance drop) than finetuned CodeGen model. Development of training strate

gies to bridge this performance gap is needed, and we leave this as future work.

Impact on different sized language models Although we use CodeGen-350-mono model in this work which consists of 350M parameters, we hypothesize that larger LMs (e.g., 2B, 6B, or 16B variants of CodeGen) would result in similar or higher performance boost due to modeling crossfile context. However, we acknowledge that our work does not substantiate that our proposed technique would boost the performance of language models of any size.

# Ethics Statement

Our work aims at improving code generation with cross-file context to improve the usability of code LMs. We highlight the limitations of our work above. We do not expect our work to have a negative broader impact, though using code LMs always comes with certain risks, e.g., generating biased, toxic, and insecure code. We refer readers to Sec. 7 in (Chen et al., 2021) for a detailed discussion on the broader impact of code LMs. In addition, we reported our usage of computational resources in Appendix D.

# References

Ibrahim Abdelaziz, Julian Dolby, James P McCusker, and Kavitha Srinivas. 2021. A toolkit for generating code knowledge graphs. The Eleventh International Conference on Knowledge Capture (K-CAP).  
Ibrahim Abdelaziz, Julian Dolby, Jamie McCusker, and Kavitha Srinivas. 2022. Can machines read coding manuals yet? – a benchmark for building better language models for code understanding. In Proceedings of the AAAI Conference on Artificial Intelligence (AAAI 2022).  
Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang. 2021. Unified pre-training for program understanding and generation. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2655-2668, Online. Association for Computational Linguistics.  
Shraddha Barke, Michael B James, and Nadia Polikarpova. 2022. Grounded copilot: How programmers interact with code-generating models. *ArXiv preprint*, abs/2206.15000.  
Sid Black, Leo Gao, Phil Wang, Connor Leahy, and Stella Biderman. 2021. Gpt-neo: Large scale autoregressive language modeling with mesh-tensorflow. If you use this software, please cite it using these metadata, 58.

Sidney Black, Stella Biderman, Eric Hallahan, Quentin Anthony, Leo Gao, Laurence Golding, Horace He, Connor Leahy, Kyle McDonell, Jason Phang, Michael Pieler, Usvsn Sai Prashanth, Shivanshu Purohit, Laria Reynolds, Jonathan Tow, Ben Wang, and Samuel Weinbach. 2022. GPT-NeoX-20B: An open-source autoregressive language model. In Proceedings of BigScience Episode #5 – Workshop on Challenges & Perspectives in Creating Large Language Models, pages 95–136, virtual+Dublin. Association for Computational Linguistics.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language models are few-shot learners. In Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual.  
Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. 2021. Evaluating large language models trained on code. ArXiv preprint, abs/2107.03374.  
Davor Cubranic and Gail C Murphy. 2003. Hipikat: Recommending pertinent software development artifacts. In 25th International Conference on Software Engineering, 2003. Proceedings., pages 408-418. IEEE.  
Yangruibo Ding, Luca Buratti, Saurabh Pajar, Alessandro Morari, Baishakhi Ray, and Saikat Chakraborty. 2022. Towards learning (dis)-similarity of source code from program contrasts. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 6300-6312, Dublin, Ireland. Association for Computational Linguistics.  
Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun Shou, Bing Qin, Ting Liu, Daxin Jiang, and Ming Zhou. 2020. CodeBERT: A pre-trained model for programming and natural languages. In Findings of the Association for Computational Linguistics: EMNLP 2020, pages 1536-1547, Online. Association for Computational Linguistics.  
Jeanne Ferrante, Karl J. Ottenstein, and Joe D. Warren. 1987. The program dependence graph and its use in optimization. ACM Trans. Program. Lang. Syst., 9(3):319-349.

Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi, Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis. 2022. Incoder: A generative model for code infilling and synthesis. ArXiv preprint, abs/2204.05999.  
Daya Guo, Shuai Lu, Nan Duan, Yanlin Wang, Ming Zhou, and Jian Yin. 2022. UniXcoder: Unified cross-modal pre-training for code representation. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 7212-7225, Dublin, Ireland. Association for Computational Linguistics.  
Scott Henninger. 1991. Retrieving software objects in an example-based programming environment. In Proceedings of the 14th annual international ACM SIGIR conference on Research and development in information retrieval, pages 251-260.  
Rosco Hill and Joe Rideout. 2004. Automatic method completion. In Proceedings. 19th International Conference on Automated Software Engineering, 2004., pages 228-235. IEEE.  
Abram Hindle, Earl T. Barr, Zhendong Su, Mark Gabel, and Premkumar Devanbu. 2012. On the naturalness of software. In Proceedings of the 34th International Conference on Software Engineering, ICSE '12, page 837-847. IEEE Press.  
Reid Holmes and Gail C Murphy. 2005. Using structural context to recommend source code examples. In Proceedings of the 27th international conference on Software engineering, pages 117-125.  
Hamel Husain, Ho-Hsiang Wu, Tiferet Gazit, Miltiadis Allamanis, and Marc Brockschmidt. 2019. Code-searchnet challenge: Evaluating the state of semantic code search. arXiv preprint arXiv:1909.09436.  
Katsuro Inoue, Reishi Yokomori, Hikaru Fujiwara, Tetsuo Yamamoto, Makoto Matsushita, and Shinji Kusumoto. 2003. Component rank: Relative significance rank for software component search. In 25th International Conference on Software Engineering, 2003. Proceedings., pages 14-24. IEEE.  
Srinivasan Iyer, Ioannis Konstas, Alvin Cheung, and Luke Zettlemoyer. 2018. Mapping language to code in programmatic context. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 1643-1652, Brussels, Belgium. Association for Computational Linguistics.  
Ganesh Jawahar, Benoit Sagot, and Djamé Seddah. 2019. What does BERT learn about the structure of language? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, pages 3651-3657, Florence, Italy. Association for Computational Linguistics.  
Ziwei Ji, Nayeon Lee, Rita Frieske, Tiezheng Yu, Dan Su, Yan Xu, Etsuko Ishii, Yejin Bang, Andrea Madotto, and Pascale Fung. 2022. Survey of hallucination in natural language generation. ACM Computing Surveys.

Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser, Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago, et al. 2022. Competition-level code generation with alphabet. ArXiv preprint, abs/2203.07814.  
Amir Michail. 2001. Codeweb: Data mining library reuse patterns. In Proceedings of the 23rd International Conference on Software Engineering. ICSE 2001, pages 827-828. IEEE.  
Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio Savarese, and Caiming Xiong. 2022. Codegen: An open large language model for code with multi-turn program synthesis. ArXiv preprint, abs/2203.13474.  
Kishore Papineni, Salim Roukos, Todd Ward, and Wei Jing Zhu. 2002. Bleu: A method for automatic evaluation of machine translation. In Proceedings of the 40th Annual Meeting on Association for Computational Linguistics, ACL '02, page 311-318, USA. Association for Computational Linguistics.  
D. L. Parnas. 1972. On the criteria to be used in decomposing systems into modules. Commun. ACM, 15(12):1053-1058.  
D.L. Parnas, P.C. Clements, and D.M. Weiss. 1985. The modular structure of complex systems. IEEE Transactions on Software Engineering, SE-11(3):259-266.  
Alec Radford, Jeff Wu, Rewon Child, David Luan, Dario Amodei, and Ilya Sutskever. 2019. Language models are unsupervised multitask learners. OpenAI preprint.  
Mary Beth Rosson and John M Carroll. 1996. The reuse of uses in smalltalk programming. ACM Transactions on Computer-Human Interaction (TOCHI), 3(3):219-253.  
Disha Shrivastava, Hugo Larochelle, and Daniel Tarlow. 2022. Repository-level prompt generation for large language models of code. ArXiv preprint, abs/2206.12839.  
Kevin J. Sullivan, William G. Griswold, Yuanfang Cai, and Ben Hallen. 2001. The structure and value of modularity in software design. ESEC/FSE-9, page 99-108, New York, NY, USA. Association for Computing Machinery.  
Michele Tufano, Cody Watson, Gabriele Bavota, Massimiliano Di Penta, Martin White, and Denys Poshanyak. 2019. An empirical study on learning bug-fixing patches in the wild via neural machine translation. ACM Transactions on Software Engineering and Methodology (TOSEM), 28(4):1-29.  
Ben Wang and Aran Komatsuzaki. 2021. GPT-J-6B: A 6 Billion Parameter Autoregressive Language Model. https://github.com/kingoflolz/mesh-transformer-jax.

Yanlin Wang, Ensheng Shi, Lun Du, Xiaodi Yang, Yuxuan Hu, Shi Han, Hongyu Zhang, and Dongmei Zhang. 2021a. Cocosum: Contextual code summarization with multi-relational graph neural network. ArXiv preprint, abs/2107.01933.  
Yue Wang, Weishi Wang, Shafiq Joty, and Steven C.H. Hoi. 2021b. CodeT5: Identifier-aware unified pretrained encoder-decoder models for code understanding and generation. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 8696-8708, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.  
Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush. 2020. Transformers: State-of-the-art natural language processing. In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38-45, Online. Association for Computational Linguistics.  
Frank F Xu, Uri Alon, Graham Neubig, and Vincent Josua Hellendoorn. 2022. A systematic evaluation of large language models of code. In Proceedings of the 6th ACM SIGPLAN International Symposium on Machine Programming, pages 1-10.  
Fabian Yamaguchi, Nico Golde, Daniel Arp, and Konrad Rieck. 2014. Modeling and discovering vulnerabilities with code property graphs. In 2014 IEEE Symposium on Security and Privacy, pages 590-604.  
Yunwen Ye and Gerhard Fischer. 2002. Supporting reuse by delivering task-relevant and personalized information. In Proceedings of the 24th international conference on Software engineering, pages 513-523.  
Yunwen Ye, Gerhard Fischer, and Brent Reeves. 2000. Integrating active information delivery and reuse repository systems. ACM SIGSOFT Software Engineering Notes, 25(6):60-68.  
Fengji Zhang, Bei Chen, Yue Zhang, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang Lou, and Weizhu Chen. 2023. Repocoder: Repository-level code completion through iterative retrieval and generation. arXiv preprint arXiv:2303.12570.  
Shuyan Zhou, Uri Alon, Frank F Xu, Zhengbao Jiang, and Graham Neubig. 2022. Doccoder: Generating code by retrieving and reading docs. ArXiv preprint, abs/2207.05987.  
Yaqin Zhou, Shangqing Liu, Jingkai Siow, Xiaoning Du, and Yang Liu. 2019. Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks. In Advances in Neural Information Processing Systems, volume 32, pages 10197-10207. Curran Associates, Inc.

# A Edge Types of Project Context Graph

We provide details of edge types that we use to build our project context graph in Table 6. The edges of the project context graph are directional, so for each edge type, we further define the expected entity type of its tail (i.e., the entity that the edge is "from") and head (i.e., the entity that the edge is "to") for each edge type. We also consider the reverse edges for certain types so that CCFINDER could retrieve entity siblings conveniently: e.g., when a function is imported, CCFINDER could retrieve global variables it depends on by visiting the function's parent, i.e., the file, with an edge of Function Reverse type, and reach the global variable with an edge of Global Var. type.

<table><tr><td>Edge Type</td><td>Tail</td><td>Head</td></tr><tr><td>Project File</td><td>root</td><td>file</td></tr><tr><td>Import</td><td>file</td><td>file</td></tr><tr><td>Global Var.</td><td>file</td><td>global var.</td></tr><tr><td>Global Var. Reverse</td><td>global var.</td><td>file</td></tr><tr><td>Function</td><td>file</td><td>function</td></tr><tr><td>Function Reverse</td><td>function</td><td>file</td></tr><tr><td>Class</td><td>file</td><td>class</td></tr><tr><td>Class Reverse</td><td>class</td><td>file</td></tr><tr><td>Member Function</td><td>class</td><td>function</td></tr></table>

# B Statistics of Retrieved Entities

This section presents the statistics of the retrieved entities of all samples in our dataset. Specifically, we hope to know two things that help us decide the experiment setup: (1) how many entities will be retrieved for the source file (2) how many tokens<sup>8</sup> are there in the retrieved entity. Table 7 shows the ratio of source files that will retrieve project entities more than a specific threshold, and Table 8 ratio of entities that will contain tokens more than a specific threshold. CoCoMIC uses 128 as the maximum number of retrieved entities to be included in the model input and tokens within each entity. Consequently, CoCoMIC can always consider most of the cross-file context without causing too expensive computational overhead.

Table 6: The list of edge types we used to build the project context graph.  

<table><tr><td>Num of entities</td><td>&gt;32</td><td>&gt;64</td><td>&gt;128</td><td>&gt;256</td></tr><tr><td>Ratio (%)</td><td>42.43</td><td>22.32</td><td>8.99</td><td>6.93</td></tr></table>

Table 7: The ratio of the number of retrieved entities for the source code file with different thresholds.  

<table><tr><td>Num of tokens</td><td>&gt;32</td><td>&gt;64</td><td>&gt;128</td><td>&gt;256</td></tr><tr><td>Ratio (%)</td><td>32.41</td><td>21.88</td><td>13.20</td><td>6.47</td></tr></table>

# C Additional Details on Data Preprocessing

As we introduced in §4.1, the model input will be source code and its retrieved cross-file context. For training, we take the code files as samples. If a code file is too long, we split the code sequence into multiple chunks with the maximum length of the model input, and each chunk is paired with the same cross-file context. As the code files are from distinct PyPI projects, we assume the duplicated samples should be rare.

We follow the standard code completion setting for testing (Nijkamp et al., 2022; Xu et al., 2022), creating incomplete programs as prompts and asking the model to predict the following pieces of code. Specifically, we create prompts by cutting the source file at the location where completion requires cross-file context. We present the details of samples' sequence length, in terms of BPE subtokens, in Table 9. For cross-file context, we concatenate the text of all retrieved entities as a sequence and count the length. We could see that the cross-file context is typically long and could not be consumed as plain text together with the prompts. CoCoMIC designs to compress the entities with the special token [SUM] that effectively alleviates this limitation.

Table 8: The ratio of the number of tokens within the retrieved entity with different thresholds.  

<table><tr><td></td><td>Mean</td><td>Max</td><td>Median</td><td>Min</td></tr><tr><td>Prompts</td><td>1,354</td><td>32,599</td><td>758</td><td>7</td></tr><tr><td>Cross-file context</td><td>4,485</td><td>186,339</td><td>1,928</td><td>22</td></tr></table>

Table 9: Length statistics of prompts and cross-file context of the test set.

# D Additional Details on Experiments

Our code is based on Transformers (Wolf et al., 2020). We train our models on a machine with 8

Nvidia A100s. Each job takes around 50 hours (i.e., 400 GPU hours) to train. We perform one round of experiments only as it is very expensive to repeat the experiments many times. The hyperparameter used is from our initial small-scale grid search on hyperparameters, where we find that the final performance is relatively stable.

# E Additional Ablation Studies

# E.1  $k$ -hop Retrieval

As we see from Table 3,  $k = 1$  underperforms comparing to  $k = 2$ . This is because  $k = 1$  fetches less comprehensive context. For example, with the import statement import FileA as A, we can access class X's static member function Y as: A.classX.funcY through 2-hop retrieval, whereas 1-hop retrieval will not fetch. In fact, 1-hop retrieval won't fetch any class member function if only the file is imported, which happens frequently in Python. Given the great coverage of  $k = 2$  (Table 2) and given we found too many unrelated entities were retrieved if we use  $k > 2$ , we decided to use  $k = 2$  throughout the work.

# E.2 Additional Baseline Variants

In addition to the CodeGen w/ Cross-file Context baseline (§5.3) which uses the same cross-file context tokens as in CoCoMIC, we experimented with a simplified setting that only takes the locales and the signature prototypes (name, arguments, and default return types, if present) to fit in more cross-file context within the limited token space.

<table><tr><td></td><td colspan="2">Code Match</td><td colspan="3">ID Match</td></tr><tr><td></td><td>EM</td><td>BLEU-4</td><td>EM</td><td>Prec.</td><td>Rec.</td></tr><tr><td>CodeGen + Finetune</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ Cross-file context (full)</td><td>17.00</td><td>36.34</td><td>25.80</td><td>48.91</td><td>54.76</td></tr><tr><td>+ Cross-file context (simp.)</td><td>17.49</td><td>37.57</td><td>26.76</td><td>51.71</td><td>54.88</td></tr><tr><td>CoCoMIC</td><td>21.39</td><td>41.65</td><td>31.26</td><td>55.45</td><td>57.83</td></tr></table>

Table 10: All baselines significantly outperforms Co-CoMIC.

We see the performance only improves marginally when using simplified cross-file context, and it still underperforms CoCoMIC significantly. This suggests that baseline models have substantial limitations of sequence lengths that the performance is subpar even if we simplify the cross-file context, while CoCoMIC is capable of compressing up to 16,384 (=128x128) tokens of cross-file context into only 128 vectors, making cross-file context readily available for the model to use.

# F Case Study

We present a case study by comparing CoCoMIC with the finetuned CodeGen model with and without cross-file context.

# F.1 CoCoMIC vs. CodeGen finetuned w/o cross-file context

We present two qualitative examples in Figure 4 and 5 where Baseline refers to the CodeGen model finetuned w/o cross-file context. From Figure 4, we see that Baseline calls get_credits() function of AssumeRoleExecutor class, which does not exist. On the other hand, being able to access cross-file context, CoCOMIC predicts the correct function execute() that returns Credentials class instance. In Figure 5, we see similar behavior from Baseline as it predicts get_key_and_secret method. In language generation literature, such an inaccurate or unfaithful generation given the input is known as hallucination (Ji et al., 2022).

# F.2 CoCoMIC vs. CodeGen finetuned with cross-file context

We present two qualitative examples in Figure 6 and 7 where Baseline refers to the CodeGen model finetuned with cross-file context. From Figure 6, we see that Baseline calls parse() method of CommandLineArgs class, which does not exist. On the other hand, due to providing cross-file context, CoCoMIC predicts the correct function get cli_args() that returns CliArgs class instance. In Figure 7, we see similar behavior from Baseline as it predicts update_from_dict(schedule) method. In both cases, the Baseline fails to make an accurate prediction due to the truncation of cross-file context, while CoCoMIC predicts correctly as it encodes each cross-file context entity individually and then utilizes their embedding in the self-attention mechanism. In contrast to cross-file context truncation, we could improve the Baseline by effectively selecting the most useful entities from the ordered list of retrieved cross-file entities. We leave this as our future work.

Prompt  
```python
import sys   
from awss assumeassume_role_executor import AssumeRoleExecutor   
from awss assumeassume-role_executorFACTORY import AssumeRoleExecutorFactory   
from awss assume command_executor import CommandExecutor   
from awss assume command_line_args import CommandLineArgs   
from awss assume.data_models import CliArgs, Credentials   
from awss assume environment variable import EnvironmentVariable   
def main(): try: cli_args: CliArg  $=$  CommandLineArgs().getCLI_args() assumerole_executor: AssumeRoleExecutor  $\equiv$  AssumeRoleExecutorFactory.get_executor的权利(args) credentials<CURSOR_POSITION>   
Cross-file Context: Entities   
["assume-role/executor.AssumeRoleExecutor\nclass AssumeRoleExecutor(ABC):", "#AssumeRoleExecutor.execute\ndef execute(self) -> Credentials:\npass", "#assume-role/executor_factor.AssumeRoleExecutorFactory\nclass AssumeRoleExecutorFactory(object):", "#AssumeRoleExecutorFactory.get_executor\ndef get_executor的权利(args): >AssumeRoleExecutor:..." , ... more entities ..]   
Baseline Prediction :Credentials  $=$  assume-role/executor.getcredentials() CoCoMIC Prediction :Credentials  $=$  assume-role/executor.execute()
```

Prompt  
```python
from botocoreexceptions import ConfigNotFound, ProfileNotFound   
from . import persistence   
def cast(bool(value): return type(value)  $= =$  type("") and value.lower() in ('1', 'yes', 'true', 'on')   
def initialized.session, \*\*kwargs): session.session_var_map["keyring"]  $=$  ("keyring", None, False, cast(bool) try: if session.get_config_variable("keyring") != False: if session.profile is not None: profile  $=$  session.profile else: profile  $=$  "default" key, secret  $=$  persistence<CURSOR POSITION>
```

Cross-file Context: Entities  
```txt
["#persistence.get_creditsals\ndef get_creditsals profile):n key = keyring.get_password(\"awscli:key\", profile)\nsecret = keyring.get_password(\"awscli:secret\", profile)\nreturn (key, secret)","#persistence.set_creditsals\ndef set_creditsals profile, key, secret):\nkeyring.set_password(\"awscli:key\", profile, key)\nkeyring.set_password(\"awscli:secret\", profile, secret)"]\n
```

Figure 4: CoCoMIC vs. CodeGen finetuned w/o cross-file context: qualitative example-1.

# Baseline Prediction

Figure 5: CoCoMIC vs. CodeGen finetuned w/o cross-file context: qualitative example-2.

# CoCoMIC Prediction

.get_key_and_secretsession.keyring，profile)

.getcredentials/profile)

# Prompt

import sys

```python
from awssassumeassume-role_executor import AssumeRoleExecutor   
from awssassumeassume-role_executorFACTORY import AssumeRoleExecutorFactory   
from awssassume.command_executor import CommandExecutor   
from awssassume command_line_args import CommandLineArgs   
from awssassume.data_models import CliArgs, Credentials   
from awssassume.environment_variable import EnvironmentVariable   
def main(): try: cli_args: Cli<CURSOR_POSI TION>
```

# Cross-file Context included by CoCoMIC

```python
["#assume-role_executor.AssumeRoleExecutor\nclass AssumeRoleExecutor(ABC):", "#AssumeRoleExecutor.execute\ndef execute(self) -> Credentials:\n pass", "#assume-role_executor_factor.AssumeRoleExecutorFactory\nclass AssumeRoleExecutorFactory(object):","#AssumeRoleExecutorFactory.get_executor\ndef get_executor(ui_args: CliArgs) -> AssumeRoleExecutor: ... else:\n response_cache_args = ResponseCacheArgs(root_arm=cli_args.root_arm,\n..."  
... more entities...,"#CommandLineArgs.getCLI_args\ndef getCLI_args(self) -> CliArgs:..."  
... more entities...]
```

# Cross-file Context included by Baseline (truncated to 128 tokens)

```python
"#assume-role_executor.AssumeRoleExecutor\nclass AssumeRoleExecutor(ABC): #AssumeRoleExecutor.execute\ndef execute(self) -> Credentials:\n pass #assume-role_executor_factor.AssumeRoleExecutorFactory\nclass AssumeRoleExecutorFactory(object):#AssumeRoleExecutorFactory.get_executor\ndef get_executor(ci_args: CliArgs) -> AssumeRoleExecutor:\n assume-role_args = AssumeRoleArgs(root_arn=cli_args.root_arn, role_session_name=cli_args.root_session_name, region_name =cli_args(region_name)\nsecurity_tokenservice:SecurityTokenService = AssumeRole (assume-role_args)\nassume-role_executor: AssumeRoleExecutor = None\nif cli_args.no_cache is True:\n assume-role_executor = AssumeRoleNoCacheExecutor (security_token_service)\n else:\n response_cache_args = ResponseCacheArgs(root_arn=cli)"
```

Figure 6: CoCoMIC vs. CodeGen finetuned with cross-file context: qualitative example-1.

# Baseline Prediction

= CommandLineArgs().parse()


# CoCoMIC Prediction

= CommandLineArgs().get cli_args()

Prompt  
```python
import asyncio   
from charlesbot.util.http import http_get_auth_request   
from charlesbot slack slack Attachment import SlackAttachment   
import logging   
import traceback   
import json   
from charlesbot_pagerduty.pagerduty_schedule import PagerdutySchedule   
from charlesbot_pagerduty.pagerduty_user import PagerdutyUser   
log  $=$  logging.getLogger(_name_)   
@asyncio.coroutine   
def get_pagerduty schedules(token,subdomain): response  $=$  yield from http_get.auth_request( auth_string  $\equiv$  "token  $\equiv \% s$  % token, url  $\equiv$  "https://s.pagerduty.com/api/v1/schedules" % subdomain ） schedules  $= [ ]$  try: json_str  $=$  json.load(response) for schedule in json_str['schedules']: pd_schedule  $=$  PagerdutySchedule() pd_schedule<CURSOR_POSITION>
```

Cross-file Context included by CoCoMIC  
```python
["#pagerduty_schedule.PagerdutySchedule\nclass PagerdutySchedule(BaseObject):...", "#PagerdutySchedule._init\_ndef _init_(self, **kwargs):\nsuper().init_(*kwargs)\nif not self.oncall_users:\nself.oncall_users = []\nif not self.escalation_policies:\nself.escalation_policies = ["]", ... more entities...
```

"#PagerdutyUser.load\ndef load(self, entries_dict):..."  
Cross-file Context included by Baseline (truncated to 128 tokens)  
Baseline Prediction  
CoCoMIC Prediction  
```python
"#pagerduty_schedule.PagerdutySchedule\nclass  
PagerdutySchedule(BaseObject):\\n\nproperties = ['description',\\n'escalation_policies',\\n 'id',\\n 'name',\\n 'time-zone',\\n 'oncall_users']  
#PagerdutySchedule._init_\\ndef __init__(self, **kwargs):\\n  
super().__init__(**kwargs)\\n if not self.oncall_users:\\n  
self.oncall_users = [_]\n if not self.escalation"
```

Figure 7: CoCoMIC vs. CodeGen finetuned with cross-file context: qualitative example-2.

```python
.update_from_dict(schedule)
```


```txt
.load(schedule)
```

# Footnotes:

Page 1: <sup>1</sup>https://github.com/amazon-science/cocomic 
Page 2: $^{2}$ https://github.com/google/importlab <sup>3</sup>https://github.com/thebjorn/pydeps 
Page 5: 4https://pypi.org/ <sup>5</sup>https://tree-sitter.github.io/ <sup>6</sup>https://pypi.org/project/import-peps/ 
Page 6: <sup>7</sup>We hypothesize that the inclusion of identifiers needed to complete a code is likely to benefit CoCoMIC. 
Page 11: Practically, these will be BPE sub-tokens from CodeGen tokenizer. 
