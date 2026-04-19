Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing
===========================================================================

Jiayi Wei, Greg Durrett, Isil Dillig  
Department of Computer Science  
University of Texas at Austin  
{jiayi,gdurrett, isil}@cs.utexas.edu

###### Abstract

Developers often dedicate significant time to maintaining and refactoring existing code. However, most prior work on generative models for code focuses solely on creating new code, neglecting the unique requirements of editing existing code. In this work, we explore a *multi-round code auto-editing* setting, aiming to predict edits to a code region based on recent changes within the same codebase. Our model, Coeditor, is a fine-tuned CodeT5 model with enhancements specifically designed for code editing tasks. We encode code changes using a line diff format and employ static analysis to form large customized model contexts, ensuring appropriate information for prediction. We collect a code editing dataset from the commit histories of 1650 open-source Python projects for training and evaluation. In a simplified single-round, single-edit task, Coeditor significantly outperforms the best code completion approach—nearly doubling its exact-match accuracy, despite using a much smaller model—demonstrating the benefits of incorporating editing history for code completion. In a multi-round, multi-edit setting, we observe substantial gains by iteratively prompting the model with additional user edits. We open-source our code, data, and model weights to encourage future research and release a VSCode extension powered by our model for interactive usage.

*Keywords*code editing, code completion, transformer models, static analysis, software engineering

1 Introduction
--------------

In recent years, there has been enormous interest in applying transformer models for code generation*(Feng et al., [2020](#bib.bib1 ""); Ahmad et al., [2021](#bib.bib2 ""); Wang et al., [2021](#bib.bib3 ""); Chen et al., [2021](#bib.bib4 ""); Fried et al., [2022](#bib.bib5 ""); Allal et al., [2023](#bib.bib6 ""))*, which has led to impressive performance on tasks such as program synthesis*(Li et al., [2022](#bib.bib7 ""); Nijkamp et al., [2022](#bib.bib8 ""))*, program translation*(Lachaux et al., [2020](#bib.bib9 ""); Szafraniec et al., [2022](#bib.bib10 ""))*, type inference*(Jesse et al., [2022](#bib.bib11 ""); Wei et al., [2023](#bib.bib12 ""))*, and code auto-completion*(Guo et al., [2021](#bib.bib13 ""); Svyatkovskiy et al., [2021](#bib.bib14 ""); Nguyen and Nadi, [2022](#bib.bib15 ""); Zhang et al., [2023](#bib.bib16 ""))*.

While these approaches effectively help programmers *creating* new code, they are not as adept at assisting with *revising* existing code. Code completion tools like GitHub Copilot do not track programmers’ changes and cannot predict where and how to make additional modifications. However, during a software project’s development cycle, developers often spend significant time editing code—changes made to one part of the codebase typically affect many others, and manually propagating these changes can be tedious and time-consuming.

In this paper, we introduce a task that we call (multi-round) *auto-editing* where the goal is to predict edits to code conditioned on the user’s previous edits. In particular, given an original codebase $U$ and a set of code changes $\Delta_{1},\ldots,\Delta_{k}$ that are semantically related (like those forming part of a commit), the auto-editing problem is to predict how to modify a specified region of code $u\in U$ by learning the following distribution:

|  | $P(\Delta u\mid\Delta_{k}\ldots\Delta_{1},U)\ .$ |  | (1) |
| --- | --- | --- | --- |

Importantly, we allow the target region region $u$ to overlap with any previous modifications $\Delta_{1},\ldots,\Delta_{k}$ to support repeated editing to the same region. This formulation enables the workflow illustrated in [Figure 1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), where a user can work alongside the model in multiple editing rounds, accepting suggestions matching the user’s intent and making additional edits manually if necessary.

To solve this problem, we propose a new model called Coeditor that builds on top of the established CodeT5*(Wang et al., [2021](#bib.bib3 ""))* model architecture and pre-trained checkpoint. Coeditor is based on two key ideas. First, it encodes all prior code edits $\Delta_{1},\ldots,\Delta_{k}$ using a line-based diffing scheme and decodes $\Delta u$ using masked span infilling; and, second, it uses lightweight static analysis to pull in relevant parts of the codebase $U$. To effectively handle large contexts with numerous code changes, we also replace CodeT5’s dense attention with a block-sparse attention pattern, allowing us to reduce the computation cost while maintaining the ability to attend to all relevant code changes.

Another challenge in developing Coeditor is the lack of suitable training data for multi-round auto-editing. We address this issue by collecting a new dataset, PyCommits, from the commit histories of 1650 open-source Python projects on GitHub. We compute tree-differences between adjacent codebase versions to identify modifications to the same Python function and randomly split some changes into the model input for training in repeated editing scenarios. During testing, we use ground truth code changes to simulate user decisions regarding when to accept partial changes suggested by the model and when to manually perform edits missed by the model.

We compare our approach against existing code infilling models and show that, even in a simplified setting that requires predicting a *single* edited line in isolation, they severely lag behind our change-aware model: our method achieves 60.4% exact match accuracy, almost twice that of the best performing code infilling model despite using a model that is 30x smaller. In the full multi-round setting, we found that Coeditor automates editing 46.7% of the changed lines, saving the user 28.6% of keystrokes measured by an edit distance metric that accounts for cursor movement.

In summary, this paper presents the following main contributions:

* •

    We introduce the multi-round code editing suggestion task, along with the corresponding PyCommits dataset and evaluation framework.

* •

    We introduce a new code editing model derived from CodeT5, using a line diff-based encoding scheme and enhancements that enable the model to condition on long contexts and appropriate other parts of the codebase, addressing key challenges in this setting.

* •

    We release our source code, dataset, model, as well as a VSCode extension that supports interactive usage to foster future research.

<img src='images/workflow.png' alt='Refer to caption' title='' width='419' height='176' />

*Figure 1:  The multi-round auto-editing task. The user inspects the model output in each editing round and can optionally perform manual editing.*

2 Motivating Example
--------------------

<img src='images/motivating_example.png' alt='Refer to caption' title='' width='598' height='387' />

*Figure 2:  An example usage of Coeditor. (a) The user first edits the pack_batch function to read an additional dictionary key, ‘‘cost’’, from each row in the input. (b) The user then removes 3 lines at the top of the group_to_batches function. (c) The user now invokes Coeditor at the bottom half of the same function. Coeditor correctly suggests adding a ‘‘cost’’ key to the dictionary variable row, but it fails to address the now undefined variables underlined in red. (d) However, if the user accepts the suggested change and manually introduces two new variables at line 209, Coeditor can then suggest the correct changes accordingly.*

In this section, we illustrate our technique using the example in [Figure 2](#S2.F2 "Figure 2 ‣ 2 Motivating Example ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), showcasing a two-round interaction between the user and our Coeditor model. Subfigures (a) and (b) display two initial user changes, while subfigures (c) and (d) illustrate two sequential Coeditor invocations with inlined model suggestions. We further analyze this example in detail below.

First, the user modifies the pack_batch function in subfigure (a) to read a new dictionary key, ‘‘cost’’, from each row in the input. The extracted values are used to compute the total cost of the batch and added to the output. Next, the user removes three lines at the top of the group_to_batches function in subfigure (b). By removing these three lines, the user wants to avoid creating these lists beforehand and instead plans to call the process_edit function inside the for loop below.

The user then scrolls down and invokes Coeditor at the bottom half of the same function (subfigure c). Here, the modified pack_batch function is called at lines 225 and 228 in subfigure (c), and its argument current_batch is iteratively constructed from row, which is a dictionary defined at line 215. Hence, the model correctly infers that row should be updated to include a ‘‘cost’’ key. Examining the surrounding context, the model also identifies that the ex_cost variable (defined at line 209) should be used as the inserted dictionary value.111Coeditor would produce the same result even if pack_batch were defined far away or in a different file, as Coeditor tracks all changes $\Delta_{i}$ the user has made since the last commit and incorporates them into the prediction context.

While Coeditor makes some useful editing suggestions so far, it does not address the now-undefined variables underlined in red by the IDE in subfigure (c). In particular, as there are no obvious alternatives nearby to replace these variables, Coeditor is unable to automatically fix these errors.
Such a situation is common when the surrounding changes alone do not provide sufficient information to derive a complete solution. Therefore, the user can accept the partial changes suggested by the model and then manually introduce two new variables at line 209, as shown in subfigure (d). Coeditor can then leverage these new variables to suggest the correct changes needed to fix the errors.

This iterative approach enables Coeditor to adapt and refine its suggestions based on additional user edits, providing a more efficient and flexible code editing experience compared to existing code completion techniques. By incorporating the editing history into the prediction context, Coeditor demonstrates its potential to assist developers in a wide range of code editing tasks, from simple modifications and refactoring to more complex codebase-wide updates.

3 Methods
---------

Recall from the introduction that we wish to model the distribution $P(\Delta u\mid\Delta_{k}\ldots\Delta_{1},U)$. To this end, we first describe how to encode the target change $\Delta u$ and contextual changes $\Delta_{1}\ldots\Delta_{k}$ ([subsection 3.1](#S3.SS1 "3.1 Encoding Code Changes ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")). We then describe how to form the context from the codebase $U$ using function signatures ([subsection 3.2](#S3.SS2 "3.2 Analyzing Relevant Signatures ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")). These choices naturally lead to a model compatible with fine-tuning CodeT5, which was pre-trained on the masked span infilling task ([subsection 3.3](#S3.SS3 "3.3 Adapting CodeT5 ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")). Finally, we describe our new dataset that is used to fine-tune this model ([subsection 3.4](#S3.SS4 "3.4 The PyCommits Dataset ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")).

### 3.1 Encoding Code Changes

A suitable format is required to map code changes into token sequences that can be processed by a seq2seq transformer language model. In our setting, we want to select a format that encodes and decodes code changes in a uniform manner while minimizing the number of tokens the model needs to produce. Hence, we adopt a line-diff-based format, enabling us to convert auto-editing into a masked span infilling problem*(Wang et al., [2021](#bib.bib3 ""))*.222Prior work has proposed various methods to produce code changes. e.g., *Zhang et al. ([2022](#bib.bib17 ""))* learns the distribution $P(u^{\prime}\mid u)$ and *Reid and Neubig ([2022](#bib.bib18 ""))* tags each input token with a label indicating deletion, insertion, or replacement. However, these methods require more copying or tagging, resulting in longer output sequences compared to our approach.

Consider a block of code $u$ to be made up of lines $l_{1},\ldots,l_{m}$ and a user-specified edit region
that spans between line $a$ and $a+n$, where $1\leq a\leq a+n\leq m$. Moreover, each line is associated with a status variable $s_{i}$ indicating what type of change (if any) has already been made; $s_{i}\in{\texttt{(empty)},\texttt{<add>},\texttt{<del>}}$.333We represent edits as line diffs output by Differ.compare using the standard difflib library.  We encode the input code by a function $\mathrm{EncInput}$ that (optionally) prepends status tokens $s_{1}\ldots s_{m}$ and placeholder tokens $\texttt{<1>}\ldots\texttt{<n>}$ at the start of each line:

|  | $\mathrm{EncInput}(u)\=s_{1}l_{1}s_{2}l_{2}\ldots\texttt{<1>}s_{a}l_{a}\texttt{<2>}s_{a+1}l_{a+1}\ldots\texttt{<n>}s_{a+n}l_{a+n}\ldots s_{m}l_{m}\ .$ |  |
| --- | --- | --- |

For contextual changes $\Delta_{1}\ldots\Delta_{k}$, we can encode them using the same format but with an empty edit region.
When the target change $\Delta u$ contains line additions, denoting the $j$th line to be inserted before line $i$ as $l^{\prime}_{ij}$, we can encode $\Delta u$ using the following expression,

|  | $\displaystyle\mathrm{EncOutput}(\Delta u)$ | $\displaystyle\=\texttt{<1>}I_{a}D_{a}\texttt{<2>}I_{a+1}D_{a+1}\ \ldots\ \texttt{<n>}I_{a+n}D_{a+n}\ ,$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle\text{where }I_{i}$ | $\displaystyle\=\texttt{<add> }l^{\prime}_{i1}\texttt{<add> }l^{\prime}_{i2}\ \ldots\ \texttt{<add> }l^{\prime}_{i|I_{i}|}\ ,$ |  |
| --- | --- | --- | --- |
|  | $\displaystyle D_{i}$ | $\displaystyle\=\text{if $l_{i}$ {is to be deleted} then {<del>} else {(empty)}}\ .$ |  |
| --- | --- | --- | --- |

Note that we add a further restriction that forbids $D_{i}$ from being <del> if $s_{i}$ is <add> in order to prevent the model from modifying a line that has just been added; we discuss this in more detail in [section 6](#S6 "6 Conclusion and Limitations ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). [Figure 3](#S3.F3 "Figure 3 ‣ 3.1 Encoding Code Changes ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing") illustrates this line-diff-based encoding scheme using the example from [Figure 2](#S2.F2 "Figure 2 ‣ 2 Motivating Example ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). This format ensures that if we replace the placeholder tokens in the input with the corresponding changes specified in the output sequence, we obtain the total change that combines $u$ and $\Delta u$.

<img src='images/example_change_encoding.png' alt='Refer to caption' title='' width='598' height='304' />

*Figure 3: Coeditor encoding format. (Left) the input sequence adds placeholder tokens to indicate code region to edit. (Top right) the output sequence specifies further changes at each placeholder token. (Bottom right) relevant signatures are retrieved from the codebase and added to the context. (In this example, the Python module is called motivating).*

### 3.2 Analyzing Relevant Signatures

Having described how we encode code changes, we must also establish a method for feeding $U$, the remaining codebase, to the model. Simply inputting the entire codebase as is would result in an excessive number of tokens, overwhelming the context. Instead, inspired by the ideas proposed in previous type inference work*(Pradel et al., [2020](#bib.bib19 ""); Wei et al., [2020](#bib.bib20 ""), [2023](#bib.bib12 ""))*, we employ lightweight static analysis to extract the most relevant information into the context, as outlined below.

For each target code region $u$, we analyze its pre-edit code and generate a list of its usages.444We use the Jedi package for this purpose: <https://github.com/davidhalter/jedi>. In the case of a function usage, we retrieve its function signature; for a variable or class member usage, we retrieve the first statement in which it was assigned. We then concatenate all these usages into a single “document”, as shown at the bottom right of [Figure 3](#S3.F3 "Figure 3 ‣ 3.1 Encoding Code Changes ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), which serves as additional input context. This approach allows the model to access the most pertinent information about the current code region and significantly improves model performance ([Table 5](#S4.T5 "Table 5 ‣ 4.3 Ablation Studies ‣ 4 Evaluation ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")), while generating only a small number of extra tokens in the context ([Table 2](#S3.T2 "Table 2 ‣ 3.4 The PyCommits Dataset ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")).

### 3.3 Adapting CodeT5

<img src='images/coeditor_sparse_attention.png' alt='Refer to caption' title='' width='299' height='130' />

*Figure 4: Coeditor encoder sparse attention pattern. All attention between the reference blocks are skipped to avoid the quadratic cost of dense attention.*

Our model is based on the architecture and pre-trained weights of CodeT5*(Wang et al., [2021](#bib.bib3 ""))*. CodeT5 was pre-trained on a large corpus of code data using the masked span infilling objective, making it a suitable choice for our problem. We employ the CodeT5-base model, containing 220M parameters, and fine-tune it for our code auto-editing setting. Although the original CodeT5 model was pre-trained with a small sequence length of 512, its relative positional encoding scheme allows us to fine-tune it on much longer sequences for our problem.

Considering that a single commit may encompass numerous code changes, concatenating all changes into a single input can lead to long token sequences that are difficult for the CodeT5 model to process with dense attention. To mitigate this issue, we replace the full attention in its encoder with a block-sparse attention pattern, illustrated in [Figure 4](#S3.F4 "Figure 4 ‣ 3.3 Adapting CodeT5 ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). This pattern divides the input sequences into multiple reference blocks and a query block. The query block contains the code to be edited, whereas each reference block encodes a contextual unit change $\Delta u_{j}$ or a chunk of the signature document. We limit the sequence length of each block to 512 tokens for references and 1024 for the query, dividing longer blocks into multiple ones if necessary.
The self-attention within each block is performed as usual, but the attention between different reference blocks is skipped to save computation, similar to other retrieval-augmented models *(Izacard and Grave, [2021](#bib.bib21 ""))*. However, we still allow the query block to attend to and be attended by all reference blocks (a global attention block *(Beltagy et al., [2020](#bib.bib22 ""); Zaheer et al., [2020](#bib.bib23 ""))*). We also set the relative distance between each reference and the query to be infinite when computing the relative positional encoding, making the model is insensitive to the ordering of the references. We are able to use a total of 16.4K reference tokens at test time, which is sufficient to cover 88.8% of problem instances in our test set without truncating the context ([Table 2](#S3.T2 "Table 2 ‣ 3.4 The PyCommits Dataset ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")). See [subsection A.2](#A1.SS2 "A.2 Discussion of Sparse Attention Mechanisms ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing") for more discussion of long-document attention mechanisms.

### 3.4 The PyCommits Dataset

To train our model, we gather real-world code changes from the commit histories of open-source Python projects, a dataset we call PyCommits.
For each commit, we first identify which changes are made to the same code unit (a unit can be either a function, a region of a class, or a region of a module) and subsequently separate the commit into a list of unit additions, unit deletions, or unit modifications. As our work primarily focuses on code editing, only unit modifications are used as training labels, while the other two types of changes remain visible to the model as context.

For each unit modification, we create a training problem instance that instructs the model to predict the code change based on all prior (but not future) changes from the same commit. Git does not record the editing order of changes within the same commit, so we employ a simple heuristic that sorts unit changes according to their source code locations and the import order between modules. Specifically, we assume that units within the same file are modified from top to bottom,
and if a module imports another module, changes in the imported module occur before those in the importing module.555Note that this ordering mainly affects how we generate the training and testing data. At test time, our model can condition on changes both above and below the edit region. To train our model for the proposed multi-round editing setting, we generate synthetic data demonstrating repeated editing to the same code unit as follows: for those code change involving least two changed lines, we randomly sample a subset of the changes as the prediction target and line the remaining changes into the input. For example, the problem instance shown in [Figure 3](#S3.F3 "Figure 3 ‣ 3.1 Encoding Code Changes ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing") can be generated by inlining 2 of the 6 changed lines in the input.

*Table 1: General statistics of the PyCommits dataset.*

|  | train | valid | test |
| --- | --- | --- | --- |
| projects | 1550 | 50 | 50 |
| used commits | 217K | 5006 | 5854 |
| modified files | 501K | 10.1K | 11.1K |
| modified functions | 958K | 20.1K | 22.5K |
| modified lines | 7.10M | 143K | 169K |

*Table 2: Additional statistics specific to our technique, computed over the test set.*

|  | definition | median | mean | max | $\geq$ max |
| --- | --- | --- | --- | --- | --- |
| query tokens | $\mathrm{EncInput}(u)$ | 258 | 361.9 | 1024 | 7.8% |
| output tokens | $\mathrm{EncOutput}(\Delta u)$ | 60 | 89.7 | 512 | 1.3% |
| prev change tokens | $\mathrm{EncInput}(\Delta_{1}\ldots\Delta_{k})$ | 1625 | 4.14K | 16.4K | 11.2% |
| signature tokens | ${\text{signature}(v)}_{v\in\text{usages}(u)}$ | 313 | 515.5 | 15.9K | 0.0% |

We construct a new code editing dataset using the commit history of 1,650 Python projects with permissive licenses (MIT, Apache, and BSD) sourced from GitHub. We use 50 of the projects for testing and 50 for validation and use the remaining 1,550 projects for training. We use at most 1000 commits per project per project to ensure that the model is trained on a diverse set of code changes. We show the general statistics in [Table 1](#S3.T1 "Table 1 ‣ 3.4 The PyCommits Dataset ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing") and the statictics that are specific to our technique in [Table 2](#S3.T2 "Table 2 ‣ 3.4 The PyCommits Dataset ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). Tokenization is performed using the CodeT5 tokenizer.

4 Evaluation
------------

In this section, we first compare Coeditor with prior code completion approaches on a simplified version of the editing task. We then report Coeditor’s performance on the proposed multi-round editing task and conduct ablation studies. Example model outputs are included in the appendix.

#### Training Setup

We initialize Coeditor with the CodeT5-base checkpoint (220M parameters) and train the model on our training set for 1.75 epoch, gradually increasing the model reference context size from 2048 tokens to 4096 tokens (at epoch 1) and then to 8192 tokens (at epoch 1.5). We use Huggingface’s Trainer implementation and the AdamW optimizer, with a linear learning rate schedule with a starting learning rate of 2e-5 and 0.01 weight decay. We train the model with a fixed batch size of 1 and a total of 1.34 million training steps. Training took about 5 days on a single NVIDIA Quadro RTX 8000 GPU with 48 GB memory.

### 4.1 Comparison with Code Completion Approaches

#### Baselines

We compare Coeditor with 3 open-source code generation models: InCoder-1B, InCoder-6B*(Fried et al., [2022](#bib.bib5 ""))*, and SantaCoder*(Allal et al., [2023](#bib.bib6 ""))*. All three code generation models are trained with the Fill-in-the-middle pre-training objective*(Aghajanyan et al., [2022](#bib.bib24 ""))* and use a context size of 2048 tokens.

#### Creating test instances

We generate code completion problem instances from real commits as follows. For each code change in PyCommits, we take the last changed line as the completion target. If the last change is a modification, we delete the modified line and let the model fill in the new version of the line. If the last change is a deletion, we simply discard the change. We then inline all changes before the target into the prediction context. This inlining process is implemented differently for each model: for our Coeditor model, the inlined changes are visible to our model following the encoding scheme described in [subsection 3.1](#S3.SS1 "3.1 Encoding Code Changes ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"); for the code completion models, we simply apply the inlined changes to the original code and use the resulting state as the model input. Also note that while our model constructs its prediction context using relevant changes and static analysis (as described in [subsection 3.2](#S3.SS2 "3.2 Analyzing Relevant Signatures ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")), the code completion models (which are unaware of code changes) only use the code surrounding the completion target as the prediction context. We call this test dataset, derived from our PyCommits test set, PyCommits-OneLine.

*Table 3: Performance on 5000 code completion instances extracted from edits (PyCommits-OneLine). Add EM and Replace EM are the (enhanced) exact-match accuracies on addition and replacement change, respectively.*

| Model | Parameters | Add EM (%) | Replace EM (%) | Overall EM (%) |
| --- | --- | --- | --- | --- |
| InCoder1B | 1.3B | 29.0 | 25.2 | 26.2 |
| InCoder6B | 6.7B | 34.0 | 30.4 | 31.3 |
| SantaCoder | 1.1B | 31.0 | 28.1 | 28.8 |
| Coeditor | 220M | 47.1 | 64.9 | 60.4 |

#### Results

We report the performance (without fine-tuning on this task) of all approaches in [Table 3](#S4.T3 "Table 3 ‣ Creating test instances ‣ 4.1 Comparison with Code Completion Approaches ‣ 4 Evaluation ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). We use an enhanced exact-match (EM) accuracy metric that performs semantic-preserving code normalization before checking for string equivalence.666We normalize Python code by (1) parsing the code into a syntax tree using the ast library, (2) removing any comments and docstrings, (3) sorting all keyword arguments in function calls, and (4) un-parsing the syntax tree. The results are measured on 5000 code completion problems sampled from our test set. We see that Coeditor significantly outperforms the other code generation models for both addition and replacement changes. Coeditor achieves an overall EM of 60.4%, which is almost twice as high as the best performing code completion model (31.3%), despite using a 30 times smaller model, demonstrating the significant benefits of incorporating editing history for code completion. We also include 3 example model outputs on this task in the appendix ([subsection A.3](#A1.SS3 "A.3 Code Completion Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")).

### 4.2 Multi-round Editing

This evaluation focuses on the editing assistant use case where we assume the user has some desired code changes in mind, and we aim to evaluate how much the model can save the user’s effort by automating as much changes as possible, potentially under the guidance of the user. The user can accept partial changes suggested by the model and make additional changes manually if needed.

#### Evaluation workflow

To evaluate the above use case automatically, we use the ground-truth code changes to simulate the user’s actions. In particular, when the model predicts a list of changes, we compare the predicted changes against the ground truth changes line-by-line and accept any line change that exactly matches the ground truth. If none of the suggested changes match the ground truth, we assume the user will manually perform the first remaining change.
In both cases, after the additional changes, we rerun the model to obtain new suggestions and repeat until all desired changes have been performed or the round limit \= 6 has been reached. In the end, we compute the total gain using the difference between the editing cost of the ground truth and the accumulative editing cost of all manually performed edits.

#### Measuring editing cost

There are multiple ways to measure the cost of performing a code change. Since there is no consensus on the best metric, we report 3 metrics in our results. Prior work*(Lavazza et al., [2023](#bib.bib25 ""))* suggests that for code understanding tasks, simple line counts-based metrics are almost as good as more complex metrics, hence our first metric, Lines, simply measures the number of changed lines before and after the edit. We also report Levenshtein, the classic Levenshtein editing distance metric that measures the minimal number of character addition, deletion, and substitution needed to transform one string into another. Although simple, the Levenshtein distance doesn’t model many important aspects of code editing, such as the cost associated with cursor movement, and it also under-count the cost of substitution and over-count the cost of large deletions. Hence, we propose an additional metric, Keystrokes, that aims to better approximate the number of needed user keystrokes than Levenshtein by allowing for batch deletion and accounting for the cost of cursor movements. We describe this metric in detail in the [subsection A.1](#A1.SS1 "A.1 Keystroke Distance ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). Note that the total gain can be negative when measured with Levenshtein and Keystrokes.777For example, the Levenshtein distance of modifying a sentence is lower than the total of first deleting the sentence and then adding a new one.

#### Results

We report the evaluation results on 5000 problems sampled from our test set in [Table 4](#S4.T4 "Table 4 ‣ Results ‣ 4.2 Multi-round Editing ‣ 4 Evaluation ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), and we also report the single-round performance for reference. We see that Coeditor achieves much larger total gains under the multi-round setting, especially when measured with the Lines and Keystrokes metric (which we believe more accurately captures the user editing effort than Levenshtein). We also show 3 examples of the model’s suggestions in the appendix ([subsection A.4](#A1.SS4 "A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")).

*Table 4: Multi-round evaluation results measured on 5000 problems from the PyCommits test set. Lines, Levenshtein, and Keystrokes are the average total gains in the corresponding metrics. Rounds is the average number of rounds needed to complete all desired changes.*

| Setting | Lines (%) | Levenshtein (%) | Keystrokes (%) | $6$plus1.5minus4plus23plus2minus2Rounds |
| --- | --- | --- | --- | --- |
| SingleRound | 28.5 | 23.1 | 19.2 | $1$ |
| MultiRound | 46.7 | 25.9 | 28.6 | $2.43$ |

### 4.3 Ablation Studies

We retrain the model with various components disabled to study their impact on the overall model performance. We report the (single-round) exact match performance of each variation on the entire PyCommits validation set in [Table 5](#S4.T5 "Table 5 ‣ 4.3 Ablation Studies ‣ 4 Evaluation ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). The results show that removing any of the components leads to a decrease in performance, highlighting the importance of each component in the overall model. Specifically, when we remove the explicit feeding of code changes (No Diffs), the EM drops the most, from 42.1% to 26.1%. When we disable the static analysis component (No Signatures), the EM decreases to 33.3%. Using a smaller limit of reference tokens impacts the model performance the least, reducing EM to 39.8%. All results reported in [Table 5](#S4.T5 "Table 5 ‣ 4.3 Ablation Studies ‣ 4 Evaluation ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing") were obtained by training the model for half amount of training steps to save compute.

*Table 5: Ablation results on the entire validation set (PyCommits). All pairwise differences are statistically significant with $p<0.05$ using a paired bootstrap test.*

| Ablation | Description | EM (%) |
| --- | --- | --- |
| No Diffs | Feeding the same input to the model except that all changes are replaced with their post-edit results alone. | 26.1 |
| No Signatures | Disabling the static analysis component and removing function and class signatures from the prediction context. | 33.3 |
| Small Context | Reducing the max number of reference tokens from 16K to 2048. | 39.8 |
| No Ablation | Model trained with our default settings. | 42.1 |

5 Related Work
--------------

The past work most similar to our setting is that of *Brody et al. ([2020](#bib.bib26 ""))*, which also targets a contextual code editing setting. However, it can only predict a restrictive set of code changes expressable as moving, deleting, or copying existing AST nodes and cannot generate novel expressions that are not present in the input. It also doesn’t make use of modern transformer architecture or pre-training techniques. In contrast, *Ni et al. ([2021](#bib.bib27 ""))* takes a rule-based approach, using program synthesis methods to distill similar change patterns in the context and make editing suggestions accordingly.

There is also prior work on non-contextual code change prediction settings. In *Chakraborty et al. ([2020](#bib.bib28 ""))*, the authors use past code patches to train the model to perform similar edits and evaluate it on future edits in the same codebase. However, since the model does not condition on relevant changes, their technique requires retraining the model for new types of edit patterns. *Panthaplackel et al. ([2020a](#bib.bib29 ""))* proposes augmenting the decoder with a direct copying mechanism to help a encoder-decoder model perform editing tasks. *Zhang et al. ([2022](#bib.bib17 ""))* proposes a de-noising pre-training scheme, in which they randomly corrupt actual code snippets and train the model to predict the uncorrupted version from the corrupted version. *Tufano et al. ([2021](#bib.bib30 ""))* focuses on predicting code review changes using developer discussions. *Reid and Neubig ([2022](#bib.bib18 ""))* focuses on modeling the iterative editing process of texts and code and proposes a different change encoding scheme that represents edits at the word-level based on the Levenshtein algorithm.

Lastly, there is prior work that focuses on on learning to update code comments*(Panthaplackel et al., [2020b](#bib.bib31 ""))* or generating natural language descriptions*(Panthaplackel et al., [2022](#bib.bib32 ""))* from code changes. This work, we focus on measuring our model’s ability to make correct code changes and remove comments and doc-strings before measuring the exact accuracy.

6 Conclusion and Limitations
----------------------------

In this paper, we presented Coeditor, a novel approach for multi-round code auto-editing. Building on the CodeT5 architecture, our model incorporates line diff format and static analysis to create large customized model contexts. We demonstrated that Coeditor significantly outperforms existing code completion methods, nearly doubling their exact-match accuracy in a simplified single-round, single-edit task. Furthermore, our model shows substantial performance improvements in a more complex multi-round, multi-edit setting. To encourage future research, we open-source our codebase, dataset, and model weights, along with a VSCode extension for practical, interactive model usage.

One limitation of our current work is assuming that the user would manually identify the region of code requiring changes and then invoke the model to predict where and how to make these changes within that region. A promising direction for future work would be to extend the model to help users identify the regions of code that need to be changed within the entire codebase, enabling new types of usage such as auto-refactoring.

Furthermore, as mentioned in [subsection 3.1](#S3.SS1 "3.1 Encoding Code Changes ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), our model is designed not to modify any lines in the input that have already been modified. While this simplifies our setting and prevents undesirable behaviors such as the model changing lines that the user has already manually modified or getting stuck in an infinite editing loop, it can also limit the practical usage of our tool. In its current state, the user cannot partially edit a line and let the model complete the rest of that line. Therefore, it would be valuable to explore methods for allowing the model to interactively collaborate with users in editing partially modified lines in future research.

References
----------

* Feng et al. [2020]Zhangyin Feng, Daya Guo, Duyu Tang, Nan Duan, Xiaocheng Feng, Ming Gong, Linjun
Shou, Bing Qin, Ting Liu, Daxin Jiang, et al.CodeBERT: A Pre-Trained Model for Programming and Natural
Languages.In *Findings of the Association for Computational Linguistics:
EMNLP 2020*, pages 1536–1547, 2020.
* Ahmad et al. [2021]Wasi Ahmad, Saikat Chakraborty, Baishakhi Ray, and Kai-Wei Chang.Unified pre-training for program understanding and generation.In *Proceedings of the 2021 Conference of the North American
Chapter of the Association for Computational Linguistics: Human Language
Technologies*, pages 2655–2668, 2021.
* Wang et al. [2021]Yue Wang, Weishi Wang, Shafiq Joty, and Steven CH Hoi.CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models
for Code Understanding and Generation.In *Proceedings of the 2021 Conference on Empirical Methods in
Natural Language Processing*, pages 8696–8708, 2021.
* Chen et al. [2021]Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira
Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg
Brockman, et al.Evaluating large language models trained on code.*arXiv preprint arXiv:2107.03374*, 2021.
* Fried et al. [2022]Daniel Fried, Armen Aghajanyan, Jessy Lin, Sida Wang, Eric Wallace, Freda Shi,
Ruiqi Zhong, Wen-tau Yih, Luke Zettlemoyer, and Mike Lewis.Incoder: A generative model for code infilling and synthesis.*arXiv preprint arXiv:2204.05999*, 2022.
* Allal et al. [2023]Loubna Ben Allal, Raymond Li, Denis Kocetkov, Chenghao Mou, Christopher Akiki,
Carlos Munoz Ferrandis, Niklas Muennighoff, Mayank Mishra, Alex Gu, Manan
Dey, et al.Santacoder: don’t reach for the stars!*arXiv preprint arXiv:2301.03988*, 2023.
* Li et al. [2022]Yujia Li, David Choi, Junyoung Chung, Nate Kushman, Julian Schrittwieser,
Rémi Leblond, Tom Eccles, James Keeling, Felix Gimeno, Agustin Dal Lago,
et al.Competition-level code generation with alphacode.*arXiv preprint arXiv:2203.07814*, 2022.
* Nijkamp et al. [2022]Erik Nijkamp, Bo Pang, Hiroaki Hayashi, Lifu Tu, Huan Wang, Yingbo Zhou, Silvio
Savarese, and Caiming Xiong.A conversational paradigm for program synthesis.*arXiv e-prints*, pages arXiv–2203, 2022.
* Lachaux et al. [2020]Marie-Anne Lachaux, Baptiste Rozière, Lowik Chanussot, and Guillaume
Lample.Unsupervised translation of programming languages.*ArXiv*, abs/2006.03511, 2020.
* Szafraniec et al. [2022]Marc Szafraniec, Baptiste Rozière, Hugh Leather Francois Charton, Patrick
Labatut, and Gabriel Synnaeve.Code translation with compiler representations.*ArXiv*, abs/2207.03578, 2022.
* Jesse et al. [2022]Kevin Jesse, Premkumar Devanbu, and Anand Ashok Sawant.Learning to predict user-defined types.*IEEE Transactions on Software Engineering*, 2022.
* Wei et al. [2023]Jiayi Wei, Greg Durrett, and Isil Dillig.TypeT5: Seq2seq Type Inference using Static Analysis.In *International Conference on Learning Representations*, 2023.URL [https://openreview.net/forum?id\=4TyNEhI2GdN](https://openreview.net/forum?id=4TyNEhI2GdN "").
* Guo et al. [2021]Daya Guo, Alexey Svyatkovskiy, Jian Yin, Nan Duan, Marc Brockschmidt, and
Miltiadis Allamanis.Learning to complete code with sketches.In *International Conference on Learning Representations*, 2021.
* Svyatkovskiy et al. [2021]Alexey Svyatkovskiy, Sebastian Lee, Anna Hadjitofi, Maik Riechert,
Juliana Vicente Franco, and Miltiadis Allamanis.Fast and memory-efficient neural code completion.In *2021 IEEE/ACM 18th International Conference on Mining
Software Repositories (MSR)*, pages 329–340. IEEE, 2021.
* Nguyen and Nadi [2022]Nhan Nguyen and Sarah Nadi.An empirical evaluation of GitHub copilot’s code suggestions.In *Proceedings of the 19th International Conference on Mining
Software Repositories*, pages 1–5, 2022.
* Zhang et al. [2023]Fengji Zhang, Bei Chen, Yue Zhang, Jin Liu, Daoguang Zan, Yi Mao, Jian-Guang
Lou, and Weizhu Chen.Repocoder: Repository-level code completion through iterative
retrieval and generation.*arXiv preprint arXiv:2303.12570*, 2023.
* Zhang et al. [2022]Jiyang Zhang, Sheena Panthaplackel, Pengyu Nie, Junyi Jessy Li, and Milos
Gligoric.CoditT5: Pretraining for Source Code and Natural Language
Editing, September 2022.URL [http://arxiv.org/abs/2208.05446](http://arxiv.org/abs/2208.05446 "").arXiv:2208.05446 [cs].
* Reid and Neubig [2022]Machel Reid and Graham Neubig.Learning to Model Editing Processes, May 2022.URL [http://arxiv.org/abs/2205.12374](http://arxiv.org/abs/2205.12374 "").arXiv:2205.12374 [cs].
* Pradel et al. [2020]Michael Pradel, Georgios Gousios, Jason Liu, and Satish Chandra.Typewriter: Neural type prediction with search-based validation.In *Proceedings of the 28th ACM Joint Meeting on European
Software Engineering Conference and Symposium on the Foundations of Software
Engineering*, pages 209–220, 2020.
* Wei et al. [2020]Jiayi Wei, Maruth Goyal, Greg Durrett, and Isil Dillig.LambdaNet: Probabilistic Type Inference using Graph Neural
Networks.In *International Conference on Learning Representations*, 2020.URL [https://openreview.net/forum?id\=Hkx6hANtwH](https://openreview.net/forum?id=Hkx6hANtwH "").
* Izacard and Grave [2021]Gautier Izacard and Edouard Grave.Leveraging passage retrieval with generative models for open domain
question answering.In *Proceedings of the 16th Conference of the European Chapter
of the Association for Computational Linguistics: Main Volume*, pages
874–880, Online, April 2021. Association for Computational Linguistics.doi: 10.18653/v1/2021.eacl-main.74.URL [https://aclanthology.org/2021.eacl-main.74](https://aclanthology.org/2021.eacl-main.74 "").
* Beltagy et al. [2020]Iz Beltagy, Matthew E. Peters, and Arman Cohan.Longformer: The Long-Document Transformer.*arXiv preprint arXiv:2004.05150*, 2020.
* Zaheer et al. [2020]Manzil Zaheer, Guru Guruganesh, Avinava Dubey, Joshua Ainslie, Chris Alberti,
Santiago Ontanon, Philip Pham, Anirudh Ravula, Qifan Wang, Li Yang, and Amr
Ahmed.Big Bird: Transformers for Longer Sequences.In *Advances in Neural Information Processing Systems*, 2020.
* Aghajanyan et al. [2022]Armen Aghajanyan, Bernie Huang, Candace Ross, Vladimir Karpukhin, Hu Xu, Naman
Goyal, Dmytro Okhonko, Mandar Joshi, Gargi Ghosh, Mike Lewis, et al.CM3: A Causal Masked Multimodal Model of the Internet.*arXiv eprint arxiv:2201.07520*, 2022.
* Lavazza et al. [2023]Luigi Lavazza, Abedallah Zaid Abualkishik, Geng Liu, and Sandro Morasca.An empirical evaluation of the “cognitive complexity” measure as
a predictor of code understandability.*Journal of Systems and Software*, 197:111561, 2023.
* Brody et al. [2020]Shaked Brody, Uri Alon, and Eran Yahav.A structural model for contextual code changes.*Proceedings of the ACM on Programming Languages*, 4(OOPSLA):1–28, November 2020.ISSN 2475-1421.doi: 10.1145/3428283.URL [https://dl.acm.org/doi/10.1145/3428283](https://dl.acm.org/doi/10.1145/3428283 "").
* Ni et al. [2021]Ansong Ni, Daniel Ramos, Aidan Z. H. Yang, Ines Lynce, Vasco Manquinho, Ruben
Martins, and Claire Le Goues.SOAR: A Synthesis Approach for Data Science API
Refactoring.In *2021 IEEE/ACM 43rd International Conference on
Software Engineering (ICSE)*, pages 112–124, Madrid, ES, May 2021.
IEEE.ISBN 978-1-66540-296-5.doi: 10.1109/ICSE43902.2021.00023.URL <https://ieeexplore.ieee.org/document/9402016/>.
* Chakraborty et al. [2020]Saikat Chakraborty, Yangruibo Ding, Miltiadis Allamanis, and Baishakhi Ray.CODIT: Code Editing with Tree-Based Neural Models.*IEEE Transactions on Software Engineering*, pages 1–1, 2020.ISSN 0098-5589, 1939-3520, 2326-3881.doi: 10.1109/TSE.2020.3020502.URL [http://arxiv.org/abs/1810.00314](http://arxiv.org/abs/1810.00314 "").arXiv: 1810.00314.
* Panthaplackel et al. [2020a]Sheena Panthaplackel, Miltiadis Allamanis, and Marc Brockschmidt.Copy that! Editing Sequences by Copying Spans, December
2020a.URL [http://arxiv.org/abs/2006.04771](http://arxiv.org/abs/2006.04771 "").arXiv:2006.04771 [cs, stat].
* Tufano et al. [2021]Rosalia Tufano, Luca Pascarella, Michele Tufano, Denys Poshyvanyk, and Gabriele
Bavota.Towards automating code review activities.In *2021 IEEE/ACM 43rd International Conference on Software
Engineering (ICSE)*, pages 163–174. IEEE, 2021.
* Panthaplackel et al. [2020b]Sheena Panthaplackel, Pengyu Nie, Milos Gligoric, Junyi Jessy Li, and Raymond
Mooney.Learning to update natural language comments based on code changes.In *Proceedings of the 58th Annual Meeting of the Association
for Computational Linguistics*, pages 1853–1868, 2020b.
* Panthaplackel et al. [2022]Sheena Panthaplackel, Junyi Jessy Li, Milos Gligoric, and Ray Mooney.Learning to describe solutions for bug reports based on developer
discussions.*Findings of the Association for Computational Linguistics: ACL
2022*, 2022.

Appendix A Appendix
-------------------

### A.1 Keystroke Distance

We developed a string distance metric incorporating the cost of cursor movement, approximating the number of keystrokes needed to transform an input string into an output string.

Given the initial state with i \= len(input), j \= len(output), cursor_dis \= init_cursor_dis, and deleting \= False, the cost is calculated using dynamic programming with the optimal combination of the following operations:

* •

    M: Match character (cost\=0), requires input[-i] \=\= output[-j] and not deleting, results in i -\= 1, j -\= 1, and cursor_dis +\= 1.

* •

    D: Delete input character (cost\=1), requires cursor_dis \=\= 0 and not deleting, results in i -\= 1.

* •

    A: Add output character (cost\=1), requires cursor_dis \=\= 0 and not deleting, results in j -\= 1.

* •

    C: Move cursor to current position (cost\=min(cursor_dis, cursor_jump_cost)), requires no conditions, results in cursor_dis \= 0.

* •

    S: Begin deletion (cost\=1), requires cursor_dis \=\= 0 and not deleting, results in deleting \= True.

* •

    K: Continue deletion (cost\=0), requires deleting, results in i -\= 1.

* •

    E: End deletion (cost\=1), requires cursor_dis \=\= 0 and deleting, results in deleting \= False.

Where cursor_jump_cost is a constant that we set to 4 when reporting our results. The worst-case complexity of this algorithm is $O(\texttt{len(input)}\times\texttt{len(output)}\times\texttt{cursor\_jump\_cost})$. Note that this model does not consider copying and pasting operations.

### A.2 Discussion of Sparse Attention Mechanisms

The block-sparse attention pattern we described in [subsection 3.3](#S3.SS3 "3.3 Adapting CodeT5 ‣ 3 Methods ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing") follows past work on retrieve-and-read models for natural language question answering. Specifically, it resembles Fusion-in-Decoder *[Izacard and Grave, [2021](#bib.bib21 "")]* with three changes. First, we have no notion of a question that is jointly encoded with each retrieved snippet. Second, our target code block $u$ is given special status in the encoder and can globally attend to each retrieved snippet. Third, we modify the relative positional encoding to make each query “infinitely” far from the reference.

Our approach also resembles Longformer *[Beltagy et al., [2020](#bib.bib22 "")]* or BigBird *[Zaheer et al., [2020](#bib.bib23 "")]*, most notably in how our query block’s cross-attention with the references can be viewed as an instance of global attention. However, our segments do not come from a coherent context, so our local attention component is a block-diagonal sparse matrix rather than a sliding window as in those methods. Our modification to relative position encoding also
makes our model invariant to the ordering of references, helping it generalize to different editing orders at inference time.

### A.3 Code Completion Examples

To help the reader see why including contextual changes can be beneficial for (editing-related) code completion problems, we compare Coeditor and InCoder6B’s outputs on 3 example problems from our test set in the next few pages ([Figure 5](#A1.F5 "Figure 5 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")–[Figure 10](#A1.F10 "Figure 10 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")). These examples are sampled from a subset that are small enough to be presentable within one or two pages and in which Coeditor outperforms InCoder6B.

### A.4 Multi-round Editing Examples

We show 3 multi-round editing examples from our test set in [Figure 11](#A1.F11 "Figure 11 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")–[Figure 15](#A1.F15 "Figure 15 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"). These examples are sampled from a subset that are small enough to be presentable within two pages and in which Coeditor achieved 50–100 total keystrokes edit gain.

<img src='images/generation-ex1.png' alt='Refer to caption' title='' width='598' height='245' />

*Figure 5: Code completion example 1. Coeditor sees from the relevant contextual changes (shown in [Figure 6](#A1.F6 "Figure 6 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing")) that some get_asynclib() calls should be replaced with get_async_backend(), so it correctly suggested the change based on the deletion before the infilling point. InCoder was not able to see the deletion and infilled the original code given only the surrounding code.*

<img src='images/generation-ex1-cont.png' alt='Refer to caption' title='' width='598' height='320' />

*Figure 6: Code completion example 1: relevant contexts. The changes highlighted in orange tell Coeditor that some get_asynclib() calls should be replaced with get_async_backend().*

<img src='images/generation-ex2.png' alt='Refer to caption' title='' width='598' height='441' />

*Figure 7: Code completion example 2. Coeditor was able to suggest the correct code based on a similar change from another file ([Figure 8](#A1.F8 "Figure 8 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), highlighted in orange), whereas InCoder was not able to see the change and suggested a wrong statement.*

<img src='images/generation-ex2-cont.png' alt='Refer to caption' title='' width='598' height='381' />

*Figure 8: Code completion example 2: relevant contexts.*

<img src='images/generation-ex3.png' alt='Refer to caption' title='' width='598' height='648' />

*Figure 9: Code completion example 3. Coeditor was able to suggest adding the correct attribute initialization based on the new usage highlighted in [Figure 10](#A1.F10 "Figure 10 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), whereas InCoder was not able to see the new usages and hallucinated a new attribute.*

<img src='images/generation-ex3-cont.png' alt='Refer to caption' title='' width='598' height='613' />

*Figure 10: Code completion example 3: relevant contexts.*

<img src='images/multi-ex1.png' alt='Refer to caption' title='' width='598' height='647' />

*Figure 11: Multi-round editing example 1. Coeditor correctly suggested a subset of the ground-truth changes. Contextual changes omitted for this example.*

<img src='images/multi-ex2.png' alt='Refer to caption' title='' width='598' height='647' />

*Figure 12: Multi-round editing example 2 (round 3). Coeditor misunderstood the user’s intention and suggested adding two more arguments to the EncodedVideo.from_path function call. Under our multi-round evaluation strategy, we assume the user would then manually add the next line from the ground truth changes (see the next figure).*

<img src='images/multi-ex2-cont.png' alt='Refer to caption' title='' width='598' height='605' />

*Figure 13: Multi-round editing example 2 (round 4). With the next line change from the ground truth added, Coeditor understood that the user intended to only change the calling style and was thus able to suggest the correct change.*

<img src='images/multi-ex3.png' alt='Refer to caption' title='' width='598' height='569' />

*Figure 14: Multi-round editing example 3. Coeditor was able to predict the correct change in the first editing round by identifying a similar change inside a different function (see [Figure 15](#A1.F15 "Figure 15 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing"), highlighted in orange).*

<img src='images/multi-ex3-cont.png' alt='Refer to caption' title='' width='598' height='616' />

*Figure 15: Multi-round editing example 3 (reference blocks). The bottom changes highlighted in orange are similar to the changes needed in [Figure 14](#A1.F14 "Figure 14 ‣ A.4 Multi-round Editing Examples ‣ Appendix A Appendix ‣ Coeditor: Leveraging Contextual Changes for Multi-round Code Auto-editing").*
