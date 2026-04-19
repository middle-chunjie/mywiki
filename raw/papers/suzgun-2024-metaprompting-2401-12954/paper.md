
HTML conversions [sometimes display errors](https://info.dev.arxiv.org/about/accessibility_html_error_messages.html) due to content that did not convert correctly from the source. This paper uses the following packages that are not yet supported by the HTML conversion tool. Feedback on these issues are not necessary; they are known and are being worked on.

* failed: arydshln
* failed: scrextend
* failed: tgpagella

Authors: achieve the best HTML results from your LaTeX submissions by following these [best practices](https://info.arxiv.org/help/submit_latex_best_practices.html).

License: CC BY 4.0

arXiv:2401.12954v1 [cs.CL] 23 Jan 2024

Meta-Prompting:  Enhancing Language Models with Task-Agnostic Scaffolding
===========================================================================

Mirac Suzgun  
Stanford University  
<msuzgun@stanford.edu>Work done while at Microsoft Research New England.Adam Tauman Kalai  
OpenAI${}^{*}$  
<adam@kal.ai>

###### Abstract

We introduce meta-prompting, an effective scaffolding technique designed to enhance the functionality of language models (LMs). This approach transforms a single LM into a multi-faceted conductor, adept at managing and integrating multiple independent LM queries. By employing high-level instructions, meta-prompting guides the LM to break down complex tasks into smaller, more manageable subtasks. These subtasks are then handled by distinct “expert” instances of the same LM, each operating under specific, tailored instructions. Central to this process is the LM itself, in its role as the conductor, which ensures seamless communication and effective integration of the outputs from these expert models. It additionally employs its inherent critical thinking and robust verification processes to refine and authenticate the end result. This collaborative prompting approach empowers a single LM to simultaneously act as a comprehensive orchestrator and a panel of diverse experts, significantly enhancing its performance across a wide array of tasks. The zero-shot, task-agnostic nature of meta-prompting greatly simplifies user interaction by obviating the need for detailed, task-specific instructions. Furthermore, our research demonstrates the seamless integration of external tools, such as a Python interpreter, into the meta-prompting framework, thereby broadening its applicability and utility. Through rigorous experimentation with GPT-4, we establish the superiority of meta-prompting over conventional scaffolding methods: When averaged across all tasks, including the Game of 24, Checkmate-in-One, and Python Programming Puzzles, meta-prompting—augmented with a Python interpreter functionality—surpasses standard prompting by 17.1%, expert (dynamic) prompting by 17.3%, and multipersona prompting by 15.2%.111The data, prompts, and the model outputs are all available at [https://github.com/suzgunmirac/meta-prompting](https://github.com/suzgunmirac/meta-prompting "").


*Figure 1: Enhancing GPT-4 with meta-prompting. In this study, we introduce and examine the effectiveness of meta-prompting, contrasting it with a range of zero-shot prompting techniques, including
standard zero-shot (Std),
zero-shot chain-of-thought (0-CoT;*Kojima et al. ([2022])*),
generic and dynamic expert (Ex-St and Ex-Dy;*Xu et al. ([2023])*),
and multipersona (MP;*Wang et al. ([2023])*).
Our research demonstrates that meta-prompting, particularly when combined with a Python interpreter, significantly improves overall accuracy and robustness in GPT-4 across a variety of tasks.*

Introduction
------------

The latest generation of language models (LMs)—notably, GPT-4*(OpenAI, [2023])*, PaLM*(Anil et al., [2023])*, and LLaMa*(Touvron et al., [2023])*—have expanded the boundaries of natural-language processing and generation. These large-scale models can tackle a wide spectrum of tasks, ranging from writing Shakespearean sonnets about hedgehogs to summarizing intricate medical reports and solving competition-level programming puzzles. Despite their versatility, these models are not infallible; they sometimes generate responses that are inaccurate, misleading, or conflicting. As the operational costs of these models become more affordable, it becomes natural to ask whether one might use scaffolding systems and leverage multiple LM queries to not only refine but also to enhance the accuracy and robustness of these model outputs.

In this work, we introduce a new technique for enhancing the functionality and performance of LMs, called meta-prompting. It involves constructing a high-level “meta” prompt that instructs an LM to: (i) break down complex tasks or problems into smaller, manageable pieces; (ii) assign these pieces to specialized “expert” models with proper and detailed natural-language instructions; (iii) oversee the communication between these expert models; and (iv) apply its own critical thinking, reasoning, and verification skills throughout the process. When presented with a query, the LM, effectively prompted under meta-prompting, serves as a conductor. It produces a message history—a narrative, if you will—comprising the responses from various expert models. The LM is originally responsible for generating the conductor’s portion of this history, which includes the selection of experts and the formulation of specific instructions for them. However, the same LM doubles itself as these independent experts as well, generating outputs based on the expertise and information chosen by the conductor for each particular query.

This approach allows for a single, uniform LM to maintain a coherent line of reasoning while also tapping into a variety of expert roles. The use of dynamically selected contexts for prompting these experts introduces fresh perspectives into the process, while the conductor model retains a bird’s-eye view of the entire history and coordination. This method, therefore, enables a single black-box LM to function effectively as both a central conductor and a diverse panel of experts to produce more accurate, reliable, and coherent responses.

Our proposed meta-prompting technique combines and expands upon various prompting ideas introduced by recent studies—including, *high-level planning and decision-making* *(Yao et al., [2023b]; Sun et al., [2023]; Hao et al., [2023a])*, *dynamic persona assignment* *(Xu et al., [2023]; Wang et al., [2023])*, *multi-agent debating* *(Du et al., [2023]; Zhuge et al., [2023])*, *self-debugging and self-reflection* *(Schick et al., [2023b]; Liu et al., [2023a]; Gou et al., [2023]; Madaan et al., [2023]; Shinn et al., [2023])*.
A key aspect of meta-prompting is its *task-agnostic* nature. Unlike traditional scaffolding methods that require specific instructions or examples tailored to each task, meta-prompting employs the same set of high-level instructions across various tasks and inputs. This universality is particularly beneficial for users who might find it cumbersome to provide detailed examples or specific guidance for every distinct task. For instance, in responding to a one-off request like “Write a Shakespearean sonnet about selfies,” the user would not need to supply examples of high-quality neoclassical poems. The meta-prompting approach elevates the utility of language models by offering a broad, flexible framework without compromising on specificity or relevance. Additionally, to demonstrate the versatility and integration capabilities of meta-prompting, we have enhanced our system with the functionality to invoke a Python interpreter. This allows for an even more dynamic and comprehensive application of the technique, further extending its potential to address a wide array of tasks and queries effectively.

We provide an illustrative visualization of a meta-prompting session in Figure[2]. It depicts how the Meta Model—our technical term for the central controlling LM (a.k.a. the conductor)—intersperses its own output with inputs and outputs from various specialized expert models or code executions. Such a configuration makes meta-prompting a nearly universal tool. It allows for the consolidation of various LM interactions and computations into a single, coherent narrative. What sets meta-prompting apart is that it leaves the decision of which prompts to use and which code snippets to execute to the discretion of the LM itself.

In our comprehensive experiments, which primarily utilize GPT-4 as the foundational LM, we compare the efficacy of meta-prompting against other task-agnostic scaffolding methods. Our findings reveal that meta-prompting not only enhances overall performance but often leads to state-of-the-art results across a diverse range of tasks. Its flexibility is noteworthy: The conductor model has the capability to call upon expert models (basically itself, albeit with fresh instructions) for performing a variety of functions. These functions might include critiquing earlier outputs, selecting specific personas for certain tasks, refining generated content, and ensuring that the final outputs meet the desired criteria in both substance and form. This approach shows a marked improvement over several existing methods, as demonstrated in Figure[1].

The core contribution of this work is the introduction of a task-agnostic scaffolding system that leverages a single LM. This LM not only carries forward the thread of the task but also dynamically selects and instructs expert models appropriate for each specific task. The effectiveness of this system is showcased across various benchmarks, including the Game of 24*(Yao et al., [2023a])*, Checkmate-in-One from the BIG-Bench suite*(BIG-Bench authors, [2023])*, and our novel task of “Shakespearean Sonnet Writing.” Overall, our empirical results underscore the versatility and robustness of meta-prompting in enhancing LM performance.

<img src='x1.png' alt='Refer to caption' title='' width='822' height='520' />

*Figure 2: An example meta-prompting history, where the prompts have been shortened for illustrative purposes. The history is initialized by a question provided by a user. Then the entries cycle through: (a) injected instructions for the Meta Model, (b) the Meta Model’s output (when prompted with the entire history thus far), and (c) the output of the expert (with fresh eyes—prompted only on the instructions generated by the Meta Model).*

Meta Prompting
--------------

Intuition and Abstract Overview. The modus operandi of meta-prompting is to use a model222Our use of the term model refers to the application of an LM with certain prompt templates to play a specified “role.” We typically only use a single LM (e.g., GPT-4) to implement all the models in an execution. to coordinate and execute multiple independent inquiries and subsequently synthesize their responses to render a final response. This mechanism, in principle, endorses an ensemble approach, drawing from the strength and diversity of independent specialized models to collaboratively address and tackle multifaceted tasks or problems. We posit that while a single, general-purpose model might deliver valuable and useful insights into generic queries, combining the perspectives and conclusions of multiple domain-specific models (which we also refer to as experts) has the potential to yield more comprehensive, robust, and accurate solutions.

Central to our meta-prompting strategy is its shallow hierarchical configuration, where a single model—called the “Meta Model”—emerges as the principal entity of authority. This prompting structure is reminiscent of an orchestra, wherein the conductor’s role is mirrored by the Meta Model and each musician corresponds to a distinct domain-specific model. Just as a conductor harmonizes multiple musical elements to craft a beautiful melody, the Meta Model combines solutions and insights from a range of models to provide an accurate and comprehensive answer to an intricate problem or task.

Conceptually, a domain-specific expert within our framework can take diverse forms, such as a finetuned LM tailored to perform a particular task, a specialized API equipped to handle specific domain-related inquiries, or even computational tools like calculators or a Python interpreter that can perform arithmetic calculations or write and execute code. These experts, despite their varying functionalities, are directed and unified under the supervision of the Meta Model.

Under our setup, experts can be called only by the Meta Model. They cannot directly interact or communicate with each other, though the Meta Model can choose to share some text from or combine the insights of various experts when interacting with a new expert. This restriction is made to simplify the communication between the experts and to put the Meta Model at the center of the operation.

Notation and Terminology. Before we delve into the specific steps involved in meta-prompting, we establish some notation and terminology. We let $\mathbb{S}$ denote the set of finite strings, with $\emptyset$ representing the empty string. We use $x\in\mathbb{S}$ to refer to a test-time query, which can be a task or a problem described in natural language. A crucial element of meta-prompting is the fixed language model, denoted as $\mathtt{LM}$, which operates from $\mathbb{S}$ to $\mathbb{S}$. This model, like GPT-4, takes an input text (a prompt history that may include a list of previous messages, symbolized by $\mathcal{H}$) and produces a corresponding output (i.e., response).
We also introduce specific template functions: $t_{\text{init}},t_{\text{mid}},$ and $t_{\text{exp}}$, each mapping from $\mathbb{S}$ to $\mathbb{S}$; each takes a string input and formats it according to a predefined template. Specifically, $t_{\text{init}}$ and $t_{\text{mid}}$ are used to format text for the history given to the Meta Model, while $t_{\text{exp}}$ wraps the output of the Meta Model in a prompt suitable for an expert model.
Furthermore, we have two string extractors, $e_{\text{exp}}$ and $e_{\text{ret}}$, each mapping from $\mathbb{S}$ to $\mathbb{S}$. These extractors are designed to retrieve a substring that is enclosed within specific delimiters, returning the first matching segment in cases where multiple segments are present.
The symbol $\oplus$ is used to represent string concatenation. Lastly, we introduce a specific string referred to as $\text{error}\in\mathbb{S}$, which is designed to denote an error message in the process.

Algorithmic Procedure. Algorithm[1] provides pseudocode of our proposed meta-prompting approach. We further provide a conceptual overview of the procedure below:

*Algorithm 1  Meta Prompting*

Input: $\texttt{LM}:\mathbb{S}\rightarrow\mathbb{S};x,\text{error}\in\mathbb{S};T\in%
\mathbb{N};t_{\text{init}},t_{\text{mid}},t_{\text{exp}},e_{\text{exp}},e_{%
\text{ret}}:\mathbb{S}\rightarrow\mathbb{S}$

1:$\mathcal{H}_{1}\leftarrow t_{\text{init}}(x)$

2:for$t\in[1,\ldots,T]$do

3:$y_{t}\leftarrow\mathtt{LM}~{}(\mathcal{H}_{t})$

4: if$e_{\text{exp}}(y_{t})\neq\emptyset$then $\triangleright$ Meta Model provided expert instructions

5:$\text{prompt}\leftarrow t_{\text{exp}}(e_{\text{exp}}(y_{t}))$

6:$z_{t}\leftarrow\mathtt{LM}~{}(\text{prompt})$

7:$\mathcal{H}_{t+1}\leftarrow\mathcal{H}_{t}\oplus t_{\text{mid}}(z_{t})$

8: else if$e_{\text{ret}}(y_{t})\neq\emptyset$then $\triangleright$ Meta Model returned a final answer

9: return $e_{\text{ret}}(y_{t})$

10: else$\triangleright$ Meta Model formatting error

11:$\mathcal{H}_{t+1}\leftarrow\mathcal{H}_{t}\oplus\text{error}$

12: end if

13:end for

1. 1.

    Transforming the Input: Using the transformation function $t_{\text{init}}$, the raw query is placed in a suitable template followed by initial instructions to the Meta Model.

2. 2.

    Loop Iteration:

    1. (a)
            Prompting the Meta Model: The current message list, namely $\mathcal{H}_{t}$, guides the Meta Model’s next action—either directly addressing the query or consulting a domain-specific expert.

        2. (b)
            Engaging Domain-Specific Expert Models: If the Meta Model does not return a result, it can conjure any expert and give it instructions, which are extracted from its output using $e_{\text{exp}}$. This process is isolated though: Each expert only sees what the Meta Model chooses to share with them, and responds accordingly. For instance, if a problem pertains to mathematics and history, the Meta Model might consult a mathematics expert for a calculation and a history expert for historical context. The output of the expert is extracted and additional instructions are appended, all using the $t_{\text{mid}}$ template.

        3. (c)
            Returning the Final Response: If the Meta Model’s response contains a final answer (highlighted by distinct special markers), the solution is extracted using $e_{\text{ret}}$ and returned.

        4. (d)
            Error Handling: In cases where the model response $y_{t}$ contains neither a final answer nor a call to an expert model, an error message appended to the message list $\mathcal{H}_{t}$. This ensures that our procedure is robust and can handle unexpected outputs.

Meta and Expert Model Specifications. In our setup, we employ the same LM, such as GPT-4, to function in both Meta and Expert capacities. Their roles are distinguished by their respective model instructions in their prompts, with the Meta Model adhering to a set of instructions provided in Figure[3], and the expert models following separate instructions dynamically determined by the Meta Model at inference time .

Experimental Setup
------------------

### 3.1 Baselines

We compare meta-prompting with the task-agnostic, zero-shot versions of the following prompting methods:

* •

    Standard prompting: This represents our most basic baseline wherein an LM is asked to directly yield a response without any specific guiding input-output exemplars or any additional guiding instructions, besides the task description already included in the input query.

* •

    Zero-shot CoT prompting *(Kojima et al., [2022])*:
    Drawing inspirations from the chain-of-thought method of *Wei et al. ([2022b])*, this zero-shot prompting approach simply appends “Let’s think step by step” to the input query, encouraging the model to have a more deliberative and iterative cognition before addressing the problem or task at hand.

* •

    Expert prompting *(Xu et al., [2023])*: This prompting approach functions through a two-step process: It first crafts an expert identity tailored to align with the specific context of the input query. It then integrates this generated expert profile into the input to generate a well-informed and authoritative response. In our experiments, we consider two versions of expert prompting, namely (a) *static* (i.e., *generic*) and (b) *dynamic* (i.e., *adaptive*); the former uses a fixed and generic expert description, whereas the latter adaptively designs a new expert identity for each input query.

* •

    Multi-persona prompting *(Du et al., [2023])*: Also known as solo-performance prompting (SPP), this method instructs an LM to perform the following: (i) Propose a small ensemble of “personas” to address the specific task or problem at hand; (ii) let these personas engage in a collective dialogue, collaboratively generating potential solutions while extending feedback to one another and refining their answers; and (iii) synthesize all the available information and deliver a final response.

### 3.2 Datasets and Tasks

To evaluate the efficacy of our proposed meta-prompting approach over other zero-shot prompting baselines, we consider a wide range of tasks and datasets that require various degrees of mathematical and algorithmic reasoning, domain-specific knowledge, and literary creativity. These include:

* •

    (a) The Game of 24 from *(Yao et al., [2023a])* where the goal is to form an arithmetic expression whose value is 24 using each of four given numbers exactly once,

* •

    Three BIG-Bench Hard (BBH; *Suzgun et al. ([2023b])*) tasks—namely, (b) Geometric Shapes, (c) Multi-Step Arithmetic Two, and (d) Word Sorting—as well as one reasoning task directly obtained from the BIG-Bench suite*(BIG-Bench authors, [2023])*, that is, (e) Checkmate-in-One;

* •

    (f) Python Programming Puzzles (P3;*Schuster et al. ([2021])*), a collection of challenging programming puzzles written in Python—with varying difficulty levels;

* •

    (g) Multilingual Grade School Math (MGSM; *Shi et al. ([2023])*), a multilingual version of the GSM8K dataset*(Cobbe et al., [2021])* with translations of a subset of examples into ten typologically diverse languages, including Bengali, Japanese, and Swahili;

* •

    (h) Shakespearean Sonnet Writing, a novel task we created where the goal is to write a sonnet with strict rhyme scheme “ABAB CDCD EFEF GG,” containing the three provided words verbatim.333While all the other tasks and datasets were previously introduced by other studies, we present this task for the first time.

### 3.3 Answer Extraction and Evaluation Protocols

As shown in Figure[3], the system instruction in our proposed meta-prompting method encourages the Meta Model to present its final answer in a specific format. This format, designed for consistent and unambiguous extraction, requires that the final answer is wrapped within triple quotes and preceded by a distinct marker (namely, “>>FINAL ANSWER:”).

Once the final answer is extracted from the model and properly post-processed, we also need to evaluate its correctness.444We have developed suitable pipelines for answer extraction and processing tailored to each task. Specific implementation details can be found in our codebase. Because we consider a wide range of tasks, there is not a single metric that allows us to measure accuracy across all. Depending on the nature and formulation of the task, we measure accuracy using one of the following three metrics:

* •

    Exact Match (EM): Under this strict metric, the correctness of an answer is determined by its precise alignment with the ground-truth label(s). An answer is deemed correct only if it is identical to a provided reference.

* •

    Soft Match (SM): This metric offers a more lenient approach than EM. For an answer to be deemed correct, it is sufficient for a ground-truth label to be present within the model’s output, regardless of any additional textual content.

* •

    Functionally Correct (FC): This metric ascertains whether the answer is functionally correct, meaning that it adheres to task-specific constraints.

We use EM for Geometric Shapes, Multi-Step Arithmetic Two, and Checkmate-in-One; SM for MGSM and Word Sorting,; and FC for Game of 24, Python Programming Puzzles, and Shakespearean Sonnet Writing.

### 3.4 Models and Inference

In our main experiments, we concentrate on GPT-4 (gpt-4-32k), which is accessible through Microsoft’s Azure OpenAI Service. Additionally, in our supplementary experiments, we include GPT-3.5 (gpt-35-turbo). Both GPT-3.5 and GPT-4 are models fine-tuned for following instructions, though GPT-4 has demonstrated significantly better reasoning and content generation abilities than GPT-3.5.555In our preliminary experiments, we also tested other OpenAI models such as text-davinci-003 and code-davinci-002, but we discovered that our meta-prompting approach yielded consequential results when applied to GPT-3.5 and GPT-4.

In all of our experiments, we consistently applied the same parameters and system instructions to the Meta Model. We set the temperature value at <!-- MathML: <math alttext="0" class="ltx_Math" display="inline" id="S3.SS4.p2.1.m1.1"><semantics id="S3.SS4.p2.1.m1.1a"><mn id="S3.SS4.p2.1.m1.1.1" xref="S3.SS4.p2.1.m1.1.1.cmml">0</mn><annotation-xml encoding="MathML-Content" id="S3.SS4.p2.1.m1.1b"><cn id="S3.SS4.p2.1.m1.1.1.cmml" type="integer" xref="S3.SS4.p2.1.m1.1.1">0</cn></annotation-xml></semantics></math> -->00, the top-p value at $0.95$, and the maximum token count at $1024$.666The temperature value, which usually ranges between 0 and 1, controls how much randomness or creativity the model exhibits. Ideally, a temperature of 0 should lead to the model producing the same output when presented with the same input. However, both GPT-3.5 and GPT-4 have shown a tendency to generate varied responses even at this setting. This means that reproducing our exact results might be challenging under identical experimental conditions. To address this issue, we are releasing all model inputs, interactions, and outputs in our GitHub repository.

<img src='x2.png' alt='Refer to caption' title='' width='789' height='944' />

*Figure 3: The instructions given to the Meta Model using the “system message” parameter in the GPT-4 API.*

Main Results and Discussion
---------------------------

*Table 1: Comparison of baselines with meta-prompting across tasks. Without a Python interpreter, meta-prompting significantly outperforms other methods on the Checkmate-in-One and Sonnet Writing tasks and is on par on most other tasks except Geometric Shapes. Meta-prompting can leverage the Python interpreter in a task-agnostic manner to improve performance significantly across many tasks.*

|  | Basic | | Expert | | SPP | Meta | | $\Delta$ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| \cdashline2-9 Task | Standard | 0-CoT | Static | Dynamic | Multi-Persona | - Python | + Python | (M-S) |
| Checkmate-in-One | 36.4 | 32.8 | 39.6 | 33.2 | 17.2 | 57.2 | 57.2 | +20.8 |
| Game of 24 | 3.0 | 11.0 | 3.0 | 2.0 | 25.0 | 11.0 | 67.0 | +64.0 |
| Geometric Shapes | 56.8 | 69.2 | 55.2 | 53.6 | 57.6 | 58.4 | 59.2 | +2.4 |
| MGSM (avg) | 84.4 | 85.5 | 83.0 | 85.0 | 85.7 | 85.4 | 84.8 | +0.4 |
| Multi-Step Arithmetic | 84.0 | 83.2 | 83.2 | 78.8 | 91.6 | 84.8 | 90.0 | +6.0 |
| Python Prog. Puzzles | 31.1 | 36.3 | 33.8 | 25.0 | 32.5 | 32.7 | 45.8 | +14.7 |
| Sonnet Writing | 62.0 | 71.2 | 74.0 | 74.0 | 73.2 | 77.6 | 79.6 | +17.6 |
| Word Sorting | 80.4 | 83.6 | 83.2 | 85.2 | 79.2 | 84.0 | 99.6 | +19.2 |
| Average (*macro*) | 54.8 | 59.1 | 56.9 | 54.6 | 57.7 | 61.4 | 72.9 | +18.1 |

The results of our experiments, summarized in Table[4], demonstrate the superior effectiveness of our meta-prompting approach compared to the standard zero-shot prompting methods. When we look at the overall performance across all tasks, there is a notable increase in accuracy with meta-prompting, especially when it is augmented with a Python interpreter. Specifically, meta-prompting outperforms standard prompting by 17.1%, expert (dynamic) prompting by 17.3%, and multipersona prompting by 15.2%. Below, we delve into four key insights that emerged from our empirical analysis.

*Table 1: Comparison of baselines with meta-prompting across tasks. Without a Python interpreter, meta-prompting significantly outperforms other methods on the Checkmate-in-One and Sonnet Writing tasks and is on par on most other tasks except Geometric Shapes. Meta-prompting can leverage the Python interpreter in a task-agnostic manner to improve performance significantly across many tasks.*
