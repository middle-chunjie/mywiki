The Dawn of LMMs:  Preliminary Explorations with GPT-4V(ision)
===============================================================

Zhengyuan Yang∗, Linjie Li∗, Kevin Lin∗, Jianfeng Wang∗, Chung-Ching Lin∗,  
Zicheng Liu, Lijuan Wang∗♠  
Microsoft Corporation  
∗Core Contributor♠Project Lead

###### Abstract

Large multimodal models (LMMs) extend large language models (LLMs) with multi-sensory skills, such as visual understanding, to achieve stronger generic intelligence. In this paper, we analyze the latest model, GPT-4V(ision)*[[99](#bib.bib99 ""), [100](#bib.bib100 ""), [101](#bib.bib101 ""), [1](#bib.bib1 "")]*111This report explores GPT-4V(ision) with the vision capability and refers to the model as “GPT-4V,” following the OpenAI reports*[[100](#bib.bib100 ""), [99](#bib.bib99 "")]*. We refer to the text-only version of the model as “GPT-4 (no vision)”*[[99](#bib.bib99 "")]*., to deepen the understanding of LMMs. The analysis focuses on the intriguing tasks that GPT-4V can perform, containing test samples to probe the quality and genericity of GPT-4V’s capabilities, its supported inputs and working modes, and the effective ways to prompt the model. In our approach to exploring GPT-4V, we curate and organize a collection of carefully designed qualitative samples spanning a variety of domains and tasks. Observations from these samples demonstrate that GPT-4V’s unprecedented ability in processing arbitrarily interleaved multimodal inputs and the genericity of its capabilities together make GPT-4V a powerful multimodal generalist system. Furthermore, GPT-4V’s unique capability of understanding visual markers drawn on input images can give rise to new
human-computer interaction methods such as visual referring prompting.
We conclude the report with in-depth discussions on the emerging application scenarios and the future research directions for GPT-4V-based systems. We hope that this preliminary exploration will inspire future research on the next-generation multimodal task formulation, new ways to exploit and enhance LMMs to solve real-world problems, and gaining better understanding of multimodal foundation models.
Finally, we acknowledge that the model under our study is solely the product of OpenAI’s innovative work, and they should be fully credited for its development. Please see the GPT-4V contributions paper*[[101](#bib.bib101 "")]* for the authorship and credit attribution: [https://cdn.openai.com/contributions/gpt-4v.pdf](https://cdn.openai.com/contributions/gpt-4v.pdf "").

###### Contents

###### List of Figures

1 Introduction
--------------

### 1.1 Motivation and Overview

The breakthroughs in large language models (LLMs)*[[23](#bib.bib23 ""), [99](#bib.bib99 ""), [30](#bib.bib30 ""), [11](#bib.bib11 ""), [123](#bib.bib123 ""), [53](#bib.bib53 "")]* have shown remarkable versatilities and capabilities across various domains and tasks. The next evolution in this field, large multimodal models (LMMs), aims to expand upon the capabilities of LLMs by integrating multi-sensory skills to achieve even stronger general intelligence. Given the dominance of the visual in human senses*[[33](#bib.bib33 ""), [58](#bib.bib58 "")]*, many LMM studies start with extending the vision capability.
Preliminary research investigations either finetune a vision encoder to align with a frozen pre-trained LLM*[[125](#bib.bib125 ""), [7](#bib.bib7 ""), [71](#bib.bib71 ""), [55](#bib.bib55 ""), [42](#bib.bib42 ""), [13](#bib.bib13 ""), [48](#bib.bib48 ""), [157](#bib.bib157 ""), [79](#bib.bib79 ""), [35](#bib.bib35 ""), [146](#bib.bib146 "")]*, or use a vision-language model to convert visual inputs to text descriptions that LLMs can understand*[[149](#bib.bib149 ""), [141](#bib.bib141 ""), [131](#bib.bib131 ""), [54](#bib.bib54 ""), [113](#bib.bib113 ""), [142](#bib.bib142 "")]*.
However, most existing models*[[13](#bib.bib13 ""), [48](#bib.bib48 ""), [157](#bib.bib157 ""), [79](#bib.bib79 ""), [35](#bib.bib35 ""), [69](#bib.bib69 "")]* are of limited model and data scales, potentially restricting the emergence of various intriguing abilities. Consequently, it remains unclear what are the status quo and emergent multimodal abilities of LMMs that are developed based on the state-of-the-art LLMs, such as GPT-4 (no vision)*[[99](#bib.bib99 "")]* and PaLM*[[30](#bib.bib30 ""), [11](#bib.bib11 "")]*.
In this paper, we report our preliminary explorations with (an early version of) GPT-4V, a state-of-the-art LMM with vision, built based on the SOTA LLM and trained with a large scale of multimodal data.

Our exploration of GPT-4V is guided by the following questions.

1. 1.

    *What are GPT-4V’s supported inputs and working modes?* The genericity of multimodal models inevitably requires the system to work with the arbitrary mix of different input modalities. GPT-4V shows unprecedented ability in understanding and processing an arbitrary mix of input images, sub-images, texts, scene texts, and visual pointers. We also demonstrate that GPT-4V well supports the test-time techniques observed in LLMs, including instruction following*[[102](#bib.bib102 "")]*, chain-of-thoughts*[[136](#bib.bib136 ""), [66](#bib.bib66 "")]*, in-context few-shot learning*[[23](#bib.bib23 "")]*, *etc*.

2. 2.

    *What are the quality and genericity of GPT-4V’s capabilities on different domains and tasks?* We sample queries covering a wide range of domains and tasks to understand GPT-4V’s capabilities, including open-world visual understanding, visual description, multimodal knowledge, commonsense, scene text understanding, document reasoning, coding, temporal reasoning, abstract reasoning, emotion understanding, and many more. GPT-4V shows impressive human-level capabilities across many of the experimented domains.

3. 3.

    *What are effective ways to use and prompt GPT-4V?* GPT-4V is strong in understanding pixel space edits, such as visual pointers and scene texts drawn on input images. Inspired by this capability, we discuss the “visual referring prompting” that directly edits input images to instruct the task of interest. Visual referring prompting can be seamlessly used together with other image and text prompts, presenting a nuanced interface for instruction and example demonstrations.

4. 4.

    *What are promising future directions?* Given GPT-4V’s strong capability across domains and tasks, we ask what is the next step for multimodal learning, and more broadly for artificial intelligence. We organize our thoughts and explorations into two perspectives, *i.e*., emergent novel application scenarios to focus on, and the future research directions for GPT-4V-based systems. We present our preliminary explorations to inspire future studies.

Guided by the aforementioned problems, we comprehensively organize and list our explored qualitative results. The report contains minimal quantitative benchmark results, and instead consists of mainly selected interesting qualitative examples. Despite being less rigorous, this design allows for providing a more comprehensive analysis covering a broad range of domains, tasks, working modes, and prompting techniques, under a fixed capacity. We believe this organized collection of explorations will inspire future works in emerging novel applications, next-generation multimodal task formulation, and developing advanced LMM-based intelligent systems.

### 1.2 Our Approach in Exploring GPT-4V

Goal of this report. The standard approach for evaluating a system is by benchmarking it against a series of carefully designed datasets, each representing a specific domain and task.
One challenge is that some of the existing benchmarks may not be suitable for evaluating LMMs anymore. For example, the image captioning outputs of LMMs are much richer and contain more detailed descriptions than the ground truths in the image captioning benchmark datasets*[[27](#bib.bib27 "")]*. There is also a lack of public information regarding GPT-4V’s large-scale pre-training, which may violate the train-test setup for certain existing datasets and invalidate those benchmark numbers. Because of this, restricting the evaluation to *existing* benchmarks and metrics may unintentionally narrow the scope of GPT-4V’s assessment.
Developing a comprehensive list of next-generation evaluation tasks and benchmarks would be the ideal ultimate solution. However, we left those as future work due to the significant efforts required.

In lieu of quantitative benchmarking, this paper focuses on using qualitative results to provide a glimpse of GPT-4V’s new capabilities and potential emerging use cases. Our goal is to discover and preview what GPT-4V might already be capable of, even though these novel capabilities may not yet be entirely reliable. We hope this collection of explorations will inspire future research in establishing quantitative benchmarks for next-generation multimodal tasks, modernizing existing benchmarks, further improving model performance and system reliability, and sparkling innovation in emerging use cases. Following this, we will delve into the core designs for our approach to exploring GPT-4V.

Sample selection guidance. This report focuses on presenting qualitative results to showcase the potential capabilities of GPT-4V, rather than providing comprehensive quantitative benchmark results. This naturally raises the question of the reliability of the showcased examples. The examples featured in this report may require careful instruction tuning to amplify GPT-4V’s corresponding capabilities. It should be noted that some complex cases may only work with the specifically designed prompts. As such, the capabilities demonstrated may not consistently work across different samples. Instead of showing only the reliable functionalities, the primary objective of this report is to provide readers with a list of our discovered potential capabilities of GPT-4V, which might otherwise be overlooked after a few unsuccessful trials.

Sample selection to prevent mere memorizing from training. A fundamental design consideration in qualitative reports*[[24](#bib.bib24 "")]* is discerning models’ true capabilities from merely memorizing responses from training samples or making educated guesses based on hints from instructions and in-context examples.
We carefully control both the images and text in the input prompts to prevent them from being seen during GPT-4V training. We generate original text queries from scratch, and try to use images that are either not accessible online or with a timestamp beyond April 2023. We will indicate instances where a specific sample does not meet this criterion, *e.g*., deliberately using samples from specific vision-language datasets.
Beyond ensuring that samples are unseen, we incorporate rationale queries into the process. These queries are designed to probe the model’s reasoning process, thereby validating GPT-4V’s possession of the intended capability.

The default working mode. As later detailed in Section[3](#S3 "3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V works effectively in different working modes, including zero-shot learning with instructions, in-context few-shot learning, *etc*. Among them, this report primarily focuses on zero-shot instruction tuning, as opposed to in-context few-shot learning. This design is to prevent potential information leakage from in-context examples. While in-context few-shot examples can enhance performance and reliability, they do not consistently engender new capabilities. As such, we designate zero-shot as the default working mode for presentation, and reduce the use of in-context examples to minimize examples’ impacts on the assessed capabilities.

### 1.3 How to Read this Report?

This report documents the explorations of GPT-4V conducted by researchers in the computer vision and vision-language multimodal field. It is primarily geared towards fellow researchers in related disciplines who seek to gain a qualitative impression of LMM’s capabilities and understand its difference from traditional vision-language models. The report is also prepared for professionals for whom AI or computer science may be outside their specialties, to assist them in conceptualizing ways LMMs can enhance their proficiency within their distinct domains of expertise.

We give an overview of the report, structured around the four core questions that guide our exploration.

1. 1.

    *What are GPT-4V’s supported inputs and working modes?* Section[2](#S2 "2 GPT-4V’s Input Modes ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") summarizes GPT-4V’s supported inputs and presents an overview of their corresponding use cases.
    Based on the flexible interleaved image-text inputs, Section[3](#S3 "3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") discusses GPT-4V’s different working modes, such as instruction tuning, in-context learning, and other emergent usages. The section covers the novel ways of using and prompting GPT-4V, aiming to provide a comprehensive overview of how we will use GPT-4V in subsequent sections.

2. 2.

    *What are the quality and genericity of GPT-4V’s capabilities on different domains and tasks?* The exploration of this question makes up a large portion of the report. Section[4](#S4 "4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") provides a comprehensive analysis covering a wide range of vision and vision-language scenarios, including image description and recognition on different domains, dense visual understanding, multimodal knowledge, commonsense, scene text understanding, document reasoning, and many more. We also separate out several novel and interesting capabilities.
    Section[6](#S6 "6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") studies GPT-4V’s capability in temporal, motion, and video understanding. Section[7](#S7 "7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") explores the abstract visual understanding and reasoning capability, and Section[8](#S8 "8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") covers the emotion and sentiment understanding.

3. 3.

    *What are effective ways to use and prompt GPT-4V?* We start the discussion on this question from the working mode and prompting method introduction in Section[3](#S3 "3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). In Section[5](#S5 "5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we highlight one novel promoting technique, namely visual referring prompting, which draws visual pointers and scene texts on input images to prompt GPT-4V. We demonstrate the flexible prompting methods, such as the combination of instruction and example demonstrations, throughout the report in the given examples.

4. 4.

    *What are promising future directions?* Section[9](#S9 "9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") focuses on the novel use cases facilitated by GPT-4V. We hope these initial examples could inspire future works to design new task setups and present rigorous benchmarks. Section[10](#S10 "10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") imagines powerful future systems that can be built based on GPT-4V, such as the multimodal plugins, multimodal chains,
    self-reflection, self-consistency, and retrieval-augmented LMMs, *etc*.

In addition to this overview and the table of contents, we have also included a list of figures. The list enumerates the qualitative examples detailed within the report, serving as an additional tool to help readers navigate to their scenarios of interest.

2 GPT-4V’s Input Modes
-----------------------

This section summarizes GPT-4V’s supported inputs, *i.e*., functioning as a uni-model language model with the text-only inputs, taking single image-text pair optionally with only a single image, and taking interleaved image-text pairs optionally with only multiple image inputs. We next highlight the representative use cases under these different input modes.

### 2.1 Text-only Inputs

GPT-4V’s strong language capability enables it to serve as an effective unimodal language model*[[38](#bib.bib38 ""), [108](#bib.bib108 ""), [23](#bib.bib23 "")]* with text-only inputs. Operating exclusively with text for both input and output, GPT-4V is capable of performing a wide variety of language and coding tasks. We refer readers to the GPT-4 technical report*[[99](#bib.bib99 "")]* for the comprehensive and in-depth analysis of GPT-4V’s language and coding capabilities, as well as the comparison with GPT-4 (no vision).

### 2.2 Single Image-text Pair

GPT-4V, the latest large multimodal model, takes images and texts as inputs to generate textual outputs. In line with existing general-purpose vision-language models*[[9](#bib.bib9 ""), [81](#bib.bib81 ""), [73](#bib.bib73 ""), [8](#bib.bib8 ""), [70](#bib.bib70 ""), [122](#bib.bib122 ""), [120](#bib.bib120 ""), [155](#bib.bib155 ""), [28](#bib.bib28 ""), [83](#bib.bib83 ""), [45](#bib.bib45 ""), [74](#bib.bib74 ""), [57](#bib.bib57 ""), [64](#bib.bib64 ""), [72](#bib.bib72 ""), [132](#bib.bib132 ""), [29](#bib.bib29 ""), [140](#bib.bib140 ""), [41](#bib.bib41 ""), [7](#bib.bib7 ""), [128](#bib.bib128 ""), [46](#bib.bib46 ""), [40](#bib.bib40 ""), [158](#bib.bib158 ""), [69](#bib.bib69 "")]*, GPT-4V can take a single image-text pair or a single image as input to perform various vision and vision-language tasks, such as image recognition*[[37](#bib.bib37 "")]*, object localization*[[153](#bib.bib153 "")]*, image captioning*[[27](#bib.bib27 "")]*, visual question answering*[[12](#bib.bib12 "")]*, visual dialogue*[[36](#bib.bib36 "")]*, dense caption*[[62](#bib.bib62 "")]*, and so on. We note that the text in the image-text pair can be used either as instruction like “describe the image” for captioning, or as the query input like the question in visual question answering. GPT-4V’s exceptional intelligence is exemplified by its significantly enhanced performance and generalizability compared to prior arts. A comprehensive analysis of its multimodal capabilities on various domains is detailed in Section[4](#S4 "4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

<img src='x1.png' alt='Refer to caption' title='' width='461' height='433' />

*Figure 1: GPT-4V can work with multi-image and interleaved image-text inputs. Check Section[2.3](#S2.SS3 "2.3 Interleaved Image-text Inputs ‣ 2 GPT-4V’s Input Modes ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 2.3 Interleaved Image-text Inputs

The generality of GPT-4V is further enhanced by its ability to handle flexibly interleaved image-text inputs. The interleaved image-text inputs can be either visually centric such as multiple images with a short question or instruction, text-centric such as a long webpage with two inserted images, or a balanced mixture of images and texts. This mode of mixed input provides flexibility for a wide array of applications. For example, it can compute the total tax paid across multiple receipt images, as shown in Figure[1](#S2.F1 "Figure 1 ‣ 2.2 Single Image-text Pair ‣ 2 GPT-4V’s Input Modes ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). It also enables processing multiple input images and extracting queried information. GPT-4V could also effectively associate information across interleaved image-text inputs, such as finding the beer price on the menu, counting the number of beers, and returning the total cost, as shown in Figure[1](#S2.F1 "Figure 1 ‣ 2.2 Single Image-text Pair ‣ 2 GPT-4V’s Input Modes ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").
Beyond direct applications, processing interleaved image-text inputs serves as a fundamental component for in-context few-shot learning and other advanced test-time prompting techniques, thereby further boosting GPT-4V’s generality. We demonstrate these intriguing novel usages in the next section, Section[3](#S3 "3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

3 GPT-4V’s Working Modes and Prompting Techniques
--------------------------------------------------

<img src='x2.png' alt='Refer to caption' title='' width='461' height='577' />

*Figure 2: GPT-4V can understand and follow text instructions, to generate the desired text outputs or learn to perform a new task. Red highlights the less informative answer. Check Section[3.1](#S3.SS1 "3.1 Following Text Instructions ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x3.png' alt='Refer to caption' title='' width='438' height='738' />

*Figure 3: Constrained prompting to return in JSON format. Images are example IDs for samples. Red highlights the wrong answer. Check Section[3.1](#S3.SS1 "3.1 Following Text Instructions ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x4.png' alt='Refer to caption' title='' width='438' height='741' />

*Figure 4: Condition on good performance to improve counting. Green(Red) highlights the correct (wrong) answer. Blue indicates different ways to prompting in addition to the basic requirement of “Count the number of apples in the image.” Check Section[3.1](#S3.SS1 "3.1 Following Text Instructions ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 3.1 Following Text Instructions

One unique strength of GPT-4V is its generality, partially achieved via
its strong capability in understanding and following text instructions*[[102](#bib.bib102 ""), [96](#bib.bib96 ""), [134](#bib.bib134 ""), [111](#bib.bib111 "")]*. Instructions provide a natural way to define and customize the desired output text for arbitrary vision-language use cases. Figure[2](#S3.F2 "Figure 2 ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows an example of image descriptions with constraints on sentence length and the words to use. Alternatively, on the input side, GPT-4V could understand the detailed instructions to perform challenging tasks, such as enabling GPT-4V to better interpret the abstract reasoning question by providing instructions on intermediate steps. The ability to learn new tasks from instructions shows great potential in adapting to various unseen applications and tasks, as detailed in Section[9](#S9 "9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). In line with recent studies*[[7](#bib.bib7 ""), [13](#bib.bib13 ""), [48](#bib.bib48 ""), [157](#bib.bib157 ""), [79](#bib.bib79 ""), [35](#bib.bib35 "")]*, the instructions discussed in this subsection are mostly in the text format, providing language descriptions of the interested task. We will discuss GPT-4V’s unique capability of following multimodal example-grounded instructions later in Section[3.3](#S3.SS3 "3.3 Visual + Text Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

In addition, we showcase how text instructions play an important role in shaping GPT-4V’s response with two techniques adopted from LLM literature*[[3](#bib.bib3 ""), [156](#bib.bib156 "")]*, ($i$) “constrained prompting” so that GPT-4V responds in a certain format; and ($ii$) “condition on good performance” that explicitly asks for good performance from GPT-4V.

#### Constrained prompting.

In Figure[3](#S3.F3 "Figure 3 ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we prompt GPT-4V to read the text in the image and return the information in a specific JSON format. Although GPT-4V makes some mistakes in extracting the corresponding information from driver’s licenses, the responses are constrained to the JSON format specified in the text instruction. We leverage this technique for certain application scenarios in Section[9](#S9 "9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

#### Condition on good performance.

One observation about LLMs is that LLMs don’t want to succeed*[[10](#bib.bib10 "")]*. Rather, they want to imitate training sets with a spectrum of performance qualities. If the user wants to succeed in a task given to the model, the user should explicitly ask for it, which has proven useful in improving the performance of LLMs*[[156](#bib.bib156 "")]*. In the context of LMMs, we have similar observations.
In Figure[4](#S3.F4 "Figure 4 ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we compare the model’s response to different text instructions for counting. We start with a simple and clear prompt: “Count the number of apples in the image.” However, GPT-4V incorrectly counts a total of 12 apples in the image. To improve its performance, we explore the use of zero-shot chain-of-thought from*[[66](#bib.bib66 "")]* for LLMs by adding the phrase “Let’s think step-by-step.” Although GPT-4V’s predicted steps are generally correct, they are not very helpful for the final count, as it still arrives at the incorrect answer of “12 apples.” Next, we modify the instruction to “Let’s count the apples row-by-row,” which is more relevant to the visual input. While GPT-4V provides the correct total count, it makes mistakes in counting the second/third row. When we further expand the instruction to “First count how many rows of apples there are, then count the apples in each row, and finally sum them up to get the total number,” the final answer deviates even more from the correct answer (15 vs. 11).
Finally, imitating “Let’s work this out in a step by step way to be sure we have the right answer.” in*[[156](#bib.bib156 "")]* for LLMs, we design the prompt as follows:
“You are an expert in counting things in the image. Let’s count the number of apples in the image below row by row to be sure we have the right answer.”. The first sentence in our prompt asks GPT-4V to assume the role of an expert in counting, and the second sentence explicitly instructs GPT-4V to succeed. With this design, GPT-4V successfully returns the correct answer for each row as well as the total count. Throughout the paper, we employ this technique in various scenarios for better performance.

### 3.2 Visual Pointing and Visual Referring Prompting

Pointing is a fundamental aspect of human-human interaction*[[89](#bib.bib89 "")]*. To provide a comparable channel of interaction, various forms of “pointing” are studied to refer to an arbitrary spatial region of interest. For example, as depicted in Figure[5](#S3.F5 "Figure 5 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), “pointing” can be represented as numerical spatial coordinates such as box coordinates and image crops, or visual markers overlaid on image pixels such as arrows, boxes, circles, and hand drawings. We observe that GPT-4V is particularly strong in understanding visual pointers drawn directly on images.
Given the flexibility of drawing on images, this capability can be used as a natural approach for future human-computer interaction in the wild*[[90](#bib.bib90 ""), [117](#bib.bib117 ""), [157](#bib.bib157 "")]*. To this end, we explore a new prompting method named visual referring prompting, where people edit the pixel space of input images to specify the desired objective, such as drawing visual pointers or handwriting scene texts.
As illustrated in Figure[6](#S3.F6 "Figure 6 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), visual referring prompting edits the image pixels, instead of the conventional text prompts, to perform the task of interest. For example, it could be a simple grounded description, which focuses on describing the pointed object while maintaining the understanding of the global image context, as shown in Figure[6](#S3.F6 "Figure 6 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")(1,2). Visual referring prompting also enables other novel use cases, such as associating the pointed object with an index written in scene text (Figure[6](#S3.F6 "Figure 6 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")(3)), or solving the question asked near the queried edge or angle (Figure[6](#S3.F6 "Figure 6 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")(4)). Section[5](#S5 "5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") will discuss visual referring prompting in more detail.

<img src='x5.png' alt='Refer to caption' title='' width='461' height='67' />

*Figure 5: Different modes of “visual pointing” in multimodal interaction.*

<img src='x6.png' alt='Refer to caption' title='' width='461' height='620' />

*Figure 6: GPT-4V demonstrates the unique capability of understanding visual pointing directly overlaid on images. Based on such capability, we explore visual referring prompting that edits input image pixels (*e.g*., drawing visual pointers and scene texts) to prompt the task of interest. Check Section[3.2](#S3.SS2 "3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x7.png' alt='Refer to caption' title='' width='461' height='778' />

*Figure 7: GPT-4V is strong in interpreting an arbitrary mix of images, sub-images, texts, scene texts, and visual pointer inputs. These elements could serve as instructions, examples, or input queries, helping GPT-4V to effectively perform novel tasks. Check Section[3.3](#S3.SS3 "3.3 Visual + Text Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 3.3 Visual + Text Prompting

Visual referring prompting can be smoothly used together with other image-text prompts,
presenting a nuanced interface that succinctly represents the problem of interest. Figure[7](#S3.F7 "Figure 7 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") presents two examples to showcase the flexibility of GPT-4V’s prompt, particularly its proficiency in integrating different input formats and seamlessly mixing instructions with examples in the inputs. GPT-4V’s genericity and flexibility result in a human-like comprehension of multimodal instructions and an unprecedented ability to adapt to unseen tasks.

#### Integrated multimodal instruction inputs.

Existing models usually have implicit constraints on how interleaved image-text inputs should be formatted, *e.g*., in-context few-shot learning requires image-text pairs to share a similar format as the query input. In contrast,
GPT-4V shows the genericity in processing an arbitrary mix of images, sub-images, texts, scene texts, and visual pointers. For example, to illustrate the “adding a line” pattern in Figure[7](#S3.F7 "Figure 7 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), one could either point to the first column in the matrix image with a circle as in sub-figure (1), or incorporate the sub-images inline as in sub-figure (2). Similarly, for input query, one could either present a large figure with the question as scene texts as in sub-figure (1), or send the mix of texts and sub-images as in sub-figure (2). In contrast to GPT-4V’s flexibility, existing multimodal models are highly restricted in terms of how they can combine images and texts, and the number of images they can process, thereby imposing limitations on the model’s capability and genericity.

#### Multimodal example-grounded instruction.

In addition to supporting more flexible input formats, GPT-4V’s genericity also opens up more effective ways of illustrating the task to perform, compared with the instruction-following mode and in-context few-shot learning. Instruction-following techniques*[[102](#bib.bib102 ""), [96](#bib.bib96 ""), [134](#bib.bib134 ""), [111](#bib.bib111 "")]*, originally proposed for NLP tasks, intuitively focus on task instructions purely in the textual format. The text instruction is loosely related to the visual query input and thus may not provide a clear task demonstration. While in-context few-shot learning*[[23](#bib.bib23 ""), [125](#bib.bib125 ""), [7](#bib.bib7 "")]* provides test-time examples that contain both images and texts, these examples must align perfectly with the format of the inference query, making them complex and lengthy to incorporate. Furthermore, in-context examples are usually used separately from instructions, requiring the model to infer the task objective and thereby compromising the demonstration’s effectiveness.
In contrast, GPT-4V’s capability to comprehend multimodal instructions enables task demonstrations to be grounded onto corresponding in-context examples, therefore more effectively illustrating the task of interest. For example, in Figure[7](#S3.F7 "Figure 7 ‣ 3.2 Visual Pointing and Visual Referring Prompting ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), grounding instructions of “finding the pattern in the first column” onto the key steps in demonstration examples (*i.e*., the circled pattern in (1) and corresponding sub-figures in (2)) simplifies the learning process and enhances the model’s performance. This approach also mirrors the human learning process, which involves abstract instructions paired with intuitive examples.

### 3.4 In-context Few-shot Learning

In-context few-shot learning is another intriguing emergent ability observed in LLMs*[[23](#bib.bib23 ""), [39](#bib.bib39 ""), [135](#bib.bib135 ""), [34](#bib.bib34 "")]*. That is, LLMs can generate desired outputs without parameter updates by prepending a few in-context examples at inference time. The examples share the same format as the input query, and serve as demonstrations to illustrate the desired outputs. Similar abilities were recently observed in multimodal models*[[125](#bib.bib125 ""), [7](#bib.bib7 ""), [55](#bib.bib55 ""), [42](#bib.bib42 ""), [151](#bib.bib151 "")]*, where query inputs are formatted image-text pairs.
Complementary to instruction tuning, in-context learning “teaches” model to perform new tasks by providing in-context examples with the same format during test time.
We demonstrate the in-context few-shot learning capacity of GPT-4V through a few compelling examples. We emphasize that in certain scenarios, in-context few-shot learning with a sufficient number of examples becomes essential, particularly when zero-shot or one-shot instruction approaches fall short. Figures[8](#S3.F8 "Figure 8 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[10](#S3.F10 "Figure 10 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") explore a challenging scenario involving the reading of a speed meter. In Figure[8](#S3.F8 "Figure 8 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), the zero-shot performance of GPT-4V on a screenshot of a speed meter image from a video is depicted. Despite numerous attempts to prompt GPT-4V in a zero-shot manner, it struggles to accurately read the current speed displayed in the image. The predictions it generates (22/30/40 mph) deviate significantly from the actual human reading of “approximately 9 mph.” Even when employing a 1-shot in-context example, as shown in Figure[9](#S3.F9 "Figure 9 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), using either a dissimilar example (Figure[9a](#S3.F9.sf1 "In Figure 9 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")) or a similar example (Figure[9b](#S3.F9.sf2 "In Figure 9 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), GPT-4V still fails to accurately locate the two numbers on the left and right sides of the yellow pointer. In contrast, Figure[10](#S3.F10 "Figure 10 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") demonstrates that when provided with 2 in-context examples, one similar to the query image and the other dissimilar, GPT-4V successfully predicts the speed reading as “around 9 mph” by recognizing that the pointer is close to 10 mph but not quite there yet.

The comparison between zero-shot, 1-shot, and 2-shot performance for reasoning over a complex line plot is illustrated in Figures[11](#S3.F11 "Figure 11 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[13](#S3.F13 "Figure 13 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").
The example we explore here presents a great difficulty level as it involves multi-hop reasoning. To answer the question “In the graph, which year has the highest average gas price for the month of June,” one needs to go through at least four steps: ($i$) locating the month of June on the x-axis, ($ii$) comparing data points for each line in June, ($iii$) identifying the color of the line with the highest value, and ($iv$) matching the color to the corresponding year in the legend at the top. Failure in any of these steps would lead to an incorrect prediction. As depicted in Figure[11](#S3.F11 "Figure 11 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), even when prompted with “text instruction, think step-by-step” in a zero-shot manner, GPT-4V fails to correctly associate the colors with the years from the legend. Furthermore, it gets distracted by the highlighted gas price of $\$3.32$ in the graph. Similarly, in Figure[12](#S3.F12 "Figure 12 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), although GPT-4V shows some improvement in reading the legend (correcting the corresponding colors for 2021 and 2022 compared to zero-shot), it still insists on answering with 2023 as the year with the highest average gas price for the month of June, despite the fact that the chart only includes data points until 01/17/2023. However, as we introduce another in-context example in Figure[13](#S3.F13 "Figure 13 ‣ 3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V finally arrives at the correct answer (2022) and provides intermediate steps that explain its reasoning process, similar to the demonstration shown in the in-context examples.

These proof-of-concept examples vividly demonstrate the rising significance of in-context few-shot learning for achieving improved performance with LMMs. This approach serves as a viable alternative to finetuning, analogous to the observations made in the context of LLMs*[[23](#bib.bib23 ""), [39](#bib.bib39 ""), [135](#bib.bib135 ""), [34](#bib.bib34 "")]*.
Despite the great importance of in-context few-shot learning in achieving better performance with LMMs, we limit its use in this report to prevent the potential information leakage or undesired hints from in-context examples. We also leave the quantitative evaluation of few-shot learning’s gain to future studies.

<img src='x8.png' alt='Refer to caption' title='' width='461' height='742' />

*Figure 8: Zero-shot performance under the challenging scenario of reading a speed meter. GPT-4V fails to read the speed meter accurately even with different ways of ZS prompting. Red highlights the wrong answer. Check Section[3.4](#S3.SS4 "3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x9.png' alt='Refer to caption' title='' width='461' height='349' />

*(a)*

<img src='x10.png' alt='Refer to caption' title='' width='461' height='353' />

*(b)*

*Figure 9:  One-shot (or prompting with multimodal example instruction) performance under the challenging scenario of reading a speed meter. GPT-4V still fails with (a) dissimilar or (b) similar 1-shot in-context example. Red highlights the wrong answer. Check Section[3.4](#S3.SS4 "3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x11.png' alt='Refer to caption' title='' width='461' height='483' />

*Figure 10: Two-shot performance under the challenging scenario of reading a speed meter. GPT-4V now can read the speed accurately. Green highlights the correct answer. Check Section[3.4](#S3.SS4 "3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x12.png' alt='Refer to caption' title='' width='530' height='810' />

*Figure 11: Zero-shot performance under the challenging scenario of reading a line plot. GPT-4V fails to answer the question even with different ways of ZS prompting. Red highlights the wrong answer. Check Section[3.4](#S3.SS4 "3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x13.png' alt='Refer to caption' title='' width='461' height='537' />

*Figure 12: One-shot (or prompting with multimodal example instruction) performance under the challenging scenario of reading a line plot. GPT-4V still fails with 1-shot in-context example. Red highlights the wrong answer. Check Section[3.4](#S3.SS4 "3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x14.png' alt='Refer to caption' title='' width='461' height='762' />

*Figure 13: Two-shot performance under the challenging scenario of reading a line plot. GPT-4V now can answer the question of “which year has the highest average gas price for the month of June?” correctly. Check Section[3.4](#S3.SS4 "3.4 In-context Few-shot Learning ‣ 3 GPT-4V’s Working Modes and Prompting Techniques ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. Green highlights the correct answer.*

4 Vision-Language Capability
-----------------------------

Understanding and describing visual information plays a crucial role in human cognition. In this section, we will investigate how GPT-4V can be utilized to comprehend and interpret the visual world. We will start by examining the model’s ability to generate open-ended descriptions for generic visual captioning.

Moving forward, in Section[4.2](#S4.SS2 "4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we will explore the application of GPT-4V in more advanced tasks, such as spatial relationship analysis, object localization, object counting, and dense captioning. In Section[4.3](#S4.SS3 "4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we will delve into the model’s capacity for multimodal knowledge and commonsense reasoning, and study whether the model can understand the context and relationships between different types of information.

Additionally, in Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we will assess the model’s capability to extract and analyze information from various sources, including scene text, tables, charts, and documents. In Section[4.5](#S4.SS5 "4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we will explore GPT-4V’s ability in comprehending and generating descriptions in multilingual scenarios. Lastly, in Section[4.6](#S4.SS6 "4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we will investigate the model’s coding proficiency with visual information, exploring its ability to perform tasks with selected examples.

### 4.1 Image Description on Diverse Domains

We access the model’s capability and generalizability by providing a single image-text pair as input. We prompt GPT-4V to generate natural language descriptions covering a variety of topics listed below.

Celebrity recognition. Recognizing human appearance*[[49](#bib.bib49 ""), [80](#bib.bib80 "")]* presents a significant challenge due to its inherent variability. To assess GPT-4V’s capabilities to recognize and describe the celebrities, we conduct an experiment by providing a text prompt, “Describe the image,” along with an input celebrity image.
In the top row of Figure[14](#S4.F14 "Figure 14 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we observe that GPT-4V accurately identifies the eight celebrities, despite their diverse backgrounds and fields. Furthermore, when we present a more specific query, “Who is the person in the image and what is the person doing?,” as shown in the bottom row of Figure[14](#S4.F14 "Figure 14 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V comprehends that the current President of the United States is delivering a speech at the 2023 G7 Summit. This illustrates the model’s ability to generalize and handle novel scenarios, such as the 2023 G7 Summit, which was not part of its training data.

Landmark recognition. Landmarks exhibit considerable variations in appearance due to factors such as viewpoint changes, lighting conditions, occlusions, and seasonal changes. Recognizing landmarks under these variations requires models to generalize well and handle the vast range of visual appearances*[[152](#bib.bib152 ""), [5](#bib.bib5 "")]*. In the experiments, we employ a straightforward text prompt, “Describe the landmark in the image,” to test the model’s capability. As shown in Figures[15](#S4.F15 "Figure 15 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[16](#S4.F16 "Figure 16 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V generates accurate and open-ended descriptions for each test image. For example, it accurately recognizes Space Needle located in Seattle, Washington, understanding that the tower was built for the 1962 World’s Fair and has since become a symbol of the city. We have similar observations for other tested photos as well. The generated descriptions go beyond simple labels or generic phrases, providing vivid and detailed narratives that capture the essence of the landmark.

Food recognition. Recognizing food or dishes is a fascinating task*[[20](#bib.bib20 ""), [95](#bib.bib95 "")]*, but it can be challenging to tackle due to the wide range of appearances and potential occlusions caused by other objects or overlapping ingredients. In our experiments, we employ a straightforward text prompt, asking the system to “Describe the name of the dish,” for testing purpose. Figure[17](#S4.F17 "Figure 17 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") demonstrates the accurate recognition of various dishes by GPT-4V. Additionally, GPT-4V effectively captures intricate details within the images, enabling it to identify specific ingredients, garnishes, or cooking techniques present in a dish.

Medical image understanding. Medical images, such as X-rays and CT scans, can have large variability due to patient populations and imaging equipment. Additionally, interpreting the visual content of these images requires expert knowledge. In Figure[18](#S4.F18 "Figure 18 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we access GPT-4V’s performance by providing the prompt, “Describe the image.” The results show that GPT-4V recognizes both the teeth and jaw bones in the given X-ray. Furthermore, when we prompt with “Are there wisdom teeth that needs to be removed in this x-ray image?” GPT-4V performs reasoning with the visual context, and explains that the wisdom teeth on the bottom left and right sides of the jaw are not fully emerged from the gum line, and this could be a reason for removal. We also conduct testing with other medical images, as shown in Figure[19](#S4.F19 "Figure 19 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). For these experiments, we use prompts such as “What’s wrong?” or “Look at the CT scan, tell me what’s wrong.” The observations reveal that GPT-4V can identify common conditions such as a Jones fracture. It could also point out potential concerns based on the CT scan of the lung. The experiments demonstrate GPT-4V’s basic understanding of medical images. We discuss the application of GPT-4V to the medical domain in Section[9.3](#S9.SS3 "9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

Logo recognition. We examine GPT-4V’s ability in logo recognition. In Figure[20](#S4.F20 "Figure 20 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we initiate the experiments by providing the text prompt, “Describe the image.” GPT-4V accurately identifies the three logos depicted in the image. We then proceed to ask a more specific question, “Describe the logos in details,” GPT-4V provides elaborate descriptions, including the design, style, and representation for each logo, respectively. Expanding the evaluation to a more challenging in-the-wild scenario, as shown in Figure[21](#S4.F21 "Figure 21 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we experiment with logos that may be partially occluded, distorted, or situated in cluttered backgrounds. We employ the text prompt “Describe both the image and logo in details” for the in-the-wild experiment. As shown in Figure[21](#S4.F21 "Figure 21 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V demonstrates strong capability in understanding logos in difficult scenarios. Notably, GPT-4V can also provide descriptions for novel or emerging logos and icons, such as the recently released Microsoft 365 Copilot.

Scene understanding. Scene understanding*[[76](#bib.bib76 ""), [32](#bib.bib32 ""), [154](#bib.bib154 "")]* is an important task in computer vision. We examine the model’s capability by providing a simple query “Describe the image.” In Figure[22](#S4.F22 "Figure 22 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V is able to describe the road and the location and color of the vehicles. It can also read the sign and notice the speed limit for this road.

Counterfactual examples. We conduct experiments by randomly selecting counterfactual examples from*[[78](#bib.bib78 "")]*. In Figure[23](#S4.F23 "Figure 23 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we observe that GPT-4V correctly describes the image contents when faced with misleading questions or instructions.

<img src='x15.png' alt='Refer to caption' title='' width='461' height='626' />

*Figure 14: Results on celebrity recognition and description. GPT-4V can recognize a variety of celebrities and describe the visual information (including their profession, action, background, and the event) in details. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x16.png' alt='Refer to caption' title='' width='461' height='706' />

*Figure 15: Results on landmark recognition and description. GPT-4V accurately recognizes the landmarks in the test images. It also generates vivid and detailed narratives that capture the essence of the landmarks. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x17.png' alt='Refer to caption' title='' width='461' height='792' />

*Figure 16: Results on landmark recognition and description. GPT-4V accurately recognizes the landmarks in the test images. It also generates vivid and detailed narratives that capture the essence of the landmarks. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x18.png' alt='Refer to caption' title='' width='461' height='671' />

*Figure 17: Results on food recognition and description. GPT-4V recognizes various dishes. It also identifies specific ingredients, garnishes, or cooking techniques present in a dish image. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x19.png' alt='Refer to caption' title='' width='461' height='729' />

*Figure 18: Results on medical image understanding. GPT-4V recognizes both the teeth and jaw bones in the given X-ray, and explains that the partially emerged wisdom teeth on the bottom left and right sides of the jaw may necessitate removal. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. The medical images are collected from the Internet.*

<img src='x20.png' alt='Refer to caption' title='' width='461' height='617' />

*Figure 19: Results on medical image understanding. GPT-4V can identify common conditions like a Jones fracture. It could also point out potential concerns based on the CT scan of the lung. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. The medical images are from the internet.*

<img src='x21.png' alt='Refer to caption' title='' width='461' height='496' />

*Figure 20: Results on logo recognition. GPT-4V correctly recognizes the logos and provides detailed descriptions, including its design, color, shape, and symbol. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x22.png' alt='Refer to caption' title='' width='507' height='771' />

*Figure 21: Results on in-the-wild logo recognition and description. GPT-4V demonstrates strong capability in understanding logos in many scenarios, including occlusions, lighting conditions, and orientations. GPT-4V can also describe novel icons, such as the recently released Microsoft 365 Copilot. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x23.png' alt='Refer to caption' title='' width='461' height='706' />

*Figure 22: Results on scene understanding. GPT-4V is able to provide a detailed description regarding the scenes and objects. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x24.png' alt='Refer to caption' title='' width='461' height='626' />

*Figure 23: Results on counterfactual examples. GPT-4V is able to provide factual descriptions regarding the scenes and objects in the images. Example images are from*[[78](#bib.bib78 "")]*. Check Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 4.2 Object Localization, Counting, and Dense Captioning

Spatial relationship understanding. Understanding the spatial relationship between humans and objects in the image is a vital aspect of visual intelligence*[[61](#bib.bib61 ""), [14](#bib.bib14 "")]*. In Figure[24](#S4.F24 "Figure 24 ‣ 4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V showcases promising capabilities in this regard. It can identify the spatial relationship between the frisbee and the man in the image. It can also recognize the spatial relationship between the man and the car in the image, and point out that the camera perspective may affect their perceived size.

Object counting. Figure[25](#S4.F25 "Figure 25 ‣ 4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") highlights our exploration of GPT-4V’s capability in object counting. In our experiments, we employ the text prompt “Count the number of X in the image” to evaluate its performance. The results indicate that GPT-4V can successfully count the number of objects, such as apples, oranges, and people, present in the image. However, challenges arise when objects are occluded, or the scene is cluttered, which can result in errors in the counting process. In the bottom left of Figure[25](#S4.F25 "Figure 25 ‣ 4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V identifies 12 people, but the correct answer should be 11. This may be due to our limited text prompt used in this experiment, and further investigation in prompting techniques is needed.

Object localization. Object localization*[[153](#bib.bib153 ""), [76](#bib.bib76 ""), [51](#bib.bib51 "")]* is a fundamental challenge in the field of computer vision. In our preliminary experiments, we address this task by utilizing a simple text prompt, “Localize each person in the image using a bounding box.” The initial results of our object localization experiments are depicted in Figure[26](#S4.F26 "Figure 26 ‣ 4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). The findings suggest that GPT-4V demonstrates the capability to generate bounding box coordinates in textual format, without separate textualized box tokens*[[25](#bib.bib25 ""), [140](#bib.bib140 ""), [129](#bib.bib129 ""), [26](#bib.bib26 ""), [82](#bib.bib82 ""), [105](#bib.bib105 "")]*. However, it is important to note that the generated bounding box coordinates are not accurate. We rescaled the predicted bounding box coordinates during visualization. Promising localization results are observed when the scene or background is relatively simpler and less cluttered. Further prompting techniques are required to enhance object localization performance in more complex and crowded environments.

Dense captioning. Dense captioning*[[62](#bib.bib62 ""), [84](#bib.bib84 "")]* involves generating detailed description for each region of interest in the given image. This advanced task in vision-language field typically requires a complex system that integrates multiple experts, such as object detector, celebrity recognition model, and image captioning model. In order to explore GPT-4V’s capabilities in dense captioning, we use an instructional prompt, as shown in Figure[27](#S4.F27 "Figure 27 ‣ 4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). The results are highly encouraging, as GPT-4V successfully localizes and recognizes the individuals within the image, and then provides concise descriptions for each scientist.

<img src='x25.png' alt='Refer to caption' title='' width='461' height='626' />

*Figure 24: Results on spatial relationship understanding. GPT-4V recognizes the spatial relationship between the objects in the images. Example images are from*[[67](#bib.bib67 ""), [14](#bib.bib14 "")]*. Check Section[4.2](#S4.SS2 "4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x26.png' alt='Refer to caption' title='' width='461' height='617' />

*Figure 25: Results on object counting. GPT-4V is able to determine the quantity of the specified objects the image. Red highlights the wrong answer. Check Section[4.2](#S4.SS2 "4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x27.png' alt='Refer to caption' title='' width='461' height='761' />

*Figure 26: Results on object localization. GPT-4V is able to generate and approximate the bounding box coordinates for the specified objects in the image. When providing a simple text prompt only, the model may encounter challenges when dealing with more complex scenarios like object occlusions and cluttered scenes. Red highlights the wrong answer. We rescaled the predictions when visualizing the bounding boxes. Check Section[4.2](#S4.SS2 "4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x28.png' alt='Refer to caption' title='' width='461' height='805' />

*Figure 27: Results on dense captioning. GPT-4V follows the text prompt and successfully generates dense captions for the input image. Red highlights the wrong answer. We rescaled the predictions when visualizing the bounding boxes. Check Section[4.2](#S4.SS2 "4.2 Object Localization, Counting, and Dense Captioning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 4.3 Multimodal Knowledge and Commonsense

Joke and meme. Jokes and memes often reference specific events, pop culture, or Internet trends. Understanding these references requires being familiar with the relevant context and cultural knowledge. Grasping the visual elements, their relationship to the text, and the intended humorous effect can be a complex task*[[99](#bib.bib99 "")]*. Moreover, memes are often user-generated, making them highly diverse and ever-expanding. To evaluate GPT-4V’s ability in this domain, we input a pair of meme and text prompt to GPT-4V. The example text prompts include “Can you explain the meme?” and “What is funny about the image?” Figure[28](#S4.F28 "Figure 28 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows the example results. We observe that GPT-4V has remarkable ability to gather information from both visual and textual modalities, and then comprehend the humor embedded within memes.

Science and knowledge. We further investigate GPT-4V’s capability in tasks that requires reasoning with scientific knowledge*[[85](#bib.bib85 "")]*. We conduct experiments by providing a text prompt question and a corresponding image. The questions cover a wide range of topics, including geography, physics, biology, and earth science. In Figures[29](#S4.F29 "Figure 29 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[31](#S4.F31 "Figure 31 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we observe that GPT-4V is able to correctly answer the science questions based on the visual context.
For instance, in the bottom row of Figure[29](#S4.F29 "Figure 29 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V recognizes the average particle speed for both sample A and sample B. By considering the relationship among particle speed, kinetic energy, and temperature, GPT-4V answers the question correctly. For another instance, as shown in the bottom row of Figure[30](#S4.F30 "Figure 30 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V takes into account the visual arrows presented in the figure to identify the producer in the specific food web. Moreover, as shown in Figure[31](#S4.F31 "Figure 31 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), when we provide a more specific prompt, such as “Suppose you are a teacher, please use the figure to explain X,” we observe the generated answer adopts a tutorial format and explains the subject step by step.

Multimodal commonsense. In Figure[32](#S4.F32 "Figure 32 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we access the ability of GPT-4V in multimodal commonsense reasoning*[[148](#bib.bib148 ""), [52](#bib.bib52 "")]*. In our experiments, we observed that GPT-4V effectively utilizes the bounding boxes presented in the image as visual prompts (e.g., [person1] and [person2]) to recognize the actions performed by the individuals. As shown in the second example in Figure[32](#S4.F32 "Figure 32 ‣ 4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), based on the formal dress worn by [person1] and [person2] and the floral decorations present in the scene, it can be inferred that they are attending a wedding ceremony. Moreover, when we provide a more specific input prompt, such as "Suppose you are a detective, what can you infer from the visual clues?", GPT-4V demonstrates the ability to discern numerous nuanced visual cues within the image and offers a list of plausible hypotheses.

<img src='x29.png' alt='Refer to caption' title='' width='461' height='783' />

*Figure 28: Results on joke and meme understanding. GPT-4V demonstrates the impressive capability to comprehend the humor embedded within memes. Check Section[4.3](#S4.SS3 "4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x30.png' alt='Refer to caption' title='' width='461' height='648' />

*Figure 29: Results on answering science questions. GPT-4V can understand the question textually and visually, and gather necessary information to answer the question. Example images are from*[[85](#bib.bib85 "")]*. Check Section[4.3](#S4.SS3 "4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x31.png' alt='Refer to caption' title='' width='461' height='658' />

*Figure 30: Results on answering science questions. GPT-4V can understand the question textually and visually, and gather necessary information to answer the question. Example images are from*[[85](#bib.bib85 "")]*. Check Section[4.3](#S4.SS3 "4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x32.png' alt='Refer to caption' title='' width='461' height='519' />

*Figure 31: Results on answering science questions. When we use a more specific text prompt like “Suppose you are a teacher, please use the figure to explain X,” we observe that GPT-4V can generate a short tutorial for explaining the subject. Check Section[4.3](#S4.SS3 "4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x33.png' alt='Refer to caption' title='' width='461' height='792' />

*Figure 32: Results on multimodal commonsense reasoning. Example images are from*[[148](#bib.bib148 ""), [52](#bib.bib52 "")]*. Check Section[4.3](#S4.SS3 "4.3 Multimodal Knowledge and Commonsense ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 4.4 Scene Text, Table, Chart, and Document Reasoning

Scene text recognition. Reading and understanding scene text in images is an important task in vision-language*[[118](#bib.bib118 ""), [119](#bib.bib119 ""), [120](#bib.bib120 ""), [17](#bib.bib17 "")]*. In our experiments, we investigate GPT-4V’s ability to recognize scene text by utilizing the input prompt “What are all the scene text in the image?” Figure[33](#S4.F33 "Figure 33 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows the example results. We observe GPT-4V accurately identifies scene text in various scenarios, including both handwritten and printed text. In Section[4.5](#S4.SS5 "4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we present further results on multilingual scenarios.

Visual math reasoning. In Figure[34](#S4.F34 "Figure 34 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V demonstrates its capability in solving visual math problems. In our experiments, we observe GPT-4V is able to extract essential information from the image. For instance, in Figure[34](#S4.F34 "Figure 34 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V correctly identifies the presence of a right triangle (or orthogonal triangle) and determines that AB is 4 units and BC is 3 units. In addition, we note that GPT-4V tends to present solutions in a well-structured manner, solving the problem step by step, thereby showcasing its ability to provide clear explanations.

Chart understanding and reasoning. We further study GPT-4V’s ability in chart understanding and reasoning. Figures[35](#S4.F35 "Figure 35 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[37](#S4.F37 "Figure 37 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") show the example results. In our preliminary explorations, GPT-4V exhibits the ability to provide detailed descriptions of charts. For example, in Figure[35](#S4.F35 "Figure 35 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), the model correctly explains the proposal process from the beginning to the end. In Figure[36](#S4.F36 "Figure 36 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), the model not only understands the program in the given flow chat, but also translates the details to a python code. In the bottom row of Figure[37](#S4.F37 "Figure 37 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V shows a clear understanding of both x- and y-axis, and explains the key insight presented in the chart. Furthermore, in our experiments, we observe that GPT-4V can answer questions based on the chart. In the top row of Figure[37](#S4.F37 "Figure 37 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V correctly calculates the average total fueling cost, excluding the Ford F150.

Table understanding and reasoning. In Figure[38](#S4.F38 "Figure 38 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we present our preliminary investigations into table understanding and reasoning. Similar to the findings from chart experiments, GPT-4V shows promising results in understanding the details in the table, as well as in reasoning and accurately responding to related questions.

Document understanding. Figure[39](#S4.F39 "Figure 39 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows the results on various types of documents, such as floor plan, poster, and exam paper. We observe GPT-4V demonstrates an understanding of the documents and provides reasonable responses. For instance, it accurately identifies the location of the bathroom for the second bedroom in the floor plan. It also recognizes the Chinese dish “Hot dry noodles,” and associates it with the city of Wuhan by following the scene text. Moreover, GPT-4V is capable of reading an exam paper. It accurately reconstructs the table in Markdown, and then fills in the table with the correct answers. We present more explorations in its coding ability in Section[4.6](#S4.SS6 "4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

In Figure[40](#S4.F40 "Figure 40 ‣ 4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we explore a more challenging case by providing a technical report*[[126](#bib.bib126 "")]* with multiple pages as input. In our limited exploration, GPT-4V exhibits impressive results. It correctly describes the main idea and their proposed method by considering the context across multiple pages. However, it may occasionally miss some implementation details. Please note that the dataset should contain 1196+665\=1861 examples, and the extracted features should include Histograms of Oriented Gradients (HOG). Instead of prompting all pages to the model simultaneously, we believe that exploring more advanced prompting techniques, such as thinking step-by-step or employing in-context few-shot approaches, could potentially enhance the model’s performance.

<img src='x34.png' alt='Refer to caption' title='' width='553' height='720' />

*Figure 33: Results on scene text recognition. GPT-4V can recognize scene text in many challenging scenarios. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x35.png' alt='Refer to caption' title='' width='461' height='626' />

*Figure 34: Results on visual math reasoning. GPT-4V is able to comprehend and solve visual math problems with a well-structured solution. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x36.png' alt='Refer to caption' title='' width='461' height='627' />

*Figure 35: Results on flow chart understanding. GPT-4V correctly describes the proposal process in details. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x37.png' alt='Refer to caption' title='' width='461' height='627' />

*Figure 36: Results on flow chart understanding. GPT-4V is able to translate the flow chart to a python code. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x38.png' alt='Refer to caption' title='' width='461' height='658' />

*Figure 37: GPT-4V shows promising results in understanding the details in the chart, as well as in reasoning and accurately responding to related questions. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x39.png' alt='Refer to caption' title='' width='461' height='541' />

*Figure 38: We observe GPT-4V can understand the details in the table, and answer related questions. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x40.png' alt='Refer to caption' title='' width='461' height='783' />

*Figure 39: Results on document understanding. GPT-4V recognizes three different types of document and answers the questions correctly. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x41.png' alt='Refer to caption' title='' width='461' height='657' />

*Figure 40: Results on document understanding. GPT-4V reads a multi-page technical report, understands the content in each section, and provides a summary of the contribution of this technical report. Red highlights the wrong answer. Check Section[4.4](#S4.SS4 "4.4 Scene Text, Table, Chart, and Document Reasoning ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 4.5 Multilingual Multimodal Understanding

We assess GPT-4V’s ability in comprehending multiple languages and modalities.
First, we explore this capability by evaluating natural images without scene text, as depicted in Figure[41](#S4.F41 "Figure 41 ‣ 4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). In the first row of the figure, we provide the input text prompt “Describe the image” in Chinese, French, and Czech, respectively. GPT-4V recognizes the input text prompts in different languages, and generates correct image descriptions in corresponding languages. In the second row of Figure[41](#S4.F41 "Figure 41 ‣ 4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we provide the input text prompt in English and specify the output language. GPT-4V follows the instruction and generates correct descriptions in the desired languages. In the bottom row of Figure[41](#S4.F41 "Figure 41 ‣ 4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we provide an input prompt in Spanish, and ask GPT-4V to generate image descriptions in 20 different languages. We observe that GPT-4V can process both the input and output text in different languages.

Furthermore, we explore a scenario involving multilingual scene text recognition, where the input image may contain scene text in various languages. As shown in Figure[42](#S4.F42 "Figure 42 ‣ 4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V correctly identifies and understands the scene text from different scenes.
As shown in the first two rows of Figure[43](#S4.F43 "Figure 43 ‣ 4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we observe that GPT-4V can recognize the scene text, and translate it to a different language. In the bottom row of Figure[43](#S4.F43 "Figure 43 ‣ 4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we provide a screenshot of a Wikipedia website which is in Catalan, and instruct GPT-4V to summarize the information in 20 different languages. GPT-4V not only recognizes the text in Catalan but also generates precise summaries and translates them into different languages. This showcases GPT-4V’s ability to comprehend and translate multilingual scene text.

We also explore the capability of multicultural understanding*[[147](#bib.bib147 ""), [77](#bib.bib77 "")]*. Figure[44](#S4.F44 "Figure 44 ‣ 4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows the example results in such a scenario. We observe that GPT-4V is capable of understanding cultural nuances and generating reasonable multilingual descriptions for the wedding images given.

In our exploration, we found that GPT-4V seamlessly comprehends and correctly generates descriptions in different languages, highlighting its versatility in handling diverse linguistic contexts.

<img src='x42.png' alt='Refer to caption' title='' width='553' height='763' />

*Figure 41: Results on multilingual image descriptions. GPT-4V is able to generate image descriptions in different languages. Check Section[4.5](#S4.SS5 "4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x43.png' alt='Refer to caption' title='' width='553' height='763' />

*Figure 42: Results on multilingual scene text recognition. GPT-4V can recognize scene text in different languages. Check Section[4.5](#S4.SS5 "4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x44.png' alt='Refer to caption' title='' width='553' height='763' />

*Figure 43: Results on multilingual text recognition, translation, and description. GPT-4V is able to recognize, translate and generate descriptions in different languages. Check Section[4.5](#S4.SS5 "4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x45.png' alt='Refer to caption' title='' width='461' height='783' />

*Figure 44: Results on multilingual multiculture understanding. Check Section[4.5](#S4.SS5 "4.5 Multilingual Multimodal Understanding ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 4.6 Coding Capability with Vision

<img src='x46.png' alt='Refer to caption' title='' width='461' height='360' />

*Figure 45:  GPT-4V’s capability to generate LaTex codes based on the hand-written input.
The instruction is ‘generate latex code.’ for each case. The output is the LaTeX code and we show the rendered result.
Although the model fails to write the code for the complex equation (bottom),
we can break it down into several simple equations, which GPT-4V is able to handle. Check Section[4.6](#S4.SS6 "4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

Figure[45](#S4.F45 "Figure 45 ‣ 4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates the ability to generate LaTeX code based on handwritten mathematical equations. This functionality can assist users in writing equations in LaTeX more efficiently.
Although the model is unable to generate code for longer equations, it can handle shorter equations effectively.
By breaking down longer equations into shorter components, the model is able to generate the appropriate code. Figure[46](#S4.F46 "Figure 46 ‣ 4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") further demonstrates how GPT-4V can reconstruct a table in the input image into MarkDown/LaTex code.

Figure[47](#S4.F47 "Figure 47 ‣ 4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows examples of writing code in Python, TikZ, and SVG to replicate the input figure.
Although the resulting output is not an exact match,
the layout is similar and the code can be easily modified to meet specific needs.

<img src='x47.png' alt='Refer to caption' title='' width='461' height='738' />

*Figure 46:  GPT-4V’s capability to generate Markdown/LaTex codes to reconstruct a table in the image. Red highlights the errors in reconstruction. Check Section[4.6](#S4.SS6 "4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x48.png' alt='Refer to caption' title='' width='461' height='793' />

*Figure 47:  GPT-4V’s capability to write codes to replicate the input figure. We directly show the rendered figures by python/TikZ/SVG as GPT-4V’s response.
The rendered figure is roughly aligned with the input figure, and the code can be easily adapted.
GPT-4V Chart. Check Section[4.6](#S4.SS6 "4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

5 Interaction with Humans: Visual Referring Prompting
-----------------------------------------------------

Pointing to a specific spatial location is an essential capability in human-computer interaction with multimodal systems, such as conducting visually grounded dialogues. As shown in Section[5.1](#S5.SS1 "5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V can well understand the visual pointers directly drawn on images. Based on this observation, we propose a novel model interaction method named “visual referring prompting.” The core idea is to directly edit image pixel space to draw visual pointers or scene texts as human referring instructions, as highlighted in Figure[50](#S5.F50 "Figure 50 ‣ 5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). We detail its usages and advantages in Section[5.2](#S5.SS2 "5.2 Visual Referring Prompting ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). Finally, Section[5.3](#S5.SS3 "5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") explores having GPT-4V generate visual pointer outputs to interact with humans. These visual pointers are intuitive for both humans and machines to generate and understand, making them a good channel for human-computer interaction.

### 5.1 Understand Pointing Inputs

As illustrated in Figure[48](#S5.F48 "Figure 48 ‣ 5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V can understand different types of visual markers directly overlaid on images as a pointer, such as circles, boxes, and hand drawings. This ability helps GPT-4V generate grounded captioning, which is a known challenging problem to have conventional vision-language models*[[128](#bib.bib128 "")]* generating visual descriptions focused on a specific area of interest. Dense captioning methods*[[62](#bib.bib62 ""), [138](#bib.bib138 "")]* use cropped boxes or mask regions to generate localized descriptions, but often ignore the global image context and produce sub-optimal descriptions. Visual pointing provides a natural way to indicate the area of interest while maintaining the global image context. For example, the top left example focuses on providing a comprehensive description of the pointed Magna beer, while also mentioning the global image context that the beer bottle is on the table.

An intuitive alternative to visual pointers overlaid on images is the region coordinates represented in the numerical text format. As shown in Figure[49](#S5.F49 "Figure 49 ‣ 5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V can understand the coordinates out-of-box, achieving the ability of spatial referring via text tokens without extra box token finetuning as in prior vision-language models*[[129](#bib.bib129 ""), [143](#bib.bib143 "")]*. Despite the promising capability, we note that our current prompt is less precise spatially. For example, in the top left example in Figure[49](#S5.F49 "Figure 49 ‣ 5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V mentions the surrounding objects napkin and water bottle, even though only the beer bottle is in the region (0.47, 0.48, 0.55, 0.87). Overall, with respect to our experimented prompts, GPT-4V works more reliably when prompted with overlaid visual pointers, compared with text coordinates. This unique capability motivates us to explore a new prompting method, namely visual referring prompting.

<img src='x49.png' alt='Refer to caption' title='' width='461' height='702' />

*Figure 48: GPT-4V understands visual pointers directly overlaid on images. Conducting grounded description with both local and global visual information is one unique application scenario. Check Section[5.1](#S5.SS1 "5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x50.png' alt='Refer to caption' title='' width='461' height='727' />

*Figure 49: An alternative to visual pointers overlaid on images is the region coordinates represented in the numerical text format. GPT-4V can understand the coordinates, *e.g*., (0.47, 0.48, 0.55, 0.87), (0.01, 0.09, 0.29, 0.21), and (0.01, 0.67, 0.36, 0.91) that correspond to the center beer bottle, top-left string lights, and bottom-left table set, respectively. We observe that GPT-4V works less reliably when prompted with text coordinates, compared with visual pointers in visual referring prompting. Check Section[5.1](#S5.SS1 "5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 5.2 Visual Referring Prompting

Inspired by GPT-4V’s strong capability in understanding visual pointing and scene text, we explore a new method to interact with GPT-4V, namely the visual referring prompting. Instead of conventional prompting techniques that edit text space, visual referring prompting is a complementary technique that directly edits the pixel space for input images for human-computer interaction. Such visual prompting could offer a more nuanced and comprehensive interaction with the image, potentially unlocking a wider array of responses from the model. For example, in Figure[50](#S5.F50 "Figure 50 ‣ 5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") (1), GPT-4V naturally associates the arrow-pointed objects with the given object indexes, easing the remaining visual reasoning and text outputs; in (2), GPT-4V understands the questions written on the image and pointed to the corresponding edge or angle, providing a nuanced interface for grounded visual dialogue; in (3), humans can point to arbitrary regions inside the figure to help GPT-4V better understand complicated documents and charts; in (4), the pattern can be concisely represented as an arrow and the scene text “+dot”, therefore helping GPT-4V to predict the next image. Complementary to text prompts that are loosely grounded to images, visual referring prompting provides a novel interaction method that could facilitate various use cases, with additional demonstrations in Figure[51](#S5.F51 "Figure 51 ‣ 5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") and Section[9](#S9 "9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

### 5.3 Generate Pointing Outputs

Section[5.1](#S5.SS1 "5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") discusses the ability of GPT-4V to understand visual pointing generated by humans. A natural question is: Can GPT-4V generate its own pointing outputs, thereby facilitating a closed-loop interaction process in human-computer interaction?

Figure[52](#S5.F52 "Figure 52 ‣ 5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") explores generating visual pointing outputs by letting GPT-4V predict region coordinates in the text format. We prompt GPT-4V to ground the object referred by text (*e.g*., the text of “blue Subaru SUV”) or a reference image (*e.g*., the image of “black Audi sedan”). Similar to the observation in having GPT-4V comprehend coordinates input, the model has a coarse understanding of spatial locations, but it wasn’t accurate with respect to the prompts used in the experiment. For example, in Figure[52](#S5.F52 "Figure 52 ‣ 5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")’s “plot visualizations,” GPT-4V can approximately identify the blue SUV and black sedan mentioned in the query, but it struggles to create a closely-fitted bounding box. We observe that including example-grounded instructions in the prompt helps GPT-4V to understand the definition of coordinates and subsequently generate better pointing outputs.

While the generated pointing outputs may not perfectly cover the queried region, they still provide a valuable tool for model interaction, interpretation, and helping multi-step visual reasoning. Specifically, the pointing outputs can be interpreted by humans to better understand GPT-4V’s references, or by GPT-4V itself to enable further reasoning based on previous outputs.
As shown in the bottom of Figure[52](#S5.F52 "Figure 52 ‣ 5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V is capable of interpreting the pointers it generates, and providing grounded descriptions with the prompts in Figure[48](#S5.F48 "Figure 48 ‣ 5.1 Understand Pointing Inputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). Furthermore, the iterative pointing generation and understanding by GPT-4V itself can help complicated multi-hop grounded visual reasoning tasks. GPT-4V is capable of deconstructing the question, generating distinct visual markers to iteratively focus on different image regions for each sub-step, ultimately collating the information to formulate the final answer.

<img src='x51.png' alt='Refer to caption' title='' width='461' height='666' />

*Figure 50: Visual referring prompting directly edits the input image as input prompts, such as drawing visual pointers and scene texts. Complementary to text prompts, visual referring prompting provides a more nuanced and natural interaction, *e.g*., (1) associating pointed objects with an index, (2) pointing to the image for questioning, (3) highlighting lines in documents and tables, (4) drawing the pattern on the image, and many other novel use cases. Check Section[5.2](#S5.SS2 "5.2 Visual Referring Prompting ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x52.png' alt='Refer to caption' title='' width='461' height='665' />

*Figure 51: Visual referring prompts enhance the seamless interaction between humans and computers. This is evident in the integration with computer and mobile Graphical User Interfaces (GUIs), and the support provided in understanding documents and slides. Check Section[5.2](#S5.SS2 "5.2 Visual Referring Prompting ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x53.png' alt='Refer to caption' title='' width='461' height='745' />

*Figure 52: GPT-4V can use its understanding of coordinates to generate visual pointing output, thereby grounding the textual or visually queried object. Using example-grounded instructions can help GPT-4V understand coordinate definitions and therefore generate better pointing. While output spatial regions are not precise, the approach enables an “understanding (*i.e*., grounded description) and generation” loop for visual pointing, leading to an effective way of human-computer interaction. Check Section[5.3](#S5.SS3 "5.3 Generate Pointing Outputs ‣ 5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

6 Temporal and Video Understanding
----------------------------------

In this section, we discuss temporal and video understanding capabilities. Even though GPT-4V operates primarily on images as inputs, evaluating its understanding of temporal sequences and video content remains a crucial aspect of its overall assessment. This is because real-world events unfold over time, and an AI system’s ability to understand these dynamic processes is instrumental in real-world applications. Capabilities like temporal anticipation, temporal ordering, temporal localization, temporal reasoning, and grounded temporal understanding help to gauge the model’s proficiency in comprehending the sequence of events, anticipating future occurrences, and contextually analyzing activities over time, all within a series of static images. In spite of its image-centric focus, GPT-4V is able to comprehend video and temporal sequences in a way that’s similar to human comprehension. To enhance the versatility and applicability of a sophisticated AI model like GPT-4V, this aspect of testing is critical to its development and refinement. For the upcoming experiments in this section, we will use multiple selected video frames as inputs to test the model’s abilities in understanding temporal sequences and video content.

### 6.1 Multi-image Sequencing

In this subsection, we demonstrate that GPT-4V can accurately comprehend and analyze sequences of video frames. Within this frame-by-frame analysis, GPT-4V recognizes the scene in which the activity is taking place, delivering a deeper contextual understanding. As shown in Figure [53](#S6.F53 "Figure 53 ‣ 6.1 Multi-image Sequencing ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), the model is not just confined to recognizing the environment; it also accurately interprets the actions being performed by individuals in the video. GPT-4V understands the sequence and context of various human poses and intelligently correlates them with the ongoing activity. By understanding pose variations beyond just identification, GPT-4V can derive meaning from the subtleties of human movement and action. As a result of this level of detailed understanding, GPT-4V can capture the essence of what’s happening in videos, offering rich and nuanced insights that go beyond just identifying objects and scenes.

<img src='x54.png' alt='Refer to caption' title='' width='461' height='510' />

*Figure 53: Sequences of video frames understanding: Interpreting human poses and deriving relevant insights from video sequences. Check Section[6.1](#S6.SS1 "6.1 Multi-image Sequencing ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 6.2 Video Understanding

#### Temporal ordering.

Temporal ordering is a crucial element of temporal commonsense and forms an essential part of GPT-4V’s capabilities evaluation. This involves providing the model with a series of shuffled images and gauging its ability to discern cause and effect relationships as well as time progressions. An understanding of such relationships requires the ability to reorder the sequence in a logically coherent and temporally accurate manner. Figure [54](#S6.F54 "Figure 54 ‣ Temporal ordering. ‣ 6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates an example of long-term temporal ordering where GPT-4V is presented with a series of shuffled image frames depicting a sushi-making event. Despite the disorder, GPT-4V effectively identifies the event and determines the appropriate temporal sequence of the sushi-making process. In addition, Figure [55](#S6.F55 "Figure 55 ‣ Temporal ordering. ‣ 6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") provides an example of short-term temporal ordering. Given a designated action, such as opening or closing a door, GPT-4V demonstrates its capability to comprehend the image’s content and determine the correct sequential order of the events. These examples highlight GPT-4V’s capability in temporal commonsense, reinforcing its ability to comprehend both long-term and short-term sequences accurately.

<img src='x55.png' alt='Refer to caption' title='' width='461' height='422' />

*Figure 54: Long-term temporal ordering: GPT-4V is presented with shuffled image frames depicting a sushi-making event. While the sushi-making process is disordered, GPT-4V is able to identify the event and determine the correct temporal sequence. Check Section[6.2](#S6.SS2 "6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x56.png' alt='Refer to caption' title='' width='461' height='535' />

*Figure 55: Short-term temporal ordering: given a specified action, such as opening or closing a door, GPT-4V demonstrates its capability to comprehend the images’ content and determine the correct sequential order corresponding to the specified action. Check Section[6.2](#S6.SS2 "6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

#### Temporal anticipation.

We demonstrate GPT-4V’s ability to anticipate future events given a set of initial frames. Long- and short-term examples are used to validate this capacity for anticipating future events. The right side of Figure [56](#S6.F56 "Figure 56 ‣ Temporal anticipation. ‣ 6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates GPT-4V’s ability to anticipate short-term events with a soccer penalty kick example. Given the first few frames, it accurately foresees the typical next actions of both the kicker and the goalkeeper, due to its understanding of the inherent structure and rules of the game. In addition, as shown in The left side of Figure [56](#S6.F56 "Figure 56 ‣ Temporal anticipation. ‣ 6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), the sushi preparation sequence illustrates GPT-4V’s long-term anticipation capability. By understanding the activity based on visual cues, GPT-4V not only recognizes the current progress in sushi preparation but also accurately anticipates the subsequent steps, demonstrating its capacity to interpret and predict complex, multi-step processes over an extended period. This combination of short-term and long-term temporal anticipation allows GPT-4V to capture and understand activities with varying temporal structures and complexities.

<img src='x57.png' alt='Refer to caption' title='' width='461' height='554' />

*Figure 56: Short-term and long-term temporal anticipation: GPT-4V captures and understands activities with varying temporal structures and complexities. Check Section[6.2](#S6.SS2 "6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

#### Temporal localization and reasoning.

Figure [57](#S6.F57 "Figure 57 ‣ Temporal localization and reasoning. ‣ 6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates GPT-4V’s capabilities in temporal localization and reasoning. It accurately identifies the precise moment the player strikes the ball. Furthermore, GPT-4V showcases its understanding of cause and effect by inferring from the relationship between the goalkeeper and the ball to determine if the goalkeeper successfully blocks the ball.
In the context of the example given, understanding whether the goalkeeper can block the ball involves not only recognizing the spatial positions of the goalkeeper and the ball but also understanding the dynamics of their interaction and predicting the outcome of these dynamics. This demonstrates a considerable level of sophistication in the model’s reasoning abilities.

<img src='x58.png' alt='Refer to caption' title='' width='461' height='513' />

*Figure 57: Temporal localization and reasoning: GPT-4V shows the capability in temporal localization by accurately identifying when the player strikes the ball. It also demonstrates cause-and-effect reasoning by determining whether the ball was blocked based on the goalkeeper-ball interaction. Check Section[6.2](#S6.SS2 "6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x59.png' alt='Refer to caption' title='' width='461' height='555' />

*Figure 58: Grounded temporal understanding: GPT-4V can apply a temporal understanding to a specific person of interest, indicated by a circle. Check Section[6.3](#S6.SS3 "6.3 Visual Referring Prompting for Grounded Temporal Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 6.3 Visual Referring Prompting for Grounded Temporal Understanding

Section [5](#S5 "5 Interaction with Humans: Visual Referring Prompting ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates GPT-4V’s capabilities in visual referring prompting. In this section, we aim to extend this capability by testing visual referring prompting for temporal understanding. This advancement offers enhanced control over video comprehension tasks.

Grounded temporal understanding. Grounded temporal understanding forms another crucial aspect of GPT-4V’s capabilities, which we explore using pointing input in a sequence of image frames. Figure[58](#S6.F58 "Figure 58 ‣ Temporal localization and reasoning. ‣ 6.2 Video Understanding ‣ 6 Temporal and Video Understanding ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") exemplifies this by demonstrating how GPT-4V can apply a temporal understanding to a specific person of interest, indicated by a circle. GPT-4V can accurately describe events in a way that aligns with the corresponding temporal order, focusing on the activities of the circled individual. Beyond this, GPT-4V demonstrates a more refined understanding of the event, recognizing the nature of the interactions. For instance, GPT-4V can distinguish between friendly interactions and violent incidents, illustrating an ability to not only comprehend the temporal flow of events but also to interpret the tone and nature of the interactions taking place. This indicates GPT-4V’s capacity to process and comprehend complex temporal and social cues within a given sequence, adding a layer of depth to its understanding.

7 Abstract Visual Reasoning and Intelligence Quotient Test
----------------------------------------------------------

<img src='x60.png' alt='Refer to caption' title='' width='461' height='777' />

*Figure 59: Understanding abstract visual stimuli such as tangram*[[59](#bib.bib59 "")]* and ASCII text art. Check Section[7.1](#S7.SS1 "7.1 Abstract Visual Stimuli ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x61.png' alt='Refer to caption' title='' width='461' height='782' />

*Figure 60: Understanding part-object association in abstract and natural images. Check Section[7.2](#S7.SS2 "7.2 Discovery and Association of Parts and Objects ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

Understanding and reasoning over abstract visual stimuli and symbols is one fundamental ability for human intelligence. This section examines if GPT-4V can abstract semantics from visual signals and can perform different types of human Intelligence Quotient (IQ) tests.

### 7.1 Abstract Visual Stimuli

Humans can infer semantics from abstract and often ambiguous visual stimuli. Figure[59](#S7.F59 "Figure 59 ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") explores having GPT-4V interpret tangram*[[31](#bib.bib31 ""), [97](#bib.bib97 ""), [43](#bib.bib43 ""), [59](#bib.bib59 "")]*. A tangram is a traditional geometric puzzle that consists of seven flat pieces called tans, which are put together to form shapes without overlapping the pieces. For example, GPT-4V interprets that sub-figure 7 in Figure[59](#S7.F59 "Figure 59 ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") best illustrates a flying goose and provides reasoning descriptions for other sub-figure, *e.g*., 4. person or robot, 9. boat or hat, and 10. dog or fox. GPT-4V also has the ability to understand other formats of abstract visual diagrams*[[127](#bib.bib127 ""), [16](#bib.bib16 ""), [150](#bib.bib150 "")]*, such as ASCII text art of cartoon characters in Figure[59](#S7.F59 "Figure 59 ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") and symbolic inputs in Figures[61](#S7.F61 "Figure 61 ‣ 7.4 Raven’s Progressive Matrices ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[62](#S7.F62 "Figure 62 ‣ 7.4 Raven’s Progressive Matrices ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

### 7.2 Discovery and Association of Parts and Objects

Discovering and associating object parts*[[139](#bib.bib139 ""), [44](#bib.bib44 "")]* is another important abstract visual reasoning capability. Humans can easily discover how object parts may compose a semantically meaningful object. Figure[60](#S7.F60 "Figure 60 ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") designs examples to probe GPT-4V’s capability in associating object parts. In the left example, we ask GPT-4V to localize an object part based on its semantic meaning. In the right example, GPT-4V is asked to associate object parts segmented by SAM*[[65](#bib.bib65 "")]*. GPT-4V can process figures for all object parts and associate them in a semantically meaningful to form the boy visualized in the bottom right.

### 7.3 Wechsler Adult Intelligence Scale

Section[7.1](#S7.SS1 "7.1 Abstract Visual Stimuli ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") demonstrates the abstract visual understanding capability of GPT-4V. As a further challenge, GPT-4V is asked to perform different abstract reasoning tasks, sourced from human Intelligence Quotient (IQ) tests.
The Wechsler Adult Intelligence Scale*[[133](#bib.bib133 "")]* is recognized as one of the “gold standard IQ tests,” and is designed to provide a comprehensive measurement of an individual’s cognitive abilities using a series of sub-tests. Figure[61](#S7.F61 "Figure 61 ‣ 7.4 Raven’s Progressive Matrices ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows representative questions and GPT-4V’s outputs from each sub-test category. GPT-4V shows promises in abstract reasoning, answering questions with texts only, symbolic visual inputs, and natural images. For example, the bottom right sample shows that GPT-4V can interpret the analogy question and find the best comparison of shoes.

### 7.4 Raven’s Progressive Matrices

Raven’s Progressive Matrices (RPM)*[[109](#bib.bib109 "")]* is another well-known non-verbal intelligence test developed to measure abstract reasoning and problem-solving abilities. The test is designed to minimize the influence of language, culture, and formal education on test performance, making it suitable for testing AI models*[[16](#bib.bib16 ""), [150](#bib.bib150 ""), [55](#bib.bib55 "")]*. Each test sample contains three or eight images, arranged in 2-by-2 or 3-by-3 matrices with one figure missing. The goal is to select the next image from multiple candidate images by identifying patterns in the provided samples. In our approach, we challenge GPT-4V by sending the entire question page as a single image, instead of converting it into interleaved image-text pairs, similar to the human approach to IQ tests. As shown in Figure[62](#S7.F62 "Figure 62 ‣ 7.4 Raven’s Progressive Matrices ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V can generate reasonable answers without processed text descriptions or sub-figures. However, we also notice that breaking down the entire question image into interleaved text and sub-figures, such as in Figure[63](#S7.F63 "Figure 63 ‣ 7.4 Raven’s Progressive Matrices ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), does simplify the task and let GPT-4V produce more reliable answers.

<img src='x62.png' alt='Refer to caption' title='' width='461' height='815' />

*Figure 61: Example questions from the Wechsler Adult Intelligence Scale (WAIS)*[[133](#bib.bib133 "")]*. Check Section[7.3](#S7.SS3 "7.3 Wechsler Adult Intelligence Scale ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x63.png' alt='Refer to caption' title='' width='461' height='673' />

*Figure 62: Example questions from the Raven’s Progressive Matrices*[[109](#bib.bib109 ""), [55](#bib.bib55 "")]*. We challenge GPT-4V by sending the entire question page as a single image, mimicking how humans look at the IQ tests. Check Section[7.4](#S7.SS4 "7.4 Raven’s Progressive Matrices ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x64.png' alt='Refer to caption' title='' width='461' height='698' />

*Figure 63: Instead of sending the entire question page as a single image, we may also process the image into multiple sub-figures and optionally provide detailed instructions and examples to further boost the answer accuracy. Check Section[7.4](#S7.SS4 "7.4 Raven’s Progressive Matrices ‣ 7 Abstract Visual Reasoning and Intelligence Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

8 Emotional Quotient Test
-------------------------

<img src='x65.png' alt='Refer to caption' title='' width='461' height='501' />

*Figure 64: GPT-4V can reliably identify and read the emotions of people from their facial expressions. Check Section[8.1](#S8.SS1 "8.1 Read Emotion from Facial Expressions ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x66.png' alt='Refer to caption' title='' width='461' height='612' />

*Figure 65: GPT-4V understands how different visual contents may arouse human emotions. Check Section[8.2](#S8.SS2 "8.2 Understand How Visual Content Arouses Emotions ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x67.png' alt='Refer to caption' title='' width='461' height='539' />

*Figure 66: GPT-4V judges image aesthetics based on societal standards and norms. Check Section[8.2](#S8.SS2 "8.2 Understand How Visual Content Arouses Emotions ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x68.png' alt='Refer to caption' title='' width='461' height='567' />

*Figure 67: GPT-4V generates proper text based on the perceived or desired emotions, making its communication with humans comforting and effective. Check Section[8.3](#S8.SS3 "8.3 Emotion Conditioned Output ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

When interacting with humans, it is important that GPT-4V has the empathy and Emotional Quotient (EQ) to understand and share the feelings of humans. Inspired by the definition of the human EQ test*[[92](#bib.bib92 ""), [91](#bib.bib91 ""), [21](#bib.bib21 "")]*, we examine GPT-4V’s capability in (1) identifying and reading human emotions from their facial expressions, (2) understanding how different visual contents may arouse emotions, and (3) generating proper text outputs conditioned on the desired emotional and sentiment.

### 8.1 Read Emotion from Facial Expressions

As shown in Figure[64](#S8.F64 "Figure 64 ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V can reliably identify and read the emotions of people from their facial expressions. It also provides reasonable rationales for the visual cues observed to make the emotion interpretation, indicating a good understanding of the facial emotions.

### 8.2 Understand How Visual Content Arouses Emotions

We next analyze GPT-4V’s ability on visual sentiment analysis, *i.e*., understanding humans’ emotional response after seeing the visual contents. Such ability is critical for GPT-4V to anticipate how visual contents may arouse human emotions and thereby react properly. As shown in Figure[65](#S8.F65 "Figure 65 ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V can interpret visual sentiments such as content, anger, awe, and fear, based on both the semantic contents and the image style. These capabilities are essential in use cases such as home robots.

In addition to interpreting visual sentiment, GPT-4V also aligns with human subjective judgments such as aesthetics. Figure[66](#S8.F66 "Figure 66 ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows examples of GPT-4V judging image aesthetics based on societal standards.

### 8.3 Emotion Conditioned Output

Based on the perceived emotions, GPT-4V effectively generates proper text outputs conditioned on the desired emotion. For example, in Figure[67](#S8.F67 "Figure 67 ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V can follow the prompt to describe the right-side scary image in a way that makes it more horrifying or becoming comforting. This demonstrates GPT-4V’s potential to enable emotion-aware human-robot communication.

9 Emerging Application Highlights
---------------------------------

In this section, we showcase a myriad of high-value application scenarios and new use cases that can be potentially enabled by the remarkable capabilities of GPT-4V. While it is true that some of these application scenarios can be accomplished by meticulously curating the training data for finetuning existing Vision and Language (VL) models, we want to emphasize that the true power of GPT-4V lies in its ability to perform effortlessly right out of the box. Moreover, we present how GPT-4V seamlessly integrates with external tools and plugins, further expanding its potential and enabling even more innovative and collaborative applications.

### 9.1 Spot the Difference

We begin with a generic use case inspired by the brain-teasing game “Spot the Difference.” In Figures[68](#S9.F68 "Figure 68 ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[69](#S9.F69 "Figure 69 ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we provide GPT-4V two visually similar images that contain subtle differences in certain regions. The task given to GPT-4V is to identify all the differences between the two images. Among the four examples, GPT-4V successfully identifies the regions or components that differ in the images. However, it falls short in providing accurate explanations for what is depicted in each image. To delve deeper into GPT-4V’s capabilities, let’s focus on the first example shown in Figure[68](#S9.F68 "Figure 68 ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").
Although GPT-4V fails to recognize that the discrepancy lies in the number of cuts in the hairband rather than the shade of the hair,
it correctly identifies that the crown, the bow of the dress, and the hair differ between the two images.
While GPT-4V’s predictions in the “Spot the Difference” game are not perfect, its ability to compare the content in two images proves valuable in real-life applications, such as defect detection, which we will explore in the following subsections.

### 9.2 Industry

<img src='x69.png' alt='Refer to caption' title='' width='461' height='773' />

*Figure 68: Spot the differences. Red highlights the inaccurate description about the differences. Check Section[9.1](#S9.SS1 "9.1 Spot the Difference ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x70.png' alt='Refer to caption' title='' width='461' height='773' />

*Figure 69: Spot the differences. Red highlights the inaccurate description about the differences. Check Section[9.1](#S9.SS1 "9.1 Spot the Difference ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

#### Defect detection.

Throughout the history of manufacturing, computer vision techniques have played a crucial role. One specific application scenario is defect detection, which is an essential step in manufacturing processes to ensure product quality. Detecting faults or defects in a timely manner and taking appropriate actions are vital for minimizing operational and quality-related costs.

In this scenario, we demonstrate the defect detection capabilities of GPT-4V by presenting images of defective products in Figures[70](#S9.F70 "Figure 70 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[71](#S9.F71 "Figure 71 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). For commonly encountered products in real-life (e.g., hazelnut, fabric, screw, and car bumper in Figure[70](#S9.F70 "Figure 70 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), GPT-4V confidently identifies the defects such as small holes in the hazelnut/fabric, stripped heads of screws, and dents in car bumpers. However, when it comes to uncommon product images (e.g., the metal parts in Figures[70](#S9.F70 "Figure 70 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[71](#S9.F71 "Figure 71 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")) or products with variations in appearance (e.g., the pill in Figure[71](#S9.F71 "Figure 71 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), GPT-4V may hesitate or even refuse to make predictions. An interesting case in Figure[71](#S9.F71 "Figure 71 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") involves a car tire, where multiple defects can be observed in the image, including dirt on the wheel, damage to the outer edge of the rim, and signs of wear on the tire. GPT-4V only focuses on the minor defect (dirt on the wheel) and fails to mention the major defect (damage to the outer edge of the rim) that would require repair.

Given the success of GPT-4V in “Spot the Difference” scenario shown in Section[9.1](#S9.SS1 "9.1 Spot the Difference ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we explore the idea of incorporating a reference image to illustrate what a defect-free product should look like, with the aim of improving the failure cases depicted in Figure[71](#S9.F71 "Figure 71 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). The results of this approach are presented in Figure[72](#S9.F72 "Figure 72 ‣ Defect detection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). By including the reference image and refining the prompt, GPT-4V successfully identifies defects in all three failure cases in single-image defect detection. These promising findings highlight a potential high-value application of GPT-4V for defect detection in the manufacturing industry.

<img src='x71.png' alt='Refer to caption' title='' width='461' height='729' />

*Figure 70: Defect detection with a single image. Yellow highlights the cases when GPT-4V is hesitating to make the predictions. Check Section[9.2](#S9.SS2 "9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x72.png' alt='Refer to caption' title='' width='461' height='582' />

*Figure 71: Failure examples of defect detection with a single image.Red highlights the cases when GPT-4V fails. Check Section[9.2](#S9.SS2 "9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x73.png' alt='Refer to caption' title='' width='461' height='805' />

*Figure 72: Defect detection with the help of a reference image.Red highlights inaccurate descriptions. Check Section[9.2](#S9.SS2 "9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

#### Safety inspection.

Figure[73](#S9.F73 "Figure 73 ‣ Safety inspection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") presents an exploration of Personal Protective Equipment (PPE) counting for safety inspection. The inadequate usage or failure to wear PPE, such as helmets, harnesses, and gloves, in work environments like construction sites, significantly increases the risk level associated with work activities. To effectively address this issue, computer vision techniques have been employed as a solution to monitor PPE compliance and promptly identify any violations of safety regulations. Taking helmets as an example, a safety inspection system is necessary to accurately detect and report the number of employees who are not wearing helmets.

In Figure[73a](#S9.F73.sf1 "In Figure 73 ‣ Safety inspection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we assess the performance of GPT-4V by directly instructing it to count the individuals wearing helmets. GPT-4V provides a response of “8 persons wearing helmets,” which matches the total count of people shown in the image, suggesting there is no alerting safety violations. Obviously, GPT-4V fails to detect the 3 individuals who are not wearing helmets, thus compromising their personal safety. This task poses a considerable challenge for GPT-4V, as it involves detecting people in the image, determining whether they are wearing helmets, and calculating the final count of people who are not wearing the helmets.

In Figure[73b](#S9.F73.sf2 "In Figure 73 ‣ Safety inspection. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), instead of presenting GPT-4V with the original image containing all 8 individuals, we provide cropped regions of the detected persons with an external person detector.
This approach divides the PPE counting workload into two steps: relying on an off-the-shelf person detector for person detection and leveraging GPT-4V’s robust visual reasoning capabilities and its ability to handle interleaved image-text inputs for identifying the safety issues.
As we can see, GPT-4V can correctly count the person who is not wearing the helmet, also demonstrating the benefit of tool use and divide-and-conquer.

<img src='x74.png' alt='Refer to caption' title='' width='428' height='233' />

*(a)*

<img src='x75.png' alt='Refer to caption' title='' width='428' height='570' />

*(b)*

*Figure 73:  Application Highlights on Safety Inspection: Personal Protective Equipment (PPE) Counting. GPT-4V fails with zero-shot prompting in (a), while succeeds with single person crops in (b). Red (Green) highlights the wrong (correct) answer. Check Section[9.2](#S9.SS2 "9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

#### Grocery checkout.

Self-checkout machines have become increasingly popular in major retailers like Walmart, Target and CVS
to expedite the checkout process for customers and reduce the workload for employees. However, the actual experience with self-checkout machines
may be frustrating for customers.
Users still need to search for the product barcode or manually enter codes for fresh items like apples, which can be time-consuming, particularly for those unfamiliar with the system. In Figure[74](#S9.F74 "Figure 74 ‣ Grocery checkout. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we provide a simplified prototype to demonstrate the potential of GPT-4V in enabling an automatic self-checkout system that can identify and ring up items without user intervention.

When presented with a photograph of a shopping basket containing five grocery items, as shown in Figure[74a](#S9.F74.sf1 "In Figure 74 ‣ Grocery checkout. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V fails to accurately identify the products within the basket. It mistakenly identifies strawberries as raspberries, crab dip as Greek yogurt, and includes salmon fillets that are not even present in the basket. However, in Figure[74b](#S9.F74.sf2 "In Figure 74 ‣ Grocery checkout. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we improve the prompt by augmenting it with catalog images of grocery products retrieved from the retail website. As a result, GPT-4V successfully identifies all five items in the basket. This successful demonstration allows the self-checkout system to proceed with retrieving the prices for each identified product from the database. While this is a simple example, it represents a significant step forward toward an automated self-checkout system. Further research and development can explore more complex and realistic scenarios to fully automate the self-checkout process, making it more efficient and convenient for customers.

<img src='x76.png' alt='Refer to caption' title='' width='415' height='225' />

*(a)*

<img src='x77.png' alt='Refer to caption' title='' width='415' height='571' />

*(b)*

*Figure 74:  Application Highlights on Grocery Checkout. GPT-4V fails with zero-shot prompting in (a), while succeeds when prompting with reference product images in (b). Red highlights the products that are not in the basket. Check Sections[9.2](#S9.SS2 "9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), [10.5](#S10.SS5 "10.5 Retrieval-Augmented LMMs ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 9.3 Medical

In Section[4.1](#S4.SS1 "4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), the effectiveness of GPT-4V in medical image understanding is demonstrated through Figures[18](#S4.F18 "Figure 18 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[19](#S4.F19 "Figure 19 ‣ 4.1 Image Description on Diverse Domains ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). Furthermore, we conducted a detailed investigation into the application of GPT-4V in radiology report generation, as depicted in Figures[75](#S9.F75 "Figure 75 ‣ 9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[78](#S9.F78 "Figure 78 ‣ 9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). In this scenario, we provided GPT-4V with various medical images and tasked it with generating complete radiology reports. Since assessing the accuracy of the generated reports requires domain knowledge, we sought the evaluation of a medical professional.

Figure[75](#S9.F75 "Figure 75 ‣ 9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") showcases two accurate examples: one involving an abdominal X-ray image and another featuring an MRI of the right knee. In both cases, GPT-4V correctly identified the study and provided an accurate diagnosis. Moving on to Figure[76](#S9.F76 "Figure 76 ‣ 9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we observe the generated radiology reports by GPT-4V for X-ray images of the hand/wrist. While GPT-4V successfully diagnosed the recommended management based on the first X-ray image, it missed the obvious distal radial fracture present in the second X-ray image. Nevertheless, the generated reports maintain a high-quality format that can serve as a template, thus reducing the workload for medical professionals when drafting reports.

In Figure[77](#S9.F77 "Figure 77 ‣ 9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we explore two additional examples involving a Chest CT and a Brain MRI. In the Chest CT case, GPT-4V mistakenly identified the mentioned nodule on the left side instead of the right side, and it also hallucinated the measurements.
The ability to process interleaved image-text pairs also allows GPT-4V to reference prior medical scans and diagnosis histories, which is shown to be critical in medical professionals’ diagnosing processes*[[15](#bib.bib15 "")]*. Figure[78](#S9.F78 "Figure 78 ‣ 9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")shows examples of understanding the symptom progression from multiple Chest X-Ray scans*[[60](#bib.bib60 ""), [15](#bib.bib15 "")]*.
These illustrations shed light on the potential of GPT-4V to serve as an AI assistant for radiology report generation. However, it is crucial to have the generated reports evaluated by medical professionals to ensure their correctness and accuracy.

<img src='x78.png' alt='Refer to caption' title='' width='530' height='749' />

*Figure 75: Application Highlights on Radiology Report Generation. The generated report is reviewed by a medical professional to evaluate its correctness. Green highlights that a medical professional has confirmed the described part of the report is correct. Check Section[9.3](#S9.SS3 "9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. The medical images are from the internet.*

<img src='x79.png' alt='Refer to caption' title='' width='530' height='792' />

*Figure 76: Application Highlights on Radiology Report Generation. The generated report is reviewed by a medical professional to evaluate its correctness. Green (Red) highlights that a medical professional has confirmed the described part of the report is correct (incorrect). Check Section[9.3](#S9.SS3 "9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. The medical images are from the Internet.*

<img src='x80.png' alt='Refer to caption' title='' width='530' height='762' />

*Figure 77: Application Highlights on Radiology Report Generation. The generated report is reviewed by a medical professional to evaluate its correctness. Green (Red) highlights that a medical professional has confirmed the described part of the report is correct (incorrect). Yellow indicates that the model is hallucinating. Check Section[9.3](#S9.SS3 "9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. The medical images are from the internet.*

<img src='x81.png' alt='Refer to caption' title='' width='530' height='753' />

*Figure 78: Application Highlights on Radiology Report Generation with Diagnosis History. Check Section[9.3](#S9.SS3 "9.3 Medical ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. The medical images are from MIMIC dataset*[[60](#bib.bib60 "")]*.*

### 9.4 Auto Insurance

In this section, we explore another practical application of GPT-4V in the field of auto insurance, focusing specifically on car accident reporting. Within this context, we can further delineate two distinct sub-categories: ($i$) Damage Evaluation and ($ii$) Insurance Reporting. The former involves the crucial task of accurately identifying and assessing the extent of damages sustained by vehicles, while the latter encompasses not only damage identification but also the recognition of vehicle-specific information depicted in images, such as the make, model, license plate, and other relevant details. By addressing both aspects, we aim to demonstrate the comprehensive capabilities of GPT-4V within the auto insurance domain.

#### Damage evaluation.

We present an image depicting car damage to GPT-4V and prompt it with “Imagine that you are an expert in evaluating the car damage from car accident for auto insurance reporting. Please evaluate the damage seen in the image below.” in Figure[79](#S9.F79 "Figure 79 ‣ Insurance reporting. ‣ 9.4 Auto Insurance ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). GPT-4V has demonstrated remarkable proficiency in accurately identifying and precisely localizing the damages depicted in all four images. Furthermore, it impresses with its ability to provide detailed descriptions of each specific damage instance. In some instances, GPT-4V even endeavors to estimate the potential cost of repair.

#### Insurance reporting.

Building on the success in damage evaluation, we modify our prompt to ask GPT-4V to identify the make, model, and license plate of the vehicle depicted in the image, and return the obtained information in JSON format. The examples depicted in Figure[80](#S9.F80 "Figure 80 ‣ Insurance reporting. ‣ 9.4 Auto Insurance ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") showcase this capability. In both instances, GPT-4V attempts to extract all the requested details from the image. However, it should be noted that certain information may be unavailable, such as the estimated cost of repair, or challenging to discern due to occlusion, as observed with the license plate in the second image. It is important to note that real-life insurance reporting typically involves multiple images capturing the car from various angles, a scenario that is usually not publicly accessible on the Internet. Nevertheless, the examples in Figures[79](#S9.F79 "Figure 79 ‣ Insurance reporting. ‣ 9.4 Auto Insurance ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[80](#S9.F80 "Figure 80 ‣ Insurance reporting. ‣ 9.4 Auto Insurance ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") vividly illustrate the potential of GPT-4V in automating the insurance reporting process for car accidents.

<img src='x82.png' alt='Refer to caption' title='' width='461' height='773' />

*Figure 79: Application Highlights on Auto Damage Evaluation. Check Section[9.4](#S9.SS4 "9.4 Auto Insurance ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x83.png' alt='Refer to caption' title='' width='461' height='635' />

*Figure 80: Application Highlights on Insurance Reporting. For the highlighted text in red, GPT-4V fails to read the license plate, potentially due to occlusion. Check Section[9.4](#S9.SS4 "9.4 Auto Insurance ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 9.5 Customized Captioner

#### Photo organization.

In this scenario, let’s picture that we have a family photo album. We demonstrate how GPT-4V can enhance the album by generating captions that explicitly mention the name of each family member shown in the photo. This personalized approach facilitates more precise and tailored photo organization, as illustrated in Figures[81](#S9.F81 "Figure 81 ‣ Dense captioning w/ segmentation. ‣ 9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[82](#S9.F82 "Figure 82 ‣ Dense captioning w/ segmentation. ‣ 9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). By providing GPT-4V with visual prompts for each family member, along with their respective names, GPT-4V can precisely identify the family members (including person, cat, and dog) to generate detailed and customized captions. Storing such captions for all the images in the family album holds the potential to enable highly personalized image search. For instance, a user could search for “a family photo of Linda, Cotton, Max, Sam, and Emma” and easily locate the corresponding family photo shown in Figure[81](#S9.F81 "Figure 81 ‣ Dense captioning w/ segmentation. ‣ 9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), or “Max lying between Sam and Emma” and locate the family photo in Figure[82](#S9.F82 "Figure 82 ‣ Dense captioning w/ segmentation. ‣ 9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").

#### Dense captioning w/ segmentation.

We demonstrate the enhanced performance of GPT-4V in dense captioning by harnessing powerful segmentation models*[[65](#bib.bib65 ""), [159](#bib.bib159 ""), [160](#bib.bib160 "")]*. Figure[83](#S9.F83 "Figure 83 ‣ Dense captioning w/ segmentation. ‣ 9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates the dense captioning results by incorporating object cut-outs generated by SAM*[[65](#bib.bib65 "")]* into the prompt to extract more detailed captions for each object of interest. In addition, we provide GPT-4V with the original image as the global context and ask it to describe four object cut-outs as detailed as possible, and incorporating references to the context image.

The results show GPT-4V can generate highly intricate dense captions for each object, some of which are accompanied by relevant references to the context image. For instance, when describing object 3 (a frog), the dense caption makes mention of a close-up shot of a frog with a snail perched on its head, despite the absence of the snail in the corresponding cut-out for object 3. Similarly, when referring to object 4 (a turtle), GPT-4V recognizes from the context image that the turtle is floating in water, thereby further enriching the generated caption.

<img src='x84.png' alt='Refer to caption' title='' width='461' height='783' />

*Figure 81: Customized Captioner for photo organization (the reference images are cropped from the query image). Blue highlights the mention of family names. Check Sections[9.5](#S9.SS5 "9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), [10.5](#S10.SS5 "10.5 Retrieval-Augmented LMMs ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x85.png' alt='Refer to caption' title='' width='461' height='769' />

*Figure 82: Customized Captioner for photo organization (the reference images are cropped from a different image than the query image) Blue highlights the mention of family names. Check Section[9.5](#S9.SS5 "9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x86.png' alt='Refer to caption' title='' width='461' height='783' />

*Figure 83: Dense captioning w/ segmentation cut-outs from SAM*[[65](#bib.bib65 "")]* Blue highlights the references to the context image. Check Section[9.5](#S9.SS5 "9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 9.6 Image Generation

In this section, we make connections with another prominent area of multimodal research: visual synthesis. By delving into the realm of image generation, we explore how GPT-4V can contribute to this field through various avenues, including evaluation and prompting.

#### Evaluation of generated images.

Figure[66](#S8.F66 "Figure 66 ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") in Section[8.2](#S8.SS2 "8.2 Understand How Visual Content Arouses Emotions ‣ 8 Emotional Quotient Test ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") demonstrates the capability of GPT-4V in assessing the aesthetics of images. Here, we show how we employ GPT-4V to evaluate the generated images based on their alignment with the given prompts for text-to-image generation, inspired by RL-Diffusion*[[18](#bib.bib18 "")]*. RL-Diffusion leverages a VL model LLAVA*[[79](#bib.bib79 "")]* to describe the generated image, followed by text similarity computation between the prompt and the image description using BERT*[[38](#bib.bib38 "")]*. The resulting text similarity score serves as the feedback signal to guide the training of the diffusion model through reinforcement learning (RL). Notably, Figures[84](#S9.F84 "Figure 84 ‣ Evaluation of generated images. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[85](#S9.F85 "Figure 85 ‣ Evaluation of generated images. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") exhibit how GPT-4V, as a single model, can effectively rate the similarity between the generated image and the prompt. Moreover, GPT-4V provides explanations for the deduction in similarity score, which can potentially be used as a feedback to improve the image generation.

In Figure[84](#S9.F84 "Figure 84 ‣ Evaluation of generated images. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we present the evaluation of image similarity using the prompt, “What is happening in the image? From a scale of 1 to 10, rate the similarity between the image and the text prompt ’a parrot driving a car’.” GPT-4V assigns a score of 1 to the most irrelevant image (a dolphin jumping over the water), while rating the most relevant image at the bottom with a score of 9. Notably, the last three images in Figure[84](#S9.F84 "Figure 84 ‣ Evaluation of generated images. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") are shown in RL-Diffusion as gradually improved generation results for the text prompt “a parrot driving a car.” The ratings assigned by GPT-4V to these three images (4 $\rightarrow$ 8 $\rightarrow$ 9) align with the refinement process.

Figure[85](#S9.F85 "Figure 85 ‣ Evaluation of generated images. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") showcases the evaluation of image generation results that involve text rendering on a cake. Leveraging its robust optical character recognition (OCR) capabilities, GPT-4V accurately recognizes the rendered texts in the generated images, such as “Azuze Research,” “ARAUIE,” and “Azure Azure,” and compares them to the text prompt requirement, which is “Azure Research.”

<img src='x87.png' alt='Refer to caption' title='' width='461' height='796' />

*Figure 84: Prompt GPT-4V to give a score from 1 to 10 on how similar the generated image is to the prompt. Blue highlights the rating given by GPT-4V. The last three images are generated from RL-Diffusion*[[18](#bib.bib18 "")]*. Check Section[9.6](#S9.SS6 "9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x88.png' alt='Refer to caption' title='' width='461' height='671' />

*Figure 85: Prompt GPT-4V to give a score from 1 to 10 on how similar the generated image is to the prompt. Blue highlights the rating given by GPT-4V. Red (Green) indicate wrong (correct) rendered text. Generated images are from DeepFloyd IF*[[2](#bib.bib2 "")]*, Midjourney V5.1*[[4](#bib.bib4 "")]*, SDXL*[[110](#bib.bib110 "")]*, and ReCo*[[143](#bib.bib143 "")]*. Check Section[9.6](#S9.SS6 "9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

#### Prompt generation for image editing.

In addition to its remarkable ability to evaluate generated images, GPT-4V offers a valuable feature that can greatly enhance image editing. By generating or rewriting the text prompt used for editing, GPT-4V can refine the edited image, resulting in a more visually appealing outcome. Figure[86](#S9.F86 "Figure 86 ‣ Prompt generation for image editing. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") provides a demonstration of how we can harness the power of GPT-4V to generate a text prompt specifically tailored for image editing. By providing the original image and text requirements that describe the desired edits, GPT-4V produces an optimized prompt for the task at hand. This optimized prompt takes into account the unique characteristics of the image, ensuring that the subsequent editing process is well-informed and effective.

Moreover, Figure[87](#S9.F87 "Figure 87 ‣ Prompt generation for image editing. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") showcases another use case of GPT-4V to improve image editing by rewriting the editing prompt. By considering the original image, the initial prompt, and the edited image, GPT-4V can generate an improved version of the prompt that incorporates the changes made during the previous editing process. One can alternate the processes depicted in Figures[86](#S9.F86 "Figure 86 ‣ Prompt generation for image editing. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[87](#S9.F87 "Figure 87 ‣ Prompt generation for image editing. ‣ 9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), allowing users to refine their edits repeatedly until they achieve a satisfying outcome. Consequently, this iterative process has the potential to significantly enhance the overall quality of the edited image, providing users with more control and creative freedom in their image editing endeavors.

<img src='x89.png' alt='Refer to caption' title='' width='461' height='640' />

*Figure 86: Improving the text prompt for image editing, given the original image and textual requirement. Blue highlights the suggested editing prompt by GPT-4V. Original image/exemplary editing prompt are from Instruct Pix2Pix*[[22](#bib.bib22 "")]*. Check Section[9.6](#S9.SS6 "9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x90.png' alt='Refer to caption' title='' width='461' height='702' />

*Figure 87: Improving the editing prompt, given the original image, the editing prompt, and the edited image. Blue highlights the suggested editing prompt by GPT-4V. Original image/editing prompt/edited image are from Instruct Pix2Pix*[[22](#bib.bib22 "")]*. Check Section[9.6](#S9.SS6 "9.6 Image Generation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 9.7 Embodied Agent

In this section, we delve into the exciting applications and implications of GPT-4V for embodied AI, exploring how it is poised to bridge the gap between multimodal understanding on static inputs and physical interaction with dynamic environments. To provide a concrete illustration, let us consider the scenario of GPT-4V assuming the role of a home robot. Within this context, we witness how it can read the menu to operate household appliances (*e.g*., coffee machine), and perform task-oriented navigation through the house.

#### Operating machine.

Imagine you’ve just acquired a brand-new coffee machine, and to your delight, your trusty home robot, GPT-4V, learns how to operate it on your behalf. In our experiment, we provide GPT-4V with a single image (Figure[88](#S9.F88 "Figure 88 ‣ Operating machine. ‣ 9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")) featuring an operating menu with both illustrations and texts. Our task for GPT-4V is to identify the button that corresponds to the “8 OZ coffee” option within the coffee machine’s operating panel. Surprisingly, GPT-4V not only accurately locates the “8 OZ coffee” button but also successfully recognizes the button for “10 OZ coffee.” However, it mistakenly identifies the power button as the “6 OZ coffee” button, potentially due to the visual confusion caused by the positioning of the “6 OZ coffee” option on both the menu and the coffee machine itself. To address this specific failure case, we devise a solution by isolating the operating menu for each button and presenting them all to GPT-4V in a single prompt (Figure[89](#S9.F89 "Figure 89 ‣ Operating machine. ‣ 9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")). With this revised approach, GPT-4V now can recognize the precise position of the “6 OZ coffee” button.

<img src='x91.png' alt='Refer to caption' title='' width='461' height='671' />

*Figure 88: Reading a full menu of coffee machine buttons, GPT-4V recognizes which button to choose for 8 OZ coffee. Green (Red) highlights the correct (wrong) answer. Check Section[9.7](#S9.SS7 "9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x92.png' alt='Refer to caption' title='' width='461' height='769' />

*Figure 89: Converting the full menu of coffee machine buttons to interleaved image-text instructions, GPT-4V can recognizes which button to choose for 6 OZ coffee, which GPT-4V failed to do so with full menu instruction. Green highlights the correct answer. Check Section[9.7](#S9.SS7 "9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

Navigation. In order to explore navigation capabilities, we utilized Redfin virtual house tour as a means to replicate interactive environments for embodied agents. The objective was to assess the performance of GPT-4V in a task-oriented scenario. To illustrate this, we present an example depicted in Figures[90](#S9.F90 "Figure 90 ‣ Operating machine. ‣ 9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[91](#S9.F91 "Figure 91 ‣ Operating machine. ‣ 9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). Initially, we provided GPT-4V with the entry image of a virtual house tour, offering a view from one corner into the living room. The task assigned to GPT-4V was to “go to the kitchen and retrieve an item from the fridge.” Our aim was to prompt GPT-4V to predict the subsequent actions.

In the first step, as shown in the first half of Figure[90](#S9.F90 "Figure 90 ‣ Operating machine. ‣ 9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V anticipated the initial action by suggesting to “turn right and move forward towards the hallway.” This prediction was based on GPT-4V’s hypothesis that the kitchen would likely be located in that direction. We then manually executed this action using the visual house touring portal, capturing the resulting view after the action was taken. This view was then used to prompt GPT-4V for the next action, as displayed in the second half of Figure[90](#S9.F90 "Figure 90 ‣ Operating machine. ‣ 9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). It’s important to note that throughout the process, we maintained a record of the immediate previous turn to provide context for GPT-4V’s subsequent actions.

As the navigation process unfolded, we successfully reached the fridge within the third turn, as indicated by the query image in the second half of Figure[91](#S9.F91 "Figure 91 ‣ Operating machine. ‣ 9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). The final action predicted by GPT-4V was to “move forward and slightly to the left in order to align myself with the fridge door. Then, use my robotic arm to open the fridge door and retrieve the requested item.” This decisive action marked the accomplishment of GPT-4V in this task-oriented navigation scenario.

<img src='x93.png' alt='Refer to caption' title='' width='461' height='706' />

*Figure 90: Acting as an embodied agent to navigate through a house to fetch something from the fridge (the 1st and 2nd turn). Blue highlights the predicted actions. Check Section[9.7](#S9.SS7 "9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x94.png' alt='Refer to caption' title='' width='461' height='773' />

*Figure 91: Acting as an embodied agent to navigate through a house to fetch something from the fridge (the 3rd and 4th turn). Blue highlights the predicted actions. Check Section[9.7](#S9.SS7 "9.7 Embodied Agent ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 9.8 GUI Navigation

Beyond navigating the physical world, this section showcases the capability of GPT-4V to interact with and navigate through the Graphical User Interface (GUI) of a computer or smartphone. We explore the potential for GPT-4V to complete complex tasks, such as web browsing, online shopping, and *etc*.

#### Web browsing.

We assess the performance of GPT-4V on computer GUI navigation under a task-oriented setting. The model was provided with the screenshot of current computer screen, the end goal of the navigation (*e.g*., finding a cooking recipe or reading today’s news), the list of possible actions (*e.g*., move the mouse, click an icon with the mouse, or type some texts with the keyboard). The model is then instructed to predict the subsequent actions (refer to Figure[92](#S9.F92 "Figure 92 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for a complete prompt example). Upon the model’s prediction, we manually execute the predicted action and capture a screenshot, which served as the input for GPT-4V for the next turn. When the predicted action is to move the mouse, GPT-4V is specifically instructed to detail the mouse’s position. Hence, the predicted actions are grounded, showing the potential of automating the whole process without human in the loop.

In Figures[92](#S9.F92 "Figure 92 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[96](#S9.F96 "Figure 96 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V predicts reasonable actions to operate a computer GUI, and finally accomplish the end goal of finding a recipe of Mapo Tofu and print out a copy of the recipe in Figure[95](#S9.F95 "Figure 95 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). We then provide GPT-4V a screenshot of the printed recipe and ask it to describe the printout as detailed as possible. As shown in Figure[96](#S9.F96 "Figure 96 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), GPT-4V is able to recognize the details presented in the printout, including the cooking time, the list of ingredients, the author of the recipe, the link to the original recipe and *etc*. Figures[97](#S9.F97 "Figure 97 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[102](#S9.F102 "Figure 102 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") present how GPT-4V can navigate through GUI to browse the web to “read today’s news”. Despite the minor errors in Figure[100](#S9.F100 "Figure 100 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") when it tries to return to the previous search result page to continue browsing for more news articles, GPT-4V can perform the navigation and read two news articles reasonably well.

#### Online shopping.

Figures[103](#S9.F103 "Figure 103 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")-[111](#S9.F111 "Figure 111 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates how GPT-4V can navigate a smartphone GUI for online shopping. Similarly, we provide GPT-4V with the screenshot of the current phone screen, the list of possible actions (*e.g*., move your finger to an icon, click an icon with your finger, scroll down a screen, or type some texts with the keyboard) and ask it to predict the subsequent actions to shop for an ergonomic keyboard with a budget between $50 and $100. GPT-4V predicts to open the Amazon app (Figure[103](#S9.F103 "Figure 103 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), search ergonomic keyboard (Figure[104](#S9.F104 "Figure 104 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), open the filter options (Figure[105](#S9.F105 "Figure 105 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), set the price range filter between $50 and $100 (Figure[106](#S9.F106 "Figure 106 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), show filtered results (Figure[107](#S9.F107 "Figure 107 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), select the top search result (Figure[108](#S9.F108 "Figure 108 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), view product details (Figure[109](#S9.F109 "Figure 109 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), add product to the shopping cart (Figure[110](#S9.F110 "Figure 110 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")) and finally proceed to checkout (Figure[111](#S9.F111 "Figure 111 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")).

#### Notification understanding.

Notifications are integral to modern human-computer interactions. GPT-4V has demonstrated its capacity to interpret notification content and respond accordingly. As shown in Figure[112](#S9.F112 "Figure 112 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), the model can read and respond to a notification, such as suggesting to open the Maps app in response to a meeting proposal in Seattle. It also handles call (Figure[113](#S9.F113 "Figure 113 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")) and message (Figure[114](#S9.F114 "Figure 114 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")) notifications on a computer screen effectively.

#### Watching videos.

Alongside web browsing, videos are a key source of online information. GPT-4V has shown its capability to describe video content based on a series of screenshots from popular short-form videos. Regardless of whether the video has subtitle overlay (Figure[115](#S9.F115 "Figure 115 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") and[116](#S9.F116 "Figure 116 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")) or not (Figure[117](#S9.F117 "Figure 117 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"),[118](#S9.F118 "Figure 118 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"),[119](#S9.F119 "Figure 119 ‣ Watching videos. ‣ 9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")), GPT-4V can generate insightful descriptions about the video content, demonstrating its potential in automatic transcript generation for user-generated video content.

<img src='x95.png' alt='Refer to caption' title='' width='461' height='706' />

*Figure 92: GPT-4V navigates through GUI to browse the web to search for the recipe of Mapo Tofu. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x96.png' alt='Refer to caption' title='' width='461' height='693' />

*Figure 93: GPT-4V navigates through GUI to browse the web to search for the recipe of Mapo Tofu. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x97.png' alt='Refer to caption' title='' width='461' height='693' />

*Figure 94: GPT-4V navigates through GUI to browse the web to search for the recipe of Mapo Tofu. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x98.png' alt='Refer to caption' title='' width='461' height='693' />

*Figure 95: GPT-4V navigates through GUI to browse the web to search for the recipe of Mapo Tofu. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x99.png' alt='Refer to caption' title='' width='461' height='738' />

*Figure 96: GPT-4V navigates through GUI to browse the web to search for the recipe of Mapo Tofu. As GPT-4V predicts to print out the recipe in the previous turn, we prompt it to read the screenshot of the printed recipe and summarize it. Red highlights the inaccurate description about the image. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x100.png' alt='Refer to caption' title='' width='461' height='706' />

*Figure 97: GPT-4V navigates through GUI to browse the web to read today’s news. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x101.png' alt='Refer to caption' title='' width='461' height='590' />

*Figure 98: GPT-4V navigates through GUI to browse the web to read today’s news. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x102.png' alt='Refer to caption' title='' width='461' height='582' />

*Figure 99: GPT-4V navigates through GUI to browse the web to read today’s news. We prompt GPT-4V to read the screenshots of the first news article and summarize it. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x103.png' alt='Refer to caption' title='' width='461' height='590' />

*Figure 100: GPT-4V navigates through GUI to browse the web to read today’s news. Upon finishing reading the first news article, GPT-4V predicts to close the tab and return to previous page to continue browsing more news articles (highlighted in blue). Red highlights the inaccurate action prediction. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x104.png' alt='Refer to caption' title='' width='461' height='590' />

*Figure 101: GPT-4V navigates through GUI to browse the web to read today’s news. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x105.png' alt='Refer to caption' title='' width='461' height='818' />

*Figure 102: GPT-4V navigates through GUI to browse the web to read today’s news. We prompt GPT-4V to read the screenshots of the second news article and summarize it. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x106.png' alt='Refer to caption' title='' width='461' height='693' />

*Figure 103: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Red highlights the inaccurate location of the Amazon icon. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x107.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 104: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x108.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 105: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x109.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 106: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x110.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 107: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x111.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 108: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Red highlights the inaccurate location of the product option to be selected. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x112.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 109: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Red highlights the inaccurate action prediction (“Buy New” is not a clickable button). Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x113.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 110: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x114.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 111: GPT-4V navigates through GUI to shop for an ergonomic keyboard online. Blue highlights the predicted actions. Red highlights the inaccurate location of the “Proceed to checkout” buttion. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x115.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 112: Prompting GPT-4V to predict the action upon receiving a notification. GPT-4V can accurately recognize the notification and the corresponding content (highlighted in green). Blue highlights the predicted actions. Red highlights the inaccurate location of the Maps app icon. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x116.png' alt='Refer to caption' title='' width='461' height='515' />

*Figure 113: Prompting GPT-4V to predict the action upon receiving a notification. GPT-4V can accurately recognize the notification and the corresponding content (highlighted in green). Blue highlights the predicted actions. Red highlights the inaccurate location of the Maps app icon. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x117.png' alt='Refer to caption' title='' width='461' height='537' />

*Figure 114: Prompting GPT-4V to predict the action upon receiving a notification. GPT-4V can accurately recognize the notification and the corresponding content (highlighted in green). Blue highlights the predicted actions. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x118.png' alt='Refer to caption' title='' width='461' height='706' />

*Figure 115: Prompting GPT-4V to watch web videos. We present GPT-4V the screenshot of the video frames following their temporal order in the original video. To save space, we illustrate the frames in a row, where the leftmost one is the first frame. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions. Red highlights the inaccurate descriptions about the video.*

<img src='x119.png' alt='Refer to caption' title='' width='461' height='716' />

*Figure 116: Watching web videos. We present GPT-4V the screenshot of the video frames following their temporal order in the original video. To save space, we illustrate the frames in a row, where the leftmost one is the first frame. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x120.png' alt='Refer to caption' title='' width='461' height='595' />

*Figure 117: Watching web videos. We present GPT-4V the screenshot of the video frames following their temporal order in the original video. To save space, we illustrate frames 1-5 in the first row, and frames 6-9 in the second row. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x121.png' alt='Refer to caption' title='' width='461' height='604' />

*Figure 118: Watching web videos. We present GPT-4V the screenshot of the video frames following their temporal order in the original video. To save space, we illustrate frames 1-5 in the first row, and frames 6-9 in the second row. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x122.png' alt='Refer to caption' title='' width='461' height='590' />

*Figure 119: Watching web videos. We present GPT-4V the screenshot of the video frames following their temporal order in the original video. To save space, we illustrate frames 1-5 in the first row, and frames 6-9 in the second row. Red highlights the inaccurate descriptions about the video. Check Section[9.8](#S9.SS8 "9.8 GUI Navigation ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

10 LMM Powered Agents
---------------------

In this section, we discuss possible future research directions that may further amplify GPT-4V’s capabilities. The discussion focuses on how the intriguing usages in LLMs may extend to the multimodal scenario and its enabled new abilities, *e.g*., multimodal plugins, multimodal chains,
self-reflection, self-consistency, and retrieval-augmented LMMs, *etc*. In the following sub-sections, we use *human-generated* examples to illustrate potential ways to enhance GPT-4V-based systems.

<img src='x123.png' alt='Refer to caption' title='' width='461' height='559' />

*Figure 120: Illustration of using the Bing Image Search*[[94](#bib.bib94 "")]* plugin to enable GPT-4V with time-sensitive knowledge (bottom, highlighted in green). Note that the earthquake happened on February 6, 2023, which is after GPT-4V’s training, thereby GPT-4V fails to identify the exact location without plugin (top). Check Section[10.1](#S10.SS1 "10.1 Multimodal Plugins ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x124.png' alt='Refer to caption' title='' width='461' height='657' />

*Figure 121: Extending GPT-4V to multimodal chains with ReAct*[[145](#bib.bib145 ""), [142](#bib.bib142 "")]* for PPE Counting scenario. Check Section[10.2](#S10.SS2 "10.2 Multimodal Chains ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 10.1 Multimodal Plugins

In the context of LLMs, plugins*[[98](#bib.bib98 ""), [56](#bib.bib56 ""), [6](#bib.bib6 ""), [112](#bib.bib112 ""), [87](#bib.bib87 ""), [103](#bib.bib103 "")]* play a crucial role in assisting LLMs for various tasks such as accessing the latest information, performing computations, or utilizing third-party services. These plugins are primarily designed to process inputs in natural language or inputs that can be interpreted as language, such as code and math equations. To illustrate the significance of multimodal plugins, such as Bing Image Search*[[94](#bib.bib94 "")]*, especially in the context of LMMs, we present Figure[120](#S10.F120 "Figure 120 ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"). By incorporating the Bing Image Search plugin, we empower GPT-4V to acquire time-sensitive knowledge related to the input image. In the upper part of the figure, we demonstrate the limitations of GPT-4V without Bing Image Search plugin. It fails to accurately answer the question, "Where was this photo taken?" due to the fact that the photo captures the aftermath of a massive earthquake that occurred on February 6, 2023, at the border of Turkey and Syria—a situation that took place after GPT-4V’s training. Since constantly retraining the model with current information can be computationally intensive and expensive, plugins like search engines prove to be invaluable resources for the model to access up-to-date information. In the lower part of Figure[120](#S10.F120 "Figure 120 ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we showcase the capabilities of GPT-4V when equipped with the Bing Image Search plugin. It effectively leverages the retrieved information from the plugin, enabling accurate identification of the location İzmir, Turkey.

### 10.2 Multimodal Chains

Chaining with LLMs has been explored extensively in recent research*[[145](#bib.bib145 ""), [47](#bib.bib47 ""), [124](#bib.bib124 ""), [107](#bib.bib107 "")]*. This approach goes beyond using a single plugin and instead establishes a system paradigm that integrates LLMs with a pool of plugins, enabling more advanced reasoning and interactions. By replacing language-only plugins with vision/multimodal experts such as image captioners, object detectors, or well-trained models for text-to-image generation and audio-to-text conversion, it becomes possible to construct a powerful multimodal chain with LLMs*[[137](#bib.bib137 ""), [142](#bib.bib142 ""), [121](#bib.bib121 ""), [114](#bib.bib114 ""), [75](#bib.bib75 ""), [86](#bib.bib86 "")]*.

However, the interactions within these chains between LLMs and the plugins typically take place in text format. Although the plugins may accept multimodal inputs, they return results in text to enhance the knowledge of LLMs. There is a notable exception in the case of image synthesis/editing*[[137](#bib.bib137 "")]*, where the plugins can generate images, but these images are not fed back into LLMs for further analysis or knowledge augmentation, as LLMs can only process language-based inputs.

In Figure[121](#S10.F121 "Figure 121 ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)"), we present an illustration of how GPT-4V, can be extended to support multimodal chains with ReAct*[[145](#bib.bib145 ""), [142](#bib.bib142 "")]*. This extension enables the plugins in the chain to provide multimodal information, which can then be collectively processed by GPT-4V to achieve advanced reasoning in scenarios such as PPE counting. The entire chaining process shown in Figure[121](#S10.F121 "Figure 121 ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") is divided into two rounds of thought, action, and observation, with each round involving the activation of a specific plugin. In the first round, GPT-4V deduces that person detection is necessary to count the number of people wearing helmets (Thought 1). Consequently, it calls the person detector tool (Action 1) and receives the coordinates of bounding boxes for each detected person in the image (Observation 1). Moving to the second round, based on the obtained bounding box information, GPT-4V infers that there are a total of 8 people in the image (Thought 2). It then utilizes the image cropping tool to crop out individual images of each person according to their corresponding bounding box coordinates (Action 2). The resulting outputs (Observation 2) consist of 8 labeled images, numbered from image 1 to image 8. GPT-4V subsequently determines whether each person in these images is wearing a helmet or not, and summarizes the total count of people wearing helmets.

Overall, this integration of LMMs with a pool of multimodal plugins opens up new possibilities for enhanced reasoning and interaction, leveraging the strengths of both language and vision capabilities. The flexibility of multimodal chains allows for a more comprehensive understanding and analysis of multimodal data, and can potentially lead to improved performance in various applications.

### 10.3 Self-Reflection

Figure[122](#S10.F122 "Figure 122 ‣ 10.3 Self-Reflection ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") demonstrates the application of self-reflection*[[116](#bib.bib116 ""), [88](#bib.bib88 ""), [63](#bib.bib63 "")]* to improve the results shown in Figure[47](#S4.F47 "Figure 47 ‣ 4.6 Coding Capability with Vision ‣ 4 Vision-Language Capability ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)").
As we can see, the self-reflected result is better aligned with the reference image.
For example, on the left side, the number of data points is corrected from 4 to 3, while on the right side, the percentage is added back above the bar.
Although the result is still not exactly identical,
it is evident that self-reflection can facilitate manual polishing.
Figure[123](#S10.F123 "Figure 123 ‣ 10.3 Self-Reflection ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows another example of self-reflection in improving the prompt generation for text-to-image models*[[106](#bib.bib106 "")]*.

<img src='x125.png' alt='Refer to caption' title='' width='461' height='652' />

*Figure 122: Illustration of using self-reflection to improve the code for figure drawing.
Left: after reflection, the number of points in the curve aligns with the reference image.
Right: the percentage is added to align with the reference image.
Check Section[10.3](#S10.SS3 "10.3 Self-Reflection ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x126.png' alt='Refer to caption' title='' width='461' height='641' />

*Figure 123: Illustration of using self-reflection to improve the generated text prompts for a text-to-image model SDXL*[[106](#bib.bib106 "")]*. GPT-4V reflects the error in the initial prompt that it does not mention the dog’s breed, and makes the correct revision. Check Section[10.3](#S10.SS3 "10.3 Self-Reflection ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

<img src='x127.png' alt='Refer to caption' title='' width='461' height='647' />

*Figure 124: Improve the counting reliability with self-consistency*[[130](#bib.bib130 "")]*, which aggregates multiple counting results repeated on the *same* image.
Check Section[10.4](#S10.SS4 "10.4 Self-Consistency ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") for detailed discussions.*

### 10.4 Self-Consistency

Self-consistency*[[130](#bib.bib130 "")]* is a decoding strategy that aggregates multiple sampled outputs to produce the final answer, such as with the majority vote. Extended from marginalizing to aggregating final answers, Tree-of-Thoughts*[[144](#bib.bib144 "")]* shows that the self-consistency idea can be applied to intermediate thoughts to improve the LLM reasoning performance. Figure[124](#S10.F124 "Figure 124 ‣ 10.3 Self-Reflection ‣ 10 LMM Powered Agents ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") illustrates the use of self-consistency on GPT-4V for counting problems. We sample multiple counting results by asking GPT-4V to count the same image multiple times, either conducting multiple runs (Samples 2-4) or rephrasing the input text instruction (Samples 1,2). The example then uses the simple majority vote to aggregate the final answer of “4 boats.” We leave the comprehensive explorations of self-consistency LMMs to future works.

### 10.5 Retrieval-Augmented LMMs

Retrieval-Augmented LMMs*[[93](#bib.bib93 ""), [68](#bib.bib68 ""), [50](#bib.bib50 ""), [19](#bib.bib19 ""), [115](#bib.bib115 ""), [104](#bib.bib104 "")]* enhances text generation by retrieving and integrating relevant information into prompts. The technique is particularly effective when specialized task-relevant information is needed, such as expert knowledge in a highly-specialized expert domain, the most recent information that may differ from LLMs’ memory, and the customizable information that varies from user to user. We imagine retrieval augmentation continues to play an essential role in LMMs. Figure[74](#S9.F74 "Figure 74 ‣ Grocery checkout. ‣ 9.2 Industry ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)") shows an example of retrieval-augmented LMMs helping grocery checkout. Since the produces’ image-text-price triplets are different in each store, it would be beneficial to retrieve them from the store’s database and yield the correct checkout information. Similarly, in Figure[81](#S9.F81 "Figure 81 ‣ Dense captioning w/ segmentation. ‣ 9.5 Customized Captioner ‣ 9 Emerging Application Highlights ‣ The Dawn of LMMs: Preliminary Explorations with GPT-4V(ision)")’s the customized captioning scenario, we imagine the system may automatically retrieve the family members’ photos from the album and achieve the customized captioning.

11 Conclusions
--------------

### 11.1 Summary and Conclusions

In this report, our primary focus is on probing GPT-4V across various application scenarios. The findings reveal its remarkable capabilities, some of which have not been investigated or demonstrated in existing approaches. While we strive to uncover as many of these capabilities as possible, we acknowledge that our presentation may not be exhaustive. Nevertheless, this report can serve as a reference for future research aimed at exploring additional uses of GPT-4V, deepening the understanding of LMMs, and building even more powerful LMMs.

### 11.2 Towards Future LMMs

The weaknesses and limitations of GPT models have been extensively discussed in related reports*[[99](#bib.bib99 ""), [100](#bib.bib100 ""), [24](#bib.bib24 "")]*.
In this section, we briefly focus on presenting our perspective on future research directions.

Models like GPT-1, GPT-2, and GPT-3 function primarily as text-in-text-out systems, capable of processing natural language only. GPT-4 (no vision) demonstrates unparalleled competence in text understanding and generation, while GPT-4V exhibits a strong ability to comprehend the image domain as well.

As a natural progression, LMMs should be able to generate
interleaved image-text content, such as producing vivid tutorials containing both text and images, to enable comprehensive multimodal content understanding and generation. Additionally, it would be beneficial to incorporate other modalities, such as video, audio, and other sensor data, to expand the capabilities of LMMs.

Regarding the learning process, current approaches predominantly rely on well-organized data, such as image-tag or image-text datasets. However, a more versatile model may be able to learn from various sources, including online web content and even real-world physical environments, to facilitate continuous self-evolution.

Acknowledgment
--------------

We express our gratitude to all contributors from OpenAI for their technical efforts on the GPT-4V project*[[99](#bib.bib99 ""), [100](#bib.bib100 ""), [101](#bib.bib101 ""), [1](#bib.bib1 "")]*, and we are profoundly thankful to OpenAI for granting early access to their remarkable tool. Our sincere appreciation goes to Misha Bilenko for his invaluable guidance and support. We also extend heartfelt thanks to our Microsoft colleagues for their insights, with special acknowledgment to John Montgomery, Marco Casalaina, Gregory Buehrer, Nguyen Bach, Gopi Kumar, Luis Vargas, Kun Wu, Meenaz Merchant, Jianfeng Gao, Matt Lungren, Sheela Agarwal, Yumao Lu, Thomas Soemo, Fisayo Okikiolu, Ce Liu, Michael Zeng, Faisal Ahmed, Ehsan Azarnasab, and Lin Liang for their constructive feedback. We also thank Yingkai Yu for helping to create screenshots on GUI Navigation.

References
----------

* [1]Chatgpt can now see, hear, and speak.[https://openai.com/blog/chatgpt-can-now-see-hear-and-speak](https://openai.com/blog/chatgpt-can-now-see-hear-and-speak ""), 2023.
* [2]Deepfloyd if.[https://github.com/deep-floyd/IF](https://github.com/deep-floyd/IF ""), 2023.
* [3]Guidance.<https://github.com/microsoft/guidance/>, 2023.
* [4]Midjourney.<https://www.midjourney.com/>, 2023.
* [5]Sameer Agarwal, Yasutaka Furukawa, Noah Snavely, Ian Simon, Brian Curless, Steven M Seitz, and Richard Szeliski.Building rome in a day.Communications of the ACM, 54(10):105–112, 2011.
* [6]Michael Ahn, Anthony Brohan, Noah Brown, Yevgen Chebotar, Omar Cortes, Byron David, Chelsea Finn, Keerthana Gopalakrishnan, Karol Hausman, Alex Herzog, et al.Do as i can, not as i say: Grounding language in robotic affordances.arXiv preprint arXiv:2204.01691, 2022.
* [7]Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katherine Millican, Malcolm Reynolds, et al.Flamingo: a visual language model for few-shot learning.Advances in Neural Information Processing Systems, 35:23716–23736, 2022.
* [8]Chris Alberti, Jeffrey Ling, Michael Collins, and David Reitter.Fusion of detected objects in text for visual question answering.In EMNLP, 2019.
* [9]Peter Anderson, Xiaodong He, Chris Buehler, Damien Teney, Mark Johnson, Stephen Gould, and Lei Zhang.Bottom-up and top-down attention for image captioning and visual question answering.In CVPR, 2018.
* [10]Karpathy Andrej.State of gpt.<https://karpathy.ai/stateofgpt.pdf>, 2023.
* [11]Rohan Anil, Andrew M Dai, Orhan Firat, Melvin Johnson, Dmitry Lepikhin, Alexandre Passos, Siamak Shakeri, Emanuel Taropa, Paige Bailey, Zhifeng Chen, et al.Palm 2 technical report.arXiv preprint arXiv:2305.10403, 2023.
* [12]Stanislaw Antol, Aishwarya Agrawal, Jiasen Lu, Margaret Mitchell, Dhruv Batra, C. Lawrence Zitnick, and Devi Parikh.VQA: Visual Question Answering.In ICCV, 2015.
* [13]Anas Awadalla, Irena Gao, Joshua Gardner, Jack Hessel, Yusuf Hanafy, Wanrong Zhu, Kalyani Marathe, Yonatan Bitton, Samir Gadre, Jenia Jitsev, Simon Kornblith, Pang Wei Koh, Gabriel Ilharco, Mitchell Wortsman, and Ludwig Schmidt.Openflamingo, March 2023.
* [14]Hessam Bagherinezhad, Hannaneh Hajishirzi, Yejin Choi, and Ali Farhadi.Are elephants bigger than butterflies? reasoning about sizes of objects.In Proceedings of the AAAI Conference on Artificial Intelligence, volume 30, 2016.
* [15]Shruthi Bannur, Stephanie Hyland, Qianchu Liu, Fernando Perez-Garcia, Maximilian Ilse, Daniel C Castro, Benedikt Boecking, Harshita Sharma, Kenza Bouzid, Anja Thieme, et al.Learning to exploit temporal structure for biomedical vision-language processing.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15016–15027, 2023.
* [16]David Barrett, Felix Hill, Adam Santoro, Ari Morcos, and Timothy Lillicrap.Measuring abstract reasoning in neural networks.In International conference on machine learning, pages 511–520. PMLR, 2018.
* [17]Ali Furkan Biten, Ruben Tito, Andres Mafla, Lluis Gomez, Marçal Rusinol, Ernest Valveny, CV Jawahar, and Dimosthenis Karatzas.Scene text visual question answering.In ICCV, 2019.
* [18]Kevin Black, Michael Janner, Yilun Du, Ilya Kostrikov, and Sergey Levine.Training diffusion models with reinforcement learning, 2023.
* [19]Sebastian Borgeaud, Arthur Mensch, Jordan Hoffmann, Trevor Cai, Eliza Rutherford, Katie Millican, George Bm Van Den Driessche, Jean-Baptiste Lespiau, Bogdan Damoc, Aidan Clark, et al.Improving language models by retrieving from trillions of tokens.In International conference on machine learning, pages 2206–2240. PMLR, 2022.
* [20]Lukas Bossard, Matthieu Guillaumin, and Luc Van Gool.Food-101–mining discriminative components with random forests.In Computer Vision–ECCV 2014: 13th European Conference, Zurich, Switzerland, September 6-12, 2014, Proceedings, Part VI 13, pages 446–461. Springer, 2014.
* [21]Marc A Brackett and Peter Salovey.Measuring emotional intelligence with the mayer-salovery-caruso emotional intelligence test (msceit).Psicothema, 18:34–41, 2006.
* [22]Tim Brooks, Aleksander Holynski, and Alexei A. Efros.Instructpix2pix: Learning to follow image editing instructions.In CVPR, 2023.
* [23]Tom B Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al.Language models are few-shot learners.In NeurIPS, 2020.
* [24]Sébastien Bubeck, Varun Chandrasekaran, Ronen Eldan, Johannes Gehrke, Eric Horvitz, Ece Kamar, Peter Lee, Yin Tat Lee, Yuanzhi Li, Scott Lundberg, et al.Sparks of artificial general intelligence: Early experiments with gpt-4.arXiv preprint arXiv:2303.12712, 2023.
* [25]Ting Chen, Saurabh Saxena, Lala Li, David J Fleet, and Geoffrey Hinton.Pix2seq: A language modeling framework for object detection.In ICLR, 2022.
* [26]Ting Chen, Saurabh Saxena, Lala Li, Tsung-Yi Lin, David J Fleet, and Geoffrey E Hinton.A unified sequence interface for vision tasks.Advances in Neural Information Processing Systems, 35:31333–31346, 2022.
* [27]Xinlei Chen, Hao Fang, Tsung-Yi Lin, Ramakrishna Vedantam, Saurabh Gupta, Piotr Dollár, and C Lawrence Zitnick.Microsoft coco captions: Data collection and evaluation server.arXiv preprint arXiv:1504.00325, 2015.
* [28]Yen-Chun Chen, Linjie Li, Licheng Yu, Ahmed El Kholy, Faisal Ahmed, Zhe Gan, Yu Cheng, and Jingjing Liu.Uniter: Learning universal image-text representations.In ECCV, 2020.
* [29]Jaemin Cho, Jie Lei, Hao Tan, and Mohit Bansal.Unifying vision-and-language tasks via text generation.In ICML, 2021.
* [30]Aakanksha Chowdhery, Sharan Narang, Jacob Devlin, Maarten Bosma, Gaurav Mishra, Adam Roberts, Paul Barham, Hyung Won Chung, Charles Sutton, Sebastian Gehrmann, et al.Palm: Scaling language modeling with pathways.arXiv preprint arXiv:2204.02311, 2022.
* [31]Herbert H Clark and Deanna Wilkes-Gibbs.Referring as a collaborative process.Cognition, 22(1):1–39, 1986.
* [32]Marius Cordts, Mohamed Omran, Sebastian Ramos, Timo Rehfeld, Markus Enzweiler, Rodrigo Benenson, Uwe Franke, Stefan Roth, and Bernt Schiele.The cityscapes dataset for semantic urban scene understanding.In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 3213–3223, 2016.
* [33]Tom Cornsweet.Visual perception.Academic press, 2012.
* [34]Damai Dai, Yutao Sun, Li Dong, Yaru Hao, Zhifang Sui, and Furu Wei.Why can gpt learn in-context? language models secretly perform gradient descent as meta optimizers.arXiv preprint arXiv:2212.10559, 2022.
* [35]Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung, and Steven Hoi.Instructblip: Towards general-purpose vision-language models with instruction tuning.arXiv preprint arXiv:2305.06500, 2023.
* [36]Abhishek Das, Satwik Kottur, Khushi Gupta, Avi Singh, Deshraj Yadav, José MF Moura, Devi Parikh, and Dhruv Batra.Visual dialog.In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 326–335, 2017.
* [37]Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, and Li Fei-Fei.Imagenet: A large-scale hierarchical image database.In CVPR, 2009.
* [38]Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova.Bert: Pre-training of deep bidirectional transformers for language understanding.In NAACL-HLT, 2019.
* [39]Qingxiu Dong, Lei Li, Damai Dai, Ce Zheng, Zhiyong Wu, Baobao Chang, Xu Sun, Jingjing Xu, and Zhifang Sui.A survey for in-context learning.arXiv preprint arXiv:2301.00234, 2022.
* [40]Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, et al.Coarse-to-fine vision-language pre-training with fusion in the backbone.In Advances in Neural Information Processing Systems.
* [41]Zi-Yi Dou, Yichong Xu, Zhe Gan, Jianfeng Wang, Shuohang Wang, Lijuan Wang, Chenguang Zhu, Pengchuan Zhang, Lu Yuan, Nanyun Peng, et al.An empirical study of training end-to-end vision-and-language transformers.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 18166–18176, 2022.
* [42]Danny Driess, Fei Xia, Mehdi S. M. Sajjadi, Corey Lynch, Aakanksha Chowdhery, Brian Ichter, Ayzaan Wahid, Jonathan Tompson, Quan Vuong, Tianhe Yu, Wenlong Huang, Yevgen Chebotar, Pierre Sermanet, Daniel Duckworth, Sergey Levine, Vincent Vanhoucke, Karol Hausman, Marc Toussaint, Klaus Greff, Andy Zeng, Igor Mordatch, and Pete Florence.Palm-e: An embodied multimodal language model.In arXiv preprint arXiv:2303.03378, 2023.
* [43]Alicia Fasquel, Angèle Brunellière, and Dominique Knutsen.A modified procedure for naming 332 pictures and collecting norms: Using tangram pictures in psycholinguistic studies.Behavior Research Methods, pages 1–23, 2022.
* [44]Samir Yitzhak Gadre, Kiana Ehsani, and Shuran Song.Act the part: Learning interaction strategies for articulated object part discovery.In Proceedings of the IEEE/CVF International Conference on Computer Vision, pages 15752–15761, 2021.
* [45]Zhe Gan, Yen-Chun Chen, Linjie Li, Chen Zhu, Yu Cheng, and Jingjing Liu.Large-scale adversarial training for vision-and-language representation learning.In NeurIPS, 2020.
* [46]Zhe Gan, Linjie Li, Chunyuan Li, Lijuan Wang, Zicheng Liu, Jianfeng Gao, et al.Vision-language pre-training: Basics, recent advances, and future trends.Foundations and Trends® in Computer Graphics and Vision, 14(3–4):163–352, 2022.
* [47]Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig.Pal: Program-aided language models.In International Conference on Machine Learning, pages 10764–10799. PMLR, 2023.
* [48]Tao Gong, Chengqi Lyu, Shilong Zhang, Yudong Wang, Miao Zheng, Qian Zhao, Kuikun Liu, Wenwei Zhang, Ping Luo, and Kai Chen.Multimodal-gpt: A vision and language model for dialogue with humans, 2023.
* [49]Yandong Guo, Lei Zhang, Yuxiao Hu, Xiaodong He, and Jianfeng Gao.Ms-celeb-1m: A dataset and benchmark for large-scale face recognition.In Computer Vision–ECCV 2016: 14th European Conference, Amsterdam, The Netherlands, October 11-14, 2016, Proceedings, Part III 14, pages 87–102. Springer, 2016.
* [50]Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, and Mingwei Chang.Retrieval augmented language model pre-training.In International conference on machine learning, pages 3929–3938. PMLR, 2020.
* [51]Kaiming He, Georgia Gkioxari, Piotr Dollár, and Ross Girshick.Mask r-cnn.In Proceedings of the IEEE international conference on computer vision, pages 2961–2969, 2017.
* [52]Jack *Hessel, Jena D *Hwang, Jae Sung Park, Rowan Zellers, Chandra Bhagavatula, Anna Rohrbach, Kate Saenko, and Yejin Choi.The Abduction of Sherlock Holmes: A Dataset for Visual Abductive Reasoning.In ECCV, 2022.
* [53]Jordan Hoffmann, Sebastian Borgeaud, Arthur Mensch, Elena Buchatskaya, Trevor Cai, Eliza Rutherford, Diego de Las Casas, Lisa Anne Hendricks, Johannes Welbl, Aidan Clark, et al.Training compute-optimal large language models.arXiv preprint arXiv:2203.15556, 2022.
* [54]Yushi Hu, Hang Hua, Zhengyuan Yang, Weijia Shi, Noah A Smith, and Jiebo Luo.Promptcap: Prompt-guided task-aware image captioning.In Proceedings of International Conference on Computer Vision (ICCV), 2023.
* [55]Shaohan Huang, Li Dong, Wenhui Wang, Yaru Hao, Saksham Singhal, Shuming Ma, Tengchao Lv, Lei Cui, Owais Khan Mohammed, Qiang Liu, et al.Language is not all you need: Aligning perception with language models.arXiv preprint arXiv:2302.14045, 2023.
* [56]Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch.Language models as zero-shot planners: Extracting actionable knowledge for embodied agents.In International Conference on Machine Learning, pages 9118–9147. PMLR, 2022.
* [57]Zhicheng Huang, Zhaoyang Zeng, Bei Liu, Dongmei Fu, and Jianlong Fu.Pixel-bert: Aligning image pixels with text by deep multi-modal transformers.arXiv preprint arXiv:2004.00849, 2020.
* [58]Fabian Hutmacher.Why is there so much more research on vision than on any other sensory modality?Frontiers in psychology, 10:2246, 2019.
* [59]Anya Ji, Noriyuki Kojima, Noah Rush, Alane Suhr, Wai Keen Vong, Robert Hawkins, and Yoav Artzi.Abstract visual reasoning with tangram shapes.In Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pages 582–601, 2022.
* [60]Alistair EW Johnson, Tom J Pollard, Seth J Berkowitz, Nathaniel R Greenbaum, Matthew P Lungren, Chih-ying Deng, Roger G Mark, and Steven Horng.Mimic-cxr, a de-identified publicly available database of chest radiographs with free-text reports.Scientific data, 6(1):317, 2019.
* [61]Justin Johnson, Bharath Hariharan, Laurens Van Der Maaten, Li Fei-Fei, C Lawrence Zitnick, and Ross Girshick.Clevr: A diagnostic dataset for compositional language and elementary visual reasoning.In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 2901–2910, 2017.
* [62]Justin Johnson, Andrej Karpathy, and Li Fei-Fei.Densecap: Fully convolutional localization networks for dense captioning.In Proceedings of the IEEE conference on computer vision and pattern recognition, pages 4565–4574, 2016.
* [63]Geunwoo Kim, Pierre Baldi, and Stephen McAleer.Language models can solve computer tasks.arXiv preprint arXiv:2303.17491, 2023.
* [64]Wonjae Kim, Bokyung Son, and Ildoo Kim.Vilt: Vision-and-language transformer without convolution or region supervision.In ICML, 2021.
* [65]Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al.Segment anything.In Proceedings of International Conference on Computer Vision (ICCV), 2023.
* [66]Takeshi Kojima, Shixiang Shane Gu, Machel Reid, Yutaka Matsuo, and Yusuke Iwasawa.Large language models are zero-shot reasoners.Advances in neural information processing systems, 35:22199–22213, 2022.
* [67]Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A Shamma, et al.Visual genome: Connecting language and vision using crowdsourced dense image annotations.IJCV, 2017.
* [68]Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, et al.Retrieval-augmented generation for knowledge-intensive nlp tasks.Advances in Neural Information Processing Systems, 33:9459–9474, 2020.
* [69]Chunyuan Li, Zhe Gan, Zhengyuan Yang, Jianwei Yang, Linjie Li, Lijuan Wang, and Jianfeng Gao.Multimodal foundation models: From specialists to general-purpose assistants.arXiv preprint arXiv:2309.10020, 2023.
* [70]Gen Li, Nan Duan, Yuejian Fang, Ming Gong, Daxin Jiang, and Ming Zhou.Unicoder-vl: A universal encoder for vision and language by cross-modal pre-training.In AAAI, 2020.
* [71]Junnan Li, Dongxu Li, Silvio Savarese, and Steven Hoi.Blip-2: Bootstrapping language-image pre-training with frozen image encoders and large language models.arXiv preprint arXiv:2301.12597, 2023.
* [72]Junnan Li, Ramprasaath R Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven Hoi.Align before fuse: Vision and language representation learning with momentum distillation.In NeurIPS, 2021.
* [73]Liunian Harold Li, Mark Yatskar, Da Yin, Cho-Jui Hsieh, and Kai-Wei Chang.Visualbert: A simple and performant baseline for vision and language.arXiv preprint arXiv:1908.03557, 2019.
* [74]Xiujun Li, Xi Yin, Chunyuan Li, Xiaowei Hu, Pengchuan Zhang, Lei Zhang, Lijuan Wang, Houdong Hu, Li Dong, Furu Wei, et al.Oscar: Object-semantics aligned pre-training for vision-language tasks.In ECCV, 2020.
* [75]Yaobo Liang, Chenfei Wu, Ting Song, Wenshan Wu, Yan Xia, Yu Liu, Yang Ou, Shuai Lu, Lei Ji, Shaoguang Mao, et al.Taskmatrix. ai: Completing tasks by connecting foundation models with millions of apis.arXiv preprint arXiv:2303.16434, 2023.
* [76]Tsung-Yi Lin, Michael Maire, Serge Belongie, James Hays, Pietro Perona, Deva Ramanan, Piotr Dollár, and C Lawrence Zitnick.Microsoft coco: Common objects in context.In ECCV, 2014.
* [77]Fangyu Liu, Emanuele Bugliarello, Edoardo Maria Ponti, Siva Reddy, Nigel Collier, and Desmond Elliott.Visually grounded reasoning across languages and cultures.In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10467–10485, Online and Punta Cana, Dominican Republic, November 2021. Association for Computational Linguistics.
* [78]Fuxiao Liu, Kevin Lin, Linjie Li, Jianfeng Wang, Yaser Yacoob, and Lijuan Wang.Aligning large multi-modal model with robust instruction tuning.arXiv preprint arXiv:2306.14565, 2023.
* [79]Haotian Liu, Chunyuan Li, Qingyang Wu, and Yong Jae Lee.Visual instruction tuning.arXiv preprint arXiv:2304.08485, 2023.
* [80]Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang.Deep learning face attributes in the wild.In Proceedings of International Conference on Computer Vision (ICCV), December 2015.
* [81]Jiasen Lu, Dhruv Batra, Devi Parikh, and Stefan Lee.Vilbert: Pretraining task-agnostic visiolinguistic representations for vision-and-language tasks.In NeurIPS, 2019.
* [82]Jiasen Lu, Christopher Clark, Rowan Zellers, Roozbeh Mottaghi, and Aniruddha Kembhavi.Unified-io: A unified model for vision, language, and multi-modal tasks.In The Eleventh International Conference on Learning Representations, 2022.
* [83]Jiasen Lu, Vedanuj Goswami, Marcus Rohrbach, Devi Parikh, and Stefan Lee.12-in-1: Multi-task vision and language representation learning.In CVPR, 2020.
* [84]Jiasen Lu, Jianwei Yang, Dhruv Batra, and Devi Parikh.Neural baby talk.In CVPR, 2018.
* [85]Pan Lu, Swaroop Mishra, Tanglin Xia, Liang Qiu, Kai-Wei Chang, Song-Chun Zhu, Oyvind Tafjord, Peter Clark, and Ashwin Kalyan.Learn to explain: Multimodal reasoning via thought chains for science question answering.Advances in Neural Information Processing Systems, 35:2507–2521, 2022.
* [86]Pan Lu, Baolin Peng, Hao Cheng, Michel Galley, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, and Jianfeng Gao.Chameleon: Plug-and-play compositional reasoning with large language models.arXiv preprint arXiv:2304.09842, 2023.
* [87]Pan Lu, Liang Qiu, Kai-Wei Chang, Ying Nian Wu, Song-Chun Zhu, Tanmay Rajpurohit, Peter Clark, and Ashwin Kalyan.Dynamic prompt learning via policy gradient for semi-structured mathematical reasoning.arXiv preprint arXiv:2209.14610, 2022.
* [88]Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, et al.Self-refine: Iterative refinement with self-feedback.arXiv preprint arXiv:2303.17651, 2023.
* [89]Bertram F Malle, Louis J Moses, and Dare A Baldwin.Intentions and intentionality: Foundations of social cognition.MIT press, 2001.
* [90]Arjun Mani, Nobline Yoo, Will Hinthorn, and Olga Russakovsky.Point and ask: Incorporating pointing into visual question answering.arXiv preprint arXiv:2011.13681, 2020.
* [91]John D Mayer.Msceit: Mayer-salovey-caruso emotional intelligence test.Toronto, Canada: Multi-Health Systems, 2002.
* [92]John D Mayer, Richard D Roberts, and Sigal G Barsade.Human abilities: Emotional intelligence.Annu. Rev. Psychol., 59:507–536, 2008.
* [93]Grégoire Mialon, Roberto Dessì, Maria Lomeli, Christoforos Nalmpantis, Ram Pasunuru, Roberta Raileanu, Baptiste Rozière, Timo Schick, Jane Dwivedi-Yu, Asli Celikyilmaz, et al.Augmented language models: a survey.arXiv preprint arXiv:2302.07842, 2023.
* [94]Microsoft.Bing image search api.[https://www.microsoft.com/en-us/bing/apis/bing-image-search-api](https://www.microsoft.com/en-us/bing/apis/bing-image-search-api ""), 2023.
* [95]Weiqing Min, Zhiling Wang, Yuxin Liu, Mengjiang Luo, Liping Kang, Xiaoming Wei, Xiaolin Wei, and Shuqiang Jiang.Large scale visual food recognition.IEEE Transactions on Pattern Analysis and Machine Intelligence, 2023.
* [96]Swaroop Mishra, Daniel Khashabi, Chitta Baral, and Hannaneh Hajishirzi.Cross-task generalization via natural language crowdsourcing instructions.In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3470–3487, 2022.
* [97]Tara Murfitt and Jan McAllister.The effect of production variables in monolog and dialog on comprehension by novel listeners.Language and Speech, 44(3):325–350, 2001.
* [98]Reiichiro Nakano, Jacob Hilton, Suchir Balaji, Jeff Wu, Long Ouyang, Christina Kim, Christopher Hesse, Shantanu Jain, Vineet Kosaraju, William Saunders, et al.Webgpt: Browser-assisted question-answering with human feedback.arXiv preprint arXiv:2112.09332, 2021.
* [99]OpenAI.Gpt-4 technical report, 2023.
* [100]OpenAI.Gpt-4v(ision) system card.2023.
* [101]OpenAI.Gpt-4v(ision) technical work and authors.2023.
* [102]Long Ouyang, Jeffrey Wu, Xu Jiang, Diogo Almeida, Carroll Wainwright, Pamela Mishkin, Chong Zhang, Sandhini Agarwal, Katarina Slama, Alex Ray, et al.Training language models to follow instructions with human feedback.Advances in Neural Information Processing Systems, 35:27730–27744, 2022.
* [103]Bhargavi Paranjape, Scott Lundberg, Sameer Singh, Hannaneh Hajishirzi, Luke Zettlemoyer, and Marco Tulio Ribeiro.Art: Automatic multi-step reasoning and tool-use for large language models.arXiv preprint arXiv:2303.09014, 2023.
* [104]Baolin Peng, Michel Galley, Pengcheng He, Hao Cheng, Yujia Xie, Yu Hu, Qiuyuan Huang, Lars Liden, Zhou Yu, Weizhu Chen, et al.Check your facts and try again: Improving large language models with external knowledge and automated feedback.arXiv preprint arXiv:2302.12813, 2023.
* [105]Zhiliang Peng, Wenhui Wang, Li Dong, Yaru Hao, Shaohan Huang, Shuming Ma, and Furu Wei.Kosmos-2: Grounding multimodal large language models to the world.arXiv preprint arXiv:2306.14824, 2023.
* [106]Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach.Sdxl: Improving latent diffusion models for high-resolution image synthesis.arXiv preprint arXiv:2307.01952, 2023.
* [107]Yujia Qin, Shengding Hu, Yankai Lin, Weize Chen, Ning Ding, Ganqu Cui, Zheni Zeng, Yufei Huang, Chaojun Xiao, Chi Han, Yi Ren Fung, Yusheng Su, Huadong Wang, Cheng Qian, Runchu Tian, Kunlun Zhu, Shihao Liang, Xingyu Shen, Bokai Xu, Zhen Zhang, Yining Ye, Bowen Li, Ziwei Tang, Jing Yi, Yuzhang Zhu, Zhenning Dai, Lan Yan, Xin Cong, Yaxi Lu, Weilin Zhao, Yuxiang Huang, Junxi Yan, Xu Han, Xian Sun, Dahai Li, Jason Phang, Cheng Yang, Tongshuang Wu, Heng Ji, Zhiyuan Liu, and Maosong Sun.Tool learning with foundation models, 2023.
* [108]Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.Exploring the limits of transfer learning with a unified text-to-text transformer.The Journal of Machine Learning Research, 21(1):5485–5551, 2020.
* [109]John C Raven and JH Court.Raven’s progressive matrices.Western Psychological Services Los Angeles, 1938.
* [110]Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.High-resolution image synthesis with latent diffusion models.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10684–10695, 2022.
* [111]Victor Sanh, Albert Webson, Colin Raffel, Stephen Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Arun Raja, Manan Dey, et al.Multitask prompted training enables zero-shot task generalization.In International Conference on Learning Representations, 2021.
* [112]Timo Schick, Jane Dwivedi-Yu, Roberto Dessì, Roberta Raileanu, Maria Lomeli, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom.Toolformer: Language models can teach themselves to use tools.arXiv preprint arXiv:2302.04761, 2023.
* [113]Zhenwei Shao, Zhou Yu, Meng Wang, and Jun Yu.Prompting large language models with answer heuristics for knowledge-based visual question answering.In CVPR, pages 14974–14983, 2023.
* [114]Yongliang Shen, Kaitao Song, Xu Tan, Dongsheng Li, Weiming Lu, and Yueting Zhuang.Hugginggpt: Solving ai tasks with chatgpt and its friends in huggingface.arXiv preprint arXiv:2303.17580, 2023.
* [115]Weijia Shi, Sewon Min, Michihiro Yasunaga, Minjoon Seo, Rich James, Mike Lewis, Luke Zettlemoyer, and Wen-tau Yih.Replug: Retrieval-augmented black-box language models.arXiv preprint arXiv:2301.12652, 2023.
* [116]Noah Shinn, Beck Labash, and Ashwin Gopinath.Reflexion: an autonomous agent with dynamic memory and self-reflection.arXiv preprint arXiv:2303.11366, 2023.
* [117]Aleksandar Shtedritski, Christian Rupprecht, and Andrea Vedaldi.What does clip know about a red circle? visual prompt engineering for vlms.arXiv preprint arXiv:2304.06712, 2023.
* [118]Oleksii Sidorov, Ronghang Hu, Marcus Rohrbach, and Amanpreet Singh.Textcaps: a dataset for image captioning with reading comprehension.In ECCV, pages 742–758, 2020.
* [119]Amanpreet Singh, Vivek Natarajan, Meet Shah, Yu Jiang, Xinlei Chen, Dhruv Batra, Devi Parikh, and Marcus Rohrbach.Towards vqa models that can read.In CVPR, 2019.
* [120]Weijie Su, Xizhou Zhu, Yue Cao, Bin Li, Lewei Lu, Furu Wei, and Jifeng Dai.Vl-bert: Pre-training of generic visual-linguistic representations.In ICLR, 2019.
* [121]Dídac Surís, Sachit Menon, and Carl Vondrick.Vipergpt: Visual inference via python execution for reasoning.arXiv preprint arXiv:2303.08128, 2023.
* [122]Hao Tan and Mohit Bansal.Lxmert: Learning cross-modality encoder representations from transformers.In EMNLP, 2019.
* [123]Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, et al.Llama: Open and efficient foundation language models.arXiv preprint arXiv:2302.13971, 2023.
* [124]Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal.Interleaving retrieval with chain-of-thought reasoning for knowledge-intensive multi-step questions.arXiv preprint arXiv:2212.10509, 2022.
* [125]Maria Tsimpoukelli, Jacob L Menick, Serkan Cabi, SM Eslami, Oriol Vinyals, and Felix Hill.Multimodal few-shot learning with frozen language models.Advances in Neural Information Processing Systems, 34:200–212, 2021.
* [126]Carven Von Bearnensquash.Paper gestalt.Secret Proceedings of Computer Vision and Pattern Recognition (CVPR), 2010.
* [127]Hong Wang, Xuan Luo, Weizhi Wang, and Xifeng Yan.Bot or human? detecting chatgpt imposters with a single question.arXiv preprint arXiv:2305.06424, 2023.
* [128]Jianfeng Wang, Zhengyuan Yang, Xiaowei Hu, Linjie Li, Kevin Lin, Zhe Gan, Zicheng Liu, Ce Liu, and Lijuan Wang.Git: A generative image-to-text transformer for vision and language.Transactions on Machine Learning Research, 2022.
* [129]Peng Wang, An Yang, Rui Men, Junyang Lin, Shuai Bai, Zhikang Li, Jianxin Ma, Chang Zhou, Jingren Zhou, and Hongxia Yang.Ofa: Unifying architectures, tasks, and modalities through a simple sequence-to-sequence learning framework.In International Conference on Machine Learning, pages 23318–23340. PMLR, 2022.
* [130]Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou.Self-consistency improves chain of thought reasoning in language models.In The Eleventh International Conference on Learning Representations, 2022.
* [131]Zhenhailong Wang, Manling Li, Ruochen Xu, Luowei Zhou, Jie Lei, Xudong Lin, Shuohang Wang, Ziyi Yang, Chenguang Zhu, Derek Hoiem, et al.Language models with image descriptors are strong few-shot video-language learners.In Advances in Neural Information Processing Systems.
* [132]Zirui Wang, Jiahui Yu, Adams Wei Yu, Zihang Dai, Yulia Tsvetkov, and Yuan Cao.Simvlm: Simple visual language model pretraining with weak supervision.In ICLR, 2022.
* [133]David Wechsler.Wais-r: Manual: Wechsler adult intelligence scale-revised.(No Title), 1981.
* [134]Jason Wei, Maarten Bosma, Vincent Y Zhao, Kelvin Guu, Adams Wei Yu, Brian Lester, Nan Du, Andrew M Dai, and Quoc V Le.Finetuned language models are zero-shot learners.In ICLR, 2022.
* [135]Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, et al.Emergent abilities of large language models.arXiv preprint arXiv:2206.07682, 2022.
* [136]Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Fei Xia, Ed Chi, Quoc V Le, Denny Zhou, et al.Chain-of-thought prompting elicits reasoning in large language models.Advances in Neural Information Processing Systems, 35:24824–24837, 2022.
* [137]Chenfei Wu, Shengming Yin, Weizhen Qi, Xiaodong Wang, Zecheng Tang, and Nan Duan.Visual chatgpt: Talking, drawing and editing with visual foundation models.arXiv preprint arXiv:2303.04671, 2023.
* [138]Jialian Wu, Jianfeng Wang, Zhengyuan Yang, Zhe Gan, Zicheng Liu, Junsong Yuan, and Lijuan Wang.Grit: A generative region-to-text transformer for object understanding.arXiv preprint arXiv:2212.00280, 2022.
* [139]Zhenjia Xu, Zhijian Liu, Chen Sun, Kevin Murphy, William T Freeman, Joshua B Tenenbaum, and Jiajun Wu.Unsupervised discovery of parts, structure, and dynamics.In International Conference on Learning Representations, 2018.
* [140]Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Faisal Ahmed, Zicheng Liu, Yumao Lu, and Lijuan Wang.Unitab: Unifying text and box outputs for grounded vision-language modeling.In European Conference on Computer Vision, pages 521–539. Springer, 2022.
* [141]Zhengyuan Yang, Zhe Gan, Jianfeng Wang, Xiaowei Hu, Yumao Lu, Zicheng Liu, and Lijuan Wang.An empirical study of gpt-3 for few-shot knowledge-based vqa.In Proceedings of the AAAI Conference on Artificial Intelligence, volume 36, pages 3081–3089, 2022.
* [142]Zhengyuan Yang, Linjie Li, Jianfeng Wang, Kevin Lin, Ehsan Azarnasab, Faisal Ahmed, Zicheng Liu, Ce Liu, Michael Zeng, and Lijuan Wang.Mm-react: Prompting chatgpt for multimodal reasoning and action.arXiv preprint arXiv:2303.11381, 2023.
* [143]Zhengyuan Yang, Jianfeng Wang, Zhe Gan, Linjie Li, Kevin Lin, Chenfei Wu, Nan Duan, Zicheng Liu, Ce Liu, Michael Zeng, et al.Reco: Region-controlled text-to-image generation.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 14246–14255, 2023.
* [144]Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Thomas L Griffiths, Yuan Cao, and Karthik Narasimhan.Tree of thoughts: Deliberate problem solving with large language models.arXiv preprint arXiv:2305.10601, 2023.
* [145]Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao.React: Synergizing reasoning and acting in language models.In The Eleventh International Conference on Learning Representations, 2022.
* [146]Qinghao Ye, Haiyang Xu, Guohai Xu, Jiabo Ye, Ming Yan, Yiyang Zhou, Junyang Wang, Anwen Hu, Pengcheng Shi, Yaya Shi, et al.mplug-owl: Modularization empowers large language models with multimodality.arXiv preprint arXiv:2304.14178, 2023.
* [147]Da Yin, Liunian Harold Li, Ziniu Hu, Nanyun Peng, and Kai-Wei Chang.Broaden the vision: Geo-diverse visual commonsense reasoning.arXiv preprint arXiv:2109.06860, 2021.
* [148]Rowan Zellers, Yonatan Bisk, Ali Farhadi, and Yejin Choi.From recognition to cognition: Visual commonsense reasoning.In CVPR, pages 6720–6731, 2019.
* [149]Andy Zeng, Maria Attarian, Krzysztof Marcin Choromanski, Adrian Wong, Stefan Welker, Federico Tombari, Aveek Purohit, Michael S Ryoo, Vikas Sindhwani, Johnny Lee, et al.Socratic models: Composing zero-shot multimodal reasoning with language.In The Eleventh International Conference on Learning Representations, 2022.
* [150]Chi Zhang, Feng Gao, Baoxiong Jia, Yixin Zhu, and Song-Chun Zhu.Raven: A dataset for relational and analogical visual reasoning.In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 5317–5327, 2019.
* [151]Zhuosheng Zhang, Aston Zhang, Mu Li, Hai Zhao, George Karypis, and Alex Smola.Multimodal chain-of-thought reasoning in language models.arXiv preprint arXiv:2302.00923, 2023.
* [152]Yan-Tao Zheng, Ming Zhao, Yang Song, Hartwig Adam, Ulrich Buddemeier, Alessandro Bissacco, Fernando Brucher, Tat-Seng Chua, and Hartmut Neven.Tour the world: building a web-scale landmark recognition engine.In 2009 IEEE Conference on Computer Vision and Pattern Recognition, pages 1085–1092. IEEE, 2009.
* [153]Bolei Zhou, Aditya Khosla, Agata Lapedriza, Aude Oliva, and Antonio Torralba.Learning deep features for discriminative localization.In CVPR, 2016.
* [154]Bolei Zhou, Agata Lapedriza, Antonio Torralba, and Aude Oliva.Places: An image database for deep scene understanding.Journal of Vision, 17(10):296–296, 2017.
* [155]Luowei Zhou, Hamid Palangi, Lei Zhang, Houdong Hu, Jason J Corso, and Jianfeng Gao.Unified vision-language pre-training for image captioning and vqa.In AAAI, 2020.
* [156]Yongchao Zhou, Andrei Ioan Muresanu, Ziwen Han, Keiran Paster, Silviu Pitis, Harris Chan, and Jimmy Ba.Large language models are human-level prompt engineers.In The Eleventh International Conference on Learning Representations, 2022.
* [157]Deyao Zhu, Jun Chen, Xiaoqian Shen, Xiang Li, and Mohamed Elhoseiny.Minigpt-4: Enhancing vision-language understanding with advanced large language models.arXiv preprint arXiv:2304.10592, 2023.
* [158]Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Harkirat Behl, Jianfeng Wang, Lu Yuan, et al.Generalized decoding for pixel, image, and language.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 15116–15127, 2023.
* [159]Xueyan Zou, Zi-Yi Dou, Jianwei Yang, Zhe Gan, Linjie Li, Chunyuan Li, Xiyang Dai, Jianfeng Wang, Lu Yuan, Nanyun Peng, Lijuan Wang, Yong Jae Lee, and Jianfeng Gao.Generalized decoding for pixel, image and language.2022.
* [160]Xueyan Zou, Jianwei Yang, Hao Zhang, Feng Li, Linjie Li, Jianfeng Gao, and Yong Jae Lee.Segment everything everywhere all at once.arXiv preprint arXiv:2304.06718, 2023.
