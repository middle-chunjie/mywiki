Fantastic Copyrighted Beasts and  How (Not) to Generate Them
============================================================

Luxi He∗♠ Yangsibo Huang∗♠ Weijia Shi∗†  
Tinghao Xie♠ Haotian Liu∘ Yue Wang◇ Luke Zettlemoyer†  
Chiyuan Zhang Danqi Chen♠ Peter Henderson♠  
  
♠Princeton University†University of Washington  
∘University of Wisconsin-Madison◇University of Southern California

###### Abstract

††footnotetext: ∗Equal contribution. Emails for leading authors: luxihe@cs.princeton.edu, yangsibo@princeton.edu, and swj0419@uw.edu. Our code is available at [https://github.com/princeton-nlp/CopyCat](https://github.com/princeton-nlp/CopyCat "").

Recent studies show that image and video generation models can be prompted to reproduce copyrighted content from their training data, raising serious legal concerns about copyright infringement. Copyrighted characters (e.g., Mario, Batman) present a significant challenge: at least one lawsuit has already awarded damages based on the generation of such characters. Consequently, commercial services like DALL·E have started deploying interventions.
However, little research has systematically examined these problems: (1) Can users easily prompt models to generate copyrighted characters, even if it is unintentional?; (2) How effective are the existing mitigation strategies?
To address these questions, we introduce a novel evaluation framework with metrics that assess both the generated image’s similarity to copyrighted characters and its consistency with user intent, grounded in a set of popular copyrighted characters from diverse studios and regions.
We show that state-of-the-art image and video generation models can still generate characters even if characters’ names are *not* explicitly mentioned, sometimes with only two generic keywords (e.g., prompting with “videogame, plumber” consistently generates Nintendo’s Mario character).
We also introduce semi-automatic techniques to identify such keywords or descriptions that trigger character generation. Using this framework, we evaluate mitigation strategies, including prompt rewriting and new approaches we propose. Our findings reveal that common methods, such as DALL·E’s prompt rewriting, are insufficient alone and require supplementary strategies like negative prompting.
Our work provides empirical grounding for discussions on copyright mitigation strategies and offers actionable insights for model deployers implementing these safeguards.

1 Introduction
--------------

State-of-the-art image and video generation models demonstrate a remarkable ability for generating high-quality visual content based on free-form user inputs *(Rombach et al., [2022]; Betker et al., [2023]; Chen et al., [2024]; Li et al., [2024a]; Blattmann et al., [2023]; Esser et al., [2024])*.
However, recent research has shown that generative models, including those for image and video, are susceptible to memorizing and generating entire datapoints or concepts from their training data*(Somepalli et al., [2023]; Carlini et al., [2023]; car, [2023])*.
Since some training data originates from copyrighted materials*(car, [2023]; Kumari et al., [2023])*, regurgitation of such content may lead to legal intellectual property liability for users and model deployers who further make use of the generated content.
In particular, this liability may stem not only from verbatim generation of training data points, but generation of some copyrightable repeating motifs highly similar to those from the training data.

As a result, copyright concerns in image generative models has been extensively discussed in both academic research*(car, [2023]; Ma et al., [2024]; Kim et al., [2024])* and litigation*(Vincent, [2023]; *Andersen et al. v. Stability AI et al.*, [N.D. Cal. 2023])*. Among the diverse copyrighted content that models may generate, copyrighted characters pose a unique *legal challenge* *(Sag, [2023]; Henderson et al., [2023]; Lee et al., [2024])*, since characters are a type of “abstractly” protected content *(Sag, [2023])* Copyrighted characters also pose a unique *technical challenge*, as they are like general concepts that can appear in many poses, sizes, and variations, making it challenging to apply many previous methods designed for (near) verbatim memorization. As a result, commercial platforms like DALL·E have started implementing tailored measures, such as prompt rewriting, with the goal of preventing the generation of copyrighted characters.


To address these questions, we present novel evaluation metrics that measure both the generated image’s resemblance to copyrighted characters and its consistency with user input. We use a diverse set of 50 popular characters from various studios and regions as the basis for our analysis. We use this framework to evaluate five image generation models – Playground v2.5*(Li et al., [2024a])*, Stable Diffusion XL*(Podell et al., [2024])*, PixArt-$\alpha$*(Chen et al., [2024])*, DeepFloyd IF*(StabilityAI, [2023])*, and DALL·E 3*(Betker et al., [2023])* – along with one video generation model, VideoFusion*(Luo et al., [2023])*, using a diverse set of popular copyrighted characters ([§ 2] to Generate Them")).

<img src='x1.png' alt='Refer to caption' title='' width='761' height='332' />

*(a) Target copyrighted character: Mario*

<img src='x2.png' alt='Refer to caption' title='' width='761' height='331' />

*(b) Target copyrighted character: Batman*

*Figure 1: Examples of copyrighted characters generated by the open-source Playground v2.5 model*(Li et al., [2024a])* and proprietary DALL·E 3 model. For Mario (a) and Batman (b), both models can generate these characters through *indirect anchoring*, using relevant descriptive keywords instead of character names. DALL·E 3 blocks explicit name prompts requests (*character name anchoring*) with content policy messages.*



Our findings suggest that generating copyrighted characters, whether intentionally or unintentionally, is relatively easy, and existing mitigation strategies are largely ineffective at preventing this. We thereby summarize the key takeaways for users and model deployers below and hope this work encourages further research in this area.

* •

    We call for more awareness of indirect anchoring, where models can generate copyrighted characters without explicitly mentioning the character’s name. Deployers should be cautious, as this could bypass safeguards that depend on direct name detection. Users should also be aware that even using generic keywords to unintentionally generate these characters may lead to legal liability.

* •

    For model deployers who adopt mitigation strategies and intend to prevent the generation of copyrighted characters, relying solely on prompt rewriting is insufficient. We recommend combining prompt rewriting with negative prompts as a more effective approach.

2 An Evaluation Framework for Copyrighted Character Generation
--------------------------------------------------------------

In this section, we describe an evaluation framework to quantify copyrighted character generation and efficacy of a mitigation strategy by combining a metric for *copyright protection* and *consistency with user intent*, respectively. Even though our main focus is copyright compliance (e.g., avoiding the generation of specific copyrighted characters like Mario), consistency with user intent is equally important (e.g., if the user requests a plumber, still generating a plumber). A balance between the two factors is usually implicitly considered in real-world intervention strategies such as prompt rewriting (see [§ 4] to Generate Them")). To explicitly characterize such inherent trade-off, we define the following metrics.

Copyright protection. As legal judgments of copyright infringement are usually multifaceted and made on a case-by-case basis (see [§ 1] to Generate Them") and [§ A] to Generate Them")), it is legally infeasible to have a universal quantitative definition of infringement. Nonetheless, many jurisdictions examine whether two characters are substantially similar *(Sag, [2023])*. Therefore, we study a highly relevant subquestion that can be better quantified: “Is the generated character so similar as to be recognized as an existing copyrighted character?” A “YES” is closely associated with a higher probability of a generation being considered infringing in many jurisdictions.  Specifically, for a copyrighted character $C$ and a corresponding generated image $\mathcal{I}\=f_{p,m}(C)$, where $f(\cdot)$ is the generation model, $p$ is the given prompt, and $m$ is mitigation (if any), the detector outputs $d(C,\mathcal{I})\in{0,1}$, indicating the presence (1) or absence (0) of character $C$ in image $\mathcal{I}$. We then calculate the metric $\mathsf{DETECT}$ by

|  | $\mathsf{DETECT}(f,p,m)\=\sum_{C\in\mathcal{D}}d(C,f_{p,m}(C)),$ |  |
| --- | --- | --- |

which sums the binary detection scores across a character list $\mathcal{D}$ we study. A lower $\mathsf{DETECT}$ score indicates that fewer copyrighted characters were generated. Ideally, a panel of humans would annotate all outputs, but this does not scale. We use GPT-4V as an automatic evaluator for its high accuracy (82.5%) and correlation with humans. We include detector choice ablation and human evaluation of copyrighted character detection in [§ C.3] to Generate Them").

Consistency with user intent. On the other hand, if a model always outputs a random image or rejects requests, it minimizes similarity to copyrighted characters but fails to meet user intent. We quantify consistency between generation and intent as a proxy for user satisfaction, testing whether key prompt characteristics appear in the image. The assumption is that as long as general traits (e.g., a “cartoon mouse”) are present, users may still be satisfied despite alterations. This evaluation reflects how model creators balance legal risks with user experience: tools like DALL·E still respond to user requests but attempt (sometimes unsuccessfully)
to steer away from exact recreations of copyrighted characters. Assuming users seek specific characters, we can generate ground-truth characteristics from the target character lists. For each copyrighted character $C$, we ask GPT-4 to automatically identify its main general characteristics $s(C)$, which we manually verify and adjust if necessary (e.g., “cartoon mouse” for Mickey Mouse). We then use VQAScore*(Lin et al., [2024])* to measure the consistency between image $\mathcal{I}$ and characteristics $s(C)$, defined as $c(s(C),\mathcal{I})\=\mathbb{P}(\texttt{``Yes''}|\mathcal{I},``\texttt{Does %
this figure show}~{}s(C)\texttt{? Please answer yes or no.}")$. For example, we calculate $\mathbb{P}(\texttt{``Yes''}|\mathcal{I},``\texttt{Does this figure show a %
cartoon mouse}\texttt{? Please answer yes or no.}")$ when the character is Mickey Mouse. The consistency metric $\mathsf{CONS}$ for a model $f$, input prompt $p$, and intervention $m$ is the average consistency score across the list of characters $\mathcal{D}$ we study:

|  | $\mathsf{CONS}(f,p,m)\=\frac{1}{|\mathcal{D}|}\sum_{C\in\mathcal{D}}c(s(C),f_{p,% m}(C)).$ |  |
| --- | --- | --- |

A higher $\mathsf{CONS}$ indicates better consistency with user intent. We show that the $\mathsf{CONS}$ evaluator has high agreement with humans in [§ C.4] to Generate Them"). We note that current definitions and metrics only capture some aspects of consistency with user requests, and future work can further improve upon this definition.

When evaluating what kind of prompt could trigger the generation of a copyrighted character in an *unprotected* system, we focus on $\mathsf{DETECT}$. When evaluating a *protected* system and intervention strategy, the trade-off becomes relevant. In this case, we consider both metrics. An effective intervention strategy $m$ should aim to minimize $\mathsf{DETECT}$ while maximizing $\mathsf{CONS}$. We omit $(f,p,m)$ for $\mathsf{DETECT}$ and $\mathsf{CONS}$ in the paper if they are clear from the context.

We apply this evaluation framework to a set of 50 diverse copyrighted characters, denoted as $\mathcal{D}$ in the paper, to obtain quantitative evaluation on character name, indirect anchoring, and mitigation strategies. They are sourced from popular studios and franchises, both U.S. and international. We cover characters from superhero movies (e.g., Batman, Iron Man, Hulk), animations (e.g., Lightning McQueen, Monkey D. Luffy, Elsa), and video games (e.g., Mario, Pikachu, Link), among others. Full list and details of the curated characters are in [§ B] to Generate Them"). We provide additional experiments in [§ C.3] to Generate Them") to show that the evaluator can generalize to a larger set of characters.

In the following sections, we discuss how we apply the evaluation framework to better understand different modes of copyrighted character generation and the effectiveness of mitigation strategies.

3 Identifying Indirect Anchors
------------------------------

Not surprisingly, prompting with “Mario” would likely generate this Nintendo character. We refer to this type of generation as *Character Name Anchoring*. However, if users ask for a generic “video game plumber” they will also receive the iconic character’s likeness from most models (Figure [1(a)] ‣ Fig. 1 ‣ 1 Introduction ‣ Fantastic Copyrighted Beasts and How (Not) to Generate Them")). We refer to this mode of generation, using keywords or descriptions without the character’s name, as *Indirect Anchoring*. We propose a systematic method to identify such indirect anchors, revealing their widespread role in generating copyrighted characters ([§ 5.1] to Generate Them")) and informing strategies using indirect anchors to mitigate copyright risks ([§ 5.2] to Generate Them")). Our two-stage approach first generates candidate descriptions and keywords for a character, then reranks them to identify the most effective triggers.

Generation. First, we use GPT-4 to generate a set of candidate descriptions and keywords pertaining to the visual appearance of the characters we apply our evaluation framework on, using the prompting template in [C.5] to Generate Them").†††The textual descriptions are around 60 words in length. This length limit provides maximal descriptive information while keeping under the 77 token limit for stable diffusion models *(Urbanek et al., [2023])*.

Ranking. Given the generated candidates, we use different ranking methods to investigate what prompts most likely trigger a character generation, even when the character’s name is not present.

*Algorithm 1  EmbeddingSim Ranking*

1:Character name $C$, $n$ candidate words $\mathcal{W}\={w_{i}}_{i\in[n]}$, text encoder $g$

2:foreach $w_{i}$ in $\mathcal{W}$do

3:Encode $w_{i}$ to $g(w_{i})$ using $g$

4:$s_{w_{i}}\leftarrow{g(C)\cdot g(w_{i})}/{\|g(C)\|\|g(w_{i})\|}$

5:end for

6:Sort $\mathcal{W}$ by $s_{w_{i}}$ in descending order

7:return Sorted $\mathcal{W}$

*Algorithm 2  Co-Occurrence Ranking*

1:Character name $C$, $n$ candidate words $\mathcal{W}\={w_{i}}_{i\in[n]}$, training corpora $\mathcal{D}$

2:foreach document $d$ in $\mathcal{D}$do

3: if$C$ and $w_{i}$ co-occur in $d$then $s_{w_{i}}\leftarrow s_{w_{i}}+1$

4: end if

5:end for

6:Sort $\mathcal{W}$ by $s_{w_{i}}$ in descending order

7:return Sorted $\mathcal{W}$

* •

    EmbeddingSim: We leverage embedding space similarity to rerank and obtain the top $k$ indirect anchor candidates, which can be descriptions or keywords. The algorithm is illustrated in [Algorithm 1] to Generate Them"), and is applicable for both descriptions and keywords. Specifically, for each character name and candidate word, we use the text encoder of the image generation model to calculate their textual embeddings. We then rank candidate keywords by their embedding’s cosine similarity with the character name embedding, computed as the averaged token embedding at the last hidden layer. We hypothesize that keywords with embeddings more similar to the character’s name may incline the model to generate that character.

* •

    Co-Occurrence: For keywords, we can also rank by their *co-occurrence with the character’s name* in popular training corpora (see [Algorithm 2] to Generate Them")). We hypothesize that models learn to associate characters with words commonly found in their descriptions or references, turning these seemingly generic adjectives into anchoring words for specific characters. We examine common training corpora, including captions from image-captioning datasets: LAION-2B*(Schuhmann et al., [2022])*), as well as text-only datasets (C4*(Raffel et al., [2020])*, OpenWebText*(Radford et al., [2019])*, and The Pile*(Gao et al., [2020])*. We follow the indexing and search procedure discussed in *Elazar et al. ([2023])* to rank and select keywords.

* •

    LM-Ranked: For keywords, we also obtain an inherently LM-ranked list as a baseline for comparison. This is achieved by obtaining the top $k$ keywords associated with certain characters using greedy decoding, based on the prompt template provided in [§ C] to Generate Them"). Note that the LM may generate words *not* present in the candidate list, but we maintain $k$ as the same for a fair comparison between LM-Ranked, EmbeddingSim, and Co-Occurrence.

While we focus on keywords re-ranking in later parts of this paper as they provide valuable information for the design of mitigation strategy, we also include relevant analysis on descriptions in [§ E.3] to Generate Them"). We provide examples of such keywords and description in [Fig. 7] to Generate Them") in [§ C.5] to Generate Them").

4 Mitigation Strategies
-----------------------

We first discuss known mitigation strategies adopted by current producion-level image generation services. We then propose new mitigation strategies, especially leveraging negative prompts, that can improve upon current implementation.

Prompt rewriting is an existing mitigation used in production-level systems such as DALL·E. Specifically, the DALL·E interface contains a prompt-rewriting step that first processes the user’s text input into a format that DALL·E can use to generate images and comply with OpenAI’s policies, such as avoiding copyrighted content. Prompt rewriting approaches currently adopted by companies reflect our evaluation framework: they encourage policy compliance (e.g., avoiding the generation of specific copyrighted characters like Mario) while staying close to the user’s intent (e.g., if the user requests a plumber, still generating a plumber). In order to simulate the prompt-rewriting pipeline, we query GPT-4 with the DALL·E’s system prompt obtained via prompt extraction methods (see full template and details in [§ C.6] to Generate Them")) and the keywords or descriptions to be rewritten.


Prompt-rewriting changes short prompts (e.g., one-word character name) most significantly, transforming them into a longer descriptive prompt that adds modification in order to create a more generic output. At a high level, such intervention is compromising faithfulness of certain visual aspects for copyright protection. The exact features to be prioritized or de-prioritized can be customized in the rewriting instructions.

Negative prompts are often used in deployed diffusion model deployments *(Playground AI, [2023])* to allow users to exclude undesired concepts or elements from the generated output. Negative prompts are incorporated through classifier-free guidance during the decoding process *(Ho \& Salimans, [2021])*. For example, the official prompt guide from Playground suggests using phrases like “ugly, deformed hands” to discourage unwanted aesthetics.‡‡‡[https://playground.com/prompt-guide/negative-prompts](https://playground.com/prompt-guide/negative-prompts "") Despite their utility, negative prompts are currently under-studied as a means to exclude specific copyrighted elements from generated outputs.

We test negative prompts as a mitigation strategy based on the important anchoring keywords selected via our methods in [§ 3] to Generate Them"). Specifically, negative prompts are “Copyrighted character” or specific target’s name paired with one of the following options: 1) $k$ LM-Ranked keywords; 2) $k$ EmbeddingSim keywords; 3) $k$ Co-Occurrence keywords; 4) $k$ EmbeddingSim + $k$ Co-Occurrence keywords.
In addition,further combine prompt rewriting and negative prompts to reduce the likelihood of generating copyright characters.

5 Experiments and Analysis
--------------------------

This section presents our empirical results, where we seek to answer the following two key questions:

* •

    Which method introduced in [§ 3] to Generate Them") most effectively identifies indirect anchors ([§ 5.1] to Generate Them"))?

* •

    How effective are the mitigation strategies discussed in [§ 4] to Generate Them"), namely prompt rewriting and negative prompting, in reducing the generation of copyrighted characters ([§ 5.2] to Generate Them"))?

#### Experimental setup.

To ensure a clear understanding and better control over model behaviors, our evaluation primarily focuses on four state-of-the-art open-source image generation models: Playground v2.5*(Li et al., [2024a])*, Stable Diffusion XL (SDXL)*(Podell et al., [2024])*, PixArt-$\alpha$*(Chen et al., [2024])*, and DeepFloyd IF*(StabilityAI, [2023])*, DALL·E 3*Betker et al. ([2023])*, as well as one video generation model, VideoFusion*(Luo et al., [2023])*.§§§Video generation pipelines can be broadly categorized into the two types: 1) image generation model followed by an image-to-video model, and 2) a direct text-to-video pipeline. For models in the first category (eg. Stable Video Diffusion *(Blattmann et al., [2023])*), our findings on image generation models are also applicable. Therefore, we focus our video experiments on models of the second type, e.g. VideoFusion*(Luo et al., [2023])*. The configuration details for each model used in our experiments can be found in [§ C.2] to Generate Them"). Our main analysis focuses on the Playground v2.5 due to its superior generation quality. We also report results for other models in [§ E.5] to Generate Them").

### 5.1 Identifying Prompts That Generate Copyrighted Characters

<img src='x3.png' alt='Refer to caption' title='' width='747' height='173' />

*(a) Prompt: Character’s name*

<img src='x4.png' alt='Refer to caption' title='' width='747' height='172' />

*(b) Prompt: Descriptions (w/o character’s name)*

<img src='x5.png' alt='Refer to caption' title='' width='747' height='174' />

*(c) Prompt: 5 Co-Occurrence-LAION keywords (w/o character’s name)*

*Figure 2: Selection of generated images by Playground v2.5 that GPT-4V detects as the requested characters. As shown, the model is able to generate images that look highly similar to the required character with (a) or w/o the character’s name in the prompt (b, c).*

First, not too surprisingly, we have verified that when using character names, ~60% of tested characters can be generated, also highlighted in the upper-left of [Tb. 1] to Generate Them").¶¶¶However, we find that models are not robust to misspellings of character names and generally do not result in generation of characters even with minor misspellings, see [§ E.4] to Generate Them"). For the remainder of this section, we focus on indirect anchoring, where the prompt does not explicitly contain the character’s name. We examine the effect of two types of indirect anchors: textual descriptions and keywords, as well as how to automatically discover them ([§ 3] to Generate Them")), by checking $\mathsf{DETECT}$, the number of detected copyrighted characters in the generation.

<img src='x6.png' alt='Refer to caption' title='' width='830' height='216' />

*Figure 3: Number of characters detected using different top keywords ranked by various methods on (a) image generation and (b) video generation models. Ranking keywords based on their co-occurrence with the character’s name in the LAION corpus is the most effective and could generate more characters than using a 60-word description when only 20 keywords are used.*

60-word descriptions lead to the generation of ~48% characters. Despite omitting character names, these descriptions often lead to successful character generation, as shown in the horizontal dashed line in [Fig. 3] to Generate Them").
Furthermore, prompts with higher embedding similarity to a character’s name tend to generate that character more reliably (see [§ E.3] to Generate Them")). Among 100 randomly generated 60-word descriptions per character, the top-ranked description by embedding similarity generates 24 characters successfully, versus only 16 for the bottom-ranked (see [§ E.3] to Generate Them")).

A few keywords, especially those with most frequent co-occurrence with character names in LAION, also easily generate copyrighted characters. We examine the effectiveness of keywords with top co-occurrence frequency with the copyrighted characters’ names ([§ 3] to Generate Them")) and visualize results in [Fig. 3] to Generate Them"). For most selections, we find that keywords chosen from LAION are more effective than using other methods. This is likely because this multimodal dataset is more common in training of image generation models compared to the other text-only ones. Notably, using 5 LAION keywords can almost match performance of using 60-word descriptions. Top 20 LAION and embedding-ranked keywords can both generate more copyrighted characters than using the more detailed paragraph descriptions. [Fig. 2] to Generate Them") shows some examples of these generated images with the descriptions and keywords discussed above.

<img src='x7.png' alt='Refer to caption' title='' width='761' height='230' />

*(a) DALL·E 3, 60-word description*

<img src='x8.png' alt='Refer to caption' title='' width='761' height='230' />

*(b) VideoFusion, 5 Co-Occurrence-LAION keywords*

*Figure 4: Example of copyrighted characters generated using (a) 60-word description with DALL·E 3, and (b) five keywords from LAION with the VideoFusion*(Luo et al., [2023])*. The video generation model also generates watermarks in its output.*

Descriptions and identified keywords also transfer to generating characters from DALL·E 3 and video models. We further test indirect anchors on production-level models, such as DALL·E 3. Surprisingly, some indirect anchors like descriptions can still bypass system safeguards and result in the generation of copyrighted characters ([Fig. 4] to Generate Them")). This further suggests that current safeguards are not fully effective. More results can be found in [§ D] to Generate Them"). In addition, we also test indirect anchors on the video generation model VideoFusion*(Luo et al., [2023])* (see [Fig. 4] to Generate Them") for examples). We compare selection methods for indirect anchors in [Fig. 3] to Generate Them"). LAION is the most useful corpus for identifying such keywords, and has a smaller gap to 60-word description on video generation compared to image generation.

### 5.2 Mitigation Effectiveness

The next question is: can we effectively prevent the models from recreating these copyrighted characters? We mainly evaluate the intervention strategies discussed in [§ 4] to Generate Them"), specifically: 1) using prompt rewriting only, 2) using negative prompts only,∥∥∥To effectively apply the proposed negative prompts, model deployers need a mechanism to detect the identity of the intended copyrighted character (if any) from the user’s prompt. As the primary focus of this work is not end-to-end system building but the evaluation of specific mitigation methods, we assume the existence of such a method. However, we provide more discussion on this in [§ E.2] to Generate Them") and demonstrate two possible implementations for detecting whether a prompt may reference (directly or indirectly) a popular character. and 3) combining negative prompts and prompt rewriting.

We evaluate these strategies using the evaluation framework as described in [§ 2] to Generate Them"): $\mathsf{DETECT}$ counts the number of detected copyrighted characters, and $\mathsf{CONS}$ measures the image’s consistency with user input. A good mitigation strategy achieves low $\mathsf{DETECT}$ and high $\mathsf{CONS}$. We run each strategy three times and report the mean and standard deviation of $\mathsf{DETECT}$ and $\mathsf{CONS}$ in [Tb. 1] to Generate Them").

Prompt rewriting alone is not entirely effective at eliminating outputs similar to copyrighted characters. Our evaluation starts with simulating prompt rewriting ([§ 4] to Generate Them")), which has been adopted as an intervention strategy for production-level models like DALL·E. However, as demonstrated in [Tb. 1] to Generate Them"), solely adopting prompt rewriting can only reduce $\mathsf{DETECT}$ from 30 to 14. Nonetheless, an advantage of prompt rewriting is that the $\mathsf{CONS}$ scores modestly improve, likely due to the rewritten prompts containing more detailed information.

*Table 1: Performance of all intervention strategies on the Playground v2.5 model. We run each strategy three times, and report the mean and standard deviation of the number of detected copyrighted characters ($\mathsf{DETECT}$, lower is better) and the consistency with user intent ($\mathsf{CONS}$, higher is better). Including the character’s name in the negative prompts is highly important for reducing $\mathsf{DETECT}$. Combining prompt rewriting and negative prompts can effectively reduce $\mathsf{DETECT}$ from 30 to 5, without significantly degrading $\mathsf{CONS}$.*

| Negative Prompt | Prompt: Target’s name | | Prompt: Rewritten prompt | |
| --- | --- | --- | --- | --- |
| | $\mathsf{DETECT}$ ($\downarrow$) | $\mathsf{CONS}$ ($\uparrow$) | $\mathsf{DETECT}$ ($\downarrow$) | $\mathsf{CONS}$ ($\uparrow$) |
| None | $30.33_{\pm 1.89}$ | $0.75_{\pm 0.01}$ | $14.33_{\pm 2.62}$ | $0.80_{\pm 0.01}$ |
| "Copyrighted character" | $30.33_{\pm 1.25}$ | $0.74_{\pm 0.01}$ | $17.33_{\pm 1.70}$ | $0.80_{\pm 0.01}$ |
| + 5 LM-Ranked keywords | $30.33_{\pm 1.89}$ | $0.71_{\pm 0.01}$ | $14.33_{\pm 1.70}$ | $0.80_{\pm 0.00}$ |
| + 5 EmbeddingSim keywords | $28.00_{\pm 1.41}$ | $0.72_{\pm 0.03}$ | $15.67_{\pm 1.25}$ | $0.80_{\pm 0.00}$ |
| + 5 Co-Occurrence-LAION keywords | $27.33_{\pm 0.00}$ | $0.73_{\pm 0.01}$ | $14.33_{\pm 0.94}$ | $0.80_{\pm 0.00}$ |
| + 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords | $23.33_{\pm 3.30}$ | $0.72_{\pm 0.03}$ | $7.00_{\pm 1.63}$ | $0.81_{\pm 0.00}$ |
| Target’s name | $23.67_{\pm 2.62}$ | $0.76_{\pm 0.01}$ | $7.67_{\pm 0.47}$ | $0.81_{\pm 0.01}$ |
| + 5 LM-Ranked keywords | $25.00_{\pm 1.63}$ | $0.74_{\pm 0.01}$ | $7.00_{\pm 1.63}$ | $0.81_{\pm 0.02}$ |
| + 5 EmbeddingSim keywords | $22.67_{\pm 2.36}$ | $0.73_{\pm 0.02}$ | $5.67_{\pm 0.47}$ | $0.80_{\pm 0.00}$ |
| + 5 Co-Occurrence-LAION keywords | $20.67_{\pm 2.05}$ | $0.75_{\pm 0.01}$ | $5.00_{\pm 0.82}$ | $0.81_{\pm 0.01}$ |
| + 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords | $20.67_{\pm 0.47}$ | $0.72_{\pm 0.03}$ | $4.33_{\pm 0.47}$ | $0.81_{\pm 0.00}$ |

*Table 2: The combination of prompt-rewriting and negative prompts (target’s name \& 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords, shown on the right half of the table) can significantly reduce $\mathsf{DETECT}$ while mostly preserving $\mathsf{CONS}$ compared to no-intervention baseline (left half of the table) across all 5 open-source models tested, making it a promising candidate for copyright-protection intervention.*

| Model | w/o Intervention | | w/ Prompt Rewriting \& Negative Prompt | |
| --- | --- | --- | --- | --- |
| | $\mathsf{DETECT}$ ($\downarrow$) | $\mathsf{CONS}$ ($\uparrow$) | $\mathsf{DETECT}$ ($\downarrow$) | $\mathsf{CONS}$ ($\uparrow$) |
| Playground v2.5(Li et al., [2024a]) | $30.33_{\pm 1.89}$ | $0.75_{\pm 0.01}$ | $4.33_{\pm 0.47}$ | $0.81_{\pm 0.00}$ |
| Stable Diffusion XL(Podell et al., [2024]) | $33.00_{\pm 1.00}$ | $0.73_{\pm 0.01}$ | $1.67_{\pm 0.94}$ | $0.77_{\pm 0.03}$ |
| PixArt-$\alpha$(Chen et al., [2024]) | $24.67_{\pm 0.58}$ | $0.79_{\pm 0.01}$ | $4.67_{\pm 0.47}$ | $0.79_{\pm 0.01}$ |
| DeepFloyd IF(StabilityAI, [2023]) | $33.67_{\pm 1.53}$ | $0.71_{\pm 0.01}$ | $2.00_{\pm 1.00}$ | $0.72_{\pm 0.01}$ |
| VideoFusion(Luo et al., [2023]) | $28.33_{\pm 1.89}$ | $0.68_{\pm 0.01}$ | $11.33_{\pm 1.53}$ | $0.76_{\pm 0.01}$ |

<img src='x9.png' alt='Refer to caption' title='' width='830' height='108' />

*(a) Prompt: Character’s name, Negative Prompt: None*

<img src='x10.png' alt='Refer to caption' title='' width='830' height='108' />

*(b) Prompt: Character’s name, Negative Prompt: Character’s name \& 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords*

<img src='x11.png' alt='Refer to caption' title='' width='830' height='108' />

*(c) Prompt: Rewritten, Negative Prompt: None*

<img src='x12.png' alt='Refer to caption' title='' width='830' height='108' />

*(d) Prompt: Rewritten, Negative Prompt: Character’s Name \& 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords*

*Figure 5: Images generated with Playground v2.5 using various prompt and negative prompt configurations. Prompt rewriting, combined with negative prompting, effectively reduces the likelihood of generating images that resemble copyrighted characters while ensuring the generated subjects align with the user’s intent (i.e., the main characteristics are preserved), as shown in (d).*

We then investigate potential reasons for prompt rewriting sometimes failing. Specifically, we calculate the average number of Top-5 LAION keywords present in rewritten prompts that result in $\mathsf{DETECT}\=0$ (success) and $\mathsf{DETECT}\=1$ (failure). We find that the failed rewritten prompts contain on average more LAION keywords—$0.667$ for failure cases compared to $0.387$ for success cases. Similarly, we also observe that failing rewritten prompts tend to share higher embedding similarity with the character’s name (see [§ E.3] to Generate Them")). This again suggests the existence of indirect anchors, and potentially their inclusion in rewritten prompts could impair this strategy.

Using negative prompts improves elimination of similar output with modest impact on consistency. In addition to existing countermeasures like prompt-rewriting, we also explore negative prompts ([§ 4] to Generate Them")). Specifically, we use keywords identified with different methods ([§ 4] to Generate Them"), with $k\=5$) as negative prompts. We generally observe that including Co-Occurrence-LAION results in higher reduction in $\mathsf{DETECT}$ compared to including LM-Ranked and EmbeddingSim ([Tb. 1] to Generate Them")).

Including character names in the negative prompt is also helpful. As shown in [Tb. 1] to Generate Them"), compared to the upper half, the lower half (target name included in negative prompt) consistently has lower $\mathsf{DETECT}$ scores.******We also examine the effectiveness of adding the character name to the negative prompt when user input does not contain the character name and also find a consistent effect. In the case of paragraph-length descriptions, the number of detected characters is reduced by over $50\%$ while maintaining consistency (see [§ E.1] to Generate Them")). Incorporating LAION keywords into the negative prompts in addition to character name further reduces $\mathsf{DETECT}$. The combination of these words in the negative prompt significantly reduces the original $\mathsf{DETECT}$ score from 30 to 4. Notably, the addition of negative prompts does not significantly impair generated image’s consistency with user’s intended prompt, as the $\mathsf{CONS}$ scores typically remain similar or only slightly lower compared to the no intervention setting, but still substantially above $0.33$, the value which indicates very high consistency (see [§ C.4] to Generate Them")). [Fig. 5] to Generate Them") and [Fig. 12] to Generate Them") (in [§ E.4] to Generate Them")) visualize some qualitative examples.

Combining prompt rewriting and negative prompts shows promise for elimination of similar output. Finally, we combine prompt rewriting and negative prompts. Specifically, we send the rewritten prompts as inputs to the image generation models. Then we apply negative prompts during generation. Surprisingly, as demonstrated in [Tb. 2] to Generate Them"), this simple technique is already quite promising in alleviating copyright concerns and is effective across all open-source models evaluated.††††††DALL·E does not allow customizing negative prompts. The number of detected copyrighted characters is significantly reduced for all models. Notably, the number of detection decreases to only 5% of the original in the case of DeepFloyd. At the same time, the $\mathsf{CONS}$ scores remain mostly stable. This suggests that despite the pressing concern of image generation models generating copyrighted characters, we can use this simple yet effective method for meaningful mitigation. [Fig. 5] to Generate Them") and [Fig. 13] to Generate Them") (in [§ E.4] to Generate Them")) present some examples. As shown, most generated images still align with the user’s intent, keeping key characteristics as the requested copyrighted character, but the generation result is already drastically different from the requested copyrighted characters.

*Table 3: Our intervention strategy combining rewriting with negative prompts has a statistically significant effect on $\mathsf{DETECT}$ scores. The negative prompt set refers to Target’s name + 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords. T-values represent one-tailed tests ($\alpha$ \= 0.05) and ∗∗ denotes $p<0.05$.*

| Prompt | Mean (Baseline) | Mean (with negative prompt set) | T-value |
| --- | --- | --- | --- |
|  | $\pm$ SD | $\pm$ SD |  |
| Target’s Name | $30.33\pm 1.89$ | $20.67\pm 0.47$ | $8.59^{**}$ |
| Rewritten Prompt | $14.33\pm 2.62$ | $4.33\pm 0.47$ | $6.51^{**}$ |

We show statistical significance of our highlighted intervention’s effect on reducing $\mathsf{DETECT}$ in [Tb. 3] to Generate Them"). Nonetheless, even this combination of strategies is not perfect at stopping the generation of copyrighted characters, which calls for more future research efforts.

6 Related Work
--------------

#### Diffusion models.

Diffusion models is a type of generative models that synthesize images through two intertwined processes: the forward and the reverse diffusion paths *(Rombach et al., [2021]; Podell et al., [2023])*. In the forward diffusion process, an image gradually transitions from its original state to a fully noised version by incrementally adding noise. The reverse process aims to reconstruct the original image from this noisy state. These models can approach the reverse process in two ways: by either predicting the clean image directly at each step or by estimating the noise to be subtracted from the noisy image. Training diffusion models requires extensive datasets, such as LAION-5B, which consists of a vast collection of publicly accessible copyrighted materials *(Schuhmann et al., [2022])*. As these models evolve, diffusion models can generate copies of samples from their training data *(car, [2023]; Vyas et al., [2023])*, which raises potential concerns regarding privacy and copyright. Recent works have explored some potential pathways to suppress certain concepts from being generated in the diffusion process *(Kumari et al., [2023]; Li et al., [2024b])*. While these methods further fine-tune models and address memorized styles and images individually, we aim to examine operationalizable ways to add copyright protection without updating the parameters.

#### Copyright and generative models.

Recent studies have delved into the copyright implications of generative models such as diffusion models and language models *(Sag, [2018]; Henderson et al., [2023]; Lee et al., [2024]; Sag, [2023]; Min et al., [2023]; Shi et al., [2024])*. *Lee et al. ([2024])*, *Sag ([2023])*, and *Henderson et al. ([2023])* in particular point to copyrighted characters as a challenging legal area.
Each of them note that it may be possible for characters to be generated even when users don’t explicitly input the character name, though they do not systematically evaluate this phenomenon.

Others have demonstrated that these models can potentially reconstruct or replicate copyrighted content from their training data*(Carlini et al., [2020]; car, [2023])*. Efforts to mitigate these risks include provable copyright protection strategies inspired by differential privacy *(Vyas et al., [2023])*, decoding-time prevention *(Golatkar et al., [2024])* that guide the generation process away from copyright concepts and model editing and unlearning that aim to remove copyrighted content from model weights *(Gong et al., [2024]; Chefer et al., [2023]; Zhang et al., [2023])*. *Ma et al. ([2024])* introduces benchmark for measuring copyright infringement unlearning from text-to-image diffusion models.
Another recent study by *Kim et al. ([2024])* also examines keywords potentially important for image generation, but only includes the character name along with the associated movie or TV program as keywords. They also show that LLM-optimized descriptions can generate images similar to copyrighted characters on proprietary models such as ChatGPT, Copilot, and Gemini. However, their optimized prompts do not explicitly exclude the characters’ names. Similarly, *Zhang et al. ([2024a])* focuses on building attacks that can generated particular concepts,including some copyrighted characters.
While these works focus on attacks and do not explore effective mitigation methods, our work focuses on building an analysis framework motivated by legal considerations like substantial similarity test for copyrighted characters and provide more in-depth understanding for both how easy it is to generate copyrighted characters as well as the effectiveness of defenses.

7 Discussions and Limitations
-----------------------------

Our work provides an initial step forward for systematically evaluating the likelihood of generating copyrighted characters and the effectiveness of inference-time mitigation strategies. While our LLM-as-a-judge method can make mistakes in judging character similarity ([§ C.3] to Generate Them")), the relatively low error rate does not affect the ranking of our methods nor our main takeaways: Current mitigation strategies are imperfect, and there are still many instances where characters are generated verbatim. There is still plenty of room for improving mitigations.

Note that whether the proposed strategies will be considered sufficient is also a jurisdiction-dependent question. There are limits to the proposed mitigation strategies and current evaluation, and the metrics we propose should not be treated as a universal optimization target. Nevertheless, our analysis suggests that current mitigations are lacking and more should be done to assess compliance with one’s policy. We hope that this work can help understand the gaps in these strategies and facilitate more complete assessment of model and system design in future research.

Future works can improve these evaluation protocols and mitigations in several ways. First, they can leverage optimization-based approaches to identify more complicated indirect anchors.
Second, they can explore improved mechanisms to identify user intent to generate copyrighted characters from prompts alone.
For example, if a user writes a complicated prompt “A video game plumber with a red hat and an M on the hat, in blue overalls….”, model improvements could better map the description to a potential character so that their name could be included in the negative prompt.
Third, future work can address additional broader types of similarly challenging visual content, like trademarks and broader sets of less-popular characters. Fourth, metrics like consistency scoring and detection could be improved to better capture legally-relevant and human-centered notions of consistency and character similarity.
While our work will likely be re-usable for these broader categories of copyrighted and trademarked content, we did not explicitly evaluate them here.

8 Conclusion
------------

In this paper, we study the important subquestion of copyrighted protection focusing on copyrighted characters. We address two main research questions: 1) Which textual prompts can trigger generation of copyrighted characters; and 2) How effective are current runtime mitigation strategies and how we can improve them? To systematically study these questions, we develop an evaluation framework that considers both elimination of similar output to copyrighted characters and generated image’s consistency with user input. We show how to leverage embedding space distance and common training corpora to extract useful indirect anchors—descriptions and keywords not explicitly mentioning the characters’ names but can be effective in triggering copyrighted character generation.
Existing mitigations, namely prompt rewriting, are not fully effective and we suggest new runtime methods to improve them.
Our work calls for more attention to the indirect anchoring challenge and the effectiveness of deployed mitigation strategies for copyrighted character protection. The insights we provide here can be operationalized by model deployers for copyright-aware image and video generation systems in the future.

Acknowledgements
----------------

We thank Yanai Elazar for his insights on the WIMBD indices. We thank Colin Wang, Mengzhou
Xia, Dan Friedman, Howard Yen, Jiayi Geng, Xindi Wu, Boyi Wei, Samyak Gupta, and Eric Wallace
for providing helpful feedback. Luxi He is supported by the Gordon Y. S. Wu Fellowship. Yangsibo
Huang is supported by the Wallace Memorial Fellowship. This research is partially supported by
a Princeton SEAS Innovation Grant. Any opinions, findings, conclusions, or recommendations
expressed in this material are those of the author(s) and do not necessarily reflect the views of the sponsors.

Ethics Statement
----------------

This work complies with the ICLR Code of Ethics. Regarding potential concerns in releasing copyrighted assets: The majority of the assets (e.g., prompts, descriptions, etc.) are generic and are likely not protected by copyright in most, if not all, jurisdictions. To the extent that we release generated images of characters (e.g., the screenshots in the paper), in many jurisdictions this would be considered acceptable under copyright law as part of research (e.g., fair use in the United States). However, to ensure compliance with more jurisdictions, beyond the paper, we can refrain from releasing certain images other than through direct sharing only with researchers on request under an agreement on proper use of the asset.

In general, our work systematically analyzes mitigation strategies and risks of generating certain copyrighted characters. We find that currently these mitigation strategies can sometimes fail, though they are actively used by different organizations.
However, there may be better ways for companies to respect the intellectual property rights of creators and their visual copyrighted characters from both a technical and structural perspectives. Leveraging the lessons we provide here will both improve likelihood that rights are respected and reduce litigation risk for companies. However, we do not address broader societal discussions on how artists should be compensated for training on images that may contain their intellectual property (such as their characters). This is a larger, worthy, discussion broader than the scope of our work. Current fair use doctrine in the United States might allow this training provided that mitigation strategies are used to prevent substantially similar outputs*(Lee et al., [2024]; Lemley \& Casey, [2020]; Henderson et al., [2023]; Pasquale \& Sun, [2024])*. But this is far from resolved. Similar fair use standards exist in other countries as well. Globally, there are also countries where even training may not be allowed and different approaches may be needed than we describe here. There are also general labor displacement concerns: mitigation strategies might successfully prevent infringement of copyright characters and model training might generally be considered fair use, yet may nonetheless impact data creators’ livelihoods. These are important issues that go beyond the scope of this work.

Reproducibility Statement
-------------------------

We are dedicated to making every aspect of our work fully open-source, offering detailed instructions to ensure reproducibility. Detailed model choice and generation configurations are included in [§ C] to Generate Them"). We also provide the original prompts we use for querying models in [§ C.5] to Generate Them") and [§ C.6] to Generate Them").

We include code and data in our supplementary material. They cover all experiments in the paper, including image and video generation code, prompt and data used, evaluation under different settings, indirect anchor studies etc. We also provide detailed instruction along with the code to ensure reproducibility of our results.

References
----------

* car (2023)Extracting training data from diffusion models, author\=Carlini, Nicolas and Hayes, Jamie and Nasr, Milad and Jagielski, Matthew and Sehwag, Vikash and Tramer, Florian and Balle, Borja and Ippolito, Daphne and Wallace, Eric.In *USENIX Security*, 2023.
* Betker et al. (2023)James Betker, Gabriel Goh, Li Jing, Tim Brooks, Jianfeng Wang, Linjie Li, Long Ouyang, Juntang Zhuang, Joyce Lee, Yufei Guo, Wesam Manassra†, Prafulla Dhariwal†, Casey Chu, Yunxin Jiao, and Aditya Ramesh.Improving Image Generation with Better Captions.2023.URL [https://cdn.openai.com/papers/dall-e-3.pdf](https://cdn.openai.com/papers/dall-e-3.pdf "").
* Blattmann et al. (2023)Andreas Blattmann, Tim Dockhorn, Sumith Kulal, Daniel Mendelevitch, Maciej Kilian, Dominik Lorenz, Yam Levi, Zion English, Vikram Voleti, Adam Letts, et al.Stable video diffusion: Scaling latent video diffusion models to large datasets.*arXiv preprint arXiv:2311.15127*, 2023.
* Carlini et al. (2020)Nicholas Carlini, Florian Tramèr, Eric Wallace, Matthew Jagielski, Ariel Herbert-Voss, Katherine Lee, Adam Roberts, Tom B. Brown, Dawn Xiaodong Song, Úlfar Erlingsson, Alina Oprea, and Colin Raffel.Extracting Training Data from Large Language Models.In *USENIX Security*, 2020.
* Carlini et al. (2023)Nicholas Carlini, Daphne Ippolito, Matthew Jagielski, Katherine Lee, Florian Tramer, and Chiyuan Zhang.Quantifying Memorization Across Neural Language Models.In *ICLR*, 2023.
* Chefer et al. (2023)Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or.Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models.*ACM Transactions on Graphics (TOG)*, 42(4):1–10, 2023.
* Chen et al. (2024)Junsong Chen, Jincheng YU, Chongjian GE, Lewei Yao, Enze Xie, Zhongdao Wang, James Kwok, Ping Luo, Huchuan Lu, and Zhenguo Li.PixArt-$\alpha$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis.In *ICLR*, 2024.
* Cohen (1960)Jacob Cohen.A coefficient of agreement for nominal scales.*Educational and psychological measurement*, 20(1):37–46, 1960.
* Dustin (2024)Spencer Dustin.ChatGPT AutoExpert - DALL·E System Prompt, 2024.URL [https://github.com/spdustin/ChatGPT-AutoExpert/blob/main/_system-prompts/gpts/dalle.md](https://github.com/spdustin/ChatGPT-AutoExpert/blob/main/_system-prompts/gpts/dalle.md "").Accessed: 2024-10-01.
* Elazar et al. (2023)Yanai Elazar, Akshita Bhagia, Ian Helgi Magnusson, Abhilasha Ravichander, Dustin Schwenk, Alane Suhr, Evan Pete Walsh, Dirk Groeneveld, Luca Soldaini, Sameer Singh, et al.What’s In My Big Data?In *ICLR*, 2023.
* *Andersen et al. v. Stability AI et al.* (N.D. Cal. 2023)*Andersen et al. v. Stability AI et al.*3:23-cv-00201, N.D. Cal. 2023.
* Esser et al. (2024)Patrick Esser, Sumith Kulal, Andreas Blattmann, Rahim Entezari, Jonas Müller, Harry Saini, Yam Levi, Dominik Lorenz, Axel Sauer, Frederic Boesel, et al.Scaling rectified flow transformers for high-resolution image synthesis.*arXiv preprint arXiv:2403.03206*, 2024.
* Gao et al. (2020)Leo Gao, Stella Biderman, Sid Black, Laurence Golding, Travis Hoppe, Charles Foster, Jason Phang, Horace He, Anish Thite, Noa Nabeshima, et al.The Pile: An 800gb dataset of diverse text for language modeling.*arXiv preprint arXiv:2101.00027*, 2020.
* Golatkar et al. (2024)Aditya Golatkar, Alessandro Achille, Luca Zancato, Yu-Xiang Wang, Ashwin Swaminathan, and Stefan 0 Soatto.CPR: Retrieval Augmented Generation for Copyright Protection.In *CVPR*, 2024.
* Gong et al. (2024)Chao Gong, Kai Chen, Zhipeng Wei, Jingjing Chen, and Yu-Gang Jiang.Reliable and Efficient Concept Erasure of Text-to-Image Diffusion Models, 2024.URL [https://arxiv.org/abs/2407.12383](https://arxiv.org/abs/2407.12383 "").
* Henderson et al. (2023)Peter Henderson, Xuechen Li, Dan Jurafsky, Tatsunori Hashimoto, Mark A. Lemley, and Percy Liang.Foundation Models and Fair Use.*ArXiv*, abs/2303.15715, 2023.URL <https://api.semanticscholar.org/CorpusID:257771630>.
* Hennessey (2020)Kaitlyn Hennessey.Intellectual Property—Mickey Mouse’s Intellectual Property Adventure: What Disney’s War on Copyrights Has to Do with Trademarks and Patents.*Western New England Law Review*, 42:25, 2020.URL <https://digitalcommons.law.wne.edu/lawreview/vol42/iss1/2>.
* Ho \& Salimans (2021)Jonathan Ho and Tim Salimans.Classifier-Free Diffusion Guidance.In *NeurIPS 2021 Workshop on Deep Generative Models and Downstream Applications*, 2021.
* Kim et al. (2024)Minseon Kim, Hyomin Lee, Boqing Gong, Huishuai Zhang, and Sung Ju Hwang.Automatic Jailbreaking of the Text-to-Image Generative AI Systems.*arXiv preprint arXiv:2405.16567*, 2024.
* Kumari et al. (2023)Nupur Kumari, Bingliang Zhang, Sheng-Yu Wang, Eli Shechtman, Richard Zhang, and Jun-Yan Zhu.Ablating Concepts in Text-to-Image Diffusion Models.In *ICCV*, 2023.
* Lee et al. (2024)Katherine Lee, A. Feder Cooper, and James Grimmelmann.Talkin’ ’Bout AI Generation: Copyright and the Generative-AI Supply Chain, 2024.
* Lee (2019)Timothy B. Lee.Mickey Mouse Will Be in the Public Domain Soon—Here’s What That Means.*Ars Technica*, January 2019.URL [https://arstechnica.com/tech-policy/2019/01/a-whole-years-worth-of-works-just-fell-into-the-public-domain/](https://arstechnica.com/tech-policy/2019/01/a-whole-years-worth-of-works-just-fell-into-the-public-domain/ "").Available at: [https://perma.cc/8M7B-ML6C](https://perma.cc/8M7B-ML6C "").
* Lemley \& Casey (2020)Mark A Lemley and Bryan Casey.Fair learning.*Tex. L. Rev.*, 99:743, 2020.
* Li et al. (2024a)Daiqing Li, Aleks Kamko, Ehsan Akhgari, Ali Sabet, Linmiao Xu, and Suhail Doshi.Playground v2. 5: Three Insights towards Enhancing Aesthetic Quality in Text-to-Image Generation.*arXiv preprint arXiv:2402.17245*, 2024a.
* Li et al. (2024b)Senmao Li, Joost van de Weijer, Taihang Hu, Fahad Shahbaz Khan, Qibin Hou, Yaxing Wang, and Jian Yang.Get What You Want, Not What You Don’t: Image Content Suppression for Text-to-Image Diffusion Models.In *ICLR*, 2024b.
* Lin et al. (2024)Zhiqiu Lin, Deepak Pathak, Baiqi Li, Jiayao Li, Xide Xia, Graham Neubig, Pengchuan Zhang, and Deva Ramanan.Evaluating Text-to-Visual Generation with Image-to-Text Generation.*arXiv preprint arXiv:2404.01291*, 2024.
* Liu (2013)Joseph P. Liu.The New Public Domain.*University of Illinois Law Review*, 2013(5):1395–1460, 2013.URL [https://illinoislawreview.org/print/volume-2013-issue-5/the-new-public-domain/](https://illinoislawreview.org/print/volume-2013-issue-5/the-new-public-domain/ "").
* Luo et al. (2023)Zhengxiong Luo, Dayou Chen, Yingya Zhang, Yan Huang, Liang Wang, Yujun Shen, Deli Zhao, Jingren Zhou, and Tieniu Tan.VideoFusion: Decomposed Diffusion Models for High-Quality Video Generation.In *CVPR*, 2023.
* Ma et al. (2024)Rui Ma, Qiang Zhou, Yizhu Jin, Daquan Zhou, Bangjun Xiao, Xiuyu Li, Yi Qu, Aishani Singh, Kurt Keutzer, Jingtong Hu, Xiaodong Xie, Zhen Dong, Shanghang Zhang, and Shiji Zhou.A Dataset and Benchmark for Copyright Infringement Unlearning from Text-to-Image Diffusion Models, 2024.URL [https://arxiv.org/abs/2403.12052](https://arxiv.org/abs/2403.12052 "").
* McGuinness (2024)Pat McGuinness.GPT-4 System Prompt Revealed, 2024.URL [https://patmcguinness.substack.com/p/gpt-4-system-prompt-revealed](https://patmcguinness.substack.com/p/gpt-4-system-prompt-revealed "").
* Min et al. (2023)Sewon Min, Suchin Gururangan, Eric Wallace, Weijia Shi, Hannaneh Hajishirzi, Noah A Smith, and Luke Zettlemoyer.SILO Language Models: Isolating Legal Risk In a Nonparametric Datastore.In *ICLR*, 2023.
* Pasquale \& Sun (2024)Frank Pasquale and Haochen Sun.Consent and Compensation: Resolving Generative AI’s Copyright Crisis.*Cornell Legal Studies Research Paper Forthcoming*, 2024.
* Playground AI (2023)Playground AI.Prompt Guide: Negative Prompts, 2023.URL [https://playground.com/prompt-guide/negative-prompts](https://playground.com/prompt-guide/negative-prompts "").
* Podell et al. (2023)Dustin Podell, Zion English, Kyle Lacey, A. Blattmann, Tim Dockhorn, Jonas Muller, Joe Penna, and Robin Rombach.SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.In *ICLR*, 2023.
* Podell et al. (2024)Dustin Podell, Zion English, Kyle Lacey, Andreas Blattmann, Tim Dockhorn, Jonas Müller, Joe Penna, and Robin Rombach.SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis.In *ICLR*, 2024.
* Radford et al. (2019)Alec Radford, Jeffrey Wu, Rewon Child, David Luan, Dario Amodei, Ilya Sutskever, et al.Language Models are Unsupervised Multitask Learners.*OpenAI blog*, 2019.
* Raffel et al. (2020)Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu.Exploring the limits of transfer learning with a unified text-to-text transformer.*Journal of machine learning research*, 2020.
* Rombach et al. (2021)Robin Rombach, A. Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.High-Resolution Image Synthesis with Latent Diffusion Models.*CVPR*, pp. 10674–10685, 2021.URL <https://api.semanticscholar.org/CorpusID:245335280>.
* Rombach et al. (2022)Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.High-resolution image synthesis with latent diffusion models.In *CVPR*, 2022.
* Sag (2018)Matthew Sag.The new legal landscape for text mining and machine learning.*J. Copyright Soc’y USA*, 66:291, 2018.
* Sag (2023)Matthew Sag.Copyright safety for generative ai.*Forthcoming in the Houston Law Review*, 2023.
* Schreyer (2015)Amanda Schreyer.An overview of legal protection for fictional characters: Balancing public and private interests.*Cybaris Intell. Prop. L. Rev.*, 6:50, 2015.
* Schuhmann et al. (2022)Christoph Schuhmann, Romain Beaumont, Richard Vencu, Cade Gordon, Ross Wightman, Mehdi Cherti, Theo Coombes, Aarush Katta, Clayton Mullis, Mitchell Wortsman, et al.Laion-5b: An open large-scale dataset for training next generation image-text models.In *NeurIPS*, 2022.
* Shi et al. (2024)Weijia Shi, Anirudh Ajith, Mengzhou Xia, Yangsibo Huang, Daogao Liu, Terra Blevins, Danqi Chen, and Luke Zettlemoyer.Detecting Pretraining Data from Large Language Models.In *The Twelfth International Conference on Learning Representations*, 2024.URL [https://openreview.net/forum?id\=zWqr3MQuNs](https://openreview.net/forum?id=zWqr3MQuNs "").
* Shimbun (2024)The Yomiuri Shimbun.China Court Awards Damages Over AI Images Resembling Ultraman; Service Provider Held Liable For Copyright Infringement.2024.URL [https://japannews.yomiuri.co.jp/society/crime-courts/20240416-180611/#:~:text\=The%20court%20ruled%20on%20Feb,halting%20generation%20of%20the%20images.](https://japannews.yomiuri.co.jp/society/crime-courts/20240416-180611/#:~:text=The%20court%20ruled%20on%20Feb,halting%20generation%20of%20the%20images. "")
* Somepalli et al. (2023)Gowthami Somepalli, Vasu Singla, Micah Goldblum, Jonas Geiping, and Tom Goldstein.Diffusion art or digital forgery? investigating data replication in diffusion models.In *CVPR*, 2023.
* StabilityAI (2023)StabilityAI.DeepFloyd IF.[https://github.com/deep-floyd/IF](https://github.com/deep-floyd/IF ""), 2023.
* Su et al. (2023)Hongjin Su, Weijia Shi, Jungo Kasai, Yizhong Wang, Yushi Hu, Mari Ostendorf, Wen-tau Yih, Noah A. Smith, Luke Zettlemoyer, and Tao Yu.One Embedder, Any Task: Instruction-Finetuned Text Embeddings.In Anna Rogers, Jordan Boyd-Graber, and Naoaki Okazaki (eds.), *Findings of the Association for Computational Linguistics: ACL 2023*, pp. 1102–1121, Toronto, Canada, July 2023. Association for Computational Linguistics.doi: 10.18653/v1/2023.findings-acl.71.URL [https://aclanthology.org/2023.findings-acl.71](https://aclanthology.org/2023.findings-acl.71 "").
* Urbanek et al. (2023)Jack Urbanek, Florian Bordes, Pietro Astolfi, Mary Williamson, Vasu Sharma, and Adriana Romero-Soriano.A Picture is Worth More Than 77 Text Tokens: Evaluating CLIP-Style Models on Dense Captions.*arXiv preprint arXiv:2312.08578*, 2023.
* Vincent (2023)James Vincent.Getty Images is suing the creators of AI art tool Stable Diffusion for scraping its content.[https://www.theverge.com/2023/1/17/23558516/ai-art-copyright-stable-diffusion-getty-images-lawsuit](https://www.theverge.com/2023/1/17/23558516/ai-art-copyright-stable-diffusion-getty-images-lawsuit ""), 2023.
* Vyas et al. (2023)Nikhil Vyas, Sham M Kakade, and Boaz Barak.On provable copyright protection for generative models.In *ICML*, 2023.
* Wang et al. (2024)Junlin Wang, Tianyi Yang, Roy Xie, and Bhuwan Dhingra.Raccoon: Prompt Extraction Benchmark of LLM-Integrated Applications, 2024.URL [https://arxiv.org/abs/2406.06737](https://arxiv.org/abs/2406.06737 "").
* Zhang et al. (2023)Eric Zhang, Kai Wang, Xingqian Xu, Zhangyang Wang, and Humphrey Shi.Forget-me-not: Learning to forget in text-to-image diffusion models.*arXiv preprint arXiv:2303.17591*, 2023.
* Zhang et al. (2024a)Yang Zhang, Teoh Tze Tzun, Lim Wei Hern, Haonan Wang, and Kenji Kawaguchi.On Copyright Risks of Text-to-Image Diffusion Models, 2024a.
* Zhang et al. (2024b)Yiming Zhang, Nicholas Carlini, and Daphne Ippolito.Effective Prompt Extraction from Language Models, 2024b.URL [https://arxiv.org/abs/2307.06865](https://arxiv.org/abs/2307.06865 "").

\appendixpage\startcontents

[sections] \printcontents[sections]l1

Appendix A Legal Background and Broader Societal Impacts
--------------------------------------------------------

In addition to the discussion in [§ 1] to Generate Them") on the unique challenges posed by copyrighted characters and the value of a quantitative study In this matter, we further expand on the legal background and broader societal impacts of our study below.

Legal Motivations Past work has studied the setting of verbatim regurgitation of images*(car, [2023])*, and some lawsuits focus on this particular legal issue*(Vincent, [2023]; *Andersen et al. v. Stability AI et al.*, [N.D. Cal. 2023])*. Some recent work builds datasets for benchmarking copyright infringement unlearning methods *(Ma et al., [2024])* and attempts to jailbreak proprietary systems to output copyrighted images *(Kim et al., [2024])*. Among the copyrighted subjects of interest, copyrighted characters, such as popular Intellectual Property (IP) from Disney, Nintendo, and Dreamworks, pose a unique legal challenge*(Sag, [2023]; Henderson et al., [2023]; Lee et al., [2024])*. At least one lawsuit in China has already resulted in liability for an image generation service that generated the copyrighted character, Ultraman*(Shimbun, [2024])*. Unlike in the verbatim memorization setting, copyrighted characters are computationally more like general concepts that can appear in many poses, sizes, and variations in the training data. Typical deduplication, or even near access free learning approaches*(Vyas et al., [2023])*, will not work for this task*(Henderson et al., [2023])*. Copyrighted characters are also in a certain area of copyright law with distinct rules to determine infringement*(Schreyer, [2015]; Hennessey, [2020])*.
To simplify the legal rules, characters are defined by key distinctive features that as a whole comprise the character. not*(Liu, [2013]; Lee, [2019])*.

As discussed earlier, to simplify the legal rules, characters are often defined by key distinctive features that as a whole compromise the character. This can lead to interesting situations. For example, in 2023 the copyright for the original version of Mickey Mouse character (Steamboat Willie) entered the public domain. But this version of the character did not wear white gloves. However, the gloved version of Mickey Mouse that is now well known has not yet entered the public domain. A number of legal scholars and commentators have pointed out that this means that using a visual depiction of the modern Mickey Mouse would likely lead to an infringement claim, but using the old style of Mickey Mouse (Steamboat Willie) would

In some cases, characters can also be trademarked, leading to other distinct legal challenges not available for memorization of datapoints as a general problem*(Hennessey, [2020])*.

The paper emphasizes that current mitigation strategies taken by companies can fall short of their intended purpose. There may be better ways for companies to respect the intellectual property rights of creators and their visual copyrighted characters from both a technical and structural perspectives. Leveraging the lessons we provide here will both improve likelihood that rights are respected and reduce litigation risk for companies. However, we do not address broader societal discussions on how artists should be compensated for training on images that may contain their intellectual property (such as their characters). This is a larger, worthy, discussion broader than the scope of our work. Current fair use doctrine in the United States might allow this training provided that mitigation strategies are used to prevent substantially similar outputs*(Lee et al., [2024]; Lemley \& Casey, [2020]; Henderson et al., [2023]; Pasquale \& Sun, [2024])*. But this is far from resolved. Similar fair use standards exist in other countries as well. Globally, there are also countries where even training may not be allowed and different approaches may be needed than we describe here. There are also general labor displacement concerns: mitigation strategies might successfully prevent infringement of copyright characters, yet may nonetheless impact data creators’ livelihoods. These are important issues that go beyond the scope of this work.

Appendix B Full List of Characters and Studios in Our Study
-----------------------------------------------------------

We source copyrighted characters from popular studios and franchises, as they are more likely to have been present in the training process of image and video generation models. In addition to U.S. studios like Disney and DreamWorks, we also include international ones like Nintendo and Shogakukan. In total, our collection includes 50 diverse popular copyrighted characters from 18 different studios and subsidiaries. The full list of characters can be found in Appendix [B] to Generate Them").

50 Characters
:   Ariel, Astro Boy, Batman, Black Panther, Bulbasaur, Buzz Lightyear, Captain America, Chun-Li, Cinderella, Cuphead, Donald Duck, Doraemon, Elsa, Goofy, Groot, Hulk, Iron Man, Judy Hopps, Kirby, Kung Fu Panda, Lightning McQueen, Link, Maleficent, Mario, Mickey Mouse, Mike Wazowski, Monkey D. Luffy, Mr. Incredible, Naruto, Nemo, Olaf, Pac-Man, Peter Pan, Piglet, Pikachu, Princess Jasmine, Puss in boots, Rapunzel, Snow White, Sonic The Hedgehog, Spider-Man, SpongeBob SquarePants, Squirtle, Thanos, Thor, Tinker Bell, Wall-E, Winnie-the-Pooh, Woody, Yoda.

18 Studios and Subsidiaries
:   Walt Disney Animation Studios, Disney subsidiaries (Marvel Studios, Pixar Animation Studios, Lucasfilm), Tezuka Productions, DC Comics (Warner Bros.), Nintendo, Capcom, Shin-Ei Animation, Studio MDHR, HAL Laboratory, DreamWorks Animation (Universal Pictures), Toei Animation, Pierrot, Bandai Namco Entertainment, Sega, Nickelodeon Animation Studio, Sony Pictures.

Appendix C Experimental details
-------------------------------

### C.1 Compute resource

All experiments are conducted on 2 NVIDIA A100 GPU cards, each with 80GB of memory. [Tb. 4] to Generate Them") provides statistics on the time cost for each image generation across all the evaluated models, using the character’s name as the input prompt.

*Table 4: Averaged time cost per generation for evaluated models using 2 NVIDIA A100 GPU cards.*

| Model | Time cost (seconds) per generation |
| --- | --- |
| Playground v2.5(Li et al., [2024a]) | $5.1$ |
| Stable Diffusion XL(Podell et al., [2024]) | $36.4$ |
| PixArt-$\alpha$(Chen et al., [2024]) | $8.3$ |
| DeepFloyd IF(StabilityAI, [2023]) | $16.4$ |
| VideoFusion(Luo et al., [2023]) | $6.7$ |

We also report the time cost per evaluation for a single image in [Tb. 5] to Generate Them"), including the cost of running the GPT-4V detector on the image and calculating the consistency score between the image and its key characteristics using VQAScore*(Lin et al., [2024])*. Note that the time cost of the GPT-4V detector is obtained via querying the API, so it may also depend on the real-time network traffic.

*Table 5: Averaged time cost per evaluation on 2 NVIDIA A100 GPU cards. Note that the GPT-4v detector does not require local computational resources, as we query the API provided by OpenAI.*

| Evaluation | Time cost (seconds) per generation |
| --- | --- |
| GPT-4V detector | $3.8$ |
| VQAScore | $<0.1$ |

### C.2 Model Generation Configurations

For Playground v2.5, Stable Diffusion XL (SDXL), and PixArt-$\alpha$, we use 50 iterative steps to progressively refine the image from noise to a coherent output. We set guidance_scale to 3 for the strength of the conditioning signal.

For DeepFloyd IF, we use the standard 3-stage set-up. Models for the 3 stages are DeepFloyd’s IF-I-XL-v1.0, IF-II-L-v1.0, and Stability AI’s stable-diffusion-x4-upscaler respectively. All generation configurations are the model’s default.

For video generation on VideoFusion, we use the model’s default parameters to generate a 16-frame video, and take the first, middle, and last frames for detailed study.

### C.3 Evaluator Set-up and Ablations

In order to automate the process of checking whether there exists a character in the image that can be recognized as an existing copyrighted character, we make use of recent Vision Language Models (VLMs). We conduct ablations study on model choice and prompt format, and choose the option with the highest accuracy and human agreement. First we discuss our human evaluation protocols as follows.

Human evaluation (authors):

To verify the reliability of judgments provided by the GPT-4V evaluator, we conduct an internal human evaluation process among the authors.

Specifically, we first sample 200 generated images (20 characters $\times$ 10 images per character) from various prompting configurations, including direct prompting with character names and indirect prompting using keywords or descriptions, with or without the application of mitigation strategies. We then ask 6 authors to independently annotate these images, following guidelines similar to those used for GPT-4V (described in [§ C.3] to Generate Them")).

For these 200 records, we examine the accuracy of GPT-4V, with the majority-human scores as ground truths. We find that the scores assigned by GPT-4V obtain a fairly high accuracy of $82.5\%$.
To further analyze the consistency and agreement, we compute the Cohen Kappa value*(Cohen, [1960])* between GPT-4V scores with the majority-human scores.
As evaluated, we observe a Cohen Kappa value of 0.648, representing a substantial agreement between human annotators and GPT-4V. We also accompany the pair-wise agreement measurements among human annotators and GPT-4V in [Fig. 6] to Generate Them").

<img src='extracted/6311776/figures/exp/matrix_agreement.png' alt='Refer to caption' title='' width='359' height='287' />

*Figure 6: Cohen Kappa score matrix between human annotators and GPT-4V. As shown, at most time, human annotators themselves reach substantial agreements (Cohen Kappa score in between $0.6\sim 0.8$). Meanwhile, GPT-4V achieves a slightly lower scores when compared with human (yet still substantial agreement on average).*

Using detector accuracy and human agreement scores, we ablate evaluator design choices and choose GPT-4v since it aligns the best with humans among other model choices and prompt format ([Tb. 6] to Generate Them")). Our ablations experiments also show that asking for generic “copyrighted characters" in the prompt instead of specific characters results in higher agreement with humans ([Tb. 6] to Generate Them")).
For each image, we prompt the multimodal model with the alongside the following prompt:

*Table 6: Ablation of Evaluator Design Choises. GPT-4v (the chosen evaluator in our paper) is the best option with highest accuracy and human agreement*

| Model | Character Name in Prompt (Yes/ No) | Accuracy | Cohen-Kappa Agreement with Human |
| --- | --- | --- | --- |
| GPT-4V | No | 82.5% | 0.65 |
| GPT-4V | Yes | 69% | 0.39 |
| Claude Sonnet | No | 71% | 0.43 |
| Claude Sonnet | Yes | 65.5% | 0.33 |
| Llava-34b | No | 78% | 0.56 |
| Llava-34b | Yes | 66.5% | 0.35 |


To further demonstrate the generalizability of our evaluator choice to beyond the selected set of 50 characters in the paper, we show additional results on an additional suite of 50 unseen characters. The ablation study shows that GPT-4V with generic “copyrighted character" in prompt is still the best option ([Tb. 7] to Generate Them")).

*Table 7: Additional ablations of evaluator designs show that GPT-4V (the chosen evaluator in our paper) is still the best option with highest accuracy and human agreement on unseen characters.*

| Model | Character Name in Prompt (Yes/ No) | Accuracy | Cohen-Kappa Agreement with Human |
| --- | --- | --- | --- |
| GPT-4V | No | 80% | 0.52 (Moderate) |
| GPT-4V | Yes | 58.4% | 0.26 (Fair) |
| Claude Sonnet | No | 78% | 0.49 (Moderate) |
| Claude Sonnet | Yes | 49.1% | 0.15 (Slight) |

The additional 50 characters list include: [’Aang’, ’Alladin’, ’Bambi’, ’Barbie ’, ’BB-8’, ’Belle’, ’Betty Boop’, ’BoJack Horseman’, ’Bugs Bunny’, ’Charlie Brown’, ’Chewbacca’, ’Cruella de Vil’, ’Daffy Duck’, ’Daisy Duck’, ’Deadpool’, ’Dumbo’, ’Eren Yeager’, ’Garfield’, ’Genie’, ’Harley Quinn’, ’Hello Kitty’, ’Jafar’, ’Minions’, ’Minnie Mouse’, ’Moana’, ’Mr. Krabs’, ’Mr. Potato Head’, ’Mulan’, ’Mushu’, ’My Little Pony’, ’Nick Wilde’, ’Optimus Prime’, ’Pink Panther’, ’Princess Peach’, ’R2-D2’, ’Rick Sanchez’, ’Sakura Haruno’, ’Sasuke Uchiha’, ’She-Ra’, ’Shrek’, ’Simba ’, ’Simpsons’, ’Snoopy’, ’Son Goku’, ’Stitch’, ’Superman ’, ’The Smurfs’, ’Totoro’, ’Ursula’, ’Wolverine’]

### C.4 Consistency Score

As described in [§ 2] to Generate Them"), we use the VQAScore *(Lin et al., [2024])* to measure consistency ($\mathsf{CONS}$) by checking if the main subject of the target copyrighted character exists in the generated image. The backbone model for computing VQAScore is CLIP-FlanT5.‡‡‡‡‡‡[https://huggingface.co/zhiqiulin/clip-flant5-xxl](https://huggingface.co/zhiqiulin/clip-flant5-xxl "")

To establish reference points, we consider two settings that yield high and low $\mathsf{CONS}$ scores:

* •

    Prompting Playground v2.5 with the character’s name yields $\mathsf{DETECT}\=33$ and $\mathsf{CONS}\=0.741$. Hence, $\mathsf{CONS}\approx 0.75$ indicates high consistency.

* •

    Prompting Playground v2.5 with the character’s name but randomly replacing 3 letters yields $\mathsf{DETECT}\=1$ and $\mathsf{CONS}\=0.329$. Hence, $\mathsf{CONS}\approx 0.33$ indicates low consistency.

#### Human Evaluation of $\mathsf{CONS}$

Authors give binary annotations (e.g., is the key characteristic present in the generated image) and we use the majority votes as the ground truth. The AUROC of the continuous CONS score against binary human annotations is 0.82, suggesting a high agreement. Note that CONS is based on VQAScore *(Lin et al., [2024])*, which has been shown to achieve SOTA performance on challenging image-text matching benchmarks.

### C.5 Generation of Indirect Anchors

50 candidate keywords for indirect anchors ranking. We use GPT-4 to generate a pool of $n$ candidate keywords related to each character, where $n\=50$ in our main experiments. Below is an example prompt for collecting $50$ keywords for Mario:


60-word descriptions. We use GPT-4 to generate 60-word descriptions for studying description-based indirect anchors. Below is an example prompt for collecting such a description for Mario:


### C.6 Prompt-Rewriting Template

For prompt-rewriting intervention, we adapt the DALL·E system prompt to simulate mitigation strategies adopted by model deployers.
The prompt is modified to highlight copyright concerns as well as restrict prompt length to be around 60 words.
The original full DALL·E system prompt can be obtained via prompt extraction methods, by directly prompting GPT-4 in a certain format, as discussed in various sources like *McGuinness ([2024]); Dustin ([2024])*. It should be noted, however, that this may not be the exact DALL·E system prompt, but prompt extraction methods have generally been successful. Regardless of whether this is the exact system prompt used in production, it is a sufficient approximation for our purposes *(Zhang et al., [2024b]; Wang et al., [2024])*.
Given our limited access to the OpenAI system, we’ve done our best to recreate the pipeline as closely as possible. We also did some manual checking on the ChatGPT interface to verify that after rewriting with this set of instructions, DALL-E will not change the wording again when doing actual generation.

The full prompt used in our experiment is shown below:


[Fig. 7] to Generate Them") shows an example of the generated keywords and descriptions for Mario.

<img src='x13.png' alt='Refer to caption' title='' width='830' height='304' />

*Figure 7: Indirect anchors (keywords and descriptions) that trigger models to generate Mario.
Both keywords and descriptions in the figure are LM-generated indirect anchors.*

Appendix D More results on DALL·E
---------------------------------

Character name anchoring does not work on DALL·E system due to its built-in filter that detects and blocks requests that explicitly mention copyrighted characters. However, indirect anchoring is still able to bypass the system guardrails and generate high-quality images that highly resemble the target copyrighted characters, as illustrated in [Fig. 8] to Generate Them").

<img src='x14.png' alt='Refer to caption' title='' width='830' height='105' />

*Figure 8: Using 60-word descriptions to circumvent built-in safeguards like character name detection and prompt rewriting, we are able to push DALL·E 3 to generate copyrighted characters.*

Appendix E More results on open-source models
----------------------------------------------

### E.1 Effect of Target’s Name as Negative Prompt for Indirect Anchoring

We also examine intervention strategies in cases where users provide keywords or descriptions to generate images. As shown in [Tb. 8] to Generate Them"), consistent with our previous observations in [Tb. 1] to Generate Them") when the character name is part of user input, adding character name as negative prompt is still a very effective method to reduce recreating copyrighted characters. In particular, when the original prompt consists of 10 keywords or descriptions, incorporating target’s name as negative reduce $\mathsf{DETECT}$ by 50% or more, while $\mathsf{CONS}$ values remain almost constant. For all experiment setup, the $\mathsf{CONS}$ values either remain stable or show a slight decrease with the addition of negative prompts. From a practical perspective, adding copyright character detection and target name as negative prompt is a simple yet effective way of reducing the recreation of copyrighted characters, at the cost slight compromise in adhering to user request.

*Table 8: Effect of adding character names as negative prompts on different indirect anchors set-up.*

|  | Negative Prompt: None | | Negative Prompt: Target’s name | |
| --- | --- | --- | --- | --- |
| Original Prompt | $\mathsf{DETECT}$ ($\downarrow$) | $\mathsf{CONS}$ ($\uparrow$) | $\mathsf{DETECT}$ ($\downarrow$) | $\mathsf{CONS}$ ($\uparrow$) |
| 10 curated keywords | $14.00_{\pm 3.0}$ | $0.76_{\pm 0.01}$ | $7.00_{\pm 2.00}$ | $0.76_{\pm 0.00}$ |
| 20 curated keywords | $28.00_{\pm 2.65}$ | $0.78_{\pm 0.00}$ | $20.67_{\pm 3.21}$ | $0.76_{\pm 0.00}$ |
| 50 curated keywords | $29.67_{\pm 2.08}$ | $0.78_{\pm 0.01}$ | $16.00_{\pm 1.00}$ | $0.76_{\pm 0.00}$ |
| 5 keywords from LAION | $19.67_{\pm 2.89}$ | $0.74_{\pm 0.00}$ | $12.33_{\pm 2.31}$ | $0.72_{\pm 0.01}$ |
| Description | $21.00_{\pm 2.65}$ | $0.78_{\pm 0.01}$ | $10.33_{\pm 0.58}$ | $0.78_{\pm 0.01}$ |

### E.2 Intent detection

In practice, user inputs can include both standard requests for generating non-copyrighted images and requests for generating copyrighted characters. In our evaluation, we assume the presence of an oracle capable of detecting whether a user input is likely to lead a text-to-image model to generate a copyrighted character. To validate this assumption, we explore two methods:

1. 1.

    LLM-based detector that uses an LLM to determine if the user input is associated with a copyrighted character. It directly queries the LLM with the prompt, "Does the following description resemble any copyrighted character?" We then compare the model’s prediction to the correct answer.

2. 2.

    Retriever-based detector that uses a retriever to compare the user input against a database of copyrighted character descriptions *(Su et al., [2023])*. For a given user query, the retriever searches for similar descriptions based on OpenAI embeddings******text-embedding-3-small. If no description with a cosine similarity greater than 0.7 is found, we conclude that the user query does not intend to generate characters substantially similar to copyrighted ones.

#### Experimental Setup

To evaluate our detection methods, we curated a dataset comprising 200 descriptions of copyrighted characters and 200 standard prompts unlikely to cause copyright issues selected from MJHQ benchmarks*†*†*†[huggingface.co/datasets/playgroundai/MJHQ-30K](huggingface.co/datasets/playgroundai/MJHQ-30K ""). We report accuracy, true positive rates (TPR) and false positive rates (FPR) as our evalaution metrics.

#### Results

As shown in [Table 9] to Generate Them"), both methods achieve over 90% accuracy. The LM-based detector achieved an accuracy of 95%, slightly outperforming the retriever-based detector. This high performance indicates that both methods are effective in identifying potential copyright issues in user inputs. It is therefore reasonable to assume that building such a detection oracle is feasible and can be done relatively easily.

*Table 9: Accuracy, true positive rate (TPR) and false positive rate (FPR) of LM-based and retriever-based detectors.*

| Detection Method | Accuracy (%) | TPR (%) | FPR (%) |
| --- | --- | --- | --- |
| LM-based detector | $95.14$ | $93.68$ | $3.32$ |
| Retriever-based detector | $93.28$ | $91.26$ | $4.36$ |

### E.3 Embedding similarity analysis

#### 60-word description as indirect anchors.

We randomly generate 100 60-word prompts per character using the template described in [§ C] to Generate Them"), and rank them by embedding similarity to the corresponding character name. As shown in [Fig. 9] to Generate Them"), the top-ranked prompt by embedding similarity generates 26 characters successfully, versus only 16 for the bottom-ranked prompt.

<img src='x15.png' alt='Refer to caption' title='' width='830' height='328' />

*Figure 9: Character generation success ($\mathsf{DETECT}$ scores) for 60-word descriptions with varying embedding similarity to the target character’s name. Prompts with higher name similarity tend to generate the desired character more often.*

#### Rewritten prompts.

We also study how the success and failure of rewritten prompts correlate with their embedding similarity to the corresponding character name. Specifically, for each character, we generate 100 rewritten prompts and rank them by their embedding similarity to the character’s name. As shown in [Fig. 10] to Generate Them"), the top-ranked rewritten prompt by embedding similarity generates 20 characters successfully, versus only 12 for the bottom-ranked rewritten prompt. This suggests that potentially, rewritten prompts that fail to avoid character generation could be due to their high similarity to the character’s name.

<img src='x16.png' alt='Refer to caption' title='' width='830' height='325' />

*Figure 10: Character generation success ($\mathsf{DETECT}$ scores) for rewritten prompts with varying embedding similarity to the target character’s name. Rewritten prompts with higher name similarity tend to generate the desired character more often (i.e., tend to fail in mitigating).*

### E.4 More results for Playground v2.5

#### Robustness analysis of character name anchoring.

Interestingly, the model exhibits high sensitivity to even minor perturbations in the character’s name. For instance, if we randomly replace a single letter in the character’s name with a different letter, the model can only generate 8 out of the 50 characters successfully. The situation is even more extreme when we randomly replace 3 letters – in this case, the model could only generate 1 out of the 50 characters accurately (see [Fig. 11(b)] to Generate Them")).

On the other hand, if the character’s name is present in the prompt, and irrelevant keywords such as "dancing" or "swimming" are added, this generally does not affect the number of characters generated (see [Fig. 11(c)] to Generate Them") and [Fig. 11(d)] to Generate Them")). These findings suggest that the character name anchoring mode heavily relies on the exact spelling of the target character’s name to generate copyrighted characters.

<img src='x17.png' alt='Refer to caption' title='' width='830' height='108' />

*(a) Prompt: Character’s name*

<img src='x18.png' alt='Refer to caption' title='' width='830' height='108' />

*(b) Prompt: Randomly replace 3 letters from the character’s name*

<img src='x19.png' alt='Refer to caption' title='' width='830' height='108' />

*(c) Prompt: Add "Swimming" to the character’s name*

<img src='x20.png' alt='Refer to caption' title='' width='830' height='108' />

*(d) Prompt: Add "Dancing" to the character’s name*

*Figure 11: The character name anchoring mode heavily relies on the exact spelling of the target character’s name to generate copyrighted characters. Randomly replacing letters in the character’s name leads to an inability to generate the character (b), while adding potentially unrelated words (while still retaining the original name) still yields the target character (c and d).*

#### More visualization.

[Fig. 12] to Generate Them") visualizes results using the character’s name as the prompt and various keywords as negative prompts. Including the character’s name in the prompt, even with detailed negative prompts, still leads to the generation of copyrighted characters. This suggests that T2I models are deeply anchored to these character names.

However, once we apply prompt rewriting and combine it with various negative prompts, the model is no longer inclined to generate these characters, as shown in [Fig. 13] to Generate Them").

<img src='x21.png' alt='Refer to caption' title='' width='830' height='108' />

*(a) Negative Prompt: None*

<img src='x22.png' alt='Refer to caption' title='' width='830' height='108' />

*(b) Negative Prompt: "Copyrighted Characters"*

<img src='x23.png' alt='Refer to caption' title='' width='830' height='108' />

*(c) Negative Prompt: Character’s Name*

<img src='x24.png' alt='Refer to caption' title='' width='830' height='108' />

*(d) Negative Prompt: Character’s Name and 5 keywords*

*Figure 12: Generated images by Playground v2.5 using the character’s name as the input prompt, along with various negative prompts. Including the character’s name in the prompt, even with detailed negative prompts, still leads to the generation of copyrighted characters. This suggests that T2I models are deeply anchored to these character names.*

<img src='x25.png' alt='Refer to caption' title='' width='830' height='108' />

*(a) Negative Prompt: None*

<img src='x26.png' alt='Refer to caption' title='' width='830' height='108' />

*(b) Negative Prompt: "Copyrighted Characters"*

<img src='x27.png' alt='Refer to caption' title='' width='830' height='108' />

*(c) Negative Prompt: Character’s Name*

<img src='x28.png' alt='Refer to caption' title='' width='830' height='108' />

*(d) Negative Prompt: Character’s Name and 5 keywords*

*Figure 13: Generated images by Playground v2.5 using the rewritten prompts as input and various negative prompts. Prompt rewriting significantly reduces instances of generating exact copies of the target, while still producing a similar entity per the user’s request. Including more detailed negative prompts further decreases the similarity to the original copyrighted characters.*

### E.5 Results for PixArt-$\alpha$, Stable Diffusion XL, and DeepFloyd IF

[Fig. 14] to Generate Them") visualizes results from the PixArt-$\alpha$ model*(Chen et al., [2024])*. With higher generation quality, the findings are also consistent with those observed using the Playground v2.5 model—adding more fine-grained negative prompts and applying prompt rewriting significantly reduces the similarity of the generated images to the original copyrighted character.

[Fig. 15] to Generate Them") visualizes results from the Stable Diffusion XL (SDXL) model*(Podell et al., [2024])*. Although the generation quality of SDXL is generally lower compared to the Playground model (see [Fig. 5] to Generate Them")), adding more fine-grained negative prompts and applying prompt rewriting significantly reduces the similarity of the generated images to the original copyrighted character.

<img src='x29.png' alt='Refer to caption' title='' width='830' height='108' />

*(a) Prompt: Character’s name, Negative Prompt: None*

<img src='x30.png' alt='Refer to caption' title='' width='830' height='108' />

*(b) Prompt: Character’s name, Negative Prompt: Character’s name \& 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords*

<img src='x31.png' alt='Refer to caption' title='' width='830' height='108' />

*(c) Prompt: Rewritten, Negative Prompt: None*

<img src='x32.png' alt='Refer to caption' title='' width='830' height='108' />

*(d) Prompt: Rewritten, Negative Prompt: Character’s name \& 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords*

*Figure 14: Images generated with PixArt-$\alpha$*(Chen et al., [2024])* using various prompt and negative prompt configurations.*

<img src='x33.png' alt='Refer to caption' title='' width='830' height='108' />

*(a) Prompt: Character’s name, Negative Prompt: None*

<img src='x34.png' alt='Refer to caption' title='' width='830' height='108' />

*(b) Prompt: Character’s name, Negative Prompt: Character’s name \& 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords*

<img src='x35.png' alt='Refer to caption' title='' width='830' height='108' />

*(c) Prompt: Rewritten, Negative Prompt: None*

<img src='x36.png' alt='Refer to caption' title='' width='830' height='108' />

*(d) Prompt: Rewritten, Negative Prompt: Character’s name \& 5 EmbeddingSim \& 5 Co-Occurrence-LAION keywords*

*Figure 15: Images generated with SDXL*(Podell et al., [2024])* using various prompt and negative prompt configurations.*
