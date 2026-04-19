[Uncaptioned image] $\mathcal{C}ola$: A Benchmark for Compositional  Text-to-image Retrieval
===============================================================================================

Arijit Ray1,2Filip Radenovic2Abhimanyu Dubey2Bryan A. Plummer1Ranjay Krishna2,3Kate Saenko1,2

###### Abstract

Compositional reasoning is a hallmark of human visual intelligence. Yet despite the size of large vision-language models, they struggle to represent simple compositions by combining objects with their attributes.
To measure this lack of compositional capability, we design $\mathcal{C}ola$, a text-to-image retrieval benchmark to Compose Objects Localized with Attributes.
To solve $\mathcal{C}ola$, a model must retrieve images with the correct configuration of attributes and objects, and avoid choosing a distractor image with the same objects and attributes but in the wrong configuration. $\mathcal{C}ola$ contains about $1.2$k composed queries of $168$ objects and $197$ attributes on around $30$K images.
Our human evaluation finds that $\mathcal{C}ola$ is $83.33\%$ accurate, similar to contemporary compositionality benchmarks.
Using $\mathcal{C}ola$ as a testbed, we explore empirical modeling designs to adapt pre-trained vision-language models to reason compositionally.
We explore $6$ adaptation strategies on $2$ seminal vision-language models, using compositionality-centric test benchmarks - $\mathcal{C}ola$ and CREPE.
We find the optimal adaptation strategy is to train a multi-modal attention layer that jointly attends over the frozen pre-trained image and language features.
Surprisingly, training multimodal layers on CLIP performs better than tuning a larger FLAVA model with already pre-trained multimodal layers.
Furthermore, our adaptation strategy improves CLIP and FLAVA to comparable levels, suggesting that training multimodal layers using contrastive attribute-object data is key, as opposed to using them pre-trained. Lastly, we show that $\mathcal{C}ola$ is harder than a closely-related contemporary benchmark, CREPE, since simpler fine-tuning strategies without multimodal layers suffice on CREPE, but not on $\mathcal{C}ola$.
However, we still see a significant gap between our best adaptation and human accuracy, suggesting considerable room for further research.
Project page: [https://cs-people.bu.edu/array/research/cola/](https://cs-people.bu.edu/array/research/cola/ "")

1 Introduction
--------------

<img src='figures/teaserfig_compositionality.png' alt='Refer to caption' title='' width='568' height='211' />

*Figure 1: We present <img src='figures/cola-icon-design-free-vector.jpeg' alt='Refer to caption' title='' width='598' height='1318' /> $\mathcal{C}ola$, where a model has to Compose Objects Localized with Attributes. To solve $\mathcal{C}ola$, a model must match the correct image to the correct caption, not a distractor image with the same objects and attributes but in the wrong configuration. We explore the design space of possible mechanisms to adapt existing models to this task; we show that a simple multimodal adaptation method to finetune pre-trained vision-language representations works best.*

Compositionality is a fundamental characteristic of human intelligence, allowing us to elicit “the meaning of the whole [as] a function of the meanings of its parts”*[[9](#bib.bib9 "")]*. In language, the whole is a sentence made up of words like nouns and adjectives. In vision, the whole is an image made up of visual elements like objects and attributes*[[28](#bib.bib28 ""), [20](#bib.bib20 "")]*. For example, the expression “round white table” is a composition of the noun “table” and adjectives “round” and “white”, visually represented in the leftmost photo of Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"). Recent work has consistently identified that this type of compositionality—that between objects and their attributes—is something existing vision-language models struggle to represent*[[55](#bib.bib55 ""), [35](#bib.bib35 ""), [25](#bib.bib25 "")]*.
Instead, they disperse attributes and ground them to distractor objects; for instance, they incorrectly match “round white table” to the second left photo in [Fig. 1](#S1.F1 "In 1 Introduction ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") by grounding the attribute “round” to the round plate instead of the table. Queries involving two objects are even more challenging, see Fig.[1](#S1.F1 "Figure 1 ‣ 1 Introduction ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") (right).

In this paper, we study the ability of large vision-language models to Compose Objects Localized with Attributes ($\mathcal{C}ola$).
Unlike related baselines that study compositionality using relationships*[[55](#bib.bib55 "")]* and scene graphs*[[35](#bib.bib35 "")]*, we focus on attribute-object bindings, since finding objects with correct attributes is crucial in many applications.
For example, an embodied AI assistant told to clean the “small wood table to the right of the large dark chair” should not start cleaning the “tan wood chair by the large brown wood smooth table.”
Additionally, object-attribute bindings should fundamentally be easier than compositions with relationships or scene graphs. However, we find that existing models still struggle with this simpler binding.

To explore these issues, we propose the $\mathcal{C}ola$ benchmark for composing objects localized with multiple attributes.
$\mathcal{C}ola$ contains two kinds of compositions: single-object queries ([Fig. 1](#S1.F1 "In 1 Introduction ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") left) and multi-object ([Fig. 1](#S1.F1 "In 1 Introduction ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") right). In each case, a model should associate the objects in the query with the correct attributes and ignore difficult distractor compositions where the query attributes are attached to distractor objects. Multi-object queries are harder since they showcase more compositions.

Unlike an image-to-text setting in a contemporary benchmark CREPE*[[35](#bib.bib35 "")]*, $\mathcal{C}ola$ evaluates models using text-to-image, where text queries are used to retrieve the correct image from a set of images. This is consistent with previous benchmarks for vision-language models*[[21](#bib.bib21 ""), [42](#bib.bib42 ""), [64](#bib.bib64 "")]*.
Further, text-to-image retrieval is harder than image-to-text retrieval because image encoders are weaker at distinguishing fine-grained differences in images for a given text than text encoders are at distinguishing fine-grained text*[[55](#bib.bib55 "")]*. Moreover, text-to-image is better aligned with practical applications, such as a user giving text instructions to a machine to find certain objects.

Using $\mathcal{C}ola$ as a development testbed, our experiments add to the ongoing discussion that pre-trained vision-language models perform poorly on compositions *[[55](#bib.bib55 "")]*.
Hence, we explore 6 finetuning strategies on 2 seminal vision-language models- CLIP*[[43](#bib.bib43 "")]* and FLAVA*[[52](#bib.bib52 "")]*- to find adaptation strategies that encourage compositionality the most.
We finetune using 3 datasets (GQA*[[18](#bib.bib18 "")]*, CLEVR*[[24](#bib.bib24 "")]*, and PACO*[[44](#bib.bib44 "")]*) and evaluate on 2 testbeds ($\mathcal{C}ola$ as well as CREPE*[[35](#bib.bib35 "")]*).
While exploring effective pre-training strategies is another valid avenue of exploration, we limit our work to adaptation strategies since training a model from scratch is expensive and can only be executed by a handful of research organizations.

We find that the best-performing architectural choice during adaptation to be a multi-modal transformer encoder-decoder*[[53](#bib.bib53 ""), [52](#bib.bib52 ""), [26](#bib.bib26 "")]* to further encode the visual and language representations from the pre-trained model. Multimodal adaption performs significantly better than tuning the unimodal encoders (that encode just vision or language features), or tuning the prompt.
Surprisingly, this adaptation improves both CLIP and FLAVA to produce comparable finetuned models, even though our CLIP model has fewer parameters and FLAVA was already pre-trained using multi-modal transformer layers. This suggests that training multimodal layers using contrastive attribute-object data is key, as opposed to using them pre-trained on web data. Our adaptation also significantly outperforms standard ways to adapt/tune foundational models such as prompt-tuning *[[65](#bib.bib65 "")]*, linear-probing *[[2](#bib.bib2 "")]*, or tuning a comparable number of split-encoder layers. Similar to recent work identifying that structural compositionality is present in models but absent in their representations*[[31](#bib.bib31 "")]*, our work finds that while pre-trained representations might not exhibit compositionality, they can be adapted to do so. However, the stark difference between human accuracy and our best adaptation suggests considerable room for further research using our benchmark.

2 Related Work
--------------

Compositionality and image retrieval. Compositionality is a key aspect of human intelligence*[[10](#bib.bib10 "")]*, especially in vision and language*[[7](#bib.bib7 "")]*.
Vision-language compositionality has been explored for visual question
answering*[[1](#bib.bib1 "")]*,
composed image retrieval (e.g., X in
the style of Y)*[[49](#bib.bib49 "")]*, and
generation*[[56](#bib.bib56 "")]*. Compositionality is one crucial
aspect of improving robustness to diverse queries, a theme heavily explored
in the vision-language community*[[46](#bib.bib46 ""), [45](#bib.bib45 ""), [51](#bib.bib51 ""), [1](#bib.bib1 ""), [8](#bib.bib8 "")]*.
With the recent popularity of foundation models, various works focus on testing their compositional reasoning*[[35](#bib.bib35 ""), [55](#bib.bib55 ""), [25](#bib.bib25 ""), [62](#bib.bib62 "")]*. Compared to CREPE*[[35](#bib.bib35 "")]* and ARO*[[62](#bib.bib62 "")]*, a model must distinguish between difficult images in our case.
Text-to-difficult-images is harder because distinguishing between difficult images (for a given caption) is harder than distinguishing between difficult captions *[[55](#bib.bib55 "")]*.
Whereas benchmarks like Winoground*[[55](#bib.bib55 "")]* primarily evaluate broad and complex relational compositionality (*e.g*., “man hugs woman from behind” vs “woman hugs man from behind”), we specifically focus on attribute object bindings in queries. This is motivated by practical applications such as an embodied agent trying to retrieve a custom object (like “a metal wrench with a red rubber handle”) in a cluttered workspace with similar distractor objects*[[39](#bib.bib39 "")]*.
Most works in the area of attribute-object image retrieval either focus on single attributes*[[19](#bib.bib19 ""), [41](#bib.bib41 "")]* or multiple attributes in very niche domains with centered images and plain backgrounds of dresses*[[16](#bib.bib16 "")]*, animals*[[60](#bib.bib60 "")]*, shoes*[[61](#bib.bib61 "")]*, or birds*[[58](#bib.bib58 "")]*.
In contrast, we focus on scenes with multiple objects and attributes where distractor objects also have the same attributes.

Vision-language aligment. Recently, there has been a flurry of image-text alignment models to learn the similarity of matched images and text in various ways. Some models use separate unimodal encoders*[[43](#bib.bib43 ""), [22](#bib.bib22 "")]* for the image and text, whereas some*[[52](#bib.bib52 ""), [33](#bib.bib33 ""), [3](#bib.bib3 ""), [11](#bib.bib11 ""), [50](#bib.bib50 "")]* use multimodal encoders as well. Various strategies such as hard negative mining*[[33](#bib.bib33 "")]*, concept distillation*[[40](#bib.bib40 "")]*, and maintaining the momentum of image-text mappings*[[17](#bib.bib17 "")]* have been employed to push performance.
We focus on testing and improving the attribute-object binding capability of such models and choose the most seminal model, CLIP*[[43](#bib.bib43 "")]*, which is widely adopted in various concurrent vision-language research/applications*[[54](#bib.bib54 ""), [27](#bib.bib27 ""), [47](#bib.bib47 "")]*.
Our approaches do not use any box annotations unlike recent text localization models*[[26](#bib.bib26 ""), [34](#bib.bib34 ""), [63](#bib.bib63 "")]*, which we also see to underperform on text-to-image retrieval.

Adapting foundational models. Since training a new VLM from scratch is expensive, we wish to formulate a simple adapter that improves the compositional attribute-object binding.
Various works explore adapting foundation models*[[4](#bib.bib4 "")]* with prompt-tuning*[[65](#bib.bib65 "")]*, linear-probing*[[2](#bib.bib2 "")]*, and fine-tuning with residual connections*[[13](#bib.bib13 "")]*. Prompt-tuning*[[32](#bib.bib32 "")]* learns the embedding layer of the word inputs and keeps the model frozen. Inspired by the success of prompt-tuning*[[32](#bib.bib32 "")]*, some works have also explored prompting in the vision*[[23](#bib.bib23 ""), [5](#bib.bib5 "")]* and vision-language*[[65](#bib.bib65 ""), [48](#bib.bib48 "")]*, and also for single attribute-object compositions*[[37](#bib.bib37 "")]*. Our optimal finetuning strategy improves significantly over prompt and fine-tuning for attribute-object compositions in even more difficult settings.
Our multi-modal strategies are similar to MAPL*[[36](#bib.bib36 "")]*, except our lightweight adapter attends over language and vision representations, whereas MAPL only attends over language.

3 [Uncaptioned image] $\mathcal{C}ola$ benchmark
-------------------------------------------------

Our goal is to adapt vision-language features to improve the compositional binding of attributes to objects. Specifically, we aim to improve the classification of a query involving single or multiple objects with multiple attributes in an image.
Images and language are composed of atomic concepts such as attributes and objects. The atomic concepts (“square”, “plate”) form certain compounds (“square plate”), and then the scene is a combination of various such compounds (“square plate on white table”).
Hence, we create a benchmark where we form queries using compositions of such atoms and test a model’s ability to distinguish between images that correctly contain the atoms in the correct composition to distractor images that contain them in the wrong composition.
In total, the $\mathcal{C}ola$ benchmark contains about 1236 composed queries from 168 objects and 197 attributes on around 30K images from 4 datasets.

$\mathcal{C}ola$ contains two query types discussed below: single-object compounds and multi-object queries.

Retrieval using single-object queries. Single-object queries have multiple attributes grounded on one object. For example, the query “square white plate,” which is of the form, $Q\=a_{1}a_{2}o$, where $a_{i}\in A$ is drawn from a finite set of possible attributes and $o\in O$ is similarly a category drawn from a finite set of objects.
With this query, a model should associate the images with the correct attachment of attributes (“square,” “white”) to the object (“plate”), and ignore incorrect attachments of the same attributes and objects (like square table but not square plates). Hence, the task is a text query for image retrieval among difficult distractors.
We first create a list of queries with more than one attribute for an object. Next, we curate a set of images where at least one of the query words is present in the image. For example, for “square white plate,” all images containing “square” objects, “white” objects, or “plates” are in the list of images to retrieve from.
The goal of the retrieval problem is to score the images having the correct attachment of the attributes to the query higher than others.
We build the test set for single object queries using three datasets with object and attribute annotations: 1) GQA *[[18](#bib.bib18 "")]*: After filtering for objects with at least 1 attribute annotated, we have 320 single-object queries composed of 114 objects and 114 attributes on 1952 images. The objects and attributes comprise common objects, making this split useful for practical applications. 2) CLEVR *[[24](#bib.bib24 "")]*: We have 3 object shapes - cubes, cylinders and spheres, composed with 8 colors, 2 materials, and 2 sizes on 15K images.
3) PACO *[[44](#bib.bib44 "")]*: This split consists of objects similar to GQA. We have 400 queries composed of 51 objects and 61 attributes on 7921 images.

Retrieval with multi-object queries. Drawing on existing literature*[[35](#bib.bib35 "")]*, a multi-object query contains multiple objects, each with its own set of attributes.
For example, “square white plate on top of brown wooden table,”
which is of the form, $Q\=a_{1}a_{2}o_{1}+a_{3}a_{4}o_{2}$, where $a_{i}\in A$ is drawn from a finite set of possible attributes and $o_{j}\in O$ from a finite set of objects.
In this setting, we want to check if the model gets confused with the wrong configuration of objects and attributes.
Thus, we find distractor image-query pairs where the attributes and objects are switched.
An example image for a query $Q\=a_{1}o_{1}+a_{2}o_{2}$ would be of the form $I^{\prime}\=a_{2}o_{1}+a_{1}o_{2}$. In other words, we switch the attributes of the two objects. We curate these distractors to ensure that $o_{1}\neq o_{2}$ and $a_{1}\neq a_{2}$.
The retrieval task, framed with this formalism, is to rank the correct images for the correct captions such that it is ranked higher than the distractor images: to learn a relevance encoding $f(I,Q)$ for image $I$ and query $Q$ such that $f(I,Q)>f(I^{\prime},Q)\And f(I^{\prime},Q^{\prime})>f(I,Q^{\prime})$. The test set is built using test split of the Visual Genome *[[29](#bib.bib29 "")]* dataset.

Filtering $\mathcal{C}ola$ multi-object with crowd workers. We use the object, attribute, and relationship annotations in the Visual Genome dataset *[[29](#bib.bib29 "")]* to create the multi-object queries. We filter the image-caption pairs with object and attribute compositions swapped as described above.
We conduct a human-annotated cleaning of this filtered test set.
We display the images $I$ and $I^{\prime}$ and queries $Q$ and $Q^{\prime}$ to $10$ crowd workers and ask them to choose which image is most relevant to which query.
We only keep the image-query pairs where the majority of crowd workers can correctly assign the correct image to the query.
After filtering, we are left with $210$ data points ($840$ image-query pairs) with 1680 possible image-query matches. The human agreement (accuracy) on our validation set is $83.88\%$ - an average of 8.33 out of 10 humans agree that the first image matches to the first caption and second image to the second caption.
Some qualitative examples are provided in Fig.[2(a)](#S3.F2.sf1 "Figure 2(a) ‣ Figure 2 ‣ 3 𝒞⁢𝑜⁢𝑙⁢𝑎 benchmark ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval").

<img src='figures/data_qual_examples_multiobj.png' alt='Refer to caption' title='' width='598' height='277' />

*(a)*

<img src='figures/model.png' alt='Refer to caption' title='' width='598' height='490' />

*(b)*

*Figure 2: a) $\mathcal{C}ola$ multi-object setting validation set: a human-cleaned difficult validation set for testing attribute-object binding. The two images have similar objects and attributes but in different configurations. A model must match the correct images to the correct captions. b) The optimal adaptation strategy (MM-Adapter): a lightweight multimodal transformer encoder on top of frozen pre-trained encoders. The multimodal encoder crafts a stronger representation by cross-attending to image patches and text tokens to attach the correct attributes to the correct objects. The stronger representation is then trained to align with the frozen text representation.*

4 Exploring finetuning strategies with $\mathcal{C}ola$
--------------------------------------------------------

Given an image ($I$) and a query ($Q$), $\mathcal{C}ola$ evaluates a model $f(I,Q)$ by measuring how well it associates the correct image to the input query.
Existing pre-trained models don’t perform well on this task since they fail to distinguish fine-grained differences in attribute-object compositions. Hence, we explore finetuning strategies that use a dataset of image-language pairs where the language descriptions contain objects and attributes. Details of finetuning datasets are described in Sec. [5.3](#S5.SS3 "5.3 Finetuning datasets ‣ 5 Evaluation Setup ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval").
We follow the standard finetuning paradigm by sampling batches of images and text from the attribute-object $\mathcal{C}ola$ finetuning dataset. Specifically, we match the correct images to the correct queries in each batch and minimize the NCELoss typically used in contrastive learning*[[43](#bib.bib43 ""), [52](#bib.bib52 "")]*.
This finetuning step aims to improve the compositional binding of attributes and objects in pretrained vision-language features. This is in contrast to training the multimodal layers on random batches of web image-text data.

Disjoint finetuning strategies Since CLIP*[[43](#bib.bib43 "")]* is commonly used for various tasks and is one of the more lightweight vision-language foundation models, we focus most of our finetuning strategies with CLIP in mind. Although, we also later use these finetuning strategies on the newer and larger FLAVA*[[52](#bib.bib52 "")]* model.
CLIP*[[43](#bib.bib43 "")]* consists of two encoders: one that encodes the input image and one that similarly encodes the input text. The output representations of the two modalities are used as separate embeddings for the image and text.
To adapt these models for a specific task/capability, researchers commonly use linear-probing or prompt-tuning.
Linear-probing trains linear layers on top of the frozen visual and text encoders using the finetuning dataset. Prompt-tuning learns the word embeddings of the query to adapt to the finetuning dataset domain without changing the weights of the model *[[65](#bib.bib65 "")]*. Other methods fine-tune the later layers of both the encoders.
All these adaptation methods tune the parameters of the two encoders separately.

Joint multimodal strategies We hypothesize that the above common adaptation strategies don’t appropriately capture the cross-modal interaction required for strong attribute-object binding.
However, CLIP is significantly more lightweight than recent multimodal models. Hence, we explore lightweight multi-modal adaptation strategies to adapt CLIP.
We describe the best-performing multimodal adaptation strategy found in our experiments which is also depicted in Fig.[2(b)](#S3.F2.sf2 "Figure 2(b) ‣ Figure 2 ‣ 3 𝒞⁢𝑜⁢𝑙⁢𝑎 benchmark ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval").
This multi-modal adaptation borrows a transformer encoder-decoder*[[53](#bib.bib53 ""), [52](#bib.bib52 ""), [26](#bib.bib26 "")]* to attend over the image and language representations jointly.
Let $M\=[I;Q]$,
denote the concatenated image patch representations extracted from CLIP’s visual encoder and the token-level representations from the query. We compute a self-attention over $M$ using a transformer encoder
$A\=Att(M)$.
Finally, we use a classify token (a token randomly initialized), referred to as [CLS], that cross-attends 111query comes from [CLS], and the keys and values come from self-attended features, $A$. to all the self-attended features $A$ using a transformer decoder to produce $\text{out}_{MM}$.
This type of cross-attending to self-attended features is similar to FLAVA*[[52](#bib.bib52 "")]*/MDETR*[[26](#bib.bib26 "")]*.

The standard practice in most multi-modal encoder-based prediction models would be to learn a linear layer to classify the output [CLS] token embedding *[[26](#bib.bib26 ""), [52](#bib.bib52 ""), [33](#bib.bib33 ""), [11](#bib.bib11 "")]*.
However, instead of learning a linear predictor, we compute the cosine similarity of [CLS] to the representations of the query tokens: $q_{i}$ from the frozen text encoder, $Q$. We posit that aligning to a frozen text encoder trained on larger scale data will act as a regularizer, helping performance on unseen compositions.
Hence, the final score for a given image-query pair is $f(I,Q)\=\frac{1}{N_{q}}\sum_{i}^{N_{q}}out_{MM}\odot q_{i}$
where $N_{q}$ is the number of tokens in the query. These two ablations are referred to as MM-pred and MM-adapter (“adapter” since we can think of the latter as adapting the image features to align better with text) in our experiments.

We also tried various flavors of computing a multi-modal adaptation inspired by FIBER*[[11](#bib.bib11 "")]* and ALBEF*[[33](#bib.bib33 "")]*, which use cross attention between text and image. We would like to stress that while exploring newer ways to fuse vision and language features is a valid avenue of research, we are interested in exploring the common themes in current fusion methods that encourage compositionality the most to drive future research. We report the best method for simplicity and include the accuracies from other strategies in Table 5 in the supplemental.

5 Evaluation Setup
------------------

We evaluate models on the two types of queries described in Sec.[3](#S3 "3 𝒞⁢𝑜⁢𝑙⁢𝑎 benchmark ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") on $\mathcal{C}ola$ and CREPE*[[35](#bib.bib35 "")]* datasets.
All models are trained using Pytorch*[[38](#bib.bib38 "")]* and use the Huggingface transformers library*[[59](#bib.bib59 "")]*. Implementation details such as hyperparameters are provided in Sec. 11 of the supplementary.

### 5.1 Metrics

Single object queries. For this type of query, we report the $\mathcal{C}ola$ MAP222we also computed the F1 score and see trends remain the same; more details in the supplementary, Sec. 10, page 18., the mean average retrieval precision over difficult distractors.
We further differentiate the mAP between seen and unseen queries.
We split $\mathcal{C}ola$ into seen and unseen sets by removing some attribute-object pairs from the training set.
For example, “square white plate” is unseen if this combination is absent in finetuning; however, “square white bowl” or a “square plate” may be present.
In our test set, we have $320$ (150 unseen, 170 seen) queries on $1950$ images for GQA, $96$ (32 unseen, 64 seen)333we report “seen” only on 32 classes to avoid disbalance with unseen. “All” MAP is on all 96 classes. “Seen” trends hold same with 64 classes as well; more details in appendix. queries on $22500$ images for CLEVR, and $400$ (200 seen, 200 unseen) queries on $7921$ images in PACO.

Multi-object queries. Recall that we have two images and two captions, and the task is to match the correct caption to the correct image.
If we denote the prediction score for an image and query to be $f(I,M)$, we regard a prediction to be correct if $f(I,M)>f(I^{\prime},M)\And f(I^{\prime},M^{\prime})>f(I,M^{\prime})$, where images $I$ and $I^{\prime}$ are paired to captions $M$ and $M^{\prime}$ respectively.
Using this criterion, we compute the $\mathcal{C}ola$ multi-object accuracy.
The random accuracy is $25\%$ since there are four ways to match the two captions to the two images.
We also evaluate on a contemporary dataset, CREPE*[[35](#bib.bib35 "")]*, where the task is inverse. For CREPE, we compute an image-to-text (I2T) accuracy, where a model must match the correct text from two choices to the given image. Note that there is only one image for the two caption choices in CREPE*[[35](#bib.bib35 "")]*. The random accuracy is $50\%$ since it is a binary task.

### 5.2 Explored finetuning strategies

Recall that the best-performing finetuning approach we found is a multimodal adaptation (MM-Adapter) for tuning pre-trained image and text features as described in Sec.[4](#S4 "4 Exploring finetuning strategies with 𝒞⁢𝑜⁢𝑙⁢𝑎 ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"). We compare against popular tuning methods like linear probing, tuning the prompt embeddings (prompt-tuning), and fine-tuning the whole model (FT all) or the last two layers (FT Late). These adaptations are applied separately to the base model for comparison. More details are in the supplementary (Sec. 11). Since our adaptation uses multimodal attention, we also compare it to a seminal model that uses multimodal attention in pretraining. We chose FLAVA *[[52](#bib.bib52 "")]* since it is one of the recent models after CLIP which is bigger, more accurate*[[52](#bib.bib52 "")]* and has easily available pre-trained weights.

|  | GQA | | | CLEVR | | | PACO | | |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
|  | All | Unseen | Seen | All | Unseen | Seen | All | Unseen | Seen |
| a. CLIP | 36.53 | 39.06 | 34.24 | 15.38 | 15.01 | 15.32 | 12.21 | 8.64 | 15.79 |
| b. +Linear | 40.44 | 42.87 | 38.24 | 47.96 | 29.43 | 46.75 | 14.22 | 6.75 | 21.68 |
| c. +prompt-tune | 37.40 | 40.69 | 34.43 | 29.61 | 23.17 | 28.05 | 12.76 | 5.92 | 19.61 |
| d. +FT all | 38.81 | 40.85 | 36.95 | 52.32 | 19.00 | 47.95 | 14.58 | 6.49 | 22.66 |
| e. +FT late | 42.19 | 44.61 | 40.01 | 64.06 | 27.53 | 67.48 | 15.66 | 8.74 | 22.58 |
| f. +MM-Pred (us) | 45.99 | 48.6 | 43.64 | 75.80 | 51.98 | 80.72 | 15.49 | 8.00 | 22.94 |
| g. +MM-Adapter (us) | 46.83 | 48.86 | 44.99 | 88.21 | 89.52 | 77.00 | 18.56 | 11.47 | 25.66 |
| h. FLAVA | 39.65 | 42.18 | 37.37 | 15.41 | 13.27 | 15.93 | 12.53 | 7.29 | 17.76 |
| i. +Linear | 37.07 | 39.96 | 34.46 | 19.30 | 17.53 | 18.52 | 11.65 | 7.90 | 15.39 |
| j. +FT-late | 39.58 | 42.26 | 37.16 | 77.95 | 72.72 | 66.42 | 12.82 | 5.79 | 19.84 |
| k. +MM-Pred (us) | 47.12 | 51.53 | 43.13 | 90.43 | 85.78 | 86.07 | 18.57 | 10.71 | 26.44 |
| l. +MM-Adapter (us) | 48.54 | 52.55 | 44.91 | 91.10 | 86.64 | 87.39 | 19.36 | 11.16 | 27.55 |

*Table 1: mAP results on the $\mathcal{C}ola$ single object compounds setting. Our multimodal adaptation (MM-Adapter) performs better than common tuning methods. Further, multimodal attention to adapt the image representation (MM-Adapter) generalizes better than using it simply as a prediction head (MM-Pred). MM-Adapter on CLIP is better than tuning the pre-trained multimodal attention layers of the bigger FLAVA (+FT late). MM-Adapter further improves FLAVA.*

### 5.3 Finetuning datasets

The $\mathcal{C}ola$ training sets are also built in the same way as described in Sec.[3](#S3 "3 𝒞⁢𝑜⁢𝑙⁢𝑎 benchmark ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") using the training splits of GQA*[[18](#bib.bib18 ""), [29](#bib.bib29 "")]*, CLEVR*[[24](#bib.bib24 "")]* PACO*[[44](#bib.bib44 "")]*, and Visual Genome *[[29](#bib.bib29 "")]*.
For GQA, the training split contains 1381 objects and 601 attributes that compose 27078 queries on 74K images. For CLEVR, we have 3 shapes composed with 8 colors and 2 sizes on 70K images. Finally, for PACO, we have 75 objects and 55 attributes that compose 18696 queries on 37883 images.
The $\mathcal{C}ola$ multi-object compounds training split has 551,980 multi-object compounds on 71,174 images. Only the test split is cleaned using human annotations.
For datasets built on GQA*[[18](#bib.bib18 "")]* and Visual Genome*[[29](#bib.bib29 "")]*, we leverage the annotations to explore the effects of different kinds of data queries. We use the region descriptions (denoted as RegionCap in the tables) in Visual Genome to test if linguistic diversity helps over templated captions ($\mathcal{C}ola$ single objects and multi-object). We also compare to hard negatives from the $\mathcal{C}ola$ multi-object pairs. We finally have a combined setting where we use all data.

6 Results
---------

Recall that we evaluate on two settings for $\mathcal{C}ola$ - the single-object compounds setting and the multi-object compounds setting as defined in Sec.[3](#S3 "3 𝒞⁢𝑜⁢𝑙⁢𝑎 benchmark ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"). We discuss the quantitative results below and some qualitative results are shown in Fig.[3](#S6.F3 "Figure 3 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval").

### 6.1 $\mathcal{C}ola$ Single-object retrieval

Multimodal adaptation is more effective than other tuning methods: In Table [1](#S5.T1 "Table 1 ‣ 5.2 Explored finetuning strategies ‣ 5 Evaluation Setup ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), compared to prompt-tuning (row c), fine-tuning all of CLIP (row d), or fine-tuning a few of the later layers (row e), tuning a multimodal attention layer of same/lesser parameters has higher mAP (row f and g). Linear probing (row b), although cheaper, significantly underperforms.
This is not surprising since multimodal attention over the image regions and text tokens offers more flexibility to the model to learn to bind the right attributes to the right object region.
Tuning the whole model is also worse than tuning the later unimodal layers (row d vs e). This might be because fine-tuning the whole model requires larger batch sizes with significantly more data.
In Fig [3](#S6.F3 "Figure 3 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), we show the comparison of tuning unimodal layers vs our multimodal adaptation (since tuning the unimodal layers is closest in performance).
Qualitative examples from each of the other adaptation methods are displayed in Figs. 9-15 in the supplementary.

|  | Multi-Obj Acc $\uparrow$ | |
| --- | --- | --- |
|  | T2I | I2T |
|  | $\mathcal{C}ola$ | CREPE [[35](#bib.bib35 "")] |
| - Random | 25.00 | 50.00 |
| - Human | 83.88 | - |
| o. CLIP | 21.42 | 77.43 |
| a. + Linear | 30.47 | 87.35 |
| b. + Prompt-tune | 27.14 | 80.81 |
| c. + FT all | 34.76 | 82.39 |
| d. + FT late | 36.19 | 87.14 |
| e. + MM-Pred (our) | 41.42 | 77.84 |
| f. + MM-Adapter (our) | 40.95 | 87.02 |
| g. FLAVA | 24.76 | 65.10 |
| h. + Linear | 22.38 | 55.10 |
| i. + FT late | 22.38 | 58.11 |
| j. + MM-Pred (our) | 39.04 | 81.37 |
| k. + MM-Adapter (our) | 40.47 | 74.81 |

*(a)*

| Data Type | Single-Object $\mathcal{C}ola$ GQA | | |
| --- | --- | --- | --- |
| | All | Unseen | Seen |
| a. RegionCap | 0.4711 | 0.4965 | 0.4481 |
| b. SingleObj | 0.4683 | 0.4886 | 0.4499 |
| c. + MultiObj | 0.4641 | 0.4795 | 0.4501 |
| d. + HardNeg | 0.4688 | 0.4843 | 0.4548 |
| e. Combined | 0.4788 | 0.4983 | 0.4612 |
|  | Multi-Object | |  |
|  | $\mathcal{C}ola$ | CREPE |  |
| a. RegionCap | 0.3114 | 0.8833 |  |
| b. SingleObj | 0.2745 | 0.9023 |  |
| c. + MultiObj | 0.3975 | 0.8702 |  |
| d. + HardNeg | 0.3483 | 0.8775 |  |
| e. Combined | 0.3893 | 0.8798 |  |

*(b)*

*Table 2: a. Results on our multi-object compounds setting on our $\mathcal{C}ola$ task and CREPE. Simpler methods suffice on CREPE, but not on $\mathcal{C}ola$, suggesting that $\mathcal{C}ola$ is harder. Red-orange-yellow is in decreasing accuracy order. MM-Adapter and MM-Pred on CLIP perform well on average on both. b. Table showing the effect of the data type used in the contrastive batch training. Having multi-object captions in the data helps $\mathcal{C}ola$ performance while maintaining CREPE performance.*

<img src='figures/qual_results_new.png' alt='Refer to caption' title='' width='598' height='312' />

*Figure 3: Qualitative results of multi-object matching (left) and retrieving a single object with multiple attributes (right).*

Using pre-trained multimodal attention layers is not enough - training them on attribute-object compositions is key: In Table [1](#S5.T1 "Table 1 ‣ 5.2 Explored finetuning strategies ‣ 5 Evaluation Setup ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), we see that MM-Adapter (row g) on CLIP ViT B-32 (151 M params) outperforms tuning the last two multimodal layers of FLAVA B-16 *[[52](#bib.bib52 "")]* (241M params) model (row j) or tuning a linear probe (row i). Surprisingly, tuning the last two FLAVA multimodal layers (row j) is worse than replacing them and training using MM-Pred and MM-Adapter layers (rows k and l). This suggests that training multimodal layers during adaptation (as opposed to pre-training) is key.

MM-Adapter is better than using multimodal attention as a prediction head: Recall that one of the ablations of our approach is aligning the output of the multimodal encoder to the frozen text embedding, making it a multimodal image-feature “adapter" (MM-Adapter). This contrasts to using the multimodal module as a prediction head with a linear layer (MM-Pred). As shown in Table [1](#S5.T1 "Table 1 ‣ 5.2 Explored finetuning strategies ‣ 5 Evaluation Setup ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), MM-Adapter outperforms MM-Pred (row g vs f), especially on unseen classes and on exhaustively annotated datasets like CLEVR *[[24](#bib.bib24 "")]* and PACO *[[44](#bib.bib44 "")]*.
We posit that aligning to the frozen text representation acts like a regularizer since it was pre-trained on more data.

### 6.2 Multi-object retrieval

Simpler methods suffice for CREPE, but not for $\mathcal{C}ola$, suggesting that $\mathcal{C}ola$ is a harder task: As shown in Table [2(a)](#S6.T2.st1 "Table 2(a) ‣ Table 2 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), linear probing (row a) or simple fine-tuning (rows c and d) suffice for CREPE *[[35](#bib.bib35 "")]*. However, our MM-Adapter improves further on $\mathcal{C}ola$ (row f vs row a, b, c, d), while maintaining performance on CREPE. This also suggests that text-to-image matching is harder than image-to-text matching, which is also reflected in Winoground *[[55](#bib.bib55 "")]*.

Baseline CLIP and FLAVA perform below chance: If we evaluate off-the-shelf CLIP *[[43](#bib.bib43 "")]* and FLAVA *[[52](#bib.bib52 "")]* on our $\mathcal{C}ola$ dataset, we see in Table [2(a)](#S6.T2.st1 "Table 2(a) ‣ Table 2 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") that it performs below random (row o and g). This is consistent with the findings in Winoground *[[55](#bib.bib55 "")]*.

Training late multimodal layers from scratch help, CLIP+MM-Adapter performs better overall: We improve performance by training multimodal layers from scratch on top of CLIP and FLAVA as shown in Table [2(a)](#S6.T2.st1 "Table 2(a) ‣ Table 2 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), row e, f, j, k.
Interestingly, as shown in Table [2(a)](#S6.T2.st1 "Table 2(a) ‣ Table 2 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), linear probing or tuning the pre-trained multimodal layers of FLAVA hurts performance (row g vs row h and i). This could be because tuning adversely perturbs the parameters trained on large-scale data.
Finally, CLIP+MM-Adapter (row f) performs comparably well as tuning multimodal layers on FLAVA (row f vs j and k).

<img src='figures/qual_examples_failure_lessons.png' alt='Refer to caption' title='' width='598' height='297' />

*Figure 4: Qualitative results on cases where models struggle with multiple object-attribute compositionality (left). Cases where we see the most improvement and the least on single-object compositional retrieval are shown on the right.*

### 6.3 Effect of fine-tuning data

Difference between free-form captions and templated $\mathcal{C}ola$ queries is minimal. Hence, $\mathcal{C}ola$ templated queries are useful: In Table [2(b)](#S6.T2.st2 "Table 2(b) ‣ Table 2 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") (row a vs c, d, e), we see that templated queries perform just as well as free-form region descriptions. This shows that $\mathcal{C}ola$ queries are still useful for attribute-object binding despite not being free-form.

Having multiple objects with multiple attributes in a caption helps: When we combine single object captions with multiple object captions, we see in Table [2(b)](#S6.T2.st2 "Table 2(b) ‣ Table 2 ‣ 6.1 𝒞⁢𝑜⁢𝑙⁢𝑎 Single-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") that performance increases especially on the multi-object $\mathcal{C}ola$ setting while maintaining performance on the single-object setting (row b vs c). The multi-object and single-object captions are formed on the same number of images.
Combining all types of data along with hard negatives (row e) doesn’t seem to affect performance much.

### 6.4 Quality of $\mathcal{C}ola$ Benchmark

Our $\mathcal{C}ola$ mAPs are harder and are more difficult to improve on: We also computed the performance based on the standard formulation of mAP, where all images are included in the list to retrieve from. In contrast, $\mathcal{C}ola$ mAP only includes hard distractors as defined in Sec.[3](#S3 "3 𝒞⁢𝑜⁢𝑙⁢𝑎 benchmark ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"). We see that trends remain the same with the standard mAP. However, our $\mathcal{C}ola$ mAP is harder to improve on. The standard MAP (supplementary, Table 4) improves by more than 2x on GQA and 10x on CLEVR. In contrast, we only improve by 1.09x on GQA and 2x on CLEVR with our harder $\mathcal{C}ola$ MAP.

Ambiguous colors, spatial relationships, and size are some common themes where models underperform in $\mathcal{C}ola$ benchmark: We analyze the types of compositional queries that models find difficult on our benchmark. In the qualitative examples shown in Fig. [4](#S6.F4 "Figure 4 ‣ 6.2 Multi-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), we see that compositions involving ambiguous colors (due to lighting) and spatial relationships are difficult for multiple-object cases. This is likely because spatial relationships are often inconsistently annotated in training data - *e.g*. “to the left of” can sometimes be from the viewer or image perspective. More examples are in the supplemental- Fig. 8 (data ambiguity), 20, 21 (prediction inaccuracy).

Significant improvements on non-salient/occluded objects. Improvements on larger objects are minimal: On attribute-object compositional retrieval for single objects, we see the most improvements using our adaptation method are on non-salient or occluded objects (like a small sign), as shown in Fig. [4](#S6.F4 "Figure 4 ‣ 6.2 Multi-object retrieval ‣ 6 Results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") (right). Queries that are commonly represented in training sets (like “clear blue sky” - skies are most commonly blue) have minimal improvements from pre-trained models.

7 Discussion and Limitations
----------------------------

This work finetunes vision-language models to test design choices for compositional attribute-object retrieval. Thus, compared to the original pre-trained models*[[43](#bib.bib43 "")]*, we may lose some other generic capabilities, such as question answering, captioning, etc. For example, our best adaptation scores $83\%$ zero-shot on CIFAR10 *[[30](#bib.bib30 "")]*, as compared to $87\%$ for pre-trained CLIP. However, by just using $5\%$ of CIFAR training data (less than 1 epoch), we reach $92\%$ accuracy while maintaining performance on our $\mathcal{C}ola$ task.
We would also like to stress that our goal is not to train a new foundation model but to explore design choices that allow for compositional reasoning. While we focus on attribute-object compositionality, there is still significant room for exploration of other types of compositional structures such as relationships, scene graphs, and counting*[[15](#bib.bib15 ""), [12](#bib.bib12 "")]*. A great avenue for future work could be collecting more detailed annotations (about the type of objects and compositions in the queries) on larger sets of images to help pinpoint themes where models are failing. Further testing is also required to see how it fares with sensitive attributes and objects— whether it is predisposed towards attaching incorrect attributes to objects because of racial/political biases in the data*[[6](#bib.bib6 "")]*.
Additionally, it is important to re-evaluate our results in the context of newer vision-language models that are being proposed.
Finally, we would like to point out that our finetuning strategy isn’t specific to compositions, suggesting its potential applicability for adaptation to other downstream tasks, especially since similar strategies and modules have been used for other computer vision tasks*[[53](#bib.bib53 ""), [52](#bib.bib52 ""), [26](#bib.bib26 "")]*.

8 Conclusion
------------

We present a new task, $\mathcal{C}ola$, to test the compositional attribute-object binding of vision-language models. This is important for various practical applications like an assistive agent requiring an understanding of fine-grained differences between objects in cluttered workplaces.
We explore the architectural choices of adapting large vision-language models that encourage such reasoning. We show that a light-weight multimodal adaptor can improve this capability in a pre-trained vision-language model as a strong baseline for further research.
We hope that $\mathcal{C}ola$ serves as a strong benchmark and our adaptation choices as strong baselines for improving compositional vision-language intelligence.

Acknowledgements. We wish to thank Dhruv Mahajan and Jang Hyun (Vincent) Cho for their valuable guidance in the initial phases of the project. We also wish to thank the anonymous reviewers for their thoughtful comments and suggestions. This material is based upon work supported, in part, by DARPA under agreement number HR00112020054 awarded to Kate Saenko and Bryan Plummer at BU. Any opinions, findings, and conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the supporting agencies.

References
----------

* [1]Aishwarya Agrawal, Aniruddha Kembhavi, Dhruv Batra, and Devi Parikh.C-VQA: A Compositional Split of the Visual Question Answering (VQA) v1.0 Dataset, Apr. 2017.arXiv:1704.08243 [cs].
* [2]Guillaume Alain and Yoshua Bengio.Understanding intermediate layers using linear classifier probes, Nov. 2018.arXiv:1610.01644 [cs, stat].
* [3]Jean-Baptiste Alayrac, Jeff Donahue, Pauline Luc, Antoine Miech, Iain Barr, Yana Hasson, Karel Lenc, Arthur Mensch, Katie Millican, Malcolm Reynolds, Roman Ring, Eliza Rutherford, Serkan Cabi, Tengda Han, Zhitao Gong, Sina Samangooei, Marianne Monteiro, Jacob Menick, Sebastian Borgeaud, Andrew Brock, Aida Nematzadeh, Sahand Sharifzadeh, Mikolaj Binkowski, Ricardo Barreira, Oriol Vinyals, Andrew Zisserman, and Karen Simonyan.Flamingo: a Visual Language Model for Few-Shot Learning, Nov. 2022.arXiv:2204.14198 [cs].
* [4]Amit Alfassy, Assaf Arbelle, Oshri Halimi, Sivan Harary, Roei Herzig, Eli Schwartz, Rameswar Panda, Michele Dolfi, Christoph Auer, Kate Saenko, PeterW J. Staar, Rogerio Feris, and Leonid Karlinsky.FETA: Towards Specializing Foundation Models for Expert Task Applications, Dec. 2022.arXiv:2209.03648 [cs].
* [5]Hyojin Bahng, Ali Jahanian, Swami Sankaranarayanan, and Phillip Isola.Exploring Visual Prompts for Adapting Large-Scale Models, June 2022.arXiv:2203.17274 [cs].
* [6]Joy Buolamwini and Timnit Gebru.Gender shades: Intersectional accuracy disparities in commercial gender classification.In Conference on fairness, accountability and transparency, pages 77–91. PMLR, 2018.
* [7]Noam Chomsky.Aspects of the Theory of Syntax.The MIT Press, 50 edition, 1965.
* [8]Gordon Christie, Ankit Laddha, Aishwarya Agrawal, Stanislaw Antol, Yash Goyal, Kevin Kochersberger, and Dhruv Batra.Resolving Language and Vision Ambiguities Together: Joint Segmentation \& Prepositional Attachment Resolution in Captioned Scenes, Sept. 2016.arXiv:1604.02125 [cs].
* [9]MJ Cresswell.Logics and languages.1973.
* [10]M. J. Cresswell.Logics and Languages.Synthese, 40(2):375–387, 1973.Publisher: Springer.
* [11]Zi-Yi Dou, Aishwarya Kamath, Zhe Gan, Pengchuan Zhang, Jianfeng Wang, Linjie Li, Zicheng Liu, Ce Liu, Yann LeCun, Nanyun Peng, Jianfeng Gao, and Lijuan Wang.Coarse-to-Fine Vision-Language Pre-training with Fusion in the Backbone, Nov. 2022.arXiv:2206.07643 [cs].
* [12]Mona Gandhi, Mustafa Omer Gul, Eva Prakash, Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala.Measuring compositional consistency for video question answering.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 5046–5055, 2022.
* [13]Peng Gao, Shijie Geng, Renrui Zhang, Teli Ma, Rongyao Fang, Yongfeng Zhang, Hongsheng Li, and Yu Qiao.CLIP-Adapter: Better Vision-Language Models with Feature Adapters, Oct. 2021.arXiv:2110.04544 [cs].
* [14]Timnit Gebru, Jamie Morgenstern, Briana Vecchione, Jennifer Wortman Vaughan, Hanna Wallach, Hal Daumé III au2, and Kate Crawford.Datasheets for datasets, 2021.
* [15]Madeleine Grunde-McLaughlin, Ranjay Krishna, and Maneesh Agrawala.Agqa: A benchmark for compositional spatio-temporal reasoning.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 11287–11297, 2021.
* [16]Sheng Guo, Weilin Huang, Xiao Zhang, Prasanna Srikhanta, Yin Cui, Yuan Li, Hartwig Adam, Matthew R. Scott, and Serge Belongie.The iMaterialist Fashion Attribute Dataset.In 2019 IEEE/CVF International Conference on Computer Vision Workshop (ICCVW), pages 3113–3116, Seoul, Korea (South), Oct. 2019. IEEE.
* [17]Kaiming He, Haoqi Fan, Yuxin Wu, Saining Xie, and Ross Girshick.Momentum Contrast for Unsupervised Visual Representation Learning, Mar. 2020.arXiv:1911.05722 [cs].
* [18]Drew A. Hudson and Christopher D. Manning.GQA: A New Dataset for Real-World Visual Reasoning and Compositional Question Answering, May 2019.arXiv:1902.09506 [cs].
* [19]Phillip Isola, Joseph J. Lim, and Edward H. Adelson.Discovering states and transformations in image collections.In 2015 IEEE Conference on Computer Vision and Pattern Recognition (CVPR), pages 1383–1391, Boston, MA, USA, June 2015. IEEE.
* [20]Jingwei Ji, Ranjay Krishna, Li Fei-Fei, and Juan Carlos Niebles.Action genome: Actions as compositions of spatio-temporal scene graphs.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 10236–10247, 2020.
* [21]Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc Le, Yun-Hsuan Sung, Zhen Li, and Tom Duerig.Scaling up visual and vision-language representation learning with noisy text supervision.In International Conference on Machine Learning, pages 4904–4916. PMLR, 2021.
* [22]Chao Jia, Yinfei Yang, Ye Xia, Yi-Ting Chen, Zarana Parekh, Hieu Pham, Quoc V. Le, Yunhsuan Sung, Zhen Li, and Tom Duerig.Scaling Up Visual and Vision-Language Representation Learning With Noisy Text Supervision, June 2021.arXiv:2102.05918 [cs].
* [23]Menglin Jia, Luming Tang, Bor-Chun Chen, Claire Cardie, Serge Belongie, Bharath Hariharan, and Ser-Nam Lim.Visual Prompt Tuning, July 2022.arXiv:2203.12119 [cs].
* [24]Justin Johnson, Bharath Hariharan, Laurens van der Maaten, Li Fei-Fei, C. Lawrence Zitnick, and Ross Girshick.CLEVR: A Diagnostic Dataset for Compositional Language and Elementary Visual Reasoning, Dec. 2016.arXiv:1612.06890 [cs].
* [25]Aishwarya Kamath, Sara Price, Jonas Pfeiffer, Yann LeCun, and Nicolas Carion.TRICD: Testing Robust Image Understanding Through Contextual Phrase Detection.2023.
* [26]Aishwarya Kamath, Mannat Singh, Yann LeCun, Gabriel Synnaeve, Ishan Misra, and Nicolas Carion.MDETR – Modulated Detection for End-to-End Multi-Modal Understanding, Oct. 2021.arXiv:2104.12763 [cs].
* [27]Gwanghyun Kim, Taesung Kwon, and Jong Chul Ye.DiffusionCLIP: Text-Guided Diffusion Models for Robust Image Manipulation, Aug. 2022.arXiv:2110.02711 [cs].
* [28]Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Li Fei-Fei.Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations.International Journal of Computer Vision, 123(1):32–73, May 2017.
* [29]Ranjay Krishna, Yuke Zhu, Oliver Groth, Justin Johnson, Kenji Hata, Joshua Kravitz, Stephanie Chen, Yannis Kalantidis, Li-Jia Li, David A. Shamma, Michael S. Bernstein, and Fei-Fei Li.Visual Genome: Connecting Language and Vision Using Crowdsourced Dense Image Annotations, Feb. 2016.arXiv:1602.07332 [cs].
* [30]Alex Krizhevsky, Geoffrey Hinton, et al.Learning multiple layers of features from tiny images.2009.
* [31]Michael A Lepori, Thomas Serre, and Ellie Pavlick.Break it down: Evidence for structural compositionality in neural networks.arXiv preprint arXiv:2301.10884, 2023.
* [32]Brian Lester, Rami Al-Rfou, and Noah Constant.The Power of Scale for Parameter-Efficient Prompt Tuning, Sept. 2021.arXiv:2104.08691 [cs].
* [33]Junnan Li, Ramprasaath R. Selvaraju, Akhilesh Deepak Gotmare, Shafiq Joty, Caiming Xiong, and Steven Hoi.Align before Fuse: Vision and Language Representation Learning with Momentum Distillation, Oct. 2021.arXiv:2107.07651 [cs].
* [34]Liunian Harold Li, Pengchuan Zhang, Haotian Zhang, Jianwei Yang, Chunyuan Li, Yiwu Zhong, Lijuan Wang, Lu Yuan, Lei Zhang, Jenq-Neng Hwang, Kai-Wei Chang, and Jianfeng Gao.Grounded Language-Image Pre-training, June 2022.arXiv:2112.03857 [cs].
* [35]Zixian Ma, Jerry Hong, Mustafa Omer Gul, Mona Gandhi, Irena Gao, and Ranjay Krishna.CREPE: Can Vision-Language Foundation Models Reason Compositionally?, Jan. 2023.arXiv:2212.07796 [cs].
* [36]Oscar Mañas, Pau Rodriguez, Saba Ahmadi, Aida Nematzadeh, Yash Goyal, and Aishwarya Agrawal.MAPL: Parameter-Efficient Adaptation of Unimodal Pre-Trained Models for Vision-Language Few-Shot Prompting, Oct. 2022.arXiv:2210.07179 [cs].
* [37]Nihal V. Nayak, Peilin Yu, and Stephen H. Bach.Learning to Compose Soft Prompts for Compositional Zero-Shot Learning, Sept. 2022.arXiv:2204.03574 [cs].
* [38]Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala.PyTorch: An Imperative Style, High-Performance Deep Learning Library, Dec. 2019.arXiv:1912.01703 [cs, stat].
* [39]Xavier Puig, Kevin Ra, Marko Boben, Jiaman Li, Tingwu Wang, Sanja Fidler, and Antonio Torralba.VirtualHome: Simulating Household Activities Via Programs.In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8494–8502, Salt Lake City, UT, June 2018. IEEE.
* [40]Filip Radenovic, Abhimanyu Dubey, Abhishek Kadian, Todor Mihaylov, Simon Vandenhende, Yash Patel, Yi Wen, Vignesh Ramanathan, and Dhruv Mahajan.Filtering, Distillation, and Hard Negatives for Vision-Language Pre-Training, Jan. 2023.arXiv:2301.02280 [cs].
* [41]Filip Radenovic, Animesh Sinha, Albert Gordo, Tamara Berg, and Dhruv Mahajan.Large-Scale Attribute-Object Compositions, May 2021.arXiv:2105.11373 [cs].
* [42]Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al.Learning transferable visual models from natural language supervision.In International Conference on Machine Learning, pages 8748–8763. PMLR, 2021.
* [43]Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, Gretchen Krueger, and Ilya Sutskever.Learning Transferable Visual Models From Natural Language Supervision, Feb. 2021.arXiv:2103.00020 [cs].
* [44]Vignesh Ramanathan, Anmol Kalia, Vladan Petrovic, Yi Wen, Baixue Zheng, Baishan Guo, Rui Wang, Aaron Marquez, Rama Kovvuri, Abhishek Kadian, Amir Mousavi, Yiwen Song, Abhimanyu Dubey, and Dhruv Mahajan.PACO: Parts and Attributes of Common Objects, Jan. 2023.arXiv:2301.01795 [cs].
* [45]Arijit Ray, Gordon Christie, Mohit Bansal, Dhruv Batra, and Devi Parikh.Question Relevance in VQA: Identifying Non-Visual And False-Premise Questions, Sept. 2016.arXiv:1606.06622 [cs].
* [46]Arijit Ray, Karan Sikka, Ajay Divakaran, Stefan Lee, and Giedrius Burachas.Sunny and Dark Outside?! Improving Answer Consistency in VQA through Entailed Question Generation, Sept. 2019.arXiv:1909.04696 [cs].
* [47]Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer.High-Resolution Image Synthesis with Latent Diffusion Models, Apr. 2022.arXiv:2112.10752 [cs].
* [48]Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister.Prefix Conditioning Unifies Language and Label Supervision, June 2022.arXiv:2206.01125 [cs].
* [49]Kuniaki Saito, Kihyuk Sohn, Xiang Zhang, Chun-Liang Li, Chen-Yu Lee, Kate Saenko, and Tomas Pfister.Pic2Word: Mapping Pictures to Words for Zero-shot Composed Image Retrieval, Feb. 2023.arXiv:2302.03084 [cs].
* [50]Madeline Chantry Schiappa, Michael Cogswell, Ajay Divakaran, and Yogesh Singh Rawat.Probing conceptual understanding of large visual-language models, 2023.
* [51]Meet Shah, Xinlei Chen, Marcus Rohrbach, and Devi Parikh.Cycle-Consistency for Robust Visual Question Answering.In 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 6642–6651, Long Beach, CA, USA, June 2019. IEEE.
* [52]Amanpreet Singh, Ronghang Hu, Vedanuj Goswami, Guillaume Couairon, Wojciech Galuba, Marcus Rohrbach, and Douwe Kiela.FLAVA: A Foundational Language And Vision Alignment Model, Mar. 2022.arXiv:2112.04482 [cs].
* [53]Hao Tan and Mohit Bansal.LXMERT: Learning Cross-Modality Encoder Representations from Transformers, Dec. 2019.arXiv:1908.07490 [cs].
* [54]Reuben Tan, Arijit Ray, Andrea Burns, Bryan A. Plummer, Justin Salamon, Oriol Nieto, Bryan Russell, and Kate Saenko.Language-guided audio-visual source separation via trimodal consistency, 2023.
* [55]Tristan Thrush, Ryan Jiang, Max Bartolo, Amanpreet Singh, Adina Williams, Douwe Kiela, and Candace Ross.Winoground: Probing Vision and Language Models for Visio-Linguistic Compositionality, Apr. 2022.arXiv:2204.03162 [cs].
* [56]Ben Usman, Dina Bashkirova, and Kate Saenko.Disentangled Unsupervised Image Translation via Restricted Information Flow, Nov. 2021.arXiv:2111.13279 [cs].
* [57]Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin.Attention Is All You Need, Dec. 2017.arXiv:1706.03762 [cs].
* [58]Catherine Wah, Steve Branson, Peter Welinder, Pietro Perona, and Serge Belongie.The Caltech-UCSD Birds-200-2011 Dataset, July 2011.Issue: 2010-001 Num Pages: 8 Number: 2010-001 Place: Pasadena, CA Publisher: California Institute of Technology.
* [59]Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Remi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander Rush.Transformers: State-of-the-Art Natural Language Processing.In Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 38–45, Online, Oct. 2020. Association for Computational Linguistics.
* [60]Yongqin Xian, Christoph H. Lampert, Bernt Schiele, and Zeynep Akata.Zero-Shot Learning – A Comprehensive Evaluation of the Good, the Bad and the Ugly, Sept. 2020.arXiv:1707.00600 [cs].
* [61]Aron Yu and Kristen Grauman.Fine-Grained Visual Comparisons with Local Learning.In 2014 IEEE Conference on Computer Vision and Pattern Recognition, pages 192–199, Columbus, OH, USA, June 2014. IEEE.
* [62]Mert Yuksekgonul, Federico Bianchi, Pratyusha Kalluri, Dan Jurafsky, and James Zou.When and why vision-language models behave like bags-of-words, and what to do about it?, Oct. 2022.arXiv:2210.01936 [cs].
* [63]Haotian Zhang, Pengchuan Zhang, Xiaowei Hu, Yen-Chun Chen, Liunian Harold Li, Xiyang Dai, Lijuan Wang, Lu Yuan, Jenq-Neng Hwang, and Jianfeng Gao.GLIPv2: Unifying Localization and Vision-Language Understanding, Oct. 2022.arXiv:2206.05836 [cs].
* [64]Pengchuan Zhang, Xiujun Li, Xiaowei Hu, Jianwei Yang, Lei Zhang, Lijuan Wang, Yejin Choi, and Jianfeng Gao.Vinvl: Revisiting visual representations in vision-language models.In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), pages 5579–5588, June 2021.
* [65]Kaiyang Zhou, Jingkang Yang, Chen Change Loy, and Ziwei Liu.Learning to Prompt for Vision-Language Models.International Journal of Computer Vision, 130(9):2337–2348, Sept. 2022.arXiv:2109.01134 [cs].

9 Additional Data Details
-------------------------

Our data is publicly available at <https://github.com/arijitray1993/COLA>.
Recall that our data is built on top of four publicly available datasets. We provide some more details on how we curated our data.

$\mathcal{C}ola$ Single Object Data

GQA GQA has annotations of objects and attributes in images. We use this to construct queries like “square white plate”. We ignore bounding boxes. For test, we filter the images with at least 2 attributes per object annotation in their test split.
We are left with 1952 images and 320 queries on our test set.
To create a challenging set where queries are unseen, we take 150 attribute object tuples from the 320 queries from the test set that are seen the least in the training set and remove those images and queries from the training set completely.
This way, we end up with 150 unseen queries with the least impact on the training set size.
We report all numbers on this test set of 150 unseen and 170 seen queries.
We train on the GQA train split (with the test unseen queries and corresponding images removed). Hence, we have around 67K training images and 27K queries. The number of paired examples is 450K image-text pairs.

CLEVR On CLEVR, we test on 96 classes on 22,500 images.
We use their compositional splits.
We train on condition A as described in their paper and dataset website and test on condition B.
In these two splits, cubes and cylinders have unseen color and size compositions. However, for spheres, all colors and shapes are seen.
Since MAP is sensitive to the number of classes, we keep the number of classes for seen and unseen the same. Hence, we leave out the spheres when reporting seen vs unseen. However, for all MAP, we report including spheres.
Hence, we have 32 unseen classes, 32 seen classes and 96 classes for “all”.
For training, we have 168 possible queries (with colors swapped for cubes and cylinders from those in the test set) on 70K images.

PACO The PACO *[[44](#bib.bib44 "")]* dataset has 55 attributes annotated on 75 object categories on 9443 images on the test set. Since all combinations of objects and attributes would result in an intractable amount of possible compositions, we sample the 400 occurring multiple attribute-object compositions in the test set. The 400 classes are sampled by sampling the top 200 seen attribute-object queries and the top 200 unseen attribute-object queries. An attribute-object query is defined as unseen if the attributes in conjunction with that object were never a subset of the attributes in conjunction with that object in the training data.
This way, we have 400 classes on 7921 images, on which we report numbers.
We have 37K training images and 18K queries and 55K paired image-text examples.

$\mathcal{C}ola$ Multi-Obj Data The multi-obj data was created on the train and test splits according to the $\mathcal{C}ola$ Single-Object GQA data splits. Only the test split is cleaned using human annotations.
We show some qualitative examples of the human-cleaned test set in Figure [7](#S11.F7 "Figure 7 ‣ 11.1 Model architecture details ‣ 11 Implementation details ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval").
We also see that some cases remain ambiguous even after human cleaning, as shown in Figure [8](#S11.F8 "Figure 8 ‣ 11.1 Model architecture details ‣ 11 Implementation details ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"). These often involve differences in perception of size (large vs small) and color (blue vs white under different lighting conditions). However, despite some minimal noise, the human accuracy of 10 independent workers on our validation set is 84%. This is opposed to our best model with an accuracy of 45%. Hence, we believe there is significant room for improvement.

### Datasheets for Datasets Answers

We believe that the majority of the questions in the datasheets paper *[[14](#bib.bib14 "")]* have already been answered in the main paper. Here, we provide some additional answers.
We also provide a histogram of the types of objects and attributes in our data in Figure [6](#S10.F6 "Figure 6 ‣ 10 More metrics and analysis ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval")

Dataset funding agency This project was supported by DARPA Semafor awarded to KS and BP. The findings and results reported in the paper are not the opinions of the US Government or the Department of Defense.

Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?  The dataset is a curated sample from larger datasets specifically aimed to test attribute-object compositionality of models.

What data does each instance consist of? Raw images, text captions, text-based scene graphs of objects and attributes in the image. Note that some objects and attribute annotations may be missing.

Is any information missing from individual instances? Yes, since it is very diffiucult to exhaustively annotate all possible objects and attributes, it is possible that some annotations are missing.

Is the dataset self-contained, or does it link to or otherwise rely on
external resources (e.g., websites, tweets, other datasets)? Since the data is built on top of publicly available datasets, some of the annotations, like scene graphs, are linked to the external dataset.

Does the dataset contain data that might be considered confidential Not that we are aware of since we use publicly available data from a published dataset.

Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety? Not that we are aware of.

Does the dataset identify any subpopulations (e.g., by age, gender)?  No personally identifiable information is present in the data. We also do not conduct any analyses with sensitive attributes like race, age, sexual orientation, or religion. Some attributes like sex and hair color may be annotated in the images, but we don’t explicitly analyze them since that is not the focus of the benchmark or the paper.

Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other
data) from the dataset? No personally identifiable information is present in the data. It may be purely coincidental that a person in real life may be present in the images of the data.

Does the dataset contain data that might be considered sensitive
in any way? Not that we are aware of.

Over what timeframe was the data collected? Visual Genome *[[29](#bib.bib29 "")]* was collected in 2016. GQA *[[18](#bib.bib18 "")]* in 2018. PACO *[[44](#bib.bib44 "")]* in 2022-23. CLEVR *[[24](#bib.bib24 "")]* in 2016.

Who was involved in the data collection process (e.g., students,
crowdworkers, contractors) and how were they compensated? Students ran the data collection and crowdworkers annotated the data. We do not know how much they were compensated for the datasets we build on top of. However, for our human cleaned multi-object $\mathcal{C}ola$ test set, we paid crowdworkers an average of 15 USD per hour with bonuses if they annotated examples correctly.

Were any ethical review processes conducted? Yes, we were exempted by IRB since the data didn’t involve any personal or sensitive information.

Has an analysis of the potential impact of the dataset and its use
on data subjects (e.g., a data protection impact analysis) been conducted? No, but this could be interesting future work.

Is the software that was used to preprocess/clean/label the data
available? Yes, we shall release the code used to curate the data.

Will the dataset be distributed under a copyright, what are the IP restrictions, and export control restrictions? No such restrictions are present. The licenses are the same as the licenses of the datasets our benchmark is built on.

Will the dataset be updated? We may update the data with more examples or more annotations periodically.

<img src='figures/CLEVR_numattributes_formatted.png' alt='Refer to caption' title='' width='568' height='309' />

*Figure 5: The MAP numbers by the number of attributes in the query on the CLEVR dataset. Note how MM-Adapter performs well even as the number of attributes is gradually increased.*

10 More metrics and analysis
----------------------------

Recall, that our goal is to adapt vision-language features to improve the compositional binding of attributes to objects. Specifically, we aim to improve the classification of a query involving single or multiple objects with multiple attributes in an image.
Hence, we perform some analysis to see how our performance is affected by increasing the number of attributes.
Recall, that we also report numbers on our $\mathcal{C}ola$ MAP, which evaluates the model’s capability to rank the images with the correct attachment of attributes to the desired object from hard distractors with incorrect attachments of attributes to objects. We also show results some other choices of MAP and the standard MAP used commonly. We show how our choice of MAP in the main paper is harder even though all trends remain the same with all choices of MAP.
Finally, we also show performance from other choices of doing multimodal fusion for our MM-Adapter and MM-pred approaches, showing that this adaptation strategy holds for various other choices as well.

Performance by number of attributes We vary the number of attributes in the query for a single object setting and check performance with increasing attributes. The results are shown in Figure [5](#S9.F5 "Figure 5 ‣ Datasheets for Datasets Answers ‣ 9 Additional Data Details ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"). We see that the baseline CLIP and finetuning has higher performance on single-attribute queries than multi-attribute queries. We show that our MM-Adapter maintains improved performance on both the single-attribute and multi-attribute cases.

Other evaluation metrics

QueryAll MAP Recall that in the main paper, we compute MAP among hard distractors for the $\mathcal{C}ola$ single-object setting. The hard distractors are images that have any of the attributes and object words in the query. The model needs to rank the images with the correct attachment of the attributes to the desired object (as opposed to simply existing somewhere) to achieve a higher MAP.
Here, we design another similar hard MAP. Here, we restrict the list of images in the pool to have all the query attributes and objects. Hence, for a query “cyan metal cylinder”, we rank among images that have “cylinders” AND “cyan” objects AND “metal” objects. In the main paper, the MAP uses an OR instead of an AND operation.
The results are shown in Table [3](#S10.T3 "Table 3 ‣ 10 More metrics and analysis ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") and we observe that all trends remain the same with this metric as presented in the main paper.
However, we observe that that this metric can only be applied to CLEVR since annotations are exhaustive. In real datasets like GQA, the number of such annotated hard distractors is limited; hence, we do an OR operation to keep a high number of images to rank from. When applied to a dataset like GQA, the trends are the same, but the numbers are spuriously high since there are very few distractor images.

Mean Rank of GT ($\checkmark$) vs distractors ($\bm{\times}$) Based on the hard distractors we made for the QueryAll MAP above, we also report the mean rank of the images with the correct attachment of attributes to the object versus the mean rank of the images with the wrong attachment of attributes. The results are also shown in Table [3](#S10.T3 "Table 3 ‣ 10 More metrics and analysis ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval").
We observe that all trends remain the same, so report only one of them in the main paper.

<img src='figures/histograms.png' alt='Refer to caption' title='' width='586' height='94' />

*Figure 6: Overview of the types of attributes and objects in our data. They correspond to practical objects in daily life.*

|  | QueryAll MAP $\uparrow$ | | | Mean Rank | |
| --- | --- | --- | --- | --- | --- |
|  | All | Seen | Unseen | ✓ $\downarrow$ | $\bm{\times}$ $\uparrow$ |
| CLIP+linear | 49.56 | 48.21 | 31.78 | 21.78 | 54.17 |
| prompt-tune | 31.07 | 29.34 | 29.43 | 31.37 | 52.72 |
| FT-all | 54.58 | 52.05 | 21.27 | 21.06 | 54.32 |
| FT-late | 66.16 | 70.15 | 30.99 | 14.03 | 55.35 |
| MM-Pred (our) | 85.51 | 77.85 | 81.18 | 9.69 | 55.99 |
| MM-Adapter (our) | 90.35 | 81.95 | 90.53 | 8.85 | 56.12 |

*Table 3: Two other choices for a hard metric computed on the CLEVR *[[24](#bib.bib24 "")]* dataset.*

|  |  | Overall MAP | | |
| --- | --- | --- | --- | --- |
|  |  | All | Unseen | Seen |
| GQA [[18](#bib.bib18 "")] | CLIP | 0.65 | 0.35 | 0.91 |
| | + prompt-tune | 8.72 | 9.63 | 7.91 |
| + Linear probe | 12.81 | 13.29 | 12.52 |
| + FT all | 11.47 | 10.87 | 11.96 |
| + FT late | 13.39 | 13.70 | 13.10 |
| + MM-Pred (our) | 16.76 | 17.45 | 16.15 |
| + MM-Adapter (our) | 17.40 | 16.79 | 17.95 |
| FLAVA | 7.33 | 6.43 | 8.15 |
| + FT-late | 9.71 | 9.49 | 9.90 |
| + MM-Pred (our) | 17.68 | 19.24 | 16.29 |
| + MM-Adapter (our) | 20.03 | 20.70 | 19.42 |
| CLEVR [[24](#bib.bib24 "")] | CLIP | 6.42 | 6.36 | 6.29 |
| | + prompt-tune | 29.42 | 23.02 | 27.79 |
| + Linear probe | 47.83 | 29.33 | 46.54 |
| + FT all | 51.99 | 18.40 | 47.63 |
| + FT late | 63.93 | 27.20 | 67.00 |
| + MM-Pred (our) | 83.40 | 76.82 | 76.10 |
| + MM-Adapter (our) | 88.15 | 89.40 | 76.90 |
| FLAVA + linear | 18.76 | 16.77 | 17.82 |
| + FT-late | 77.59 | 71.91 | 66.25 |
| + MM-Pred (our) | 90.41 | 85.74 | 86.05 |
| + MM-Adapter (our) | 91.08 | 86.60 | 87.39 |
| PACO [[44](#bib.bib44 "")] | CLIP | 0.71 | 0.11 | 1.31 |
| | + prompt-tune | 6.19 | 2.78 | 9.61 |
| + Linear probe | 8.22 | 3.83 | 12.61 |
| + FT all | 7.19 | 3.00 | 11.38 |
| + FT late | 9.29 | 5.37 | 13.21 |
| + MM-Pred (our) | 9.63 | 4.00 | 15.26 |
| + MM-Adapter (our) | 10.00 | 6.50 | 15.22 |
| FLAVA + linear | 3.45 | 1.73 | 5.17 |
| + FT-late | 6.31 | 2.00 | 10.60 |
| + MM-Pred (our) | 10.77 | 4.77 | 16.76 |
| + MM-Adapter (our) | 12.02 | 6.36 | 17.67 |

*Table 4: Standard MAP on all images with multiple attributes on objects annotated in the test set (not just hard distractors like our $\mathcal{C}ola$ MAP). Note how we can improve significantly (eg, 0.65 to 17.40 on the GQA split - 10x), but by a much lesser fraction on our $\mathcal{C}ola$ MAP which is only among hard distractors.*

Standard MAP In contrast to our hard $\mathcal{C}ola$ MAP’s, we also compute the MAP on all images in the validation set regardless of hard or easy distractors. Once again, we see all trends remain the same as shown in Table [4](#S10.T4 "Table 4 ‣ 10 More metrics and analysis ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"). However, we note that the MAP numbers on all images are much lower. This is becuase of two reasons - a) the number of images to rank from is higher, and b) datasets like GQA have missing annotations, hence there are many images that get denoted as a negative retrieval becuase of a missing annotation. When we restrict the images to at least have one of the query words, this noise reduces somewhat.
However, note how it is easier to improve on this overall MAP than on the harder $\mathcal{C}ola$ MAP reported in the main paper. This shows models can quickly improve on distinguishing coarse-grained differences but differentiating between the fine-grained ones (as evaluated by the $\mathcal{C}ola$ MAP) is harder.

|  |  |  | $\mathcal{C}ola$ Single-Obj MAP | | |
| --- | --- | --- | --- | --- | --- |
|  |  | Params | All | Unseen | Seen |
| GQA [[18](#bib.bib18 "")] | Unimodal | 13M | 42.19 | 44.61 | 40.01 |
| | FLAVA | 9.9M | 47.43 | 48.95 | 46.05 |
| ALBEF | 10.9M | 45.2 | 48.23 | 42.6 |
| MDETR | 7.8M | 46.83 | 48.86 | 44.99 |
| FIBER | 15M | 47.08 | 50.01 | 44.43 |
| FIBER-MM | 12M | 46.05 | 48.91 | 43.47 |
| CLEVR [[24](#bib.bib24 "")] | Unimodal | 13M | 64.05 | 27.53 | 67.48 |
| | FLAVA | 9.9M | 88.21 | 89.52 | 77 |
| ALBEF | 10.9M | 85.56 | 85.47 | 72.97 |
| MDETR | 7.8M | 89.35 | 89.4 | 80.2 |
| FIBER | 15M | 82.9 | 76.97 | 73.5 |
| FIBER-MM | 12M | 86.6 | 88.56 | 72.98 |
| PACO [[44](#bib.bib44 "")] | Unimodal | 13M | 15.66 | 8.74 | 22.58 |
| | FLAVA | 9.9M | 18.56 | 11.47 | 25.66 |
| ALBEF | 10.9M | 18.22 | 10.57 | 25.8 |
| MDETR | 7.8M | 19 | 11.13 | 26.87 |
| FIBER | 15M | 12.34 | 5.34 | 19.35 |
| FIBER-MM | 12M | 11.83 | 4.49 | 19.17 |

*Table 5: Different choices for multimodal fusion inspired from ways researchers have done multimodal fusion in literature. Note that these are not numbers from the models proposed in their papers, but the accuracy of using the style of multimodal fusion, which we use on top of frozen CLIP features. Most multimodal variants perform better than tuning similar or more number of parameters on unimodal attention layers. The main paper numbers are from the MDETR-style multimodal fusion.*

F1 Score We also ran a sample of the evaluation using F1 on the GQA split of COLA single-objects, and we see that all trends remain the same when comparing F1. For instance, F1 of CLIP baseline is 0.28, whereas FT-all is 0.31, FT-Late is 0.33, and our MM-Adapt is 0.39, MM-Pred is 0.40. Our conclusion stays the same: adapting the multimodal attention layers is better than tuning the split-modal attention layers (FT-Late), fine-tuning the entire model, or linear probing.

### 10.1 Other choices of multimodal fusion

In our MM-Adapter and MM-Pred approaches, we use multimodal fusion. There are various ways to do multimodal fusion.
Some of the salient choices are inspired by FLAVA*[[52](#bib.bib52 "")]*, ALBEF*[[33](#bib.bib33 "")]*, MDETR*[[26](#bib.bib26 "")]*, and FIBER*[[11](#bib.bib11 "")]*.
We describe some of the ways we try multimodal fusion:

1. –

    FLAVA-inspired - self-attention on a [CLS] token concatenated with image patch and text tokens- Here, we take the image patch features and text token features and employ self-attention transformer *[[57](#bib.bib57 "")]* on the concatenated image, text and [CLS] tokens.

2. –

    MDETR-inspired - self-attention over image patch and text tokens and then, a [CLS] token cross attending to the image and text tokens- In the MDETR *[[26](#bib.bib26 "")]* paper, they use self-attention over image and text features and then multiple task tokens that cross attend to the self-attended image-text features for various tasks. Since, we have only one task here, which is retrieval, we use one [CLS]. We have also experimented with using multiple (100) [CLS] tokens to see if they learn different things. We observe that all the [CLS] tokens learn the same thing with minimal performance gap. This is the choice of multimodal fusion that we report in the paper for both our MM-Adapter and MM-Pred approaches.

3. –

    ALBEF-inspired - text cross-attends to image - Here, first we have separate unimodal self-attention layers on the image patch and text token features. Then, the text token features cross-attend to the image patch features along with a [CLS] token. The [CLS] output is then used for MM-Pred (prediction using fully-connected layer) or MM-Adapter (cosine similarity to frozen text features).

4. –

    FIBER-inspired - text cross-attends to image and vice versa- Here, first we have separate unimodal self-attention layers on the image patch and text token features. Then, have text token features cross-attend to the image patch features along with a [CLS] token. We also have the image patch features cross-attend to text token features along with another [CLS] token. We finally measure the cosine similarity of the two [CLS] tokens.

5. –

    FIBER-MM - In the above FIBER and ALBEF style fusion, we used separate unimodal self-attention layers on the image patch and text token features before the cross attention. Here, we design a modification, we use a multimodal self-attention on the image patch and text tokens first, like FLAVA. Then, we do cross-attention like FIBER as described above.

Accuracies on GQA, CLEVR and PACO for $\mathcal{C}ola$ single-object case on the above-described multimodal choices are shown in Table [5](#S10.T5 "Table 5 ‣ 10 More metrics and analysis ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval").
We see similar trends as the choice of multimodal attention reported in the paper.
All the methods of doing multimodal fusion work better than unimodal fusion.
Also, while some choices work better than others, note how using the multimodal layers as a feature adaptor (MM-Adapter) works better than using it as a prediction head (MM-Pred) for all design choices.

11 Implementation details
-------------------------

Now, we present more implementation details of the models and adaptation strategies used in the main paper. We also provide more details on the datasets used.

### 11.1 Model architecture details

Recall that we have use a CLIP *[[43](#bib.bib43 "")]* image and text encoder to extract image and text region features.
Here are some additional details for each of the choices of adaption we tried:

1. –

    Linear: We train a linear probe on the pre-trained representations. We train a separate linear layer on top of the image and text pooled features for CLIP*[[43](#bib.bib43 "")]*. Each linear layer transforms the 512-dimensional image and text representation to another 512-dimensional embeddings. Finally, we compute the cosine similarity between the two transformed embeddings.

2. –

    Prompt-tune: We tune the embedding layer of the text words used in our training queries while keeping everything else frozen.

3. –

    FT all: We fine-tune the whole model. This involves tuning 151M parameters in the case of CLIP.

4. –

    FT Late: We take the second-last layer features from the image and text encoders of CLIP. There are 49 image patch features and K text token features (K depends on the input query length, but it is capped to 77). We train a separate transformer encoder layer on the 49 image patch embeddings and the K text tokens. The transformer encoder has 2 transformer encoder self-attention layers with 4 heads each. We tried variations of 1 layer, 2 layers and 3 layers and report the best performance. This design is chosen to be the most similar in the number of parameters and approach to our multimodal adaptation approach to be a strong baseline.

5. –

    MM-Pred: Here, we use multimodal attention as a prediction head like common multimodal models*[[52](#bib.bib52 ""), [26](#bib.bib26 "")]*, but train it on the frozen CLIP*[[43](#bib.bib43 "")]* base image and text encoders. Once again, the multimodal transformer encoder has 2 layers with 4 heads each. We predict a score using a fully-connected layer on the [CLS] token output of the multimodal attention that maps the 512-dimension embedding to a 1-dimensional score.

6. –

    MM-Adapter: This differs from our MM-Adapter approach, where we use multimodal attention to adapt the image representation and use their cosine similarity to the text features.

For the image-text-matching loss, we get a score for each image-text pair in a batch. For each score, we compute the binary sigmoidal cross entropy and take the average in the batch.
We use a sigmoidal cross entropy since for each image, there can be multiple text queries that are true and vice versa.
We train using a learning rate of 0.00001 and a weight decay of 0.0001 for the models on top of CLIP. For adaptations on top of FLAVA, we see that we need a higher learning rate to converge quicker, hence, we use a learning rate of 0.001 and a weight decay of 0.0001.

<img src='figures/multiobj_examples_good.png' alt='Refer to caption' title='' width='586' height='444' />

*Figure 7: Some examples from the $\mathcal{C}ola$ multi-obj setting.*

<img src='figures/multiobj_examples_ambiguous.png' alt='Refer to caption' title='' width='586' height='434' />

*Figure 8: Some examples from the $\mathcal{C}ola$ multi-obj setting that are somewhat ambiguous even after human cleaning. Note how these mostly have to do with color (which can look different under different lighting) and size (which is subjective).*

12 Qualitative results
----------------------

Single-object case Figures [9](#S12.F9 "Figure 9 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), [10](#S12.F10 "Figure 10 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), [11](#S12.F11 "Figure 11 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), [12](#S12.F12 "Figure 12 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") show examples of top 5 retrievals based on common adaptation methods and our MM-Adapter method on the $\mathcal{C}ola$ single-object setting on the GQA *[[18](#bib.bib18 "")]* dataset. Each row in the image is a different adaptation method (based on the methods shown in Table 1 in the main paper).
Note how we improve on multiple attributes attached to non-salient and small objects.
Figures [13](#S12.F13 "Figure 13 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), [14](#S12.F14 "Figure 14 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), [15](#S12.F15 "Figure 15 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") show some cases where we see marginal improvements from off-the-shelf CLIP or simpler adaptation techniques like fine-tuning or linear probing. We observe that marginal improvements are mostly on queries with large areas of the image like sky and water. The existing CLIP *[[43](#bib.bib43 "")]* model is fairly good at such large salient objects, especially when paired with common attributes like “green” for the object “leaf”.

Multi-object case Figures [16](#S12.F16 "Figure 16 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), [17](#S12.F17 "Figure 17 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), [18](#S12.F18 "Figure 18 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), and [19](#S12.F19 "Figure 19 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") show some results on the $\mathcal{C}ola$ multi-object setting. Similar to the observations in the single-object setting, we improve the attribute-object binding capability even when the objects are non-salient in the image.
In addition to relational compositionality, as shown in Figures [20](#S12.F20 "Figure 20 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval") and [21](#S12.F21 "Figure 21 ‣ 12 Qualitative results ‣ 𝒞⁢𝑜⁢𝑙⁢𝑎: A Benchmark for Compositional Text-to-image Retrieval"), our method also fails to understand fine differences in the relative strength of attributes and when objects are occluded to a high degree.

All images we use are from publicly available datasets, and we are unaware of any correspondences with identifiable humans in real life.

<img src='figures/qual1.png' alt='Refer to caption' title='' width='586' height='473' />

*Figure 9: Qualitative results on multiple attributes attached to an object. Note how we improve on many attributes attached to small non-salient objects in the cluttered scene. The round white table in the test images were often small and hence, the original model had trouble finding them. Note how the original CLIP only find the slaient black metal chair (first row), and in comparison, we find smaller non-salient ones as well (last row)*

<img src='figures/qual2.png' alt='Refer to caption' title='' width='586' height='473' />

*Figure 10: Qualitative results on multiple attributes attached to an object. Note how we improve on many attributes attached to small non-salient objects in the cluttered scene.*

<img src='figures/qual3.png' alt='Refer to caption' title='' width='586' height='475' />

*Figure 11: Qualitative results on multiple attributes attached to an object. Note how we improve on many attributes attached to small non-salient objects in the cluttered scene.*

<img src='figures/qual4.png' alt='Refer to caption' title='' width='586' height='477' />

*Figure 12: Qualitative results on multiple attributes attached to an object. Note how we improve on many attributes attached to small non-salient objects in the cluttered scene.*

<img src='figures/qual_fail1.png' alt='Refer to caption' title='' width='586' height='475' />

*Figure 13: Queries with attributes that cover a wide area with common attributes, like blue sky, have minimal improvements from off-the-shelf or simple adaptation strategies since existing models perform well on such queries already.*

<img src='figures/qual_fail2.png' alt='Refer to caption' title='' width='586' height='476' />

*Figure 14: Queries with attributes that cover a wide area, like water bodies, have minimal improvements from off-the-shelf or simple adaptation strategies since existing models perform well on such queries already.*

<img src='figures/qual_fail3.png' alt='Refer to caption' title='' width='586' height='458' />

*Figure 15: Queries with attributes that cover a wide area with common attributes, like a green large leaf, have minimal improvements from off-the-shelf or simple adaptation strategies since existing models perform well on such queries already.*

<img src='figures/multiObj_qual1.png' alt='Refer to caption' title='' width='538' height='404' />

*Figure 16: Qualitative results on multi-object cases. Once again, we see significant improvements on compositions involving small non-salient objects such as a small sign.*

<img src='figures/multiObj_qual2.png' alt='Refer to caption' title='' width='538' height='434' />

*Figure 17: Qualitative results on multi-object cases.*

<img src='figures/multiObj_qual3.png' alt='Refer to caption' title='' width='538' height='399' />

*Figure 18: Qualitative results on multi-object cases. We see significant improvements on compositions where the images have a lot of clutter and distractor objects - many things are white and brown in the scenes.*

<img src='figures/multiObj_qual4.png' alt='Refer to caption' title='' width='538' height='443' />

*Figure 19: Qualitative results on multi-object cases.*

<img src='figures/multiObj_failure_qual2.png' alt='Refer to caption' title='' width='538' height='480' />

*Figure 20: Our method performs somewhat poorly on very fine-grained relative differences. In the example above, a brown chair is underneath a brown desk, but the desk is not empty. In fact, even the desk in the correct image for that caption is not technically empty, but it is more empty than the distractor and our model fails to understand the relative difference.*

<img src='figures/multiObj_failure_qual3.png' alt='Refer to caption' title='' width='538' height='478' />

*Figure 21: Our method also performs poorly on occluded objects or when objects have some of the attributes of the distractor as well. In the example above, the doors are not clearly in view. In addition, the brown door also has a white stripe, which further confuses the model.*
