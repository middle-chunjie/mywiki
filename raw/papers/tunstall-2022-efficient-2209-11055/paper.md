# Efficient Few-Shot Learning Without Prompts

Lewis Tunstall<sup>1</sup>, Nils Reimers<sup>2</sup>, Unso Eun Seo Jo<sup>1</sup>, Luke Bates<sup>3</sup>, Daniel Korat<sup>4</sup>, Moshe Wasserblat<sup>4</sup>, Oren Pereg<sup>4</sup>

Hugging Face 2cohere.ai

<sup>3</sup>Ubiquitous Knowledge Processing Lab, Technical University of Darmstadt

$^{4}$ Emergent AI Lab, Intel Labs

$^{1}$ firstname@huggingface.com  $^{2}$ info@nils-reimers.de

${}^{3}$  bates@ukp.informatik.tu-darmstadt.de

$^{4}$ firstname.lastname@intel.com

# Abstract

Recent few-shot methods, such as parameter-efficient fine-tuning (PEFT) and pattern exploiting training (PET), have achieved impressive results in label-scarce settings. However, they are difficult to employ since they are subject to high variability from manually crafted prompts, and typically require billion-parameter language models to achieve high accuracy. To address these shortcomings, we propose SETFIT (Sentence Transformer Finetuning), an efficient and prompt-free framework for few-shot fine-tuning of Sentence Transformers (ST). SETFIT works by first finetuning a pretrained ST on a small number of text pairs, in a contrastive Siamese manner. The resulting model is then used to generate rich text embeddings, which are used to train a classification head. This simple framework requires no prompts or verbalizers, and achieves high accuracy with orders of magnitude less parameters than existing techniques. Our experiments show that SETFIT obtains comparable results with PEFT and PET techniques, while being an order of magnitude faster to train. We also show that SETFIT can be applied in multilingual settings by simply switching the ST body. Our code<sup>1</sup> and datasets<sup>2</sup> are made publicly available.

# 1 Introduction

Few-shot learning methods have emerged as an attractive solution to label-scarce scenarios, where data annotation can be time-consuming and costly. These methods are designed to work with a small number of labeled training examples, and typically involve adapting pretrained language models (PLMs) for specific downstream tasks.

Today, there exist several approaches to few-shot learning with PLMs. These include in-context learning (ICL), parameter-efficient finetuning (PEFT), and pattern exploiting training

Figure 1: Compared to standard fine-tuning, SETFIT is more sample efficient and exhibits less variability when trained on a small number of labeled examples.

(PET). Unfortunately, these approaches can be impractical for many researchers and practitioners. One disadvantage is that these approaches typically rely on the use of large-scale language models to achieve high performance. For example, T-Few (Liu et al., 2022) is based on the 11 billion parameter model T0 (Sanh et al., 2021), while GPT-3 (Brown et al., 2020a) is an order of magnitude larger. Secondly, training and deploying these few-shot methods typically requires specialized infrastructure with limited accessibility. Moreover, PET and the prominent PEFT methods require, as part of their training, the input of manually generated prompts, yielding varying outcomes depending on the level of manual prompt-engineering.

In this paper, we propose SETFIT, an approach based on Sentence Transformers (ST) (Reimers and Gurevych, 2019) that dispenses with prompts altogether and does not require large-scale PLMs to achieve high accuracy. For example, with only 8 labeled examples in the Customer Reviews (CR) sentiment dataset, SETFIT is competitive with fine-tuning on the full training set, despite the fine-tuned model being three times larger (see Figure 1).

We demonstrate SETFIT's efficacy in few-shot text classification over a range of NLP datasets

and in multiple scenarios including distillation and non-English data. We compare our method to standard PLM fine-tuning, state-of-the-art PET- and PEFT-based methods such as ADAPET (Tam et al., 2021) and T-Few (Liu et al., 2022), as well as recent prompt-free techniques such as PERFECT (Karimi Mahabadi et al., 2022a).

We summarize our contributions as follows:

1. We propose SETFIT- a simple and prompt-free method - and provide a comprehensive guide for applying it in practical few-shot settings.  
2. We evaluate SETFIT's performance on a number of few-shot text classifications tasks and show that it outperforms the state-of-the-art prompt-free method and ranks alongside much larger prompt-based, few-shot models.  
3. We make the code and data used in our work publicly available.

# 2 Related Work

SETFIT engages with two related lines of literature. We first extend the small but significant body of work on Sentence Transformers (ST) for text classification. Perone et al. (2018) introduced the idea of using sentence embeddings for text classification. Piao (2021) used 'out-of-the-box' STs for text classification without fine-tuning them. SETFIT differs from these works in two aspects: First, we fine-tune the ST in a Siamese manner for a text classification objective showing that it significantly enhances performance; second, we demonstrate this approach in few-shot setups.

SETFIT is also related to the recently emerging few-shot and zero-shot training line of literature as few-shot and zero-shot approaches have recently received a great deal of interest in the research community due to the availability of pretrained language models and the untapped capacity to use them in resource-constrained domains. Specifically, we discuss ICL, PEFT, and prompt-based fine-tuning.

ICL models directly generate predictions based on input-to-output training examples provided as prompts, without any parameter updates. Perhaps the best known example is GPT-3 (Brown et al., 2020b), which achieves remarkable few-shot performance. However, GPT-3 contains 175 billion

parameters and requires massive computational resources, prompt engineering, and can only utilize pretrained knowledge.

PEFT methods, such as adapters (Rebuffi et al., 2017), hold the majority of parameters fixed during training and only update small feed-forward networks that are inserted within the larger model architecture. A recent example is T-FEW (Liu et al., 2022), which outperforms GPT-3 at much lower computational cost. It accomplishes this by adding learned vectors that rescale the network's internal activations. T-FEW is 16 times smaller than GPT-3, but is still too large to be utilized as a practical tool in industry. It also requires a set of handcrafted prompts for each dataset.

Another alternative to ICL is prompt-based fine-tuning. This approach converts the downstream classification task into a masked-language modeling (MLM) objective. The model outputs tokens in a cloze-style format that maps to the corresponding labels via a predefined template. A well known example of this method is Pattern Exploiting Training (PET) (Schick and Schütze, 2021b,a). Like GPT-3, PET relies on manually-crafted prompts, but since the model can be fine-tuned to specific tasks, PET-based approaches typically outperform GPT-3 in few-shot scenarios, even with far smaller PLM backbones. PET has since been extended in two main directions: ADAPET (Tam et al., 2021), which improves PET with a decoupled label objective and label-conditioned MLM objective, and PERFECT (Karimi Mahabadi et al., 2022b) which uses task-specific adapters (Houlsby et al., 2019; Pfeiffer et al., 2021) and multi-token label-embeddings eliminate task prompts and verbalizers.

# 3 SetFit: Sentence Transformer Fine-Tuning

SETFIT is based on Sentence Transformers (Reimers and Gurevych, 2019) which are modifications of pretrained transformer models that use Siamese and triplet network structures to derive semantically meaningful sentence embeddings. The goal of these models is to minimize the distance between pairs of semantically similar sentences and maximize the distance between sentence pairs that are semantically distant. Standard STs output a fixed, dense vector that is meant to represent textual data and can then be used by machine learning algorithms.

Figure 2: SETFIT's fine-tuning and training block diagram.

# 3.1 The SETFIT approach for few-shot text classification

SETFIT uses a two-step training approach in which we first fine-tune an ST and then train a classifier head. In the first step, an ST is fine-tuned on the input data in a contrastive, Siamese manner on sentence pairs. In the second step, a text classification head is trained using the encoded training data generated by the fine-tuned ST from the first step. Figure 2 illustrates this process, and we discuss these two steps in the following sections.

ST fine-tuning To better handle the limited amount of labeled training data in few-shot scenarios, we adopt a contrastive training approach that is often used for image similarity (Koch et al., 2015). Formally, given a small set of  $K$  labeled examples  $D = \{(x_{i},y_{i})\}$ , where  $x_{i}$  and  $y_{i}$  are sentences and their class labels, respectively. For each class label  $c \in C$ , we generate a set of  $R$  positive triplets  $T_{p}^{c} = \{(x_{i},x_{j},1)\}$ , where  $x_{i}$  and  $x_{j}$  are pairs of randomly chosen sentences from the same class  $c$  such that  $(y_{i} = y_{j} = c)$ . Similarly, we also generate a set of  $R$  negative triplets  $T_{n}^{c} = \{(x_{i},x_{j},0)\}$ , where  $x_{i}$  are sentences from class  $c$  and  $x_{j}$  are randomly chosen sentences from different classes such that  $(y_{i} = c, y_{j} \neq c)$ . Finally, the contrastive fine tuning data set  $T$  is produced by concatenating the positive and negative triplets across all class labels;  $T = \{(T_{p}^{0},T_{n}^{0}), (T_{p}^{1},T_{n}^{1}), \dots, (T_{p}^{|C|},T_{n}^{|C|})\}$ , where  $|C|$  is the number of class labels,  $|T| = 2R|C|$  is the number of pairs in  $T$  and  $R$  is a hyperparameter. Unless stated otherwise, we used  $R = 20$  in all the evaluations.

This contrastive fine-tuning approach enlarges the size of training data in few-shot scenarios. Assuming that a small number  $(K)$  of labeled examples are given for a binary classification task, the potential size of the ST fine-tuning set  $T$  is derived from the number of unique sentence pairs that can be generated, namely  $K(K - 1) / 2$ , which is sig

nificantly larger than just  $K$ .

Classification head training In this second step, the fine-tuned ST encodes the original labeled training data  $\{x_{i}\}$ , yielding a single sentence embedding per training sample;  $Emb^{x_i} = ST(x_i)$  where  $ST()$  is the function representing the fine-tuned ST. The embeddings, along with their class labels, constitute the training set for the classification head  $T^{CH} = \{(Emb^{x_i},y_i)\}$  where  $|T^{CH}| = |D|$ . A logistic regression model is used as the text classification head throughout this work.

Inference At inference time, the fine-tuned ST encodes an unseen input sentence  $(x_{i})$  and produces a sentence embedding. Next, the classification head that was trained in the training step, produces the class prediction of the input sentence based on its sentence embedding. Formally this is  $x_{i}^{pred} = CH(ST(x_{i}))$ , where  $CH$  represents the classification head prediction function.

# 4 Experiments

# 4.1 Data

We conduct experiments on available text classification datasets. We split the datasets into development and test datasets (See Table 6 in Appendix A). The development datasets are utilized for setting SETFIT's hyperparameters such as the number of training pairs  $(|T|)$ , the loss function and the optimal number of training epochs. In order to test the robustness of SETFIT to various types of text, we choose test datasets that represent different text classification tasks with a varying number of classes. All datasets used are available on the Hugging Face Hub under the SETFIT organisation. In addition we evaluate SETFIT on the RAFT benchmark (Alex et al., 2021), a real-world few-shot text-classification benchmark composed of 11 practical tasks, where each task has only 50 training

examples.

# 4.2 SETFIT models

We evaluate three variations of SETFIT each one uses different underlying model of different size (Shown in Table 1)

<table><tr><td>Variation</td><td>Underlying ST Model</td><td>Size*</td></tr><tr><td>SETFITROBERTA</td><td>all-roberta-large-v14</td><td>355M</td></tr><tr><td>SETFITMPNET</td><td>paraphrase/mpnet-base-v25</td><td>110M</td></tr><tr><td>SETFITMINILM</td><td>paraphrase-MiniLM-L3-v26</td><td>15M</td></tr></table>

Table 1: SETFIT model variations using three different underlying ST models. *Number of parameters.

# 4.3 Baselines

We compare SETFIT's performance against standard transformer fine-tuning and recent best-performing few-shot approaches: ADAPET (Tam et al., 2021), PERFECT (Karimi Mahabadi et al., 2022b), and T-Few (Liu et al., 2022).

Standard fine-tuning Our first baseline is ROBERTALARGE (Liu et al., 2019), a standard, encoder-only transformer that is fine-tuned for sequence classification. Since we assume no validation sets, we constructed validation splits by randomly selecting equally sized portions from the train split. We perform a hyperparameter search on the number of epochs in the range [25,75] and pick the best performing model on a validation split.

We use a learning rate of  $2e^{-5}$  and batch size of 4 in all our experiments.

ADAPET Pattern exploiting training (PET) (Schick and Schütze, 2021b,a) is a method for improving PLM performance in few-shot setups on downstream tasks by converting textual input into a cloze-style question intended to be reminiscent of the masked language modelling (MLM) objective under which large PLMs such as BERT (Devlin et al., 2019) are trained. To determine SETFIT's performance relative to PET-based approaches, we compare our method to ADAPET (Tam et al., 2021), an extension of PET. In recent work

(Schick and Schütze, 2021), the authors show that PET-based classification methods excel on the RAFT benchmark, placing second only to much larger models such as T-Few. In our experiments, we used ADAPET with default hyperparameters and examined its performance with different PLM backbones, reporting the PLM which resulted in the best performance, albert-xxlarge-v2 (see Appendix A.2 in the Appendix for further details).

PERFECT PERFECT (Karimi Mahabadi et al., 2022b) is another cloze-based fine-tuning method, but unlike PET or ADAPET, it does not require handcrafted task prompts and verbalizers. Instead, PERFECT uses task-specific adapters (Houlsby et al., 2019; Pfeiffer et al., 2021) and multi-token label-embeddings which are independent from the language model vocabulary during fine-tuning. To run PERFECT on our test datasets, we adapted the configurations provided in the PERFECT codebase.

T-FeW T-FeW (Liu et al., 2022) is a PEFT-based few-shot learning method based on T0 (Sanh et al., 2021). The authors provide two versions of T-FeW: 11 and 3 billion parameters. Due to compute constraints, we were unable to run the 11 billion version, which requires an 80GB A100 GPU. Running tests on T-FeW as opposed to SETFIT posed several hurdles. First, because T-FeW's performance varies significantly depending on the input prompts, we run each experiment using 5 random seeds, and report the median result, as in the original paper. Second, T-FeW relies on dataset-specific prompts, made available on P3 (Public Pool of Prompts) (Bach et al., 2022). Only one of our test datasets had prompts in P3. For the rest of the datasets, we adapt standardized P3 prompts of similar tasks or implement prompts ourselves (See Appendix A.3).

# 4.4 Experimental Setup

Systematically evaluating few-shot performance can be challenging, because fine-tuning on small datasets may incur instability (Dodge et al., 2020; Zhang et al., 2021). To address this issue, in our experiments we use 10 random training splits for each dataset and sample size. These splits are used as training data across all tested methods. For each method, we report the average measure (depending on the dataset) and the standard deviation across these splits. We fine-tune SETFIT's ST model using

cosine-similarity loss with a learning rate of  $1e^{-3}$ , a batch size of 16 and a maximum sequence length of 256 tokens, for 1 epoch.

# 5 Results

Table 2 shows a comparison between SETFIT<sub>MPNET</sub> and the baselines for  $N = 8$  and  $N = 64$  labeled training samples per class. For reference purposes, standard fine-tuning results using the full training data are also shown (in all cases higher scores indicate stronger performance; see Table 6 in Appendix A for dataset metric details). We find that SETFIT<sub>MPNET</sub> significantly outperforms the FINETUNE baseline for  $N = 8$  by an average of 19.3 points. However, as the number of training samples increases to  $N = 64$ , the gap decreases to 5.6 points.

Similarly, we find that  $\mathrm{SETFIT}_{\mathrm{MPNET}}$  outperforms PERFECT by 13.6 and 2.6 points.  $\mathrm{SETFIT}_{\mathrm{MPNET}}$  also outperforms ADAPET by 4.0 and 1.5 points for  $N = 8$  and  $N = 64$  respectively. For  $N = 8$ ,  $\mathrm{SETFIT}_{\mathrm{MPNET}}$  is on par with T-FeW 3B whereas for  $N = 64$ $\mathrm{SETFIT}_{\mathrm{MPNET}}$  outperforms T-FeW 3B by 5 points on average, despite being prompt-free and more than 27 times smaller.

RAFT results The test datasets listed in Table 2 were not specifically designed for few-shot benchmarking. In order to better benchmark SETFIT, we used the RAFT benchmark (Alex et al., 2021) which is specifically designed for benchmarking few-shot methods. Table 3 shows the average accuracy of SETFITMPNET and SETFITROBERTA and four prominent methods. SETFITROBERTA outperforms GPT3 and PET by 8.6 and 1.7 points respectively while alleviating the need for hand crafting prompts. It also surpasses the human baseline in 7 out of 11 tasks. SETFITROBERTA falls short of T-FEW 11B by 4.5 points. however, SETFITROBERTA is more than 30 times smaller than T-FeW 11B, does not require manual prompt crafting and is much more efficient in training and inference (see Table 5).

# 6 Multilingual Experiments

To determine SETFIT's performance in a multilingual, few-shot text classification scenario, we conducted development and test experiments on multilingual datasets and compared SETFIT to standard transformer fine-tuning and ADAPET. To the best of our knowledge, this is the first work to ex

amine ADAPET on non-English data (see Appendix A for details).

Experimental Setup For the multilingual experiments, we use the Multilingual Amazon Reviews Corpus (MARC) (Keung et al., 2020). This dataset consists of Amazon reviews in six languages (English, Japanese, German, French, Spanish, and Chinese), where each review is labeled according to a 5-star rating scale. We chose this corpus for its typological diversity in order to examine the generalizability of SETFIT and other methods across a variety of languages.

For the SETFIT underlying model, we use paraphrase-multilingual-mpnet-base-v2, which is a multilingual version of paraphrase-mpnet-base-v2 that is trained on parallel data in over 50 languages.

For the FINETUNE and ADAPET baselines, we use XLM-ROBERTA<sub>BASE</sub> (Conneau et al., 2019), which has a similar size to the SETFIT model. We compare the performance of each method using the same settings as (Conneau et al., 2019):

- each: Train and evaluate on monolingual data to measure per-language performance.  
- en: Train on the English training data and then evaluate on each language's test set.  
- all: Train on all the training data and evaluate on each language's test set.

Method For SETFIT standard fine-tuning, and ADAPET, we adopt the same methodology and hyperparameters used for the monolingual English experiments in 4. We evaluate each method in the few-shot regime ( $N = 8$  samples per class) and compare against performance of fine-tuning on the full training set of 20,000 examples.

Results Table 4 shows the results of SETFIT standard fine-tuning, and ADAPET on each language in MARC, where a higher MAE indicates weaker performance. In the few-shot regime of  $N = 8$  samples per class, we find that SETFIT significantly outperforms FINETUNE and ADAPET in all settings (each, en, all), with the best average performance obtained when training on English data only.

<table><tr><td>Method</td><td>SST-5</td><td>AmazonCF</td><td>CR</td><td>Emotion</td><td>EnronSpam</td><td>AGNews</td><td>Average†</td></tr><tr><td></td><td></td><td></td><td>|N|=8*</td><td></td><td></td><td></td><td></td></tr><tr><td>FINETUNE</td><td>33.52.1</td><td>9.24.9</td><td>58.86.3</td><td>28.76.8</td><td>85.06.0</td><td>81.73.8</td><td>43.05.2</td></tr><tr><td>PERFECT</td><td>34.93.1</td><td>18.15.3</td><td>81.58.6</td><td>29.85.7</td><td>79.37.4</td><td>80.85.0</td><td>48.76.0</td></tr><tr><td>ADAPET</td><td>50.01.9</td><td>19.47.3</td><td>91.01.3</td><td>46.23.7</td><td>85.13.7</td><td>85.12.7</td><td>58.33.6</td></tr><tr><td>T-FEW 3B</td><td>55.01.4</td><td>19.03.9</td><td>92.11.0</td><td>57.41.8</td><td>93.11.6</td><td>-</td><td>63.41.9</td></tr><tr><td>SETFITMPNET</td><td>43.63.0</td><td>40.311.8</td><td>88.51.9</td><td>48.84.5</td><td>90.13.4</td><td>82.92.8</td><td>62.34.9</td></tr><tr><td></td><td></td><td></td><td>|N|=64*</td><td></td><td></td><td></td><td></td></tr><tr><td>FINETUNE</td><td>45.96.9</td><td>52.812.1</td><td>88.91.9</td><td>65.017.2</td><td>95.90.8</td><td>88.40.9</td><td>69.77.8</td></tr><tr><td>PERFECT</td><td>49.10.7</td><td>65.15.2</td><td>92.20.5</td><td>61.72.7</td><td>95.41.1</td><td>89.00.3</td><td>72.71.9</td></tr><tr><td>ADAPET</td><td>54.10.8</td><td>54.16.4</td><td>92.60.7</td><td>72.02.2</td><td>96.00.9</td><td>88.00.6</td><td>73.82.2</td></tr><tr><td>T-FEW 3B</td><td>56.00.6</td><td>34.74.5</td><td>93.11.0</td><td>70.91.1</td><td>97.00.3</td><td>-</td><td>70.31.5</td></tr><tr><td>SETFITMPNET</td><td>51.90.6</td><td>61.92.9</td><td>90.40.6</td><td>76.21.3</td><td>96.10.8</td><td>88.00.7</td><td>75.31.3</td></tr><tr><td></td><td></td><td></td><td>|N| = Full**</td><td></td><td></td><td></td><td></td></tr><tr><td>FINETUNE</td><td>59.8</td><td>80.1</td><td>92.4</td><td>92.6</td><td>99.0</td><td>93.8</td><td>84.8</td></tr></table>

Table 2: SETFIT performance score and standard deviation compared to the baselines across 6 test datasets for three training set sizes  $\left| N\right|$  . *Number of training samples per class. **Entire available training data used. †The AGNews dataset is excluded from the average score to enable fair comparison with T-FEW (which has AGNews in its training set). *The inputs of SST-5 (but not its labels) appeared in T-FEW's training set, as part of Rotten Tomatoes dataset.  

<table><tr><td>Rank</td><td>Method</td><td>Score</td><td>Size*</td></tr><tr><td>1</td><td>YIWISE</td><td>76.8</td><td>-</td></tr><tr><td>2</td><td>T-Few 11B</td><td>75.8</td><td>11B</td></tr><tr><td>4</td><td>Human baseline</td><td>73.5</td><td>-</td></tr><tr><td>6</td><td>SETFITROBERTA</td><td>71.3</td><td>355M</td></tr><tr><td>9</td><td>PET</td><td>69.6</td><td>235M</td></tr><tr><td>11</td><td>SETFITMPNET</td><td>66.9</td><td>110M</td></tr><tr><td>12</td><td>GPT-3</td><td>62.7</td><td>175B</td></tr></table>

Table 3: SETFIT compared to prominent methods on the RAFT leaderboard (as of Sept. 5, 2022). *Number of parameters.

# 7 SETFIT Model Efficiency

# 7.1 Few-shot distillation

We have shown that SETFIT achieves state-of-the-art results in few-shot setups using underlying base models such as paraphrase-mpnet-base-v2 and ROBERTALARGE, containing 110M parameters and 355M parameters respectively; but in real-world deployments, where cost and sustainability are prioritized, the use on even more efficient models is desirable. Previous works have shown model distillation to be effective in reducing computational load while preserving much of the original model's performance (Ba and Caruana, 2014; Hinton et al., 2015). In this section we evaluate the performance

of SETFIT as a student model compared to a standard transformer student model in few-shot distillation setups when the amount of unlabeled training data is limited.

Experimental Setup For the distillation tests we use the datasets AGNews, Emotion and SST-5 described in Appendix A.1. For the SETFIT teacher we chose SETFITMPNET, which contains 110M parameters, whereas for the SETFIT student we chose SETFITMINILM, which is a much smaller model (15M parameters). For fair comparison, we use as the baseline student MiniLM-L3-H384-uncased $^{10}$ , a standard transformer encoder of the same size as our SETFIT student model. For each of the three datasets we train the SETFIT teacher model using only 16 labeled samples per class, and the student models are trained using the same 16 labeled samples per class together with various amounts of additional unlabeled data. We follow the same data-split policy and SETFIT training parameters' settings described in Section 4.4.

Method The SETFIT student is trained using sentence pairs and the level of similarity between each pair as input. The similarity is generated by using the underlying ST of the teacher to produce sen

<table><tr><td>Method</td><td>Train</td><td>En</td><td>De</td><td>Ja</td><td>Zh</td><td>Fr</td><td>Es</td><td>Average</td></tr><tr><td rowspan="3">FINETUNE</td><td>each</td><td>122.914.0</td><td>119.913.6</td><td>120.58.0</td><td>128.610.7</td><td>123.213.0</td><td>116.38.3</td><td>121.911.3</td></tr><tr><td>en</td><td>115.911.3</td><td>115.212.0</td><td>121.612.3</td><td>123.08.8</td><td>117.313.0</td><td>113.112.4</td><td>117.711.6</td></tr><tr><td>all</td><td>117.84.9</td><td>116.39.7</td><td>121.512.4</td><td>120.56.7</td><td>117.39.9</td><td>110.19.5</td><td>117.28.8</td></tr><tr><td rowspan="3">ADAPET</td><td>each</td><td>129.913.6</td><td>136.410.6</td><td>130.413.4</td><td>135.010.9</td><td>141.810.1</td><td>136.010.4</td><td>134.911.5</td></tr><tr><td>en</td><td>138.917.8</td><td>151.517.8</td><td>160.816.7</td><td>158.816.3</td><td>152.015.7</td><td>149.817.1</td><td>152.016.9</td></tr><tr><td>all</td><td>150.812.0</td><td>136.27.0</td><td>150.810.0</td><td>152.810.2</td><td>140.014.0</td><td>145.14.5</td><td>146.011.3</td></tr><tr><td rowspan="3">SETFIT</td><td>each</td><td>82.94.3</td><td>80.02.4</td><td>95.52.8</td><td>95.32.8</td><td>85.36.0</td><td>80.85.4</td><td>86.64.9</td></tr><tr><td>en</td><td>82.64.8</td><td>83.45.9</td><td>93.26.6</td><td>93.93.6</td><td>82.24.8</td><td>83.45.9</td><td>86.45.2</td></tr><tr><td>all</td><td>83.05.3</td><td>84.07.6</td><td>97.19.2</td><td>97.46.5</td><td>83.56.5</td><td>84.96.1</td><td>88.36.9</td></tr><tr><td rowspan="4">FINETUNE</td><td></td><td></td><td></td><td colspan="5">|N|=Full**</td></tr><tr><td>each</td><td>46.2</td><td>43.7</td><td>46.8</td><td>56.6</td><td>47.8</td><td>45.3</td><td>47.7</td></tr><tr><td>en</td><td>46.1</td><td>46.6</td><td>61.0</td><td>69.4</td><td>55.6</td><td>52.9</td><td>55.3</td></tr><tr><td>all</td><td>46.6</td><td>49.4</td><td>61.0</td><td>69.4</td><td>55.6</td><td>55.0</td><td>56.2</td></tr></table>

Table 4: Average performance (MAE × 100) on the Multilingual Amazon Reviews Corpus for two training set sizes  $|N|$ . * No. of training samples per class. **Entire available training data used (20,000 samples).

Figure 3: Average accuracy as a function of the unlabeled training set size  $N$  of the SETFIT student and the baseline student on AG News, Emotion and SST5 datasets.



tence embeddings for each pair and to calculate the cosine-similarity between them. The underlying ST of the SETFIT student is trained to mimic the ST of the teacher output by minimizing the error between the SETFIT teacher-produced cosine-similarity and its output. The classification head of the student is then trained using the embeddings produced by the student's ST and the logits produced by the SETFIT teacher classification head. The baseline student is trained to mimic the teacher output by minimizing the error between the logits produced by the SETFIT teacher classification head and its output.

Results Figure 3 shows a comparison between the SETFIT student model and the baseline student model for various amounts of unlabeled training data  $(N)$ . The SETFIT student significantly

outperforms the baseline student when only small amounts of unlabeled data are available. For example, for  $N = 8$ , the SETFIT student outperforms the baseline student by 24.8, 25.1, and 8.9 average accuracy on the AGNews, Emotion and SST5 datasets, respectively. As  $N$  increases, the performance gains decrease and are on par for  $N = 1K$ .

# 7.2 Computational costs

Comparing the relative computational costs of SET-FIT versus PET and PEFT methods isn't straightforward since each method typically has different hardware and memory requirements.

To simplify the comparison, we follow the approach adopted by Liu et al. (2022) and use FLOPsper-token estimates to compare SETFIT to T-FeW. These estimates can be obtained from Kaplan et al.

(2020), who show that encoder-only models with  $N$  parameters have approximately  $2N$  FLOPs-per-token for inference and  $6N$  FLOPs-per-token for training. The resulting cost for inference and training is then given by:

$$
\begin{array}{l} \mathcal {C} _ {\mathrm {i n f}} = 2 N \cdot \ell_ {\mathrm {s e q}}, \\ \mathcal {C} _ {\mathrm {t r a i n}} = 6 N \cdot \ell_ {\mathrm {s e q}} \cdot n _ {\mathrm {s t e p s}} \cdot n _ {\mathrm {b a t c h}}, \\ \end{array}
$$

where  $\ell_{\mathrm{seq}}$  is the input sequence length,  $n_{\mathrm{steps}}$  is the number of training steps, and  $n_{\mathrm{batch}}$  is the batch size. For encoder-decoder models like T-FeW, these estimates are halved, since the model only processes each token with either the encoder or decoder.

For the inference and training estimates shown in Table 5, we use  $\ell_{\mathrm{seq}} = 38$  and  $\ell_{\mathrm{seq}} = 54$  as the input sequence length for SETFITMPNET (T-FEW); this is the median number of tokens across all the test datasets in Table 2. We also use  $n_{\mathrm{steps}} = 1000$  and  $n_{\mathrm{batch}} = 8$  for all training estimates. As shown in the table, the SETFITMPNET model is approximately an order of magnitude faster at inference and training than T-FEW, despite having comparable performance on the test datasets of Table 2. SETFITMINILM is two orders of magnitude faster than T-FEW, with an average score reduction of 3.1 accuracy points. Moreover, the storage cost of the SETFIT models (70MB and 420MB respectively) is 163 to 26 times smaller than the T0-3B checkpoint used by T-FEW 3B (11.4GB), making these models much better suited for real-world deployment.

These estimates are borne out by comparing the time needed to train each method to convergence on  $N = 8$  examples. For our datasets, SETFITMPNET takes approximately 30 seconds to train on a p3.2xlarge AWS instance (16GB GPU memory), at a cost of \ $0.025 per split. On the other hand, T-Few 3B requires at least 40GB GPU memory, and training on a p4d.24xlarge AWS instance takes approximately 700 seconds, at a cost of \$ 0.7 per split.

# 8 Conclusion

This paper introduces SETFIT, a new few-shot text classification approach. We show that SETFIT has several advantages over comparable approaches such as T-FEW, ADAPET and PERFECT. In particular, SETFIT is much faster at inference and training; SETFIT requires much smaller base models to be performant, not requiring external compute; and

<table><tr><td>Method</td><td>Inf. FLOPs</td><td>Train FLOPs</td><td>Speed-up</td><td>Score</td></tr><tr><td>T-FEW 3B</td><td>1.6e11</td><td>3.9e15</td><td>1x</td><td>63.41.9</td></tr><tr><td>SETFITMPNET</td><td>8.3e9</td><td>2.0e14</td><td>19x</td><td>62.34.9</td></tr><tr><td>SETFITMINILM†</td><td>1.3e9</td><td>3.2e13</td><td>123x</td><td>60.31.6</td></tr></table>

Table 5: Relative computational cost and average scores of SETFIT and T-FEW using  $|N| = 8$  on the test datasets listed in Table 2.  ${}^{ \dagger }$  Trained in the distillation setup as described in Section 7.1,using  $\left| N\right|  = 8$  for teacher training and the rest of the available training data as unlabeled student training data. For fixed  ${n}_{\text{steps }}$  and  ${n}_{\text{batch }}$  ,the relative speed-up  $\left( {{N}^{\prime } \cdot  {\ell }_{\text{seq }}^{\prime }}\right) /\left( {{2N} \cdot  {\ell }_{\text{seq }}}\right)$  is the same for inference and training.

SETFIT is additionally not subject to the instability and inconvenience of prompting. We have also demonstrated that SETFIT is a robust few-shot text classifier in languages other than English across varying typologies. Finally, SETFIT has proven useful in few-shot distillation setups.

# Acknowledgements

The authors thank Hugging Face Inc and Intel Inc. for providing computing resources and the German Federal Ministry of Education and Research and the Hessian Ministry of Science and the Arts (HMWK) within the projects "The Third Wave of Artificial Intelligence - 3AI", hessian.AI, and within their joint support of the National Research Center for Applied Cybersecurity ATHENE.

# References

Neel Alex, Eli Lifland, Lewis Tunstall, Abhishek Thakur, Pegah Maham, C. Jess Riedel, Emmie Hine, Carolyn Ashurst, Paul Sedille, Alexis Carlier, Michael Noetel, and Andreas Stuhlmuller. 2021. RAFT: A real-world few-shot text classification benchmark. In Thirty-fifth Conference on Neural Information Processing Systems Datasets and Benchmarks Track (Round 2).  
Jimmy Ba and Rich Caruana. 2014. Do deep nets really need to be deep? In Z. Ghahramani, M. Welling, C. Cortes, N. D. Lawrence, and K. Q. Weinberger, editors, Advances in Neural Information Processing Systems 27, pages 2654-2662. Curran Associates, Inc.  
Stephen H. Bach, Victor Sanh, Zheng-Xin Yong, Albert Webson, Colin Raffel, Nihal V. Nayak, Abheesht Sharma, Taewoon Kim, M Saiful Bari, Thibault Fevry, Zaid Alyafeai, Manan Dey, Andrea Santilli, Zhiqing Sun, Srulik Ben-David, Canwen Xu, Gunjan Chhablani, Han Wang, Jason Alan

Fries, Maged S. Al-shaibani, Shanya Sharma, Ur-mish Thakker, Khalid Almubarak, Xiangru Tang, Dragomir Radev, Mike Tian-Jian Jiang, and Alexander M. Rush. 2022. Promptsource: An integrated development environment and repository for natural language prompts.  
Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel Ziegler, Jeffrey Wu, Clemens Winter, Chris Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020a. Language models are few-shot learners. In Advances in Neural Information Processing Systems, volume 33, pages 1877-1901. Curran Associates, Inc.  
Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020b. Language models are few-shot learners. CoRR, abs/2005.14165.  
Alexis Conneau, Kartikay Khandelwal, Naman Goyal, Vishrav Chaudhary, Guillaume Wenzek, Francisco Guzmán, Edouard Grave, Myle Ott, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Unsupervised cross-lingual representation learning at scale. CoRR, abs/1911.02116.  
Alexis Conneau and Douwe Kiela. 2018. SentEval: An evaluation toolkit for universal sentence representations. In Proceedings of the Eleventh International Conference on Language Resources and Evaluation (LREC 2018), Miyazaki, Japan. European Language Resources Association (ELRA).  
Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. BERT: Pre-training of deep bidirectional transformers for language understanding. In Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers), pages 4171-4186, Minneapolis, Minnesota. Association for Computational Linguistics.  
Jesse Dodge, Gabriel Ilharco, Roy Schwartz, Ali Farhadi, Hannaneh Hajishirzi, and Noah Smith. 2020. Fine-tuning pretrained language models: Weight initializations, data orders, and early stopping. arXiv preprint arXiv:2002.06305.

Derek Greene and Pádraig Cunningham. 2006. Practical solutions to the problem of diagonal dominance in kernel document clustering. In Proc. 23rd International Conference on Machine learning (ICML'06), pages 377-384. ACM Press.  
Geoffrey Hinton, Oriol Vinyals, and Jeff Dean. 2015. Distilling the knowledge in a neural network. Cite arxiv:1503.02531 Comment: NIPS 2014 Deep Learning Workshop.  
Neil Houlsby, Andrei Giurgiu, Stanislaw Jastrzebski, Bruna Morrone, Quentin de Laroussilhe, Andrea Gesmundo, Mona Attariyan, and Sylvain Gelly. 2019. Parameter-efficient transfer learning for NLP. CoRR, abs/1902.00751.  
Minqing Hu and Bing Liu. 2004. Mining and summarizing customer reviews. In Proceedings of the Tenth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, KDD '04, page 168-177, New York, NY, USA. Association for Computing Machinery.  
Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. CoRR, abs/2001.08361.  
Rabeeh Karimi Mahabadi, Luke Zettlemoyer, James Henderson, Lambert Mathias, Marzieh Saeidi, Veselin Stoyanov, and Majid Yazdani. 2022a. Prompt-free and efficient few-shot learning with language models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3638-3652, Dublin, Ireland. Association for Computational Linguistics.  
Rabeeh Karimi Mahabadi, Luke Zettlemoyer, James Henderson, Lambert Mathias, Marzieh Saeidi, Veselin Stoyanov, and Majid Yazdani. 2022b. Prompt-free and efficient few-shot learning with language models. In Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3638-3652, Dublin, Ireland. Association for Computational Linguistics.  
Phillip Keung, Yichao Lu, György Szarvas, and Noah A. Smith. 2020. The multilingual amazon reviews corpus.  
Gregory Koch, Richard Zemel, Ruslan Salakhutdinov, et al. 2015. Siamese neural networks for one-shot image recognition. In ICML deep learning workshop, volume 2, page 0. Lille.  
Haokun Liu, Derek Tam, Mohammed Muqeeth, Jay Mohta, Tenghao Huang, Mohit Bansal, and Colin Raffel. 2022. Few-shot parameter-efficient finetuning is better and cheaper than in-context learning.

Yinhan Liu, Myle Ott, Naman Goyal, Jingfei Du, Mandar Joshi, Danqi Chen, Omer Levy, Mike Lewis, Luke Zettlemoyer, and Veselin Stoyanov. 2019. Roberta: A robustly optimized bert pretraining approach. arXiv preprint arXiv:1907.11692.  
Andrew L. Maas, Raymond E. Daly, Peter T. Pham, Dan Huang, Andrew Y. Ng, and Christopher Potts. 2011. Learning word vectors for sentiment analysis. In Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies, pages 142-150, Portland, Oregon, USA. Association for Computational Linguistics.  
James O'Neill, Polina Rozenshtein, Ryuichi Kiryo, Motoko Kubota, and Danushka Bollegala. 2021. I wish I would have loved this one, but I didn't - A multilingual dataset for counterfactual detection in product reviews. CoRR, abs/2104.06893.  
Christian S. Perone, Roberto Pereira Silveira, and Thomas S. Paula. 2018. Evaluation of sentence embeddings in downstream and linguistic probing tasks. CoRR, abs/1806.06259.  
Jonas Pfeiffer, Aishwarya Kamath, Andreas Rückle, Kyunghyun Cho, and Iryna Gurevych. 2021. AdapterFusion: Non-destructive task composition for transfer learning. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 487-503, Online. Association for Computational Linguistics.  
Guangyuan Piao. 2021. Scholarly text classification with sentence bert and entity embeddings. In *Trends and Applications in Knowledge Discovery and Data Mining*, pages 79–87, Cham. Springer International Publishing.  
Sylvestre-Alvise Rebuffi, Hakan Bilen, and Andrea Vedaldi. 2017. Learning multiple visual domains with residual adapters.  
Nils Reimers and Iryna Gurevych. 2019. SentenceBERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 3982-3992, Hong Kong, China. Association for Computational Linguistics.  
Victor Sanh, Albert Webson, Colin Raffel, Stephen H. Bach, Lintang Sutawika, Zaid Alyafeai, Antoine Chaffin, Arnaud Stiegler, Teven Le Scao, Arun Raja, Manan Dey, M. Saiful Bari, Canwen Xu, Urmish Thakker, Shanya Sharma, Eliza Szczechla, Taewoon Kim, Gunjan Chhablani, Nihal V. Nayak, Debajyoti Datta, Jonathan Chang, Mike Tian-Jian Jiang, Han Wang, Matteo Manica, Sheng Shen, Zheng Xin Yong, Harshit Pandey, Rachel Bawden, Thomas Wang, Trishala Neeraj, Jos Rozen, Abheesht Sharma, Andrea Santilli, Thibault Févry,

Jason Alan Fries, Ryan Teehan, Stella Biderman, Leo Gao, Tali Bers, Thomas Wolf, and Alexander M. Rush. 2021. Multitask prompted training enables zero-shot task generalization. CoRR, abs/2110.08207.  
Elvis Saravia, Hsien-Chi Toby Liu, Yen-Hao Huang, Junlin Wu, and Yi-Shin Chen. 2018. CARER: Contextualized affect representations for emotion recognition. In Proceedings of the 2018 Conference on Empirical Methods in Natural Language Processing, pages 3687-3697, Brussels, Belgium. Association for Computational Linguistics.  
Timo Schick and Hinrich Schütze. 2021a. Exploiting cloze-questions for few-shot text classification and natural language inference. In Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume, pages 255–269, Online. Association for Computational Linguistics.  
Timo Schick and Hinrich Schütze. 2021b. It's not just size that matters: Small language models are also few-shot learners. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 2339-2352, Online. Association for Computational Linguistics.  
Timo Schick and Hinrich Schütze. 2021. True few-shot learning with prompts - A real-world perspective. CoRR, abs/2111.13440.  
Richard Socher, Alex Perelygin, Jean Wu, Jason Chuang, Christopher D. Manning, Andrew Ng, and Christopher Potts. 2013. Recursive deep models for semantic compositionality over a sentiment treebank. In Proceedings of the 2013 Conference on Empirical Methods in Natural Language Processing, pages 1631-1642, Seattle, Washington, USA. Association for Computational Linguistics.  
Derek Tam, Rakesh R. Menon, Mohit Bansal, Shashank Srivastava, and Colin Raffel. 2021. Improving and simplifying pattern exploiting training. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 4980-4991, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.  
I. Androutsopoulos V. Metsis and G. Paliouras. 2006. Spam filtering with naive bayes - which naive bayes? In Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006).  
Tianyi Zhang, Felix Wu, Arzoo Katiyar, Kilian Q Weinberger, and Yoav Artzi. 2021. Revisiting few-sample {bert} fine-tuning. In International Conference on Learning Representations.  
Xiang Zhang, Junbo Zhao, and Yann LeCun. 2015a. Character-level convolutional networks for text classification. In Advances in Neural Information Processing Systems, volume 28. Curran Associates, Inc.

Xiang Zhang, Junbo Jake Zhao, and Yann LeCun. 2015b. Character-level convolutional networks for text classification. In NIPS.

# A Appendix

# A.1 Datasets

Table 6 shows the development and test datasets that are used for setting SETFIT's hyperparameters. Following is a description of the datasets used:

SST2 The Stanford Sentiment Treebank 2 is a collection of single sentence movie reviews with positive-negative sentiment class labels. (Socher et al., 2013).

IMDB The Internet Movie Database dataset is a collection of single sentence movie reviews with positive-negative sentiment class labels. (Maas et al., 2011).

BBC News The BBC News dataset is a collection of articles from the news outlet BBC with one of 5 topic classifications: Politics, Sports, Entertainment, Tech, and Business. (Greene and Cunningham, 2006).

Enron Spam The Enron spam email dataset consists of emails from the internal Enron correspondence channel where emails are classified as spam or not spam. (V. Metsis and Paliouras, 2006).

Student Question Categories This is a set of questions from university entrance exams in India that are classified into 4 subjects: Math, Biology, Chemistry, Physics.

TREC-QC The Text Retrieval Conference Question Answering dataset.

Toxic Conversations<sup>12</sup> The Toxic Conversations dataset is a set of comments from Civil Comments, a platform for reader comments for independent news outlets. Human raters have given them toxicity attributes.

Amazon Polarity<sup>13</sup> The Amazon Polarity dataset consists of customer reviews from Amazon taken over 18 years with binary sentiment labels. Examples are either positive ("Great Read") or negative ("The Worst!") labelled. (Zhang et al., 2015a).

Following is a description of the test datasets:

# Stanford Sentiment Treebank-5 (SST5)

The SST-5 dataset is the fine-grained version of the Stanford Sentiment Treebank, where each example is given one of five labels: very positive, positive, neutral, negative, very negative.

Amazon Counterfactual The Amazon Counterfactual dataset is set of Amazon customer reviews with professionally labeled binary labels of counterfactual detection. Counterfactual statements are statements that denote something that did not happen or cannot (e.g. "They are much bigger than I thought they would be."). We used the English subset for our experiments. (O'Neill et al., 2021).

Customer Reviews The Customer Reviews (Hu and Liu, 2004) dataset is part of the of SentEval (Conneau and Kiela, 2018) benchmark. It is composed of positive and negative opinions mined from the web and written by customers about a variety of products.

Emotion<sup>14</sup> The Emotion dataset consists of tweets from Twitter that display clear emotions (e.g. "i am now nearly finished [with] the week detox and i feel amazing"). Labels are one of six categories: anger, fear, joy, love, sadness, and surprise. (Saravia et al., 2018).

AG News AG News is a dataset of news titles from AG news with one of 4 classifications (World, Entertainment, Sports, and Business). (Zhang et al., 2015b).

# A.2 ADAPET Training Procedure

By default, ADAPET assumes access to a training, development, and test dataset. It trains for 1,000 batches, runs predictions on the development data every 250 batches and checkpoints, keeping the model state which performed best on the development dataset. In our case, where we assume few-shot training and no development data, we ran ADAPET for 1,000 batches and disabled the checkpointing, using the model state that resulted after training for 1,000 batches. For the English data in Table 2, we used the pattern "[TEXT1] this is [LBL]", where "[TEXT1]" and "[LBL]" are placeholders for a given piece of text and the corresponding label, respectively. We constructed the

<table><tr><td>Dataset Name</td><td>Type of Task</td><td>Cls.*</td><td>Label Dist.**</td><td>Metric</td><td>Split</td></tr><tr><td>SST5</td><td>Sentiment</td><td>5</td><td>Approx. equal</td><td>Accuracy</td><td>Test</td></tr><tr><td>Amazon Counterfactual</td><td>Counterfactual</td><td>2</td><td>10% counterfactual</td><td>MCC</td><td>Test</td></tr><tr><td>CR</td><td>Sentiment</td><td>2</td><td>Equal</td><td>Accuracy</td><td>Test</td></tr><tr><td>Emotion</td><td>Emotion</td><td>6</td><td>Equal</td><td>Accuracy</td><td>Test</td></tr><tr><td>Enron Spam</td><td>Unwanted Language</td><td>2</td><td>Equal</td><td>Accuracy</td><td>Test</td></tr><tr><td>AG News</td><td>Topic</td><td>4</td><td>Equal</td><td>Accuracy</td><td>Test</td></tr><tr><td>SST2</td><td>Sentiment</td><td>2</td><td>Equal</td><td>Accuracy</td><td>Dev</td></tr><tr><td>IMDB</td><td>Sentiment</td><td>2</td><td>Equal</td><td>Accuracy</td><td>Dev</td></tr><tr><td>BBC News</td><td>Topic</td><td>5</td><td>Equal</td><td>Accuracy</td><td>Dev</td></tr><tr><td>Student Question Categories</td><td>Topic</td><td>4</td><td>ApproxEQUAL</td><td>Accuracy</td><td>Dev</td></tr><tr><td>TREC-QC</td><td>Topic</td><td>50</td><td>N/A</td><td>Accuracy</td><td>Dev</td></tr><tr><td>Toxic Conversations</td><td>Unwanted Language</td><td>2</td><td>8% Toxic</td><td>Avg. Precision</td><td>Dev</td></tr><tr><td>Amazon Polarity</td><td>Sentiment</td><td>2</td><td>Equal</td><td>Accuracy</td><td>Dev</td></tr></table>

Table 6: English datasets used for development and test experiments. *No. of classes per dataset. **Distribution of the examples across classes.

verbalizer from the "label" and "label text" columns that are available in all of our datasets. For the multilingual datasets in Table 4, we used the same pattern, but asked native speakers of each language to translate this pattern into their language. We additionally constructed the verbalizer by mapping labels to a star rating, for example,  $0 = 1$  star and  $4 = 5$  stars, again asking native speakers of each language to translate the verbalizer into their language.

# A.3 Prompts used in T-Few

The Emotion dataset is the only one that had existing prompts in P3 (Public Pool of Prompts) (Bach et al., 2022). For three other datasets, we had to adapt existing prompts designed for similar datasets on P3, by making minimal required changes to address the differences in data domains or label names:

- Prompts for Enron Spam, a spam e-mail detection dataset, were adapted from sms_spam dataset prompts.  
- CR prompts were adapted from amazon_polarity.  
- SST5 prompts were adapted from yelp_review_full.

The Amazon Counterfactual dataset does not have any relevant prompts on P3. Hence, we manually generated prompts ourselves, based on standard practices for prompt creation published in P3.

We also added two new prompts for SST5, to make it compatible with the label names of SST5. Following is a list of prompts we created for each dataset:

# Amazon Counterfactual Prompts

Input template:

{{text}}}Is the statement factual?

Target template:

{ answer Choices[label] }

Answer choices template:

Yes | No

Input template:

{{text}}Does the statement describe a fact?

Target template:

{ answer Choices[label] }

Answer choices template:

Yes | | No

Input template:

{{text}} Is the statement

non-counterfactual or counterfactual?

Target template:

{ answer Choices[label] }}

Answer choices template:

non-counterfactual || counterfactual

Input template:

```txt
{{text}}} Is the statement counterfactual?
```

Target template:  
```txt
{ answer Choices[label] }
```

Answer choices template:  
```txt
No | | | Yes
```

Input template:  
```txt
{{text}}Does the sentence express an event that did not happen?
```

Target template:  
```txt
{ answer Choices[label] }
```

Answer choices template:  
```txt
No | | | Yes
```

Input template:  
```txt
{{text}}Does this describe an actual event?
```

Target template:  
```txt
{ answer Choices[label] }
```

Answer choices template:  
```txt
Yes || No
```

Input template:  
```txt
{{text}}} Does the sentence contain events that did not or cannot take place?
```

Target template:  
```txt
{ answer Choices[label] }
```

Answer choices template:  
```txt
Yes || No
```

Input template:  
```handlebars
Is the label for the following sentence non-counterfactual or counterfactual? {{text}}
```

Target template:  
```txt
{ answer Choices[label] }
```

Answer choices template:  
```txt
non-counterfactual || counterfactual
```

# New prompts for SST5

Input template:  
```txt
How do you feel about the following sentence? {{text}}
```

Target template:  
```txt
{ answer Choices[label] }
```

Answer choices template:  
```typescript
very negative | | negative | | neutral | | positive | | very positive
```

Input and target templates:  
```handlebars
{{text}}Thismovieisavery {{answerchoices[label]}}}one
```

Answer choices template:  
```txt
terrible | bad okay good great
```

# Footnotes:

Page 0: https://github.com/huggingface/setfit 2https://huggingface.co/setfit 
Page 2: huggingface.co/SetFit 
Page 3: <sup>7</sup>https://huggingface.co/ albert-xxlarge-v2 4https://huggingface. co/sentence-transformers/ all-roberta-large-v1 5https://huggingface. co/sentence-transformers/ paraphrase-mpnet-base-v2 6https://huggingface. co/sentence-transformers/ paraphrase-MiniLM-L3-v2 
Page 4: huggingface.co/sentence-transformers/ paraphrase-multilingual-mpnet-base-v2 huggingface.co/xlm-roberta-base 
Page 5: 10huggingface.co/nreimers/ MiniLM-L3-H384-uncased 
Page 10: 14hf.co/datasets/emotion 11www.kaggle.com/datasets/mrutyunjaybiswal/iitjee-neet  
aims-students-questions-data  
12https://www.kaggle.com/competitions/jigsaw  
unintended-bias-in-toxicity-classification/data  
13hf.co/datasets/amazon_polarity 
