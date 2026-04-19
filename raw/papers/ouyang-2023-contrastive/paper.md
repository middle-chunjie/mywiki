# Contrastive Learning for Conversion Rate Prediction

Wentao Ouyang

Alibaba Group

maiwei.oywt@alibaba-inc.com

Rui Dong

Alibaba Group

kailu.dr@alibaba-inc.com

Xiuwu Zhang

Alibaba Group

xiuwu.zxw@alibaba-inc.com

Chaofeng Guo

Alibaba Group

chaofeng.gcf@alibaba-inc.com

Jinmei Luo

Alibaba Group

cathy.jm@alibaba-inc.com

Xiangzheng Liu

Alibaba Group

xiangzheng.lxz@alibaba-inc.com

Yanlong Du

Alibaba Group

yanlong.dyl@alibaba-inc.com

# ABSTRACT

Conversion rate (CVR) prediction plays an important role in advertising systems. Recently, supervised deep neural network-based models have shown promising performance in CVR prediction. However, they are data hungry and require an enormous amount of training data. In online advertising systems, although there are millions to billions of ads, users tend to click only a small set of them and to convert on an even smaller set. This data sparsity issue restricts the power of these deep models. In this paper, we propose the Contrastive Learning for CVR prediction (CL4CVR) framework. It associates the supervised CVR prediction task with a contrastive learning task, which can learn better data representations exploiting abundant unlabeled data and improve the CVR prediction performance. To tailor the contrastive learning task to the CVR prediction problem, we propose embedding masking (EM), rather than feature masking, to create two views of augmented samples. We also propose a false negative elimination (FNE) component to eliminate samples with the same feature as the anchor sample, to account for the natural property in user behavior data. We further propose a supervised positive inclusion (SPI) component to include additional positive samples for each anchor sample, in order to make full use of sparse but precious user conversion events. Experimental results on two real-world conversion datasets demonstrate the superior performance of CL4CVR. The source code is available at https://github.com/DongRuiHust/CL4CVR.

# CCS CONCEPTS

- Information systems  $\rightarrow$  Online advertising.

# KEYWORDS

Online advertising; Conversion rate (CVR) prediction; Contrastive learning

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

SIGIR '23, July 23-27, 2023, Taipei, Taiwan

© 2023 Copyright held by the owner/author(s). Publication rights licensed to ACM.

ACM ISBN 978-1-4503-9408-6/23/07...$15.00

https://doi.org/10.1145/3539618.3591968

# ACM Reference Format:

Wentao Ouyang, Rui Dong, Xiuwu Zhang, Chaofeng Guo, Jinmei Luo, Xiangzheng Liu, and Yanlong Du. 2023. Contrastive Learning for Conversion Rate Prediction. In Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR '23), July 23-27, 2023, Taipei, Taiwan. ACM, New York, NY, USA, 5 pages. https://doi.org/10.1145/3539618.3591968

# 1 INTRODUCTION

Conversion rate (CVR) prediction [4, 9, 11, 12] is an essential task in online advertising systems. The predicted CVR impacts both the ad ranking strategy and the ad charging model [13, 15, 16, 28].

Recently, deep neural network-based models have achieved promising performance in CVR prediction [9, 13, 20, 24]. However, deep models are data hungry and require an enormous amount of training data. In online advertising systems, although there may be millions to billions of ads, users tend to click only a small set of them and to convert on an even smaller set. This data sparsity issue restricts the prediction power of these deep models.

Contrastive learning (CL) [6, 26] offers a new way to conquer the data sparsity issue via unlabeled data. The idea is to impose different transformations on the original data and to obtain two augmented views for each sample. It then pulls views of the same sample close in the latent space and pushes views of different samples apart in order to learn discriminative and generalizable representations.

In this paper, we propose the Contrastive Learning for CVR prediction (CL4CVR) framework, which associates the supervised CVR prediction task with a CL task. The CL task can learn better data representations and improve the CVR prediction performance. The way to create different data augmentations highly impacts the performance of CL. In recommender systems, most data augmentation methods are ID-based sequence or graph approaches [21, 23, 26], which do not apply to CVR prediction which is a feature-rich problem. The most relevant work is [25], which proposes feature masking for item recommendation. The aim is that two differently masked views, each containing part of item features, can still well represent the same item. However, feature masking does not work well for CVR prediction, because the input features are diverse, which relate to the user, the item, the context and the interaction, rather than only the item. We cannot make a good CVR prediction if we only know the target user but not the target item (which is masked).

Figure 1: (a) Structure of the CL4CVR framework. (b) ESMM as the supervised prediction model. (c) MLP as the encoder.

To tailor the CL task to the CVR prediction problem, we propose embedding masking (EM), rather than feature masking, to generate two views of augmented samples. In this way, each augmented view contains all the features, except that some embedding dimensions are masked. The CL loss will force the learned embeddings to be more representative. We also propose a false negative elimination (FNE) component to account for the natural property in user behavior data. We further propose a supervised positive inclusion (SPI) component to make full use of sparse but precious user conversion events. Experimental results show that the proposed EM, FNE and SPI strategies all improve the CVR prediction performance.

In summary, the main contributions of this paper are

- We propose the CL4CVR framework, which leverages a contrastive learning task to learn better data representations and to improve the CVR prediction performance.  
- We propose embedding masking for data augmentation that is tailored to feature-rich CVR prediction.  
- We propose a false negative elimination component and a supervised positive inclusion component to further improve the contrastive learning performance.

# 2 MODEL DESIGN

We propose the CL4CVR framework that combines contrastive learning (CL) with supervised learning (SL) to improve the performance of CVR prediction. The structure is shown in Fig. 1(a).

# 2.1 Problem Formulation

In typical advertising systems, user actions follow an impression  $\rightarrow$  click  $\rightarrow$  conversion path. Denote the input feature vector as  $\mathbf{x}$ , which contains multiple fields such as user ID, gender, age group, ad ID, ad title, ad industry, city, OS, etc. If a click event occurs, the click label is  $y = 1$ , otherwise,  $y = 0$ . If a conversion event occurs, the conversion label is  $z = 1$ , otherwise,  $z = 0$ . The (post-click) CVR prediction problem is to estimate the probability  $\hat{z} = p(z = 1 | y = 1, \mathbf{x})$ .

Figure 2: (a) Feature masking. (b) Embedding masking.

# 2.2 Supervised Prediction Model

Our focus in this paper is on the design of the CL task, and we use existing CVR prediction model as the SL task. In particular, we use ESMM [13] as the supervised prediction model because of its popularity and versatility. More sophisticated models such as  $\mathrm{ESM}^2$  [20], GMCM [3] and  $\mathrm{HM}^3$  [19] require additional post-click behaviors (e.g., favorite, add to cart and read reviews), which are not always available in different advertising systems.

Fig. 1(b) shows the structure of ESMM. It has a shared embedding layer, a CTR tower and a CVR tower. Assume there are  $N$  samples in a mini-batch. Denote the predicted CTR as  $\hat{y}$  and the predicted CVR as  $\hat{z}$ , the supervised loss is defined as

$$
L _ {p r e d} = \frac {1}{N} \sum_ {n = 1} ^ {N} l \left(\hat {y} _ {n}, y _ {n}\right) + \frac {1}{N} \sum_ {n = 1} ^ {N} l \left(\hat {y} _ {n} \hat {z} _ {n}, y _ {n} z _ {n}\right), \tag {1}
$$

where  $l(\hat{y}_n,y_n) = -y_n\log (\hat{y}_n) - (1 - y_n)\log (1 - \hat{y}_n)$ .

# 2.3 Embedding Masking

We now turn our attention to the CL task. Data augmentation is an important step that highly impacts the CL performance. In [25], the authors propose to create two views of each original sample by feature masking for item recommendation (Fig. 2(a)). The aim is that two differently masked views, each containing part of item features, can still well represent the same item.

However, feature masking does not work well in CVR prediction, because the input features are diverse, rather than only about the item. We cannot decide whether a user would like to convert on an ad if the ad features are masked. Therefore, we propose embedding masking (EM) in this paper, which is illustrated in Fig. 2(b).

In EM, we apply two different element-wise masks on the concatenated long embedding vector  $\mathbf{e}$  rather than on the raw features  $\mathbf{x}$ . Assume there are  $F$  features and the embedding dimension for each feature is  $K$ . Then a feature mask has dimension  $F$ , but an embedding mask has dimension  $FK$ . By EM, each masked view contains all (rather than part of) the features, except that some random embedding dimensions are masked. The aim is that the remaining embedding dimensions can still well represent the whole sample and the CL loss will force the learned embeddings to be more representative. We denote the two augmented embedding vectors of the same sample as  $\tilde{\mathbf{e}}_i$  and  $\tilde{\mathbf{e}}_j$ , which form a positive pair.

# 2.4 Encoder and Traditional Contrastive Loss

We map the two views  $\tilde{\mathbf{e}}_i$  and  $\tilde{\mathbf{e}}_j$  of the same sample to two high-level representation vectors  $\mathbf{h}_i$  and  $\mathbf{h}_j$  through the same encoder  $f$ . That is,  $\mathbf{h}_i = f(\tilde{\mathbf{e}}_i)$  and  $\mathbf{h}_j = f(\tilde{\mathbf{e}}_j)$ . For simplicity, we use an MLP as the encoder, which contains several fully connected (FC) layers with the ReLU activation [14] except the last layer (Fig. 1(c)).

<table><tr><td>Original</td><td>View 1</td><td>View 2</td><td>Anchor sample</td><td>Positive sample</td><td>Negative samples</td></tr><tr><td>Sample x1</td><td>ˆ1</td><td>ˆ2</td><td>ˆ1</td><td>ˆ2</td><td></td></tr><tr><td>Sample x2</td><td>ˆ3</td><td>ˆ4</td><td></td><td></td><td>ˆ3 ˆ4</td></tr><tr><td>Sample x3</td><td>ˆ5</td><td>ˆ6</td><td></td><td></td><td>ˆ5 ˆ6</td></tr><tr><td>Sample x4</td><td>ˆ7</td><td>ˆ8</td><td></td><td></td><td>ˆ7 ˆ8</td></tr></table>

(a) Traditional contrastive learning  

<table><tr><td>Original</td><td>View 1</td><td>View 2</td><td>Duplication indicator</td><td>Anchor sample</td><td>Positive sample</td><td>Negative samples</td></tr><tr><td>Sample x1</td><td>ˆe1</td><td>ˆe2</td><td></td><td>ˆe1</td><td>ˆe2</td><td></td></tr><tr><td>Sample x2</td><td>ˆe3</td><td>ˆe4</td><td>I(o(ˆe1),o(ˆe3)) = 0</td><td></td><td></td><td>ˆe3 ˆe4</td></tr><tr><td>Sample x3</td><td>ˆe5</td><td>ˆe6</td><td>I(o(ˆe1),o(ˆe5)) = 1</td><td></td><td></td><td>ˆe5 ˆe6</td></tr><tr><td>Sample x4</td><td>ˆe7</td><td>ˆe8</td><td>I(o(ˆe1),o(ˆe7)) = 0</td><td></td><td></td><td>ˆe7 ˆe8</td></tr></table>

(b) False negative elimination  

<table><tr><td>Original</td><td>View 1</td><td>View 2</td><td>Label</td><td>Anchor sample</td><td>Positive samples</td><td>Negative samples</td></tr><tr><td>Sample x1</td><td>ˆe1</td><td>ˆe2</td><td>z(ˆe1) = z(ˆe2) = 1</td><td>ˆe1</td><td>ˆe2</td><td></td></tr><tr><td>Sample x2</td><td>ˆe3</td><td>ˆe4</td><td>z(ˆe3) = z(ˆe4) = 0</td><td></td><td></td><td>ˆe3</td></tr><tr><td>Sample x3</td><td>ˆe5</td><td>ˆe6</td><td>z(ˆe5) = z(ˆe6) = 0</td><td></td><td></td><td>ˆe5</td></tr><tr><td>Sample x4</td><td>ˆe7</td><td>ˆe8</td><td>z(ˆe7) = z(ˆe8) = 1</td><td></td><td>ˆe7</td><td>ˆe8</td></tr></table>

(c) Supervised positive inclusion

Figure 3: Anchor sample, positive sample(s) and negative samples in (a) traditional contrastive learning, (b) false negative elimination and (c) supervised positive inclusion.

Given  $N$  original samples in a mini-batch, there are  $2N$  augmented samples. Given an anchor sample  $\tilde{\mathbf{e}}_i$ , the authors in [6] treat the other augmented sample  $\tilde{\mathbf{e}}_j$  of the same original sample as the positive and treat other augmented samples as negatives. We illustrate it in Fig. 3(a). The traditional contrastive loss [6] is

$$
L _ {0} = - \frac {1}{2 N} \sum_ {i = 1} ^ {2 N} \log \frac {\exp (s (\mathbf {h} _ {i} , \mathbf {h} _ {j}) / \tau)}{\sum_ {k \neq i} \exp (s (\mathbf {h} _ {i} , \mathbf {h} _ {k}) / \tau)}, \tag {2}
$$

where  $s(\mathbf{h}_i, \mathbf{h}_j)$  is the cosine similarity function and  $\tau$  is a tunable temperature hyper-parameter. This loss function aims to learn robust data representations such that similar samples are close to each other and random samples are pushed away in the latent space.

# 2.5 False Negative Elimination

In advertising systems, it is common that an ad is shown to a user multiple times at different time epochs. Because user behaviors naturally contain uncertainty, it is possible that the user clicks the ad  $a_1$  times and converts  $a_2$  times  $(a_2 \leq a_1)$ . This results in  $a_1$  click samples with the same features but possibly different conversion labels. When such samples are included for CL, contradiction happens. It is because augmented samples corresponding to original samples with different indices will be treated as negatives. However, their original samples actually have the same features.

Therefore, we propose a false negative elimination (FNE) component. It generates a set  $\mathcal{M}(i)$  for an anchor sample index  $i$  (Fig. 3(b)). Note that, FNE only impacts the CL task and the supervised prediction model still uses all the original samples for training, as otherwise, the learned conversion probabilities are incorrect. We use  $o(\tilde{\mathbf{e}}_i)$  to denote the original sample of  $\tilde{\mathbf{e}}_i$ . We introduce a duplication indicator where  $I(o(\tilde{\mathbf{e}}_i), o(\tilde{\mathbf{e}}_k)) = 1$  indicates that  $o(\tilde{\mathbf{e}}_i)$  and  $o(\tilde{\mathbf{e}}_k)$  have the same features and it is 0 otherwise. Given an anchor sample index  $i$ , we define the set

$$
\mathcal {M} (i) = \{j \} \cup \left\{k \mid I \left(o \left(\tilde {\mathbf {e}} _ {i}\right), o \left(\tilde {\mathbf {e}} _ {k}\right)\right) = 0 \right\}. \tag {3}
$$

$\mathcal{M}(i)$  contains the indices of samples that should be included in the denominator of the CL loss function.

# 2.6 Supervised Positive Inclusion

As the conversion label is sparse but also precious, we further propose a supervised positive inclusion (SPI) component to effectively leverage label information. It generates a set  $S(i)$  with supervised positive included for an anchor sample index  $i$ .

Inspired by supervised contrastive learning [10], we include additional positive samples for an anchor sample when its conversion label is 1 (Fig. 3(c)). Note that in traditional contrastive learning [6], an anchor sample  $\tilde{\mathbf{e}}_i$  has a single positive sample  $\tilde{\mathbf{e}}_j$ .

Given an anchor sample index  $i$ , we define the set

$$
\mathcal {S} (i) = \{j \} \cup \left\{k \mid z \left(\tilde {\mathbf {e}} _ {k}\right) = z \left(\tilde {\mathbf {e}} _ {i}\right) = 1, k \neq i, k \neq j \right\}, \tag {4}
$$

where  $z(\tilde{\mathbf{e}}_i)$  denotes the label of  $\tilde{\mathbf{e}}_i$ , which is the same as the original sample. In other words,  $S(i) = \{j\}$  if  $z(\tilde{\mathbf{e}}_i) = 0$  (i.e., the anchor sample has label 0) and  $S(i)$  may contain more positive samples if  $z(\tilde{\mathbf{e}}_i) = 1$ . We do not include supervised positive samples when  $z(\tilde{\mathbf{e}}_i) = 0$  because of the data sparsity issue. It is possible that all the samples in a mini-batch has  $z = 0$ , which makes all the samples supervised positives and there is no negative and no contrast at all.

# 2.7 Contrastive Loss and Overall Loss

$\mathcal{M}(i)$  generated by FNE and  $S(i)$  generated by SPI impact the contrastive loss. In particular, we define the contrastive loss used in this paper as

$$
L _ {c l} = - \frac {1}{2 N} \sum_ {i = 1} ^ {2 N} \left[ \frac {1}{| Q (i) |} \sum_ {q \in Q (i)} \log \frac {\exp \left(s \left(\mathbf {h} _ {i} , \mathbf {h} _ {q}\right) / \tau\right)}{\sum_ {k \in \mathcal {M} (i)} \exp \left(s \left(\mathbf {h} _ {i} , \mathbf {h} _ {k}\right) / \tau\right)} \right], \tag {5}
$$

where  $Q(i) = S(i)\cap \mathcal{M}(i)$ . For each anchor sample, we average over all its positives. The overall loss is the combination of the supervised CVR prediction loss and the contrastive loss as  $L = L_{pred} + \alpha L_{cl}$ , where  $\alpha$  is a tunable balancing hyper-parameter.

# 3 EXPERIMENTS

# 3.1 Datasets

Table 1: Statistics of experimental datasets.  

<table><tr><td>Dataset</td><td># Fields</td><td># Train</td><td># Val</td><td># Test</td><td># Show</td><td># Click</td><td># Conv</td></tr><tr><td>Industrial</td><td>60</td><td>278.8M</td><td>49.2M</td><td>48.4M</td><td>376.4M</td><td>64.5M</td><td>0.67M</td></tr><tr><td>Public</td><td>17</td><td>2.3M</td><td>0.98M</td><td>3.3M</td><td>6.6M</td><td>3.3M</td><td>0.018M</td></tr></table>

The statistics of the datasets are listed in Table 1. Both datasets contain samples from advertising systems with rich features, and are tagged with click and conversion labels. 1) Industrial dataset: This dataset contains a random sample of user behavior logs from an industrial news feed advertising system in 2022. 2) Public dataset: This dataset is gathered from the traffic logs in Taobao<sup>1</sup>.

# 3.2 Compared Methods

We compare the following methods for CVR prediction. Base is the supervised prediction model. Other methods associate the same base model with different data regularization or CL algorithms.

- Base. The supervised CVR prediction model. In this paper, we use ESMM [13] as the base.  
- FD. Base model with random Feature Dropout [18] in the supervised task.

Table 2: Test AUCs on experimental datasets. The best result is in bold font. A small improvement in AUC (e.g., 0.0020) can lead to a significant increase in online CVR (e.g.,  $3\%$ ). * indicates the statistical significance for  $p \leq 0.01$  compared with the best baseline over paired t-test.  

<table><tr><td></td><td colspan="2">Industrial dataset</td><td colspan="2">Public dataset</td></tr><tr><td></td><td>CVR AUC</td><td>Gain</td><td>CVR AUC</td><td>Gain</td></tr><tr><td>Base</td><td>0.8558</td><td>-</td><td>0.6524</td><td>-</td></tr><tr><td>FD</td><td>0.8452</td><td>-0.0106</td><td>0.6469</td><td>-0.0055</td></tr><tr><td>SO</td><td>0.8563</td><td>+0.0005</td><td>0.6534</td><td>+0.0010</td></tr><tr><td>RFM</td><td>0.8522</td><td>-0.0036</td><td>0.6536</td><td>+0.0012</td></tr><tr><td>CFM</td><td>0.8539</td><td>-0.0019</td><td>0.6541</td><td>+0.0017</td></tr><tr><td>CL4CVR</td><td>0.8637*</td><td>+0.0079</td><td>0.6590*</td><td>+0.0066</td></tr></table>

Table 3: Ablation study. Test AUCs on experimental datasets. EM - embedding masking. FNE - false negative elimination. SPI - supervised positive inclusion.  

<table><tr><td></td><td colspan="2">Industrial dataset</td><td colspan="2">Public dataset</td></tr><tr><td></td><td>CVR AUC</td><td>Gain</td><td>CVR AUC</td><td>Gain</td></tr><tr><td>Base</td><td>0.8558</td><td>-</td><td>0.6524</td><td>-</td></tr><tr><td>EM</td><td>0.8586</td><td>+0.0028</td><td>0.6572</td><td>+0.0048</td></tr><tr><td>EM + FNE</td><td>0.8605</td><td>+0.0047</td><td>0.6581</td><td>+0.0057</td></tr><tr><td>EM + SPI</td><td>0.8617</td><td>+0.0059</td><td>0.6580</td><td>+0.0056</td></tr><tr><td>EM + FNE + SPI</td><td>0.8637</td><td>+0.0079</td><td>0.6590</td><td>+0.0066</td></tr></table>

- SO. Base model with Spread-Out regularization [27] on original examples.  
- RFM. Random Feature Masking [25]. Base model with a CL task. It randomly splits features into two disjoint sets.  
- CFM. Correlated Feature Masking [25]. Base model with a CL task. It splits features according to feature correlation.  
- CL4CVR. The framework proposed in this paper.

# 3.3 Settings

Parameter Settings. We set the dimensions of fully connected layers in prediction towers and those in the CL encoder as {512, 256, 128}. The training batch size is set to 64. All the methods are implemented in Tensorflow [1] and optimized by Adagrad [7]. We run each method 3 times and report the average results.

Evaluation Metric. The Area Under the ROC Curve (AUC) is a widely used metric for CVR prediction. It reflects the probability that a model ranks a randomly chosen positive sample higher than a randomly chosen negative sample. The larger the better.

# 3.4 Experimental Results

3.4.1 Effectiveness. Table 2 shows the AUCs of different methods. It is observed that FD performs worst because it operates on the supervised task. SO performs better than RFM and CFM on the industrial dataset, but they have comparable performance on the public dataset. CFM performs better than RFM because it further considers feature correlation. CL4CVR performs best on both datasets, showing its effectiveness to cope with the data sparsity issue and to improve the CVR prediction performance.

3.4.2 Ablation Study. Table 3 lists the AUCs of three components in CL4CVR. It is observed that EM itself outperforms RFM

(a) Industrial dataset

(b) Public dataset

Figure 4: Impact of the temperature  $\tau$ .  
(a) Industrial dataset  
Figure 5: Impact of the CL loss weight  $\alpha$ .

(b) Public dataset

and CFM, showing that embedding masking is more suitable than feature masking for CVR prediction. The incorporation of the FNE component or the SPI component leads to further improvement. CL4CVR that uses all the three components perform best, showing that these components complement each other and improve the prediction performance from different perspectives.

3.4.3 Impact of the Temperature and the CL Loss Weight. Fig. 4 plots the impact of the temperature  $\tau$ . It is observed that generally a large  $\tau$  works well on the two datasets. Fig. 5 plots the impact of the CL loss weight  $\alpha$ , where 0 denotes the supervised base model. It is observed that when  $\alpha$  increases initially, performance improvement is observed. But when  $\alpha$  is too large, too much emphasis on the CL task will degrade the performance.

# 4 RELATED WORK

CVR prediction. The task of CVR prediction [4, 11, 12] in online advertising is to estimate the probability of a user makes a conversion event on a specific ad. [11] estimates CVR based on past performance observations along data hierarchies. [5] proposes an LR model and [2] proposes a log-linear model for CVR prediction. [17] proposes a model in non-guaranteed delivery advertising. [13] proposes ESMM to exploit click and conversion data in the entire sample space.  $\mathrm{ESM}^2$  [20], GMCM [3] and  $\mathrm{HM}^3$  [19] exploit additional purchase-related behaviors after click (e.g., favorite, add to cart and read reviews) for CVR prediction.

Contrastive learning. Contrastive learning [6, 8, 26] offers a new way to conquer the data sparsity issue via unlabeled data. It is able to learn more discriminative and generalizable representations. Contrastive learning has been applied to a wide range of domains such as computer vision [6], natural language processing [8] and recommendation [26]. In the recommendation domain, most data augmentation methods are ID-based sequence or graph approaches [21-23, 26], which do not apply to CVR prediction which is a feature-rich problem. The most relevant work is [25] which proposes feature masking for item recommendation.

# 5 CONCLUSION

In this paper, we propose the Contrastive Learning for CVR prediction (CL4CVR) framework. It associates the supervised CVR prediction task with a contrastive learning task, which can learn better data representations exploiting abundant unlabeled data and improve the CVR prediction performance. To tailor the contrastive learning task to the CVR prediction problem, we propose embedding masking, false negative elimination and supervised positive inclusion strategies. Experimental results on two real-world conversion datasets demonstrate the superior performance of CL4CVR.

# REFERENCES

[1] Martin Abadi, Paul Barham, Jianmin Chen, Zhifeng Chen, Andy Davis, Jeffrey Dean, Matthieu Devin, Sanjay Ghemawat, Geoffrey Irving, Michael Isard, et al. 2016. Tensorflow: A system for large-scale machine learning. In 12th USENIX Symposium on Operating Systems Design and Implementation (OSDI). USENIX, 265-283.  
[2] Deepak Agarwal, Rahul Agrawal, Rajiv Khanna, and Nagaraj Kota. 2010. Estimating rates of rare events with multiple hierarchies through scalable log-linear models. In Proceedings of the 16th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 213-222.  
[3] Wentian Bao, Hong Wen, Sha Li, Xiao-Yang Liu, Quan Lin, and Keping Yang. 2020. GMCM: Graph-based micro-behavior conversion model for post-click conversion rate estimation. In Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). 2201-2210.  
[4] Olivier Chapelle. 2014. Modeling delayed feedback in display advertising. In Proceedings of the 20th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD), 1097-1105.  
[5] Olivier Chapelle, Eren Manavoglu, and Romer Rosales. 2014. Simple and scalable response prediction for display advertising. ACM Transactions on Intelligent Systems and Technology (TIST) 5, 4 (2014), 1-34.  
[6] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geoffrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In International Conference on Machine Learning (ICML). PMLR, 1597-1607.  
[7] John Duchi, Elad Hazan, and Yoram Singer. 2011. Adaptive subgradient methods for online learning and stochastic optimization. Journal of Machine Learning Research 12, Jul (2011), 2121-2159.  
[8] Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. SimCSE: Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (EMNLP). 6894-6910.  
[9] Siyuan Guo, Lixin Zou, Yiding Liu, Wenwen Ye, Suqi Cheng, Shuaiqiang Wang, Hechang Chen, Dawei Yin, and Yi Chang. 2021. Enhanced doubly robust learning for debiasing post-click conversion rate estimation. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). 275-284.  
[10] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan. 2020. Supervised contrastive learning. Advances in Neural Information Processing Systems 33 (2020), 18661-18673.  
[11] Kuang-chih Lee, Burkay Orten, Ali Dasdan, and Wentong Li. 2012. Estimating conversion rate in display advertising from past erformance data. In Proceedings of the 18th ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 768-776.  
[12] Quan Lu, Shengjun Pan, Liang Wang, Junwei Pan, Fengdan Wan, and Hongxia Yang. 2017. A practical framework of conversion rate prediction for online display advertising. In Proceedings of the International Workshop on Data Mining for Online Advertising (ADKDD). 1-9.  
[13] Xiao Ma, Liqin Zhao, Guan Huang, Zhi Wang, Zelin Hu, Xiaogiang Zhu, and Kun Gai. 2018. Entire space multi-task model: An effective approach for estimating post-click conversion rate. In The 41st International ACM SIGIR Conference on Research & Development in Information Retrieval (SIGIR). ACM, 1137-1140.  
[14] Vinod Nair and Geoffrey E Hinton. 2010. Rectified linear units improve restricted boltzmann machines. In Proceedings of the 27th International Conference on Machine Learning (ICML). 807-814.  
[15] Junwei Pan, Yizhi Mao, Alfonso Lobos Ruiz, Yu Sun, and Aaron Flores. 2019. Predicting different types of conversions with multi-task learning in online

advertising. In Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (KDD). 2689-2697.  
[16] Xiaofeng Pan, Ming Li, Jing Zhang, Keren Yu, Hong Wen, Luping Wang, Chengjun Mao, and Bo Cao. 2022. MetaCVR: Conversion Rate Prediction via Meta Learning in Small-Scale Recommendation Scenarios. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). 2110-2114.  
[17] Rómer Rosales, Haibin Cheng, and Eren Manavoglu. 2012. Post-click conversion modeling and analysis for non-guaranteed delivery display advertising. In Proceedings of the fifth ACM international conference on Web Search and Data Mining (WSDM), 293-302.  
[18] Maksims Volkovs, Guangwei Yu, and Tomi Poutanen. 2017. Dropoutnet: Addressing cold start in recommender systems. Advances in Neural Information Processing Systems 30 (2017).  
[19] Hong Wen, Jing Zhang, Fuyu Lv, Wentian Bao, Tianyi Wang, and Zulong Chen. 2021. Hierarchically modeling micro and macro behaviors via multi-task learning for conversion rate prediction. In Proceedings of the 44th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). 2187-2191.  
[20] Hong Wen, Jing Zhang, Yuan Wang, Fuyu Lv, Wentian Bao, Quan Lin, and Keping Yang. 2020. Entire space multi-task modeling via post-click behavior decomposition for conversion rate prediction. In Proceedings of the 43rd International ACM SIGIR conference on Research and Development in Information Retrieval (SIGIR). 2377-2386.  
[21] Jiancan Wu, Xiang Wang, Fuli Feng, Xiangnan He, Liang Chen, Jianxun Lian, and Xing Xie. 2021. Self-supervised graph learning for recommendation. In Proceedings of the 44th international ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). 726-735.  
[22] Lianghao Xia, Chao Huang, Yong Xu, Jiashu Zhao, Dawei Yin, and Jimmy Huang. 2022. Hypergraph contrastive collaborative filtering. In Proceedings of the 45th International ACM SIGIR Conference on Research and Development in Information Retrieval (SIGIR). 70-79.  
[23] Xu Xie, Fei Sun, Zhaoyang Liu, Shiwen Wu, Jinyang Gao, Jiandong Zhang, Bolin Ding, and Bin Cui. 2022. Contrastive learning for sequential recommendation. In 2022 IEEE 38th International Conference on Data Engineering (ICDE). IEEE, 1259-1273.  
[24] Zixuan Xu, Penghui Wei, Weimin Zhang, Shaoguo Liu, Liang Wang, and Bo Zheng. 2022. UKD: Debiasing Conversion Rate Estimation via Uncertainty-regularized Knowledge Distillation. In Proceedings of the ACM Web Conference 2022 (WWW). 2078-2087.  
[25] Tiansheng Yao, Xinyang Yi, Derek Zhiyuan Cheng, Felix Yu, Ting Chen, Aditya Menon, Lichan Hong, Ed H Chi, Steve Tjoa, Jieqi Kang, et al. 2021. Self-supervised learning for large-scale item recommendations. In Proceedings of the 30th ACM International Conference on Information & Knowledge Management (CIKM). 4321-4330.  
[26] Junliang Yu, Hongzhi Yin, Xin Xia, Tong Chen, Jundong Li, and Zi Huang. 2022. Self-Supervised Learning for Recommender Systems: A Survey. arXiv preprint arXiv:2203.15876 (2022).  
[27] Xu Zhang, Felix X Yu, Sanjiv Kumar, and Shih-Fu Chang. 2017. Learning spreadout local feature descriptors. In Proceedings of the IEEE International Conference on Computer Vision (ICCV). 4595-4603.  
[28] Han Zhu, Junqi Jin, Chang Tan, Fei Pan, Yifan Zeng, Han Li, and Kun Gai. 2017. Optimized cost per click in taobao display advertising. In Proceedings of the 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 2191-2200.

# Footnotes:

Page 2: 1https://tianchi.aliyun.com/dataset/dataDetail?dataId=408 
