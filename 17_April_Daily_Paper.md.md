3D: 4 papers, one of them is published in CVPR2020.

Graph: 3 paper.



---

##### [1] RGBD-Dog: Predicting Canine Pose from RGBD Sensors

- [CVPR20] University of Bath: Sinéad Kearney, et al.
- <https://arxiv.org/pdf/2004.07788> 

The automatic extraction of animal \reb{3D} pose from images without markers is of interest in a range of scientific fields. Most work to date predicts animal pose from RGB images, based on 2D labelling of joint positions. However, due to the difficult nature of obtaining training data, no ground truth dataset of 3D animal motion is available to quantitatively evaluate these approaches. In addition, a lack of 3D animal pose data also makes it difficult to train 3D pose-prediction methods in a similar manner to the popular field of body-pose prediction. In our work, we focus on the problem of 3D canine pose estimation from RGBD images, recording a diverse range of dog breeds with several Microsoft Kinect v2s, simultaneously obtaining the 3D ground truth skeleton via a motion capture system. We generate a dataset of synthetic RGBD images from this data. A stacked hourglass network is trained to predict 3D joint locations, which is then constrained using prior models of shape and pose. We evaluate our model on both synthetic and real RGBD images and compare our results to previously published work fitting canine models to images. Finally, despite our training set consisting only of dog data, visual inspection implies that our network can produce good predictions for images of other quadrupeds -- e.g. horses or cats -- when their pose is similar to that contained in our training set.

![RGBDfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/RGBDfig2.png)

------

##### [2] Top-Down Networks: A coarse-to-fine reimagination of CNNs

- [CVPR20W] Delft University of Technology: Ioannis Lelekas, et al.
- <https://arxiv.org/pdf/2004.07629.pdf> 

Biological vision adopts a coarse-to-fine information processing pathway, from initial visual detection and binding of salient features of a visual scene, to the enhanced and preferential processing given relevant stimuli. On the contrary, CNNs employ a fine-to-coarse processing, moving from local, edge-detecting filters to more global ones extracting abstract representations of the input. In this paper we reverse the feature extraction part of standard bottom-up architectures and turn them upside-down: We propose top-down networks. Our proposed coarse-to-fine pathway, by blurring higher frequency information and restoring it only at later stages, offers a line of defence against adversarial attacks that introduce high frequency noise. Moreover, since we increase image resolution with depth, the high resolution of the feature map in the final convolutional layer contributes to the explainability of the network's decision making process. This favors object-driven decisions over context driven ones, and thus provides better localized class activation maps. This paper offers empirical evidence for the applicability of the top-down resolution processing to various existing architectures on multiple visual tasks.

![TDNfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/TDNfig2.png)

------

##### [3] Combinatorial 3D Shape Generation via Sequential Assembly

- POSTECH: Jungtaek Kim, et al.
- https://arxiv.org/pdf/2004.07414.pdf

3D shape generation has drawn attention in computer vision and machine learning since it opens an inspiring way to designing or creating new objects. Existing methods, however, do not reflect an important aspect of human generation processes in real life -- we often create a 3D shape by sequentially assembling geometric primitives into a combinatorial configuration. In this work, we propose a new 3D shape generation algorithm that aims to create such a combinatorial configuration from a set of volumetric primitives. To tackle the exponential growth of feasible combinations in terms of the number of primitives, we adopt sequential model-based optimization. Our method sequentially assembles primitives by exploiting and exploring adequate regions that are constrained by the current primitive placements. The evaluation function conveys global structure guidance for the assembling process to follow. Experimental results demonstrate that our method successfully generates combinatorial objects and simulates more realistic generation processes. We also introduce a new dataset for combinatorial 3D shape generation.

![C3DSGfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/C3DSGfig3.png)

------

##### [4] Joint Supervised and Self-Supervised Learning for 3D Real-World Challenges

- Politecnico di Torino: Antonio Alliegro, et al.
- https://arxiv.org/pdf/2004.07392.pdf

Point cloud processing and 3D shape understanding are very challenging tasks for which deep learning techniques have demonstrated great potentials. Still further progresses are essential to allow artificial intelligent agents to interact with the real world, where the amount of annotated data may be limited and integrating new sources of knowledge becomes crucial to support autonomous learning. Here we consider several possible scenarios involving synthetic and real-world point clouds where supervised learning fails due to data scarcity and large domain gaps. We propose to enrich standard feature representations by leveraging self-supervision through a multi-task model that can solve a 3D puzzle while learning the main task of shape classification or part segmentation. An extensive analysis investigating few-shot, transfer learning and cross-domain settings shows the effectiveness of our approach with state-of-the-art results for 3D shape classification and part segmentation.

![JS&SSLfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/JS&SSLfig1.png)

------

##### [5] Learning Furniture Compatibility with Graph Neural Networks

- [CVPR20W] Target Corporatio: Luisa F. Polanıa, et al.
- https://arxiv.org/pdf/2004.07268.pdf

We propose a graph neural network (GNN) approach to the problem of predicting the stylistic compatibility of a set of furniture items from images. While most existing results are based on siamese networks which evaluate pairwise compatibility between items, the proposed GNN architecture exploits relational information among groups of items. We present two GNN models, both of which comprise a deep CNN that extracts a feature representation for each image, a gated recurrent unit (GRU) network that models interactions between the furniture items in a set, and an aggregation function that calculates the compatibility score. In the first model, a generalized contrastive loss function that promotes the generation of clustered embeddings for items belonging to the same furniture set is introduced. Also, in the first model, the edge function between nodes in the GRU and the aggregation function are fixed in order to limit model complexity and allow training on smaller datasets; in the second model, the edge function and aggregation function are learned directly from the data. We demonstrate state-of-the art accuracy for compatibility prediction and "fill in the blank" tasks on the Bonn and Singapore furniture datasets. We further introduce a new dataset, called the Target Furniture Collections dataset, which contains over 6000 furniture items that have been hand-curated by stylists to make up 1632 compatible sets. We also demonstrate superior prediction accuracy on this dataset.

![LFCfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/LFCfig1.png)

------

##### [6] Representation Learning of Histopathology Images using Graph Neural Networks

- [CVPR20W] University of Waterloo: Mohammed Adnan, et al.
- https://arxiv.org/pdf/2004.07399.pdf

Representation learning for Whole Slide Images (WSIs) is pivotal in developing image-based systems to achieve higher precision in diagnostic pathology. We propose a two-stage framework for WSI representation learning. We sample relevant patches using a color-based method and use graph neural networks to learn relations among sampled patches to aggregate the image information into a single vector representation. We introduce attention via graph pooling to automatically infer patches with higher relevance. We demonstrate the performance of our approach for discriminating two sub-types of lung cancers, Lung Adenocarcinoma (LUAD) & Lung Squamous Cell Carcinoma (LUSC). We collected 1,026 lung cancer WSIs with the 40× magnification from The Cancer Genome Atlas (TCGA) dataset, the largest public repository of histopathology images and achieved state-of-the-art accuracy of 88.8% and AUC of 0.89 on lung cancer sub-type classification by extracting features from a pre-trained DenseNet

![RLfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/RLfig1.png)

------

##### [7] PICK: Processing Key Information Extraction from Documents using Improved Graph Learning-Convolutional Networks

- Xuzhou Medical University & Ping An Property: Wenwen Yu, Ning Lu, et al.
- <https://arxiv.org/pdf/2004.07464.pdf> 

Computer vision with state-of-the-art deep learning models has achieved huge success in the field of Optical Character Recognition (OCR) including text detection and recognition tasks recently. However, Key Information Extraction (KIE) from documents as the downstream task of OCR, having a large number of use scenarios in real-world, remains a challenge because documents not only have textual features extracting from OCR systems but also have semantic visual features that are not fully exploited and play a critical role in KIE. Too little work has been devoted to efficiently make full use of both textual and visual features of the documents. In this paper, we introduce PICK, a framework that is effective and robust in handling complex documents layout for KIE by combining graph learning with graph convolution operation, yielding a richer semantic representation containing the textual and visual features and global layout without ambiguity. Extensive experiments on real-world datasets have been conducted to show that our method outperforms baselines methods by significant margins.

![PICKfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/PICKfig3.png)

---

