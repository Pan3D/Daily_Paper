3D: 6 papers.

Graph: 2 papers, one of them is published in CVPR2020.

---

##### [1] A2D2: Audi Autonomous Driving Dataset

- Audi AG: Jakob Geyer, et al.
- <https://arxiv.org/pdf/2004.06320.pdf> 
- https://www.a2d2.audi/ [Project]

Research in machine learning, mobile robotics, and autonomous driving is accelerated by the availability of high quality annotated data. To this end, we release the Audi Autonomous Driving Dataset (A2D2). Our dataset consists of simultaneously recorded images and 3D point clouds, together with 3D bounding boxes, semantic segmentation, instance segmentation, and data extracted from the automotive bus. Our sensor suite consists of six cameras and five LiDAR units, providing full 360 degree coverage. The recorded data is time synchronized and mutually registered. Annotations are for non-sequential frames: 41,277 frames with semantic segmentation image and point cloud labels, of which 12,497 frames also have 3D bounding box annotations for objects within the field of view of the front camera. In addition, we provide 392,556 sequential frames of unannotated sensor data for recordings in three cities in the south of Germany. These sequences contain several loops. Faces and vehicle number plates are blurred due to GDPR legislation and to preserve anonymity. A2D2 is made available under the CC BY-ND 4.0 license, permitting commercial use subject to the terms of the license.

![A2D2fig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/A2D2fig1.png)

------

##### [2] Self6D: Self-Supervised Monocular 6D Object Pose Estimation

- Tsinghua University & Technical University of Munich: Gu Wang, Fabian Manhardt, et al.
- https://arxiv.org/pdf/2004.06468.pdf

Estimating the 6D object pose is a fundamental problem in computer vision. Convolutional Neural Networks (CNNs) have recently proven to be capable of predicting reliable 6D pose estimates even from monocular images. Nonetheless, CNNs are identified as being extremely data-driven, yet, acquiring adequate annotations is oftentimes very time-consuming and labor intensive. To overcome this shortcoming, we propose the idea of monocular 6D pose estimation by means of self-supervised learning, which eradicates the need for real data with annotations. After training our proposed network fully supervised with synthetic RGB data, we leverage recent advances in neural rendering to further self-supervise the model on unannotated real RGB-D data, seeking for a visually and geometrically optimal alignment. Extensive evaluations demonstrate that our proposed self-supervision is able to significantly enhance the model's original performance, outperforming all other methods relying on synthetic data or employing elaborate techniques from the domain adaptation realm. 

![Self6Dfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/Self6Dfig2.png)

------

##### [3] Few-Shot Single-View 3-D Object Reconstruction with Compositional Priors

- University of Queensland: Mateusz Michalkiewicz, et al.
- https://arxiv.org/pdf/2004.06302.pdf

The impressive performance of deep convolutional neural networks insingle- view 3D reconstruction suggests that these models perform non-trivial reasoning about the 3D structure of the output space. However, recent work has challenged this belief, showing that complex encoder-decoder architectures perform similarly to nearest-neighbor baselines or simple linear decoder models that exploit large amounts of per category data in standard benchmarks. On the other hand settings where 3D shape must be inferred for new categories with few examples are more natural and require models that generalize about shapes. In this work we demonstrate experimentally that naive baselines do not apply when the goal is to learn to reconstruct novel objects using very few examples, and that in a \emph{few-shot} learning setting, the network must learn concepts that can be applied to new categories, avoiding rote memorization. To address deficiencies in existing approaches to this problem, we propose three approaches that efficiently integrate a class prior into a 3D reconstruction model, allowing to account for intra-class variability and imposing an implicit compositional structure that the model should learn. Experiments on the popular ShapeNet database demonstrate that our method significantly outperform existing baselines on this task in the few-shot setting.

![FSSV3D.fig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/FSSV3D.fig1.png)

------

##### [4] RealMonoDepth: Self-Supervised Monocular Depth Estimation for General Scenes

- University of Surrey: Mertalp Ocal, Armin Mustafa.
- https://arxiv.org/pdf/2004.06267.pdf

We present a generalised self-supervised learning approach for monocular estimation of the real depth across scenes with diverse depth ranges from 1--100s of meters. Existing supervised methods for monocular depth estimation require accurate depth measurements for training. This limitation has led to the introduction of self-supervised methods that are trained on stereo image pairs with a fixed camera baseline to estimate disparity which is transformed to depth given known calibration. Self-supervised approaches have demonstrated impressive results but do not generalise to scenes with different depth ranges or camera baselines. In this paper, we introduce RealMonoDepth a self-supervised monocular depth estimation approach which learns to estimate the real scene depth for a diverse range of indoor and outdoor scenes. A novel loss function with respect to the true scene depth based on relative depth scaling and warping is proposed. This allows self-supervised training of a single network with multiple data sets for scenes with diverse depth ranges from both stereo pair and in the wild moving camera data sets. A comprehensive performance evaluation across five benchmark data sets demonstrates that RealMonoDepth provides a single trained network which generalises depth estimation across indoor and outdoor scenes, consistently outperforming previous self-supervised approaches.

![RMDfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/RMDfig2.png)

------

##### [5] End-to-End Estimation of Multi-Person 3D Poses from Multiple Cameras

- Microsoft Research Asia: Hanyue Tu, et al.
- https://arxiv.org/pdf/2004.06239.pdf
- https://github.com/microsoft/multiperson-pose-estimation-pytorch [Code]

We present an approach to estimate 3D poses of multiple people from multiple camera views. In contrast to the previous efforts which require to establish cross-view correspondence based on noisy and incomplete 2D pose estimations, we present an end-to-end solution which directly operates in the 3D space, therefore avoids making incorrect decisions in the 2D space. To achieve this goal, the features in all camera views are warped and aggregated in a common 3D space, and fed into Cuboid Proposal Network (CPN) to coarsely localize all people. Then we propose Pose Regression Network (PRN) to estimate a detailed 3D pose for each proposal. The approach is robust to occlusion which occurs frequently in practice. Without bells and whistles, it outperforms the state-of-the-arts on the public datasets.

![E2EEfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/E2EEfig2.png)

------

##### [6] Unsupervised Performance Analysis of 3D Face Alignment

- Inria Grenoble Rhone-Alpes: Mostafa Sadeghi, et al.
- https://arxiv.org/pdf/2004.06550.pdf

We address the problem of analyzing the performance of 3D face alignment (3DFA) algorithms. Traditionally, performance analysis relies on carefully annotated datasets. Here, these annotations correspond to the 3D coordinates of a set of pre-defined facial landmarks. However, this annotation process, be it manual or automatic, is rarely error-free, which strongly biases the analysis. In contrast, we propose a fully unsupervised methodology based on robust statistics and a parametric confidence test. We revisit the problem of robust estimation of the rigid transformation between two point sets and we describe two algorithms, one based on a mixture between a Gaussian and a uniform distribution, and another one based on the generalized Student's t-distribution. We show that these methods are robust to up to 50\% outliers, which makes them suitable for mapping a face, from an unknown pose to a frontal pose, in the presence of facial expressions and occlusions. Using these methods in conjunction with large datasets of face images, we build a statistical frontal facial model and an associated parametric confidence metric, eventually used for performance analysis. We empirically show that the proposed pipeline is neither method-biased nor data-biased, and that it can be used to assess both the performance of 3DFA algorithms and the accuracy of annotations of face datasets.

![UPAfig8](https://github.com/Pan3D/Daily_Paper/blob/master/images/UPAfig8.png)

------

##### [7] Bidirectional Graph Reasoning Network for Panoptic Segmentation

- [CVPR20] Sun Yat-sen University: Yangxin Wu, Gengwei Zhang, et al.
- <https://arxiv.org/pdf/2004.06272.pdf> 

Recent researches on panoptic segmentation resort to a single end-to-end network to combine the tasks of instance segmentation and semantic segmentation. However, prior models only unified the two related tasks at the architectural level via a multi-branch scheme or revealed the underlying correlation between them by unidirectional feature fusion, which disregards the explicit semantic and co-occurrence relations among objects and background. Inspired by the fact that context information is critical to recognize and localize the objects, and inclusive object details are significant to parse the background scene, we thus investigate on explicitly modeling the correlations between object and background to achieve a holistic understanding of an image in the panoptic segmentation task. We introduce a Bidirectional Graph Reasoning Network (BGRNet), which incorporates graph structure into the conventional panoptic segmentation network to mine the intra-modular and intermodular relations within and between foreground things and background stuff classes. In particular, BGRNet first constructs image-specific graphs in both instance and semantic segmentation branches that enable flexible reasoning at the proposal level and class level, respectively. To establish the correlations between separate branches and fully leverage the complementary relations between things and stuff, we propose a Bidirectional Graph Connection Module to diffuse information across branches in a learnable fashion. Experimental results demonstrate the superiority of our BGRNet that achieves the new state-of-the-art performance on challenging COCO and ADE20K panoptic segmentation benchmarks.

![BGRfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/BGRfig2.png)

------

##### [8] DialGraph: Sparse Graph Learning Networks for Visual Dialog

- Seoul National University & SK T-Brain: Gi-Cheon Kang, et al.
- <https://arxiv.org/pdf/2004.06698.pdf>

Visual dialog is a task of answering a sequence of questions grounded in an image utilizing a dialog history. Previous studies have implicitly explored the problem of reasoning semantic structures among the history using softmax attention. However, we argue that the softmax attention yields dense structures that could distract to answer the questions requiring partial or even no contextual information. In this paper, we formulate the visual dialog tasks as graph structure learning tasks. To tackle the problem, we propose Sparse Graph Learning Networks (SGLNs) consisting of a multimodal node embedding module and a sparse graph learning module. The proposed model explicitly learn sparse dialog structures by incorporating binary and score edges, leveraging a new structural loss function. Then, it finally outputs the answer, updating each node via a message passing framework. As a result, the proposed model outperforms the state-of-the-art approaches on the VisDial v1.0 dataset, only using 10.95% of the dialog history, as well as improves interpretability compared to baseline methods.

![DialGraphfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/DialGraphfig2.png)

---

