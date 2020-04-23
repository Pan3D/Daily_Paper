##### [1] How to track your dragon: A Multi-Attentional Framework for real-time RGB-D 6-DOF Object Pose Tracking

- [ICIP 2020] National Technical University of Athens: Isidoros Marougkas, et al.
- <https://arxiv.org/pdf/2004.10335.pdf> 

We present a novel multi-attentional convolutional architecture to tackle the problem of real-time RGB-D 6D object pose tracking of single, known objects. Such a problem poses multiple challenges originating both from the objects' nature and their interaction with their environment, which previous approaches have failed to fully address. The proposed framework encapsulates methods for background clutter and occlusion handling by integrating multiple parallel soft spatial attention modules into a multitask Convolutional Neural Network (CNN) architecture. Moreover, we consider the special geometrical properties of both the object's 3D model and the pose space, and we use a more sophisticated approach for data augmentation for training. The provided experimental results confirm the effectiveness of the proposed multi-attentional architecture, as it improves the State-of-the-Art (SoA) tracking performance by an average score of 40.5% for translation and 57.5% for rotation, when testing on the dataset presented in [1], the most complete dataset designed, up to date, for the problem of RGB-D object tracking.

![HTfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/HTfig2.png)

------

##### [2] Combining Deep Learning Classifiers for 3D Action Recognition

- Masaryk University: Jan Sedmidubsky, Pavel Zezula.
- <https://arxiv.org/pdf/2004.10314.pdf> 

The popular task of 3D human action recognition is almost exclusively solved by training deep-learning classifiers. To achieve a high recognition accuracy, the input 3D actions are often pre-processed by various normalization or augmentation techniques. However, it is not computationally feasible to train a classifier for each possible variant of training data in order to select the best-performing subset of pre-processing techniques for a given dataset. In this paper, we propose to train an independent classifier for each available pre-processing technique and fuse the classification results based on a strict majority vote rule. Together with a proposed evaluation procedure, we can very efficiently determine the best combination of normalization and augmentation techniques for a specific dataset. For the best-performing combination, we can retrospectively apply the normalized/augmented variants of input data to train only a single classifier. This also allows us to decide whether it is better to train a single model, or rather a set of independent classifiers.

![CDfig4](https://github.com/Pan3D/Daily_Paper/blob/master/images/CDfig4.png)

------

##### [3] Pseudo RGB-D for Self-Improving Monocular SLAM and Depth Prediction

- IIIT-Delhi: Lokender Tiwari, et al.
- <https://arxiv.org/pdf/2004.10681.pdf> 

Classical monocular Simultaneous Localization And Mapping (SLAM) and the recently emerging convolutional neural networks (CNNs) for monocular depth prediction represent two largely disjoint approaches towards building a 3D map of the surrounding environment. In this paper, we demonstrate that the coupling of these two by leveraging the strengths of each mitigates the other's shortcomings. Specifically, we propose a joint narrow and wide baseline based self-improving framework, where on the one hand the CNN-predicted depth is leveraged to perform pseudo RGB-D feature-based SLAM, leading to better accuracy and robustness than the monocular RGB SLAM baseline. On the other hand, the bundle-adjusted 3D scene structures and camera poses from the more principled geometric SLAM are injected back into the depth network through novel wide baseline losses proposed for improving the depth prediction network, which then continues to contribute towards better pose and 3D structure estimation in the next iteration. We emphasize that our framework only requires unlabeled monocular videos in both training and inference stages, and yet is able to outperform state-of-the-art self-supervised monocular and stereo depth prediction networks (e.g, Monodepth2) and feature-based monocular SLAM system (i.e, ORB-SLAM). Extensive experiments on KITTI and TUM RGB-D datasets verify the superiority of our self-improving geometry-CNN framework.

![PRfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/PRfig2.png)

------

##### [4] Real-time Simultaneous 3D Head Modeling and Facial Motion Capture with an RGB-D camera

- Kyushu University: Diego Thomas.
- <https://arxiv.org/pdf/2004.10557.pdf> 

We propose a method to build in real-time animated 3D head models using a consumer-grade RGB-D camera. Our proposed method is the first one to provide simultaneously comprehensive facial motion tracking and a detailed 3D model of the user's head. Anyone's head can be instantly reconstructed and his facial motion captured without requiring any training or pre-scanning. The user starts facing the camera with a neutral expression in the first frame, but is free to move, talk and change his face expression as he wills otherwise. The facial motion is captured using a blendshape animation model while geometric details are captured using a Deviation image mapped over the template mesh. We contribute with an efficient algorithm to grow and refine the deforming 3D model of the head on-the-fly and in real-time. We demonstrate robust and high-fidelity simultaneous facial motion capture and 3D head modeling results on a wide range of subjects with various head poses and facial expressions.

![RSfig6](https://github.com/Pan3D/Daily_Paper/blob/master/images/RSfig6.png)

------

##### [5] TetraTSDF: 3D human reconstruction from a single image with a tetrahedral outer shell

- Kyushu University: Hayato Onizuka, et al.
- <https://arxiv.org/pdf/2004.10534.pdf> 

Recovering the 3D shape of a person from its 2D appearance is ill-posed due to ambiguities. Nevertheless, with the help of convolutional neural networks (CNN) and prior knowledge on the 3D human body, it is possible to overcome such ambiguities to recover detailed 3D shapes of human bodies from single images. Current solutions, however, fail to reconstruct all the details of a person wearing loose clothes. This is because of either (a) huge memory requirement that cannot be maintained even on modern GPUs or (b) the compact 3D representation that cannot encode all the details. In this paper, we propose the tetrahedral outer shell volumetric truncated signed distance function (TetraTSDF) model for the human body, and its corresponding part connection network (PCN) for 3D human body shape regression. Our proposed model is compact, dense, accurate, and yet well suited for CNN-based regression task. Our proposed PCN allows us to learn the distribution of the TSDF in the tetrahedral volume from a single image in an end-to-end manner. Results show that our proposed method allows to reconstruct detailed shapes of humans wearing loose clothes from single RGB images.

![TTfig5](https://github.com/Pan3D/Daily_Paper/blob/master/images/TTfig5.png)

------

##### [6] Graph-based Kinship Reasoning Network

- [ICME 2020] Tsinghua University: Wanhua Li, et al.
- <https://arxiv.org/pdf/2004.10375.pdf> 

In this paper, we propose a graph-based kinship reasoning (GKR) network for kinship verification, which aims to effectively perform relational reasoning on the extracted features of an image pair. Unlike most existing methods which mainly focus on how to learn discriminative features, our method considers how to compare and fuse the extracted feature pair to reason about the kin relations. The proposed GKR constructs a star graph called kinship relational graph where each peripheral node represents the information comparison in one feature dimension and the central node is used as a bridge for information communication among peripheral nodes. Then the GKR performs relational reasoning on this graph with recursive message passing. Extensive experimental results on the KinFaceW-I and KinFaceW-II datasets show that the proposed GKR outperforms the state-of-the-art methods.

![GKfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/GKfig2.png)

---

##### [7] Multi-view Self-Constructing Graph Convolutional Networks with Adaptive Class Weighting Loss for Semantic Segmentation

- [CVPRW 2020] Oslo: Qinghui Liu, et al.
- <https://arxiv.org/pdf/2004.10327.pdf> 
- <https://github.com/samleoqh/MSCG-Net> 

We propose a novel architecture called the Multi-view Self-Constructing Graph Convolutional Networks (MSCG-Net) for semantic segmentation. Building on the recently proposed Self-Constructing Graph (SCG) module, which makes use of learnable latent variables to self-construct the underlying graphs directly from the input features without relying on manually built prior knowledge graphs, we leverage multiple views in order to explicitly exploit the rotational invariance in airborne images. We further develop an adaptive class weighting loss to address the class imbalance. We demonstrate the effectiveness and flexibility of the proposed method on the Agriculture-Vision challenge dataset and our model achieves very competitive results (0.547 mIoU) with much fewer parameters and at a lower computational cost compared to related pure-CNN based work.

![MSfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/MSfig2.png)

------

##### [8] Graph Convolutional Subspace Clustering: A Robust Subspace Clustering Framework for Hyperspectral Image

- [TGRS] China University of Geosciences: Zijia Zhang, et al.
- <https://arxiv.org/pdf/2004.10476.pdf> 

Hyperspectral image (HSI) clustering is a challenging task due to the high complexity of HSI data. Subspace clustering has been proven to be powerful for exploiting the intrinsic relationship between data points. Despite the impressive performance in the HSI clustering, traditional subspace clustering methods often ignore the inherent structural information among data. In this paper, we revisit the subspace clustering with graph convolution and present a novel subspace clustering framework called Graph Convolutional Subspace Clustering (GCSC) for robust HSI clustering. Specifically, the framework recasts the self-expressiveness property of the data into the non-Euclidean domain, which results in a more robust graph embedding dictionary. We show that traditional subspace clustering models are the special forms of our framework with the Euclidean data. Basing on the framework, we further propose two novel subspace clustering models by using the Frobenius norm, namely Efficient GCSC (EGCSC) and Efficient Kernel GCSC (EKGCSC). Both models have a globally optimal closed-form solution, which makes them easier to implement, train, and apply in practice. Extensive experiments on three popular HSI datasets demonstrate that EGCSC and EKGCSC can achieve state-of-the-art clustering performance and dramatically outperforms many existing methods with significant margins.

![GCfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/GCfig2.png)

------

##### [9] Recursive Social Behavior Graph for Trajectory Prediction

- Shanghai Jiao Tong University: Jianhua Sun, et al.
- <https://arxiv.org/pdf/2004.10402.pdf> 

Social interaction is an important topic in human trajectory prediction to generate plausible paths. In this paper, we present a novel insight of group-based social interaction model to explore relationships among pedestrians. We recursively extract social representations supervised by group-based annotations and formulate them into a social behavior graph, called Recursive Social Behavior Graph. Our recursive mechanism explores the representation power largely. Graph Convolutional Neural Network then is used to propagate social interaction information in such a graph. With the guidance of Recursive Social Behavior Graph, we surpass state-of-the-art method on ETH and UCY dataset for 11.1% in ADE and 10.8% in FDE in average, and successfully predict complex social behaviors.

![RSfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/RSfig3.png)

----

