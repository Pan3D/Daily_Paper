### 7 April, 2020

3D: 13 papers，three of them are publishhed in CVPR.

3D medical image: 2 papers，published in MICCAI19 and ICPR20 seperately.

Graph: 2 papers, one of them published in CVPR.

---

##### [1] Robust 3D Self-portraits in Seconds

- [CVPR20 Oral] Tsinghua University: Zhe Li, et al.
- <https://arxiv.org/pdf/2004.02460.pdf>

In this paper, we propose an efficient method for robust 3D self-portraits using a single RGBD camera. Benefiting from the proposed PIFusion and lightweight bundle adjustment algorithm, our method can generate detailed 3D self-portraits in seconds and shows the ability to handle subjects wearing extremely loose clothes. To achieve highly efficient and robust reconstruction, we propose PIFusion, which combines learning-based 3D recovery with volumetric non-rigid fusion to generate accurate sparse partial scans of the subject. Moreover, a non-rigid volumetric deformation method is proposed to continuously refine the learned shape prior. Finally, a lightweight bundle adjustment algorithm is proposed to guarantee that all the partial scans can not only "loop" with each other but also remain consistent with the selected live key observations. The results and experiments show that the proposed method achieves more robust and efficient 3D self-portraits compared with state-of-the-art methods.

![RSPfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/RSPfig2.png)

------

##### [2] Weakly-Supervised Mesh-Convolutional Hand Reconstruction in the Wild

- [CVPR20 Oral] Imperial College London: Dominik Kulon, et al.
- <https://arxiv.org/pdf/2004.01946.pdf> 
- <https://www.arielai.com/mesh_hands/> [Project] 

We introduce a simple and effective network architecture for monocular 3D hand pose estimation consisting of an image encoder followed by a mesh convolutional decoder that is trained through a direct 3D hand mesh reconstruction loss. We train our network by gathering a large-scale dataset of hand action in YouTube videos and use it as a source of weak supervision. Our weakly-supervised mesh convolutions-based system largely outperforms state-of-the-art methods, even halving the errors on the in the wild benchmark.

![WSMCfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/WSMCfig1.png)

------

##### [3] Lightweight Multi-View 3D Pose Estimation through Camera-Disentangled Representation

- [CVPR20] EPFL: Edoardo Remelli, et al.
- <https://arxiv.org/pdf/2004.02186.pdf> 

We present a lightweight solution to recover 3D pose from multi-view images captured with spatially calibrated cameras. Building upon recent advances in interpretable representation learning, we exploit 3D geometry to fuse input images into a unified latent representation of pose, which is disentangled from camera view-points. This allows us to reason effectively about 3D pose across different views without using compute-intensive volumetric grids. Our architecture then conditions the learned representation on camera projection operators to produce accurate per-view 2d detections, that can be simply lifted to 3D via a differentiable Direct Linear Transform (DLT) layer. In order to do it efficiently, we propose a novel implementation of DLT that is orders of magnitude faster on GPU architectures than standard SVD-based triangulation methods. We evaluate our approach on two large-scale human pose datasets (H36M and Total Capture): our method outperforms or performs comparably to the state-of-the-art volumetric methods, while, unlike them, yielding real-time performance.

![LMfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/LMfig2.png)

------

##### [4] Deformable 3D Convolution for Video Super-Resolution

- National University of Defense Technology: Xinyi Ying, et al.
- <https://arxiv.org/pdf/2004.02803.pdf> 
- <https://github.com/XinyiYing/D3Dnet> [Torch]

The spatio-temporal information among video sequences is significant for video super-resolution (SR). However, the spatio-temporal information cannot be fully used by existing video SR methods since spatial feature extraction and temporal motion compensation are usually performed sequentially. In this paper, we propose a deformable 3D convolution network (D3Dnet) to incorporate spatio-temporal information from both spatial and temporal dimensions for video SR. Specifically, we introduce deformable 3D convolutions (D3D) to integrate 2D spatial deformable convolutions with 3D convolutions (C3D), obtaining both superior spatio-temporal modeling capability and motion-aware modeling flexibility. Extensive experiments have demonstrated the effectiveness of our proposed D3D in exploiting spatio-temporal information. Comparative results show that our network outperforms the state-of-the-art methods.

![D3DCfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/D3DCfig2.png)

------

##### [5] Light3DPose: Real-time Multi-Person 3D PoseEstimation from Multiple Views

- Checkout Technologies: Alessio Elmi, Davide Mazzini, et al.
- <https://arxiv.org/pdf/2004.02688.pdf>

We present an approach to perform 3D pose estimation of multiple people from a few calibrated camera views. Our architecture, leveraging the recently proposed unprojection layer, aggregates feature-maps from a 2D pose estimator backbone into a comprehensive representation of the 3D scene. Such intermediate representation is then elaborated by a fully-convolutional volumetric network and a decoding stage to extract 3D skeletons with sub-voxel accuracy. Our method achieves state of the art MPJPE on the CMU Panoptic dataset using a few unseen views and obtains competitive results even with a single input view. We also assess the transfer learning capabilities of the model by testing it against the publicly available Shelf dataset obtaining good performance metrics. The proposed method is inherently efficient: as a pure bottom-up approach, it is computationally independent of the number of people in the scene. Furthermore, even though the computational burden of the 2D part scales linearly with the number of input views, the overall architecture is able to exploit a very lightweight 2D backbone which is orders of magnitude faster than the volumetric counterpart, resulting in fast inference time. The system can run at 6 FPS, processing up to 10 camera views on a single 1080Ti GPU.

![Light3DPosefig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/Light3DPosefig2.png)

------

##### [6] Finding Your (3D) Center: 3D Object Detection Using a Learned Loss

- University College London: David Griffiths, et al.
- <https://arxiv.org/pdf/2004.02693.pdf>
- <https://dgriffiths3.github.io/> [Project]

Massive semantic labeling is readily available for 2D images, but much harder to achieve for 3D scenes. Objects in 3D repositories like ShapeNet are labeled, but regrettably only in isolation, so without context. 3D scenes can be acquired by range scanners on city-level scale, but much fewer with semantic labels. Addressing this disparity, we introduce a new optimization procedure, which allows training for 3D detection with raw 3D scans while using as little as 5% of the object labels and still achieve comparable performance. Our optimization uses two networks. A scene network maps an entire 3D scene to a set of 3D object centers. As we assume the scene not to be labeled by centers, no classic loss, such as chamfer can be used to train it. Instead, we use another network to emulate the loss. This loss network is trained on a small labeled subset and maps a non-centered 3D object in the presence of distractions to its own center. This function is very similar - and hence can be used instead of - the gradient the supervised loss would have. Our evaluation documents competitive fidelity at a much lower level of supervision, respectively higher quality at comparable supervision.

![FYCfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/FYCfig1.png)

------

##### [7]Reconfigurable Voxels: A New Representation for LiDAR-Based Point Clouds

- The Chinese University of Hong Kong: Tai Wang, et al.
- <https://arxiv.org/pdf/2004.02724.pdf> 

LiDAR is an important method for autonomous driving systems to sense the environment. The point clouds obtained by LiDAR typically exhibit sparse and irregular distribution, thus posing great challenges to the detection of 3D objects, especially those that are small and distant. To tackle this difficulty, we propose Reconfigurable Voxels, a new approach to constructing representations from 3D point clouds. Specifically, we devise a biased random walk scheme, which adaptively covers each neighborhood with a fixed number of voxels based on the local spatial distribution and produces a representation by integrating the points in the chosen neighbors. We found empirically that this approach effectively improves the stability of voxel features, especially for sparse regions. Experimental results on multiple benchmarks, including nuScenes, Lyft, and KITTI, show that this new representation can remarkably improve the detection performance for small and distant objects, without incurring noticeable overhead costs.

![RVfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/RVfig2.png)

------

##### [8] Guiding Monocular Depth Estimation Using Depth-Attention Volume

- University of Oulu: Lam Huynh, et al.
- <https://arxiv.org/pdf/2004.02760.pdf> 

Recovering the scene depth from a single image is an ill-posed problem that requires additional priors, often referred to as monocular depth cues, to disambiguate different 3D interpretations. In recent works, those priors have been learned in an end-to-end manner from large datasets by using deep neural networks. In this paper, we propose guiding depth estimation to favor planar structures that are ubiquitous especially in indoor environments. This is achieved by incorporating a non-local coplanarity constraint to the network with a novel attention mechanism called depth-attention volume (DAV). Experiments on two popular indoor datasets, namely NYU-Depth-v2 and ScanNet, show that our method achieves state-of-the-art depth estimation results while using only a fraction of the number of parameters needed by the competing methods.

![GMDEfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/GMDEfig3.png)

------

##### [9] Anisotropic Convolutional Networks for 3D Semantic Scene Completion

- Nanjing University of Science and Technology: Jie Li, et al.
- <https://waterljwant.github.io/SSC/> [Project]

As a voxel-wise labeling task, semantic scene completion (SSC) tries to simultaneously infer the occupancy and semantic labels for a scene from a single depth and/or RGB image. The key challenge for SSC is how to effectively take advantage of the 3D context to model various objects or stuffs with severe variations in shapes, layouts and visibility. To handle such variations, we propose a novel module called anisotropic convolution, which properties with flexibility and power impossible for the competing methods such as standard 3D convolution and some of its variations. In contrast to the standard 3D convolution that is limited to a fixed 3D receptive field, our module is capable of modeling the dimensional anisotropy voxel-wisely. The basic idea is to enable anisotropic 3D receptive field by decomposing a 3D convolution into three consecutive 1D convolutions, and the kernel size for each such 1D convolution is adaptively determined on the fly. By stacking multiple such anisotropic convolution modules, the voxel-wise modeling capability can be further enhanced while maintaining a controllable amount of model parameters. Extensive experiments on two SSC benchmarks, NYU-Depth-v2 and NYUCAD, show the superior performance of the proposed method.

![ACNfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/ACNfig1.png)

------

##### [10] Learning and Recognizing Archeological Features from LiDAR Data

- [IC Big Data] IBM: Conrad M Albrecht, et al.
- <https://arxiv.org/pdf/2004.02099.pdf> 

We present a remote sensing pipeline that processes LiDAR (Light Detection And Ranging) data through machine & deep learning for the application of archeological feature detection on big geo-spatial data platforms such as e.g. IBM PAIRS Geoscope. Today, archeologists get overwhelmed by the task of visually surveying huge amounts of (raw) LiDAR data in order to identify areas of interest for inspection on the ground. We showcase a software system pipeline that results in significant savings in terms of expert productivity while missing only a small fraction of the artifacts. Our work employs artificial neural networks in conjunction with an efficient spatial segmentation procedure based on domain knowledge. Data processing is constraint by a limited amount of training labels and noisy LiDAR signals due to vegetation cover and decay of ancient structures. We aim at identifying geo-spatial areas with archeological artifacts in a supervised fashion allowing the domain expert to flexibly tune parameters based on her needs.

![LRAFfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/LRAFfig1.png)

------

##### [11] SimAug: Learning Robust Representations from 3D Simulation for Pedestrian Trajectory Prediction in Unseen Cameras

- Carnegie Mellon University: Junwei Liang, et al.
- <https://arxiv.org/pdf/2004.02022.pdf>

This paper focuses on the problem of predicting future trajectories of people in unseen scenarios and camera views. We propose a method to efficiently utilize multi-view 3D simulation data for training. Our approach finds the hardest camera view to mix up with adversarial data from the original camera view in training, thus enabling the model to learn robust representations that can generalize to unseen camera views. We refer to our method as SimAug. We show that SimAug achieves best results on three out-of-domain real-world benchmarks, as well as getting state-of-the-art in the Stanford Drone and the VIRAT/ActEV dataset with in-domain training data. We will release our models and code.

![SimAugfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/SimAugfig2.png)

---

##### [12] SSN: Shape Signature Networks for Multi-class Object Detection from Point Clouds

- The Chinese University of Hong Kong: Xinge Zhu, et al.

- <https://arxiv.org/pdf/2004.02774.pdf> 
- <https://github.com/xinge008/SSN> 

Multi-class 3D object detection aims to localize and classify objects of multiple categories from point clouds. Due to the nature of point clouds, i.e. unstructured, sparse and noisy, some features benefit-ting multi-class discrimination are underexploited, such as shape information. In this paper, we propose a novel 3D shape signature to explore the shape information from point clouds. By incorporating operations of symmetry, convex hull and chebyshev fitting, the proposed shape sig-nature is not only compact and effective but also robust to the noise, which serves as a soft constraint to improve the feature capability of multi-class discrimination. Based on the proposed shape signature, we develop the shape signature networks (SSN) for 3D object detection, which consist of pyramid feature encoding part, shape-aware grouping heads and explicit shape encoding objective. Experiments show that the proposed method performs remarkably better than existing methods on two large-scale datasets. Furthermore, our shape signature can act as a plug-and-play component and ablation study shows its effectiveness and good scalability.

![SSNfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/SSNfig2.png)

---

##### [13] SqueezeSegV3: Spatially-Adaptive Convolution for Efficient Point-Cloud Segmentation

- UC Berkeley: Chenfeng Xu, et al.
- <https://arxiv.org/pdf/2004.01803.pdf> 
- <https://github.com/chenfengxu714/SqueezeSegV3> 

LiDAR point-cloud segmentation is an important problem for many applications. For large-scale point cloud segmentation, the \textit{de facto} method is to project a 3D point cloud to get a 2D LiDAR image and use convolutions to process it. Despite the similarity between regular RGB and LiDAR images, we discover that the feature distribution of LiDAR images changes drastically at different image locations. Using standard convolutions to process such LiDAR images is problematic, as convolution filters pick up local features that are only active in specific regions in the image. As a result, the capacity of the network is under-utilized and the segmentation performance decreases. To fix this, we propose Spatially-Adaptive Convolution (SAC) to adopt different filters for different locations according to the input image. SAC can be computed efficiently since it can be implemented as a series of element-wise multiplications, im2col, and standard convolution. It is a general framework such that several previous methods can be seen as special cases of SAC. Using SAC, we build SqueezeSegV3 for LiDAR point-cloud segmentation and outperform all previous published methods by at least 3.7% mIoU on the SemanticKITTI benchmark with comparable inference speed.

![SqueezeSegV3fig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/SqueezeSegV3fig1.png)

---

##### [14] Volumetric Attention for 3D Medical Image Segmentation and Detection

- [MICCAI19] Sigma Technologies: Xudong Wang, et al.
- <https://arxiv.org/pdf/2004.01997.pdf>

A volumetric attention(VA) module for 3D medical image segmentation and detection is proposed. VA attention is inspired by recent advances in video processing, enables 2.5D networks to leverage context information along the z direction, and allows the use of pretrained 2D detection models when training data is limited, as is often the case for medical applications. Its integration in the Mask R-CNN is shown to enable state-of-the-art performance on the Liver Tumor Segmentation (LiTS) Challenge, outperforming the previous challenge winner by 3.9 points and achieving top performance on the LiTS leader board at the time of paper submission. Detection experiments on the DeepLesion dataset also show that the addition of VA to existing object detectors enables a 69.1 sensitivity at 0.5 false positive per image, outperforming the best published results by 6.6 points.

![VAfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/VAfig2.png)

------

##### [15] Semantic Segmentation of highly class imbalanced fully labelled 3D volumetric biomedical images and unsupervised Domain Adaptation of the pre-trained Segmentation Network to segment another fully unlabelled Biomedical 3D Image stack

- [ICPR20] Indian Institute of Science: Shreya Roy, Anirban Chakraborty.
- <https://arxiv.org/pdf/2004.02748.pdf> 

The goal of our work is to perform pixel label semantic segmentation on 3D biomedical volumetric data. Manual annotation is always difficult for a large bio-medical dataset. So, we consider two cases where one dataset is fully labeled and the other dataset is assumed to be fully unlabelled. We first perform Semantic Segmentation on the fully labeled isotropic biomedical source data (FIBSEM) and try to incorporate the the trained model for segmenting the target unlabelled dataset(SNEMI3D)which shares some similarities with the source dataset in the context of different types of cellular bodies and other cellular components. Although, the cellular components vary in size and shape. So in this paper, we have proposed a novel approach in the context of unsupervised domain adaptation while classifying each pixel of the target volumetric data into cell boundary and cell body. Also, we have proposed a novel approach to giving non-uniform weights to different pixels in the training images while performing the pixel-level semantic segmentation in the presence of the corresponding pixel-wise label map along with the training original images in the source domain. We have used the Entropy Map or a Distance Transform matrix retrieved from the given ground truth label map which has helped to overcome the class imbalance problem in the medical image data where the cell boundaries are extremely thin and hence, extremely prone to be misclassified as non-boundary.

![SSfig4](https://github.com/Pan3D/Daily_Paper/blob/master/images/SSfig4.png)

---

##### [16] Geometrically Principled Connections in Graph Neural Networks

- [CVPR20] Imperial College London: Shunwang Gong, Mehdi Bahri, et al.
- <https://arxiv.org/pdf/2004.02658.pdf> 

Graph convolution operators bring the advantages of deep learning to a variety of graph and mesh processing tasks previously deemed out of reach. With their continued success comes the desire to design more powerful architectures, often by adapting existing deep learning techniques to non-Euclidean data. In this paper, we argue geometry should remain the primary driving force behind innovation in the emerging field of geometric deep learning. We relate graph neural networks to widely successful computer graphics and data approximation models: radial basis functions (RBFs). We conjecture that, like RBFs, graph convolution layers would benefit from the addition of simple functions to the powerful convolution kernels. We introduce affine skip connections, a novel building block formed by combining a fully connected layer with any graph convolution operator. We experimentally demonstrate the effectiveness of our technique and show the improved performance is the consequence of more than the increased number of parameters. Operators equipped with the affine skip connection markedly outperform their base performance on every task we evaluated, i.e., shape reconstruction, dense shape correspondence, and graph classification. We hope our simple and effective approach will serve as a solid baseline and help ease future research in graph neural networks.

![GPCfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/GPCfig1.png)

---

##### [17] Iterative Context-Aware Graph Inference for Visual Dialog

- Hefei University of Technology: Dan Guo, et al.
- <https://arxiv.org/pdf/2004.02194.pdf> 

Visual dialog is a challenging task that requires the comprehension of the semantic dependencies among implicit visual and textual contexts. This task can refer to the relation inference in a graphical model with sparse contexts and unknown graph structure (relation descriptor), and how to model the underlying context-aware relation inference is critical. To this end, we propose a novel Context-Aware Graph (CAG) neural network. Each node in the graph corresponds to a joint semantic feature, including both object-based (visual) and history-related (textual) context representations. The graph structure (relations in dialog) is iteratively updated using an adaptive top-K message passing mechanism. Specifically, in every message passing step, each node selects the most K relevant nodes, and only receives messages from them. Then, after the update, we impose graph attention on all the nodes to get the final graph embedding and infer the answer. In CAG, each node has dynamic relations in the graph (different related K neighbor nodes), and only the most relevant nodes are attributive to the context-aware relational graph inference. Experimental results on VisDial v0.9 and v1.0 datasets show that CAG outperforms comparative methods. Visualization results further validate the interpretability of our method.

![ICGfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/ICGfig2.png)

----

