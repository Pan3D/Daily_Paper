3D: 12 papers, seven of them are published in CVPR2020.

Graph: 1 paper.

---

##### [1] Leveraging 2D Data to Learn Textured 3D Mesh Generation

- [CVPR20 Oral] IST Austria: Paul Henderson, et al.
- <https://arxiv.org/pdf/2004.04180.pdf>

Numerous methods have been proposed for probabilistic generative modelling of 3D objects. However, none of these is able to produce textured objects, which renders them of limited use for practical tasks. In this work, we present the first generative model of textured 3D meshes. Training such a model would traditionally require a large dataset of textured meshes, but unfortunately, existing datasets of meshes lack detailed textures. We instead propose a new training methodology that allows learning from collections of 2D images without any 3D information. To do so, we train our model to explain a distribution of images by modelling each image as a 3D foreground object placed in front of a 2D background. Thus, it learns to generate meshes that when rendered, produce images similar to those in its training set.
A well-known problem when generating meshes with deep networks is the emergence of self-intersections, which are problematic for many use-cases. As a second contribution we therefore introduce a new generation process for 3D meshes that guarantees no self-intersections arise, based on the physical intuition that faces should push one another out of the way as they move.
We conduct extensive experiments on our approach, reporting quantitative and qualitative results on both synthetic data and natural images. These show our method successfully learns to generate plausible and diverse textured 3D samples for five challenging object classes.![L2DDfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/L2DDfig2.png)

---

##### [2] X3D: Expanding Architectures for Efficient Video Recognition

- [CVPR20 Oral] FAIR: Christoph Feichtenhofer. 
- https://arxiv.org/pdf/2004.04730
- https://github.com/facebookresearch/SlowFast[Code]

This paper presents X3D, a family of efficient video networks that progressively expand a tiny 2D image classification architecture along multiple network axes, in space, time, width and depth. Inspired by feature selection methods in machine learning, a simple stepwise network expansion approach is employed that expands a single axis in each step, such that good accuracy to complexity trade-off is achieved. To expand X3D to a specific target complexity, we perform progressive forward expansion followed by backward contraction. X3D achieves state-of-the-art performance while requiring 4.8x and 5.5x fewer multiply-adds and parameters for similar accuracy as previous work. Our most surprising finding is that networks with high spatiotemporal resolution can perform well, while being extremely light in terms of network width and parameters. We report competitive accuracy at unprecedented efficiency on video  classification and detection benchmarks.![X3Dfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/X3Dfig1.png)

------

##### [3] Self-Supervised 3D Human Pose Estimation via Part Guided Novel Image Synthesis

- [CVPR20 Oral] Indian Institute of Science: Jogendra Nath Kundu, Siddharth Seth, et al.
- <https://arxiv.org/pdf/2004.04400.pdf> 
- <http://val.cds.iisc.ac.in/pgp-human/> [Project]

Camera captured human pose is an outcome of several sources of variation. Performance of supervised 3D pose estimation approaches comes at the cost of dispensing with variations, such as shape and appearance, that may be useful for solving other related tasks. As a result, the learned model not only inculcates task-bias but also dataset-bias because of its strong reliance on the annotated samples, which also holds true for weakly-supervised models. Acknowledging this, we propose a self-supervised learning framework to disentangle such variations from unlabeled video frames. We leverage the prior knowledge on human skeleton and poses in the form of a single part-based 2D puppet model, human pose articulation constraints, and a set of unpaired 3D poses. Our differentiable formalization, bridging the representation gap between the 3D pose and spatial part maps, not only facilitates discovery of interpretable pose disentanglement but also allows us to operate on videos with diverse camera movements. Qualitative results on unseen in-the-wild datasets establish our superior generalization across multiple tasks beyond the primary tasks of 3D pose estimation and part segmentation. Furthermore, we demonstrate state-of-the-art weakly-supervised 3D pose estimation performance on both Human3.6M and MPI-INF-3DHP datasets.![SS3DHPEfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/SS3DHPEfig2.png)

------

##### [4] ARCH: Animatable Reconstruction of Clothed Humans

- [CVPR20] Facebook: Zeng Huang, et al.
- <https://arxiv.org/pdf/2004.04572.pdf> 

In this paper, we propose ARCH (Animatable Reconstruction of Clothed Humans), a novel end-to-end framework for accurate reconstruction of animation-ready 3D clothed humans from a monocular image. Existing approaches to digitize 3D humans struggle to handle pose variations and recover details. Also, they do not produce models that are animation ready. In contrast, ARCH is a learned pose-aware model that produces detailed 3D rigged full-body human avatars from a single unconstrained RGB image. A Semantic Space and a Semantic Deformation Field are created using a parametric 3D body estimator. They allow the transformation of 2D/3D clothed humans into a canonical space, reducing ambiguities in geometry caused by pose variations and occlusions in training data. Detailed surface geometry and appearance are learned using an implicit function representation with spatial local features. Furthermore, we propose additional per-pixel supervision on the 3D reconstruction using opacity-aware differentiable rendering. Our experiments indicate that ARCH increases the fidelity of the reconstructed humans. We obtain more than 50% lower reconstruction errors for standard metrics compared to state-of-the-art methods on public datasets. We also show numerous qualitative examples of animated, high-quality reconstructed avatars unseen in the literature so far.![ARCHfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/ARCHfig2.png)

------

##### [5] MoreFusion: Multi-object Reasoning for 6D Pose Estimation from Volumetric Fusion

- [CVPR20] Imperial College London: Kentaro Wada, et al.
- <https://arxiv.org/pdf/2004.04336.pdf>

Robots and other smart devices need efficient object-based scene representations from their on-board vision systems to reason about contact, physics and occlusion. Recognized precise object models will play an important role alongside non-parametric reconstructions of unrecognized structures. We present a system which can estimate the accurate poses of multiple known objects in contact and occlusion from real-time, embodied multi-view vision. Our approach makes 3D object pose proposals from single RGB-D views, accumulates pose estimates and non-parametric occupancy information from multiple views as the camera moves, and performs joint optimization to estimate consistent, non-intersecting poses for multiple objects in contact.
We verify the accuracy and robustness of our approach experimentally on 2 object datasets: YCB-Video, and our own challenging Cluttered YCB-Video. We demonstrate a real-time robotics application where a robot arm precisely and orderly disassembles complicated piles of objects, using only on-board RGB-D vision.![MoreFusionfig4](https://github.com/Pan3D/Daily_Paper/blob/master/images/MoreFusionfig4.png)

------

##### [6] 3D Photography using Context-aware Layered Depth Inpainting

- [CVPR20] Virginia Tech: Meng-Li Shih, et al.
- https://arxiv.org/pdf/2004.04727.pdf
- https://github.com/vt-vl-lab/3d-photo-inpainting [Code]

We propose a method for converting a single RGB-D input image into a 3D photo a multi-layer representation for novel view synthesis that contains hallucinated color and depth structures in regions occluded in the original view. We use a Layered Depth Image with explicit pixel connectivity as underlying representation, and present a learning-based inpainting model that synthesizes new local color-and-depth content into the occluded region in a spatial context-aware manner. The resulting 3D photos can be efficiently rendered with motion parallax using standard graphics engines. We validate the effectiveness of our method on a wide range of challenging everyday scenes and show fewer artifacts compared with the state of the arts.![3DPfig6](https://github.com/Pan3D/Daily_Paper/blob/master/images/3DPfig6.png)

------

##### [7] Where Does It End? -- Reasoning About Hidden Surfaces by Object Intersection Constraints

- [CVPR20] Embodied Vision Group: Michael Strecke and Jorg Stuckler.
- https://arxiv.org/pdf/2004.04630.pdf

Dynamic scene understanding is an essential capability in robotics and VR/AR. In this paper we propose Co-Section, an optimization-based approach to 3D dynamic scene reconstruction, which infers hidden shape information from intersection constraints. An object-level dynamic SLAM frontend detects, segments, tracks and maps dynamic objects in the scene. Our optimization backend completes the shapes using hull and intersection constraints between the objects. In experiments, we demonstrate our approach on real and synthetic dynamic scene datasets. We also assess the shape completion performance of our method quantitatively. To the best of our knowledge, our approach is the first method to incorporate such physical plausibility constraints on object intersections for shape completion of dynamic objects in an energy minimization framework.![WDfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/WDfig2.png)

---

##### [8] Cortical surface registration using unsupervised learning

- Harvard Medical School: Jieyu Cheng, et al.
- <https://arxiv.org/pdf/2004.04617.pdf> 

Non-rigid cortical registration is an important and challenging task due to the geometric complexity of the human cortex and the high degree of inter-subject variability. A conventional solution is to use a spherical representation of surface properties and perform registration by aligning cortical folding patterns in that space. This strategy produces accurate spatial alignment but often requires a high computational cost. Recently, convolutional neural networks (CNNs) have demonstrated the potential to dramatically speed up volumetric registration. However, due to distortions introduced by projecting a sphere to a 2D plane, a direct application of recent learning-based methods to surfaces yields poor results. In this study, we present SphereMorph, a diffeomorphic registration framework for cortical surfaces using deep networks that addresses these issues. SphereMorph uses a UNet-style network associated with a spherical kernel to learn the displacement field and warps the sphere using a modified spatial transformer layer. We propose a resampling weight in computing the data fitting loss to account for distortions introduced by polar projection, and demonstrate the performance of our proposed method on two tasks, including cortical parcellation and group-wise functional area alignment. The experiments show that the proposed SphereMorph is capable of modeling the geometric registration problem in a CNN framework and demonstrate superior registration accuracy and computational efficiency.![CSRfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/CSRfig2.png)

------

##### [9] Masked GANs for Unsupervised Depth and Pose Prediction with Scale Consistency

- East China University of Science and Technology: Chaoqiang Zhao, et al.
- <https://arxiv.org/pdf/2004.04345.pdf> 

Previous works have shown that adversarial learning can be used for unsupervised monocular depth and visual odometry (VO) estimation. However, the performance of pose and depth networks is limited by occlusions and visual field changes. Because of the incomplete correspondence of visual information between frames caused by motion, target images cannot be synthesized completely from source images via view reconstruction and bilinear interpolation. The reconstruction loss based on the difference between synthesized and real target images will be affected by the incomplete reconstruction. Besides, the data distribution of unreconstructed regions will be learned and help the discriminator distinguish between real and fake images, thereby causing the case that the generator may fail to compete with the discriminator. Therefore, a MaskNet is designed in this paper to predict these regions and reduce their impacts on the reconstruction loss and adversarial loss. The impact of unreconstructed regions on discriminator is tackled by proposing a boolean mask scheme, as shown in Fig. 1. Furthermore, we consider the scale consistency of our pose network by utilizing a new scale-consistency loss, therefore our pose network is capable of providing the full camera trajectory over the long monocular sequence. Extensive experiments on KITTI dataset show that each component proposed in this paper contributes to the performance, and both of our depth and trajectory prediction achieve competitive performance.![MGfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/MGfig3.png)

------

##### [10] LightConvPoint: convolution for points

- Valeo: Alexandre Boulch, et al.
- <https://arxiv.org/pdf/2004.04462.pdf> 
- <https://github.com/valeoai/LightConvPoint> [Code]

Recent state-of-the-art methods for point cloud semantic segmentation are based on convolution defined for point clouds. In this paper, we propose a formulation of the convolution for point cloud directly designed from the discrete convolution in image processing. The resulting formulation underlines the separation between the discrete kernel space and the geometric space where the points lies. The link between the two space is done by a change space matrix A which distributes the input features on the convolution kernel. Several existing methods fall under this formulation. We show that the matrix A can be easily estimated with neural networks. Finally, we show competitive results on several semantic segmentation benchmarks while being efficient both in computation time and memory.![LCPfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/LCPfig2.png)

---

##### [11] Spatial Information Guided Convolution for Real-Time RGBD Semantic Segmentation

- Nankai University: Chaoqiang Zhao, et al.
- https://arxiv.org/pdf/2004.04534.pdf

3D spatial information is known to be beneficial to the semantic segmentation task. Most existing methods take 3D spatial data as an additional input, leading to a two-stream segmentation network that processes RGB and 3D spatial information separately. This solution greatly increases the inference time and severely limits its scope for real-time applications. To solve this problem, we propose Spatial information guided Convolution (S-Conv), which allows efficient RGB feature and 3D spatial information integration. S-Conv is competent to infer the sampling offset of the convolution kernel guided by the 3D spatial information, helping the convolutional layer adjust the receptive field and adapt to geometric transformations. S-Conv also incorporates geometric information into the feature learning process by generating spatially adaptive convolutional weights. The capability of perceiving geometry is largely enhanced without much affecting the amount of parameters and computational cost. We further embed S-Conv into a semantic segmentation network, called Spatial information Guided convolutional Network (SGNet), resulting in real-time inference and state-of-the-art performance on NYUDv2 and SUNRGBD datasets.![SIGCfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/SIGCfig2.png)

------

##### [12] Neural Object Descriptors for Multi-View Shape Reconstruction

- Dyson Robotics Lab: Edgar Sucar, et al.
- https://arxiv.org/pdf/2004.04485.pdf

The choice of scene representation is crucial in both the shape inference algorithms it requires and the smart applications it enables. We present efficient and optimisable multi-class learned object descriptors together with a novel probabilistic and differential rendering engine, for principled full object shape inference from one or more RGB-D images. Our framework allows for accurate and robust 3D object reconstruction which enables multiple applications including robot grasping and placing, augmented reality, and the first object level SLAM system capable of optimising object poses and shapes jointly with camera trajectory.![NODfig4](https://github.com/Pan3D/Daily_Paper/blob/master/images/NODfig4.png)

------

##### [13] Learning to Recognizing Spatial Configurations of Objects with Graph Neural Networks

- Inria: Laetitia Teodorescu, et al.
- <https://arxiv.org/pdf/2004.04546.pdf>

Deep learning algorithms can be seen as compositions of functions acting on learned representations encoded as tensor-structured data. However, in most applications those representations are monolithic, with for instance one single vector encoding an entire image or sentence. In this paper, we build upon the recent successes of Graph Neural Networks (GNNs) to explore the use of graph-structured representations for learning spatial configurations. Motivated by the ability of humans to distinguish arrangements of shapes, we introduce two novel geometrical reasoning tasks, for which we provide the datasets. We introduce novel GNN layers and architectures to solve the tasks and show that graph-structured representations are necessary for good performance.![LRSCfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/LRSCfig2.png)

---

