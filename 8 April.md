### 8 April, 2020

3D: 9 papers, fourof them are published in CVPR。

Graph: 1 papers.

---

##### [1] Disp R-CNN: Stereo 3D Object Detection via Shape Prior Guided Instance Disparity Estimation

- [CVPR20] Zhejiang University & SenseTime: Jiaming Sun, Linghao Chen, et al.
- <https://arxiv.org/pdf/2004.03572.pdf> 
- <https://github.com/zju3dv/disprcnn> [Torch]

In this paper, we propose a novel system named Disp R-CNN for 3D object detection from stereo images. Many recent works solve this problem by first recovering a point cloud with disparity estimation and then apply a 3D detector. The disparity map is computed for the entire image, which is costly and fails to leverage category-specific prior. In contrast, we design an instance disparity estimation network (iDispNet) that predicts disparity only for pixels on objects of interest and learns a category-specific shape prior for more accurate disparity estimation. To address the challenge from scarcity of disparity annotation in training, we propose to use a statistical shape model to generate dense disparity pseudo-ground-truth without the need of LiDAR point clouds, which makes our system more widely applicable. Experiments on the KITTI dataset show that, even when LiDAR ground-truth is not available at training time, Disp R-CNN achieves competitive performance and outperforms previous state-of-the-art methods by 20% in terms of average precision.

![DispRCNNfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/DispRCNNfig2.png)

------

##### [2] Cascaded Refinement Network for Point Cloud Completion

- [CVPR20] National University of Singapore: Xiaogang Wang, et al.
- <https://arxiv.org/pdf/2004.03327.pdf> 
- <https://github.com/xiaogangw/cascaded-point-completion> 

Point clouds are often sparse and incomplete. Existing shape completion methods are incapable of generating details of objects or learning the complex point distributions. To this end, we propose a cascaded refinement network together with a coarse-to-fine strategy to synthesize the detailed object shapes. Considering the local details of partial input with the global shape information together, we can preserve the existing details in the incomplete point set and generate the missing parts with high fidelity. We also design a patch discriminator that guarantees every local area has the same pattern with the ground truth to learn the complicated point distribution. Quantitative and qualitative experiments on different datasets show that our method achieves superior results compared to existing state-of-the-art approaches on the 3D point cloud completion task.

![CRNfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/CRNfig2.png)

------

##### [3] End-to-End Pseudo-LiDAR for Image-Based 3D Object Detection

- [CVPR20] Cornell Univeristy: Rui Qian, Divyansh Garg, et al.
- <https://arxiv.org/pdf/2004.03080.pdf> 
- <https://github.com/mileyan/pseudo-LiDAR_e2e> 

Reliable and accurate 3D object detection is a necessity for safe autonomous driving. Although LiDAR sensors can provide accurate 3D point cloud estimates of the environment, they are also prohibitively expensive for many settings. Recently, the introduction of pseudo-LiDAR (PL) has led to a drastic reduction in the accuracy gap between methods based on LiDAR sensors and those based on cheap stereo cameras. PL combines state-of-the-art deep neural networks for 3D depth estimation with those for 3D object detection by converting 2D depth map outputs to 3D point cloud inputs. However, so far these two networks have to be trained separately. In this paper, we introduce a new framework based on differentiable Change of Representation (CoR) modules that allow the entire PL pipeline to be trained end-to-end. The resulting framework is compatible with most state-of-the-art networks for both tasks and in combination with PointRCNN improves over PL consistently across all benchmarks -- yielding the highest entry on the KITTI image-based 3D object detection leaderboard at the time of submission.

![E2EPLfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/E2EPLfig3.png)

------

##### [4] Depth Sensing Beyond LiDAR Range

- [CVPR20] Cornell University: Kai Zhang, et al.
- <https://arxiv.org/pdf/2004.03048.pdf> 

Depth sensing is a critical component of autonomous driving technologies, but today's LiDAR- or stereo camera-based solutions have limited range. We seek to increase the maximum range of self-driving vehicles' depth perception modules for the sake of better safety. To that end, we propose a novel three-camera system that utilizes small field of view cameras. Our system, along with our novel algorithm for computing metric depth, does not require full pre-calibration and can output dense depth maps with practically acceptable accuracy for scenes and objects at long distances not well covered by most commercial LiDARs.

![DSBLRfig23](https://github.com/Pan3D/Daily_Paper/blob/master/images/DSBLRfig23.png)

------

##### [5] MNEW: Multi-domain Neighborhood Embedding and Weighting for Sparse Point Clouds Segmentation

- Aptiv Corporation: Yang Zheng, et al.
- <https://arxiv.org/pdf/2004.03401.pdf> 

Point clouds have been widely adopted in 3D semantic scene understanding. However, point clouds for typical tasks such as 3D shape segmentation or indoor scenario parsing are much denser than outdoor LiDAR sweeps for the application of autonomous driving perception. Due to the spatial property disparity, many successful methods designed for dense point clouds behave depreciated effectiveness on the sparse data. In this paper, we focus on the semantic segmentation task of sparse outdoor point clouds. We propose a new method called MNEW, including multi-domain neighborhood embedding, and attention weighting based on their geometry distance, feature similarity, and neighborhood sparsity. The network architecture inherits PointNet which directly process point clouds to capture pointwise details and global semantics, and is improved by involving multi-scale local neighborhoods in static geometry domain and dynamic feature space. The distance/similarity attention and sparsity-adapted weighting mechanism of MNEW enable its capability for a wide range of data sparsity distribution. With experiments conducted on virtual and real KITTI semantic datasets, MNEW achieves the top performance for sparse point clouds, which is important to the application of LiDAR-based automated driving perception.

![MNEWfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/MNEWfig3.png)

------

##### [6] Differential 3D Facial Recognition: Adding 3D to Your State-of-the-Art 2D Method

- Universidad de la Republica: J. Matias Di Martino, et al.
- <https://arxiv.org/pdf/2004.03385.pdf> 

Active illumination is a prominent complement to enhance 2D face recognition and make it more robust, e.g., to spoofing attacks and low-light conditions. In the present work we show that it is possible to adopt active illumination to enhance state-of-the-art 2D face recognition approaches with 3D features, while bypassing the complicated task of 3D reconstruction. The key idea is to project over the test face a high spatial frequency pattern, which allows us to simultaneously recover real 3D information plus a standard 2D facial image. Therefore, state-of-the-art 2D face recognition solution can be transparently applied, while from the high frequency component of the input image, complementary 3D facial features are extracted. Experimental results on ND-2006 dataset show that the proposed ideas can significantly boost face recognition performance and dramatically improve the robustness to spoofing attacks.

![D3DFRfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/D3DFRfig3.png)

------

##### [7] SC4D: A Sparse 4D Convolutional Network for Skeleton-Based Action Recognition

- Chinese Academy of Sciences: Lei Shi, et al.
- <https://arxiv.org/pdf/2004.03259.pdf> 

In this paper, a new perspective is presented for skeleton-based action recognition. Specifically, we regard the skeletal sequence as a spatial-temporal point cloud and voxelize it into a 4-dimensional grid. A novel sparse 4D convolutional network (SC4D) is proposed to directly process the generated 4D grid for high-level perceptions. Without manually designing the hand-crafted transformation rules, it makes better use of the advantages of the convolutional network, resulting in a more concise, general and robust framework for skeletal data. Besides, by processing the space and time simultaneously, it largely keeps the spatial-temporal consistency of the skeletal data, and thus brings better expressiveness. Moreover, with the help of the sparse tensor, it can be efficiently executed with less computations. To verify the superiority of SC4D, extensive experiments are conducted on two challenging datasets, namely, NTU-RGBD and SHREC, where SC4D achieves state-of-the-art performance on both of them.

![SC4Dfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/SC4Dfig1.png)

------

##### [8] Predicting Camera Viewpoint Improves Cross-dataset Generalization for 3D Human Pose Estimation

- UC Irvine: Zhe Wang, et al.
- <https://arxiv.org/pdf/2004.03143.pdf> 
- <http://wangzheallen.github.io/cross-dataset-generalization> [Project]

Monocular estimation of 3d human pose has attracted increased attention with the availability of large ground-truth motion capture datasets. However, the diversity of training data available is limited and it is not clear to what extent methods generalize outside the specific datasets they are trained on. In this work we carry out a systematic study of the diversity and biases present in specific datasets and its effect on cross-dataset generalization across a compendium of 5 pose datasets. We specifically focus on systematic differences in the distribution of camera viewpoints relative to a body-centered coordinate frame. Based on this observation, we propose an auxiliary task of predicting the camera viewpoint in addition to pose. We find that models trained to jointly predict viewpoint and pose systematically show significantly improved cross-dataset generalization.

![PCVIfig4](https://github.com/Pan3D/Daily_Paper/blob/master/images/PCVIfig4.png)

------

##### [9] Learning to Accelerate Decomposition for Multi-Directional 3D Printing

- Tsinghua University: Chenming Wu, et al.
- <https://arxiv.org/pdf/2004.03450.pdf> 

As a strong complementary of additive manufacturing, multi-directional 3D printing has the capability of decreasing or eliminating the need for support structures. Recent work proposed a beam-guided search algorithm to find an optimized sequence of plane-clipping, which gives volume decomposition of a given 3D model. Different printing directions are employed in different regions so that a model can be fabricated with tremendously less supports (or even no support in many cases). To obtain optimized decomposition, a large beam width needs to be used in the search algorithm, which therefore leads to a very time-consuming computation. In this paper, we propose a learning framework that can accelerate the beam-guided search by using only 1/2 of the original beam width to obtain results with similar quality. Specifically, we train a classifier for each pair of candidate clipping planes based on six newly proposed feature metrics from the results of beam-guided search with large beam width. With the help of these feature metrics, both the current and the sequence-dependent information are captured by the classifier to score candidates of clipping. As a result, we can achieve around 2 times acceleration. We test and demonstrate the performance of our accelerated decomposition on a large dataset of models for 3D printing.

![LADfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/LADfig3.png)

------

##### [10] Generative Adversarial Zero-shot Learning via Knowledge Graphs

- Zhejiang University: Yuxia Geng, et al.
- <https://arxiv.org/pdf/2004.03109.pdf> 

Zero-shot learning (ZSL) is to handle the prediction of those unseen classes that have no labeled training data. Recently, generative methods like Generative Adversarial Networks (GANs) are being widely investigated for ZSL due to their high accuracy, generalization capability and so on. However, the side information of classes used now is limited to text descriptions and attribute annotations, which are in short of semantics of the classes. In this paper, we introduce a new generative ZSL method named KG-GAN by incorporating rich semantics in a knowledge graph (KG) into GANs. Specifically, we build upon Graph Neural Networks and encode KG from two views: class view and attribute view considering the different semantics of KG. With well-learned semantic embeddings for each node (representing a visual category), we leverage GANs to synthesize compelling visual features for unseen classes. According to our evaluation with multiple image classification datasets, KG-GAN can achieve better performance than the state-of-the-art baselines.

![GAZLfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/GAZLfig2.png)

----

