3D: 9 papers, three of them are published in CVPR2020, two of them are published in IROS2020.

3D medical image: 1 paper.

Graph: 1 paper.

---

##### [1]UC-Net: Uncertainty Inspired RGB-D Saliency Detection via Conditional Variational Autoencoders

- [CVPR2020 Oral] Australian National University: Jing Zhang, et al.
- <https://arxiv.org/pdf/2004.05763.pdf> 
- <https://github.com/JingZhang617/UCNet> [Code]

In this paper, we propose the first framework (UCNet) to employ uncertainty for RGB-D saliency detection by learning from the data labeling process. Existing RGB-D saliency detection methods treat the saliency detection task as a point estimation problem, and produce a single saliency map following a deterministic learning pipeline. Inspired by the saliency data labeling process, we propose probabilistic RGB-D saliency detection network via conditional variational autoencoders to model human annotation uncertainty and generate multiple saliency maps for each input image by sampling in the latent space. With the proposed saliency consensus process, we are able to generate an accurate saliency map based on these multiple predictions. Quantitative and qualitative evaluations on six challenging benchmark datasets against 18 competing algorithms demonstrate the effectiveness of our approach in learning the distribution of saliency maps, leading to a new state-of-the-art in RGB-D saliency detection.

![UCNetfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/UCNetfig2.png)

---

##### [2] MLCVNet: Multi-Level Context VoteNet for 3D Object Detection

- [CVPR20] Nanjing University : Qian Xie, et al.
- <https://arxiv.org/pdf/2004.05679.pdf> 
- <https://github.com/NUAAXQ/MLCVNet> [Code]

In this paper, we address the 3D object detection task by capturing multi-level contextual information with the self-attention mechanism and multi-scale feature fusion. Most existing 3D object detection methods recognize objects individually, without giving any consideration on contextual information between these objects. Comparatively, we propose Multi-Level Context VoteNet (MLCVNet) to recognize 3D objects correlatively, building on the state-of-the-art VoteNet. We introduce three context modules into the voting and classifying stages of VoteNet to encode contextual information at different levels. Specifically, a Patch-to-Patch Context (PPC) module is employed to capture contextual information between the point patches, before voting for their corresponding object centroid points. Subsequently, an Object-to-Object Context (OOC) module is incorporated before the proposal and classification stage, to capture the contextual information between object candidates. Finally, a Global Scene Context (GSC) module is designed to learn the global scene context. We demonstrate these by capturing contextual information at patch, object and scene levels. Our method is an effective way to promote detection accuracy, achieving new state-of-the-art detection performance on challenging 3D object detection datasets, i.e., SUN RGBD and ScanNet.

![MLCVNetfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/MLCVNetfig3.png)

------

##### [3] Probabilistic Orientated Object Detection in Automotive Radar

- [CVPR20] Xsense.ai: Xu Dong, Pengluo Wang, et al.
- <https://arxiv.org/pdf/2004.05310.pdf> 

Autonomous radar has been an integral part of advanced driver assistance systems due to its robustness to adverse weather and various lighting conditions. Conventional automotive radars use digital signal processing (DSP) algorithms to process raw data into sparse radar pins that do not provide information regarding the size and orientation of the objects. In this paper, we propose a deep-learning based algorithm for radar object detection. The algorithm takes in radar data in its raw tensor representation and places probabilistic oriented bounding boxes around the detected objects in bird's-eye-view space. We created a new multimodal dataset with 102544 frames of raw radar and synchronized LiDAR data. To reduce human annotation effort we developed a scalable pipeline to automatically annotate ground truth using LiDAR as reference. Based on this dataset we developed a vehicle detection pipeline using raw radar data as the only input. Our best performing radar detection model achieves 77.28\% AP under oriented IoU of 0.3. To the best of our knowledge, this is the first attempt to investigate object detection with raw radar data for conventional corner automotive radars.

![POODfig5](https://github.com/Pan3D/Daily_Paper/blob/master/images/POODfig5.png)

------

##### [4] Object-oriented SLAM using Quadrics and Symmetry Properties for Indoor Environments

- [IROS20] Beihang University: Ziwei Liao, Wei Wang, et al.
- <https://www.youtube.com/watch?v=u9zRBp4TPIs&feature=youtu.be> [Object]

Aiming at the application environment of indoor mobile robots, this paper proposes a sparse object-level SLAM algorithm based on an RGB-D camera. A quadric representation is used as a landmark to compactly model objects, including their position, orientation, and occupied space. The state-of-art quadric-based SLAM algorithm faces the observability problem caused by the limited perspective under the plane trajectory of the mobile robot. To solve the problem, the proposed algorithm fuses both object detection and point cloud data to estimate the quadric parameters. It finishes the quadric initialization based on a single frame of RGB-D data, which significantly reduces the requirements for perspective changes. As objects are often observed locally, the proposed algorithm uses the symmetrical properties of indoor artificial objects to estimate the occluded parts to obtain more accurate quadric parameters. Experiments have shown that compared with the state-of-art algorithm, especially on the forward trajectory of mobile robots, the proposed algorithm significantly improves the accuracy and convergence speed of quadric reconstruction. Finally, we made available an opensource implementation to replicate the experiments.

![OSLAMfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/OSLAMfig2.png)

---

##### [5] Monocular Depth Estimation with Self-supervised Instance Adaptation

- [IROS20] University of Oxford: Robert McCraith, et al.
- <https://arxiv.org/pdf/2004.05821.pdf> 

Recent advances in self-supervised learning havedemonstrated that it is possible to learn accurate monoculardepth reconstruction from raw video data, without using any 3Dground truth for supervision. However, in robotics applications,multiple views of a scene may or may not be available, depend-ing on the actions of the robot, switching between monocularand multi-view reconstruction. To address this mixed setting,we proposed a new approach that extends any off-the-shelfself-supervised monocular depth reconstruction system to usemore than one image at test time. Our method builds on astandard prior learned to perform monocular reconstruction,but uses self-supervision at test time to further improve thereconstruction accuracy when multiple images are available.When used to update the correct components of the model, thisapproach is highly-effective. On the standard KITTI bench-mark, our self-supervised method consistently outperformsall the previous methods with an average 25% reduction inabsolute error for the three common setups (monocular, stereoand monocular+stereo), and comes very close in accuracy whencompared to the fully-supervised state-of-the-art methods.

![MDEfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/MDEfig2.png)

---

##### [6] CVPR 2019 WAD Challenge on Trajectory Prediction and 3D Perception

- Baidu: Sibo Zhang, et al.
- <https://arxiv.org/pdf/2004.05966.pdf> 
- <http://wad.ai/2019/challenge.html> [Workshop]

This paper reviews the CVPR 2019 challenge on Autonomous Driving. Baidu's Robotics and Autonomous Driving Lab (RAL) providing 150 minutes labeled Trajectory and 3D Perception dataset including about 80k lidar point cloud and 1000km trajectories for urban traffic. The challenge has two tasks in (1) Trajectory Prediction and (2) 3D Lidar Object Detection. There are more than 200 teams submitted results on Leaderboard and more than 1000 participants attended the workshop.

![CWfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/CWfig2.png)

------

##### [7] Deep Learning for Image and Point Cloud Fusion in Autonomous Driving: A Review

- University of Waterloo: Yaodong Cui, et al.

- <https://arxiv.org/pdf/2004.05224.pdf>

Autonomous vehicles are experiencing rapid development in the past few years. However, achieving full autonomy is not a trivial task, due to the nature of the complex and dynamic driving environment. Therefore, autonomous vehicles are equipped with a suite of different sensors to ensure robust, accurate environmental perception. In particular, camera-LiDAR fusion is becoming an emerging research theme. However, so far there is no critical review that focuses on deep-learning-based camera-LiDAR fusion methods. To bridge this gap and motivate future research, this paper devotes to review recent deep-learning-based data fusion approaches that leverage both image and point cloud. This review gives a brief overview of deep learning on image and point cloud data processing. Followed by in-depth reviews of camera-LiDAR fusion methods in depth completion, object detection, semantic segmentation and tracking, which are organized based on their respective fusion levels. Furthermore, we compare these methods on publicly available datasets. Finally, we identified gaps and over-looked challenges between current academic researches and real-world applications. Based on these observations, we provide our insights and point out promising research directions.

![DLIPCFfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/DLIPCFfig2.png)

------

##### [8] Multi-View Matching (MVM): Facilitating Multi-Person 3D Pose Estimation Learning with Action-Frozen People Video

- University of Southern California: Yeji Shen, C.-C. Jay Kuo.
- <https://arxiv.org/pdf/2004.05275.pdf> 

To tackle the challeging problem of multi-person 3D pose estimation from a single image, we propose a multi-view matching (MVM) method in this work. The MVM method generates reliable 3D human poses from a large-scale video dataset, called the Mannequin dataset, that contains action-frozen people immitating mannequins. With a large amount of in-the-wild video data labeled by 3D supervisions automatically generated by MVM, we are able to train a neural network that takes a single image as the input for multi-person 3D pose estimation. The core technology of MVM lies in effective alignment of 2D poses obtained from multiple views of a static scene that has a strong geometric constraint. Our objective is to maximize mutual consistency of 2D poses estimated in multiple frames, where geometric constraints as well as appearance similarities are taken into account simultaneously. To demonstrate the effectiveness of 3D supervisions provided by the MVM method, we conduct experiments on the 3DPW and the MSCOCO datasets and show that our proposed solution offers the state-of-the-art performance.

![MVMfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/MVMfig2.png)

------

##### [9] Toward Hierarchical Self-Supervised Monocular Absolute Depth Estimation for Autonomous Driving Applications

- Tongji University: Feng Xue, Guirong Zhuo, et al.
- <https://arxiv.org/pdf/2004.05560.pdf> 

In recent years, self-supervised methods for monocular depth estimation has rapidly become an significant branch of depth estimation task, especially for autonomous driving applications. Despite the high overall precision achieved, current methods still suffer from a) imprecise object-level depth inference and b) uncertain scale factor. The former problem would cause texture copy or provide inaccurate object boundary, and the latter would require current methods to have an additional sensor like LiDAR to provide depth groundtruth or stereo camera as additional training inputs, which makes them difficult to implement. In this work, we propose to address these two problems together by introducing DNet. Our contributions are twofold: a) a novel dense connected prediction (DCP) layer is proposed to provide better object-level depth estimation and b) specifically for autonomous driving scenarios, dense geometrical constrains (DGC) is introduced so that precise scale factor can be recovered without additional cost for autonomous vehicles. Extensive experiments have been conducted and, both DCP layer and DGC module are proved to be effectively solving the aforementioned problems respectively. Thanks to DCP layer, object boundary can now be better distinguished in the depth map and the depth is more continues on object level. It is also demonstrated that the performance of using DGC to perform scale recovery is comparable to that using ground-truth information, when the camera height is given and the ground point takes up more than 1.03% of the pixels. Code will be publicly available once the paper is accepted.

![THfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/THfig2.png)

------

##### [10] Attend and Decode: 4D fMRI Task State Decoding Using Attention Models

- Lawrence Livermore National Lab: Sam Nguyen, et al.
- <https://arxiv.org/pdf/2004.05234.pdf> 

Functional magnetic resonance imaging (fMRI) is a neuroimaging modality that captures the blood oxygen level in a subject's brain while the subject performs a variety of functional tasks under different conditions. Given fMRI data, the problem of inferring the task, known as task state decoding, is challenging due to the high dimensionality (hundreds of million sampling points per datum) and complex spatio-temporal blood flow patterns inherent in the data. In this work, we propose to tackle the fMRI task state decoding problem by casting it as a 4D spatio-temporal classification problem. We present a novel architecture called Brain Attend and Decode (BAnD), that uses residual convolutional neural networks for spatial feature extraction and self-attention mechanisms for temporal modeling. We achieve significant performance gain compared to previous works on a 7-task benchmark from the large-scale Human Connectome Project (HCP) dataset. We also investigate the transferability of BAnD's extracted features on unseen HCP tasks, either by freezing the spatial feature extraction layers and retraining the temporal model, or finetuning the entire model. The pre-trained features from BAnD are useful on similar tasks while finetuning them yields competitive results on unseen tasks/conditions.

![ADfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/ADfig1.png)

------

##### [11] Principal Neighbourhood Aggregation for Graph Nets

- University of Cambridge: Gabriele Corso, Luca Cavalleri, et al.
- <https://arxiv.org/pdf/2004.05718.pdf>

Graph Neural Networks (GNNs) have been shown to be effective models for different predictive tasks on graph-structured data. Recent work on their expressive power has focused on isomorphism tasks and countable feature spaces. We extend this theoretical framework to include continuous features - which occur regularly in real-world input domains and within the hidden layers of GNNs - and we demonstrate the requirement for multiple aggregation functions in this setting. Accordingly, we propose Principal Neighbourhood Aggregation (PNA), a novel architecture combining multiple aggregators with degree-scalers (which generalize the sum aggregator). Finally, we compare the capacity of different models to capture and exploit the graph structure via a benchmark containing multiple tasks taken from classical graph theory, which demonstrates the capacity of our model.

![PNAfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/PNAfig1.png)

----

