3D: 3 papers, one of them is published in ICLR2020.

Graph: 1 paper.

翻阅公众号历史记录，查看每日 arXiv 论文更新。

3D方向3篇，其中1篇ICLR。图结构方向有1篇文章。

文末更多 CVPR2020 3D方向论文整理。

题目为机器翻译，仅供参考。

---

##### [1] Learning to Explore using Active Neural SLAM

- [ICLR20] Carnegie Mellon University: Devendra Singh Chaplot, et al.
- <https://arxiv.org/pdf/2004.05155.pdf> 
- <https://www.cs.cmu.edu/~dchaplot/projects/neural-slam.html> [Project]
- <https://github.com/devendrachaplot/Neural-SLAM> [Torch]

This work presents a modular and hierarchical approach to learn policies for exploring 3D environments, called `Active Neural SLAM'. Our approach leverages the strengths of both classical and learning-based methods, by using analytical path planners with learned SLAM module, and global and local policies. The use of learning provides flexibility with respect to input modalities (in the SLAM module), leverages structural regularities of the world (in global policies), and provides robustness to errors in state estimation (in local policies). Such use of learning within each module retains its benefits, while at the same time, hierarchical decomposition and modular training allow us to sidestep the high sample complexities associated with training end-to-end policies. Our experiments in visually and physically realistic simulated 3D environments demonstrate the effectiveness of our approach over past learning and geometry-based approaches. The proposed model can also be easily transferred to the PointGoal task and was the winning entry of the CVPR 2019 Habitat PointGoal Navigation Challenge.

![LEANfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/LEANfig2.png)

------

##### [2] 3D IoU-Net: IoU Guided 3D Object Detector for Point Clouds

- Zhejiang University: Jiale Li, et al.
- <https://arxiv.org/pdf/2004.04962.pdf>

Most existing point cloud based 3D object detectors focus on the tasks of classification and box regression. However, another bottleneck in this area is achieving an accurate detection confidence for the Non-Maximum Suppression (NMS) post-processing. In this paper, we add a 3D IoU prediction branch to the regular classification and regression branches. The predicted IoU is used as the detection confidence for NMS. In order to obtain a more accurate IoU prediction, we propose a 3D IoU-Net with IoU sensitive feature learning and an IoU alignment operation. To obtain a perspective-invariant prediction head, we propose an Attentive Corner Aggregation (ACA) module by aggregating a local point cloud feature from each perspective of eight corners and adaptively weighting the contribution of each perspective with different attentions. We propose a Corner Geometry Encoding (CGE) module for geometry information embedding. To the best of our knowledge, this is the first time geometric embedding information has been introduced in proposal feature learning. These two feature parts are then adaptively fused by a multi-layer perceptron (MLP) network as our IoU sensitive feature. The IoU alignment operation is introduced to resolve the mismatching between the bounding box regression head and IoU prediction, thereby further enhancing the accuracy of IoU prediction. The experimental results on the KITTI car detection benchmark show that 3D IoU-Net with IoU perception achieves state-of-the-art performance.

![3DIoUNetfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/3DIoUNetfig2.png)

------

##### [3] 6D Camera Relocalization in Ambiguous Scenes via Continuous Multimodal Inference

- Technical University of Munich: Mai Bui, et al.
- <https://arxiv.org/pdf/2004.04807.pdf>
- <https://multimodal3dvision.github.io/> [Project]

We present a multimodal camera relocalization framework that captures ambiguities and uncertainties with continuous mixture models defined on the manifold of camera poses. In highly ambiguous environments, which can easily arise due to symmetries and repetitive structures in the scene, computing one plausible solution (what most state-of-the-art methods currently regress) may not be sufficient. Instead we predict multiple camera pose hypotheses as well as the respective uncertainty for each prediction. Towards this aim, we use Bingham distributions, to model the orientation of the camera pose, and a multivariate Gaussian to model the position, with an end-to-end deep neural network. By incorporating a Winner-Takes-All training scheme, we finally obtain a mixture model that is well suited for explaining ambiguities in the scene, yet does not suffer from mode collapse, a common problem with mixture density networks. We introduce a new dataset specifically designed to foster camera localization research in ambiguous environments and exhaustively evaluate our method on synthetic as well as real data on both ambiguous scenes and on non-ambiguous benchmark datasets.

![6DCRfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/6DCRfig2.png)

------

##### [4] Robust Line Segments Matching via Graph Convolution Networks

- Xidian University: QuanMeng Ma, Guang Jiang, et al.
- <https://arxiv.org/pdf/2004.04993.pdf> 
- <https://github.com/mameng1/> [Code]

Line matching plays an essential role in structure from motion (SFM) and simultaneous localization and mapping (SLAM), especially in low-textured and repetitive scenes. In this paper, we present a new method of using a graph convolution network to match line segments in a pair of images, and we design a graph-based strategy of matching line segments with relaxing to an optimal transport problem. In contrast to hand-crafted line matching algorithms, our approach learns local line segment descriptor and the matching simultaneously through end-to-end training. The results show our method outperforms the state-of-the-art techniques, and especially, the recall is improved from 45.28% to 70.47% under a similar presicion.

![RLSMfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/RLSMfig2.png)

----

