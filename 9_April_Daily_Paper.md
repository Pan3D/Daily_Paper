

3D: 4 papers，two of them are published in CVPR2020.

Graph: 1 papers, published in CVPR2020.



---

##### [1] Learning 3D Semantic Scene Graphs from 3D Indoor Reconstructions

- [CVPR20] Technische Universitat Munchen: Johanna Wald, Helisa Dhamo, et al.
- <https://arxiv.org/pdf/2004.03967.pdf> 
- <https://www.youtube.com/watch?v=8D3HjYf6cYw&feature=youtu.be> [Video]

Scene understanding has been of high interest in computer vision. It encompasses not only identifying objects in a scene, but also their relationships within the given context. With this goal, a recent line of works tackles 3D semantic segmentation and scene layout prediction. In our work we focus on scene graphs, a data structure that organizes the entities of a scene in a graph, where objects are nodes and their relationships modeled as edges. We leverage inference on scene graphs as a way to carry out 3D scene understanding, mapping objects and their relationships. In particular, we propose a learned method that regresses a scene graph from the point cloud of a scene. Our novel architecture is based on PointNet and Graph Convolutional Networks (GCN). In addition, we introduce 3DSSG, a semi-automatically generated dataset, that contains semantically rich scene graphs of 3D scenes. We show the application of our method in a domain-agnostic retrieval task, where graphs serve as an intermediate representation for 3D-3D and 2D-3D matching.

![L3DSSGfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/L3DSSGfig2.png)

------

##### [2] Weakly Supervised Semantic Point Cloud Segmentation:Towards 10X Fewer Labels

- [CVPR20] National University of Singapore: Xun Xu, Gim Hee Lee.
- <https://arxiv.org/pdf/2004.04091.pdf> 
- <https://github.com/alex-xun-xu/WeakSupPointCloudSeg> [Code]

Point cloud analysis has received much attention recently; and segmentation is one of the most important tasks. The success of existing approaches is attributed to deep network design and large amount of labelled training data, where the latter is assumed to be always available. However, obtaining 3d point cloud segmentation labels is often very costly in practice. In this work, we propose a weakly supervised point cloud segmentation approach which requires only a tiny fraction of points to be labelled in the training stage. This is made possible by learning gradient approximation and exploitation of additional spatial and color smoothness constraints. Experiments are done on three public datasets with different degrees of weak supervision. In particular, our proposed method can produce results that are close to and sometimes even better than its fully supervised counterpart with 10× fewer labels.

![WSSfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/WSSfig2.png)

------

##### [3] Exemplar Fine-Tuning for 3D Human Pose Fitting Towards In-the-Wild 3D Human Pose Estimation

- Facebool: Hanbyul Joo, et al.
- <https://arxiv.org/pdf/2004.03686.pdf> 

We propose a method for building large collections of human poses with full 3D annotations captured `in the wild', for which specialized capture equipment cannot be used. We start with a dataset with 2D keypoint annotations such as COCO and MPII and generates corresponding 3D poses. This is done via Exemplar Fine-Tuning (EFT), a new method to fit a 3D parametric model to 2D keypoints. EFT is accurate and can exploit a data-driven pose prior to resolve the depth reconstruction ambiguity that comes from using only 2D observations as input. We use EFT to augment these large in-the-wild datasets with plausible and accurate 3D pose annotations. We then use this data to strongly supervise a 3D pose regression network, achieving state-of-the-art results in standard benchmarks, including the ones collected outdoor. This network also achieves unprecedented 3D pose estimation quality on extremely challenging Internet videos.

![EFTfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/EFTfig2.png)

------

##### [4] Multi-Person Absolute 3D Human Pose Estimation with Weak Depth Supervision

- Eotvos Lorand University: Marton Veges, et al.
- <https://arxiv.org/pdf/2004.03989.pdf> 
- <https://github.com/vegesm/wdspose> [Torch] 

In 3D human pose estimation one of the biggest problems is the lack of large, diverse datasets. This is especially true for multi-person 3D pose estimation, where, to our knowledge, there are only machine generated annotations available for training. To mitigate this issue, we introduce a network that can be trained with additional RGB-D images in a weakly supervised fashion. Due to the existence of cheap sensors, videos with depth maps are widely available, and our method can exploit a large, unannotated dataset. Our algorithm is a monocular, multi-person, absolute pose estimator. We evaluate the algorithm on several benchmarks, showing a consistent improvement in error rates. Also, our model achieves state-of-the-art results on the MuPoTS-3D dataset by a considerable margin.

![MPAfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/MPAfig1.png)

------

##### [5] Semantic Image Manipulation Using Scene Graphs

- [CVPR20] Technische Universitat Munchen: Helisa Dhamo, Azade Farshad, et al.
- <https://arxiv.org/pdf/2004.03677.pdf> 
- <https://he-dhamo.github.io/SIMSG/> [Project]

Image manipulation can be considered a special case of image generation where the image to be produced is a modification of an existing image. Image generation and manipulation have been, for the most part, tasks that operate on raw pixels. However, the remarkable progress in learning rich image and object representations has opened the way for tasks such as text-to-image or layout-to-image generation that are mainly driven by semantics. In our work, we address the novel problem of image manipulation from scene graphs, in which a user can edit images by merely applying changes in the nodes or edges of a semantic graph that is generated from the image. Our goal is to encode image information in a given constellation and from there on generate new constellations, such as replacing objects or even changing relationships between objects, while respecting the semantics and style from the original image. We introduce a spatio-semantic scene graph network that does not require direct supervision for constellation changes or image edits. This makes it possible to train the system from existing real-world datasets with no additional annotation effort.

![SIMfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/SIMfig2.png)

----

