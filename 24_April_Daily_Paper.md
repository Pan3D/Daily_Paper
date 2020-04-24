3D: 2 papers, one is CVPR2020 Oral. Another is published in CVPR2020 Workshop.

Graph: 2 papers.

---

##### [1] Through the Looking Glass: Neural 3D Reconstruction of Transparent Shapes 

- [CVP2020 Oral] UC, San Diego: Zhengqin Li, Yu-Ying Yeh, et al.
- <https://arxiv.org/pdf/2004.10904.pdf> 
- <https://github.com/lzqsd/TransparentShapeReconstruction> [Code]

Recovering the 3D shape of transparent objects using a small number of unconstrained natural images is an ill-posed problem. Complex light paths induced by refraction and reflection have prevented both traditional and deep multiview stereo from solving this challenge. We propose a physically-based network to recover 3D shape of transparent objects using a few images acquired with a mobile phone camera, under a known but arbitrary environment map. Our novel contributions include a normal representation that enables the network to model complex light transport through local computation, a rendering layer that models refractions and reflections, a cost volume specifically designed for normal refinement of transparent shapes and a feature mapping based on predicted normals for 3D point cloud reconstruction. We render a synthetic dataset to encourage the model to learn refractive light transport across different views. Our experiments show successful recovery of high-quality 3D geometry for complex transparent shapes using as few as 5-12 natural images. Code and data are publicly released.

![TLfig3](https://github.com/Pan3D/Daily_Paper/blob/master/image2/TLfig3.png)

------

##### [2] Improved Noise and Attack Robustness for Semantic Segmentation by Using Multi-Task Training with Self-Supervised Depth Estimation

- [CVPRW 2020] Technische Universitat Braunschweig: Marvin Klingner, et al.
- <https://arxiv.org/pdf/2004.11072.pdf> 

While current approaches for neural network training often aim at improving performance, less focus is put on training methods aiming at robustness towards varying noise conditions or directed attacks by adversarial examples. In this paper, we propose to improve robustness by a multi-task training, which extends supervised semantic segmentation by a self-supervised monocular depth estimation on unlabeled videos. This additional task is only performed during training to improve the semantic segmentation model's robustness at test time under several input perturbations. Moreover, we even find that our joint training approach also improves the performance of the model on the original (supervised) semantic segmentation task. Our evaluation exhibits a particular novelty in that it allows to mutually compare the effect of input noises and adversarial attacks on the robustness of the semantic segmentation. We show the effectiveness of our method on the Cityscapes dataset, where our multi-task training approach consistently outperforms the single-task semantic segmentation baseline in terms of both robustness vs. noise and in terms of adversarial attacks, without the need for depth labels in training.

![INfig2](https://github.com/Pan3D/Daily_Paper/blob/master/image2/INfig2.png)

------

##### [3] Fast Convex Relaxations using Graph Discretizations

- University of Siegen: Jonas Geiping, et al.
- <https://arxiv.org/pdf/2004.11075.pdf>

Matching and partitioning problems are fundamentals of computer vision applications with examples in multilabel segmentation, stereo estimation and optical-flow computation. These tasks can be posed as non-convex energy minimization problems and solved near-globally optimal by recent convex lifting approaches. Yet, applying these techniques comes with a significant computational effort, reducing their feasibility in practical applications. We discuss spatial discretization of continuous partitioning problems into a graph structure, generalizing discretization onto a Cartesian grid. This setup allows us to faithfully work on super-pixel graphs constructed by SLIC or Cut-Pursuit, massively decreasing the computational effort for lifted partitioning problems compared to a Cartesian grid, while optimal energy values remain similar: The global matching is still solved near-globally optimally. We discuss this methodology in detail and show examples in multi-label segmentation by minimal partitions and stereo estimation, where we demonstrate that the proposed graph discretization technique can reduce the runtime as well as the memory consumption by up to a factor of 10 in comparison to classical pixelwise discretizations.

![FCfig2](https://github.com/Pan3D/Daily_Paper/blob/master/image2/FCfig2.png)

------

##### [4] Visual Commonsense Graphs: Reasoning about the Dynamic Context of a Still Image

- Allen Institute for Artificial Intelligence: Jae Sung Park, et al.
- http://visualcomet.xyz/ [Project] 

Even from a single frame of a still image, people can reason about the dynamic story of the image before, after, and beyond the frame. For example, given an image of a man struggling to stay afloat in water, we can reason that the man fell into the water sometime in the past, the intent of that man at the moment is to stay alive, and he will need help in the near future or else he will get washed away. We propose VisualComet, the novel framework of visual commonsense reasoning tasks to predict events that might have happened before, events that might happen next, and the intents of the people at present. To support research toward visual commonsense reasoning, we introduce the first large-scale repository of Visual Commonsense Graphs that consists of over 1.4 million textual descriptions of visual commonsense inferences carefully annotated over a diverse set of 60,000 images, each paired with short video summaries of before and after. In addition, we provide person-grounding (i.e., co-reference links) between people appearing in the image and people mentioned in the textual commonsense descriptions, allowing for tighter integration between images and text. We establish strong baseline performances on this task and demonstrate that integration between visual and textual commonsense reasoning is the key and wins over non-integrative alternatives.

![VCfig2](https://github.com/Pan3D/Daily_Paper/blob/master/image2/VCfig2.png)

----

