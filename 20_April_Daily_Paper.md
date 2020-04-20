3D: 4 papers, one of them is published in CVPR.

Graph: 2 papers.

---

##### [1] Detailed 2D-3D Joint Representation for Human-Object Interaction

- [CVPR20] Shanghai Jiao Tong University: Yong-Lu Li, et al.
- <https://arxiv.org/pdf/2004.08154.pdf> 
- <https://github.com/DirtyHarryLYL/DJ-RN> [TF]

Human-Object Interaction (HOI) detection lies at the core of action understanding. Besides 2D information such as human/object appearance and locations, 3D pose is also usually utilized in HOI learning since its view-independence. However, rough 3D body joints just carry sparse body information and are not sufficient to understand complex interactions. Thus, we need detailed 3D body shape to go further. Meanwhile, the interacted object in 3D is also not fully studied in HOI learning. In light of these, we propose a detailed 2D-3D joint representation learning method. First, we utilize the single-view human body capture method to obtain detailed 3D body, face and hand shapes. Next, we estimate the 3D object location and size with reference to the 2D human-object spatial configuration and object category priors. Finally, a joint learning framework and cross-modal consistency tasks are proposed to learn the joint HOI representation. To better evaluate the 2D ambiguity processing capacity of models, we propose a new benchmark named Ambiguous-HOI consisting of hard ambiguous images. Extensive experiments in large-scale HOI benchmark and Ambiguous-HOI show impressive effectiveness of our method.

![DJfig5](D:\20_Spring\Paper\Paper_All\Account\4.20 Paper\DJfig5.png)

------

##### [2] IDDA: a large-scale multi-domain dataset for autonomous driving

- Politecnico di Torino: Emanuele Alberti, Antonio Tavera, et al.
- <https://arxiv.org/pdf/2004.08298.pdf> 
- <https://idda-dataset.github.io/home/> [Project]

Semantic segmentation is key in autonomous driving. Using deep visual learning architectures is not trivial in this context, because of the challenges in creating suitable large scale annotated datasets. This issue has been traditionally circumvented through the use of synthetic datasets, that have become a popular resource in this field. They have been released with the need to develop semantic segmentation algorithms able to close the visual domain shift between the training and test data. Although exacerbated by the use of artificial data, the problem is extremely relevant in this field even when training on real data. Indeed, weather conditions, viewpoint changes and variations in the city appearances can vary considerably from car to car, and even at test time for a single, specific vehicle. How to deal with domain adaptation in semantic segmentation, and how to leverage effectively several different data distributions (source domains) are important research questions in this field. To support work in this direction, this paper contributes a new large scale, synthetic dataset for semantic segmentation with more than 100 different source visual domains. The dataset has been created to explicitly address the challenges of domain shift between training and test data in various weather and view point conditions, in seven different city types. Extensive benchmark experiments assess the dataset, showcasing open challenges for the current state of the art.

![IDDAfig2](D:\20_Spring\Paper\Paper_All\Account\4.20 Paper\IDDAfig2.png)

------

##### [3] DepthNet Nano: A Highly Compact Self-Normalizing Neural Network for Monocular Depth Estimation

- University of Waterloo: Linda Wang, et al. 
- <https://arxiv.org/pdf/2004.08008.pdf> 

Depth estimation is an active area of research in the field of computer vision, and has garnered significant interest due to its rising demand in a large number of applications ranging from robotics and unmanned aerial vehicles to autonomous vehicles. A particularly challenging problem in this area is monocular depth estimation, where the goal is to infer depth from a single image. An effective strategy that has shown considerable promise in recent years for tackling this problem is the utilization of deep convolutional neural networks. Despite these successes, the memory and computational requirements of such networks have made widespread deployment in embedded scenarios very challenging. In this study, we introduce DepthNet Nano, a highly compact self normalizing network for monocular depth estimation designed using a human machine collaborative design strategy, where principled network design prototyping based on encoder-decoder design principles are coupled with machine-driven design exploration. The result is a compact deep neural network with highly customized macroarchitecture and microarchitecture designs, as well as self-normalizing characteristics, that are highly tailored for the task of embedded depth estimation. The proposed DepthNet Nano possesses a highly efficient network architecture (e.g., 24X smaller and 42X fewer MAC operations than Alhashim et al. on KITTI), while still achieving comparable performance with state-of-the-art networks on the NYU-Depth V2 and KITTI datasets. Furthermore, experiments on inference speed and energy efficiency on a Jetson AGX Xavier embedded module further illustrate the efficacy of DepthNet Nano at different resolutions and power budgets (e.g., ~14 FPS and >0.46 images/sec/watt at 384 X 1280 at a 30W power budget on KITTI).

![DNfig3](D:\20_Spring\Paper\Paper_All\Account\4.20 Paper\DNfig3.png)

------

##### [4] Learning visual policies for building 3D shape categories

- Inria: Alexander Pashevich, Igor Kalevatykh, et al.

- <https://arxiv.org/abs/2004.07950> 

Manipulation and assembly tasks require non-trivial planning of actions depending on the environment and the final goal. Previous work in this domain often assembles particular instances of objects from known sets of primitives. In contrast, we here aim to handle varying sets of primitives and to construct different objects of the same shape category. Given a single object instance of a category, e.g. an arch, and a binary shape classifier, we learn a visual policy to assemble other instances of the same category. In particular, we propose a disassembly procedure and learn a state policy that discovers new object instances and their assembly plans in state space. We then render simulated states in the observation space and learn a heatmap representation to predict alternative actions from a given input image. To validate our approach, we first demonstrate its efficiency for building object categories in state space. We then show the success of our visual policies for building arches from different primitives. Moreover, we demonstrate (i) the reactive ability of our method to re-assemble objects using additional primitives and (ii) the robust performance of our policy for unseen primitives resembling building blocks used during training. Our visual assembly policies are trained with no real images and reach up to 95% success rate when evaluated on a real robot.

![LVPfig2](D:\20_Spring\Paper\Paper_All\Account\4.20 Paper\LVPfig2.png)

------

##### [5] Structured Landmark Detection via Topology-Adapting Deep Graph Learning

- University of Rochester: Weijian Li, et al.
- <https://arxiv.org/pdf/2004.08190.pdf> 

Image landmark detection aims to automatically identify the locations of predefined fiducial points. Despite recent success in this filed, higher-ordered structural modeling to capture implicit or explicit relationships among anatomical landmarks has not been adequately exploited. In this work, we present a new topology-adapting deep graph learning approach for accurate anatomical facial and medical (e.g., hand, pelvis) landmark detection. The proposed method constructs graph signals leveraging both local image features and global shape features. The adaptive graph topology naturally explores and lands on task-specific structures which is learned end-to-end with two Graph Convolutional Networks (GCNs). Extensive experiments are conducted on three public facial image datasets (WFLW, 300W and COFW-68) as well as three real-world X-ray medical datasets (Cephalometric (public), Hand and Pelvis). Quantitative results comparing with the previous state-of-the-art approaches across all studied datasets indicating the superior performance in both robustness and accuracy. Qualitative visualizations of the learned graph topologies demonstrate a physically plausible connectivity laying behind the landmarks.

![SLDfig1](D:\20_Spring\Paper\Paper_All\Account\4.20 Paper\SLDfig1.png)

------

##### [6] MPLP++: Fast, Parallel Dual Block-Coordinate Ascent for Dense Graphical Models

- [ECCV2018] Uni. Heidelberg: Siddharth Tourani, et al.
- <https://arxiv.org/pdf/2004.08227.pdf> 

Dense, discrete Graphical Models with pairwise potentials are a powerful class of models which are employed in state-of-the-art computer vision and bio-imaging applications. This work introduces a new MAP-solver, based on the popular Dual Block-Coordinate Ascent principle. Surprisingly, by making a small change to the low-performing solver, the Max Product Linear Programming (MPLP) algorithm, we derive the new solver MPLP++ that significantly outperforms all existing solvers by a large margin, including the state-of-the-art solver Tree-Reweighted Sequential (TRWS) message-passing algorithm. Additionally, our solver is highly parallel, in contrast to TRWS, which gives a further boost in performance with the proposed GPU and multi-thread CPU implementations. We verify the superiority of our algorithm on dense problems from publicly available benchmarks, as well, as a new benchmark for 6D Object Pose estimation. We also provide an ablation study with respect to graph density.

![MPLPfig1](D:\20_Spring\Paper\Paper_All\Account\4.20 Paper\MPLPfig1.png)

----

