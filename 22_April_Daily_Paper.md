

##### [1] PAI-GCN: Permutable Anisotropic Graph Convolutional Networks for 3D Shape Representation Learning

- Shanghai Jiao Tong University: Zhongpai Gao, et al.
- <https://arxiv.org/pdf/2004.09995.pdf>

Demand for efficient 3D shape representation learning is increasing in many 3D computer vision applications. The recent success of convolutional neural networks (CNNs) for image analysis suggests the value of adapting insight from CNN to 3D shapes. However, unlike images that are Euclidean structured, 3D shape data are irregular since each node's neighbors are inconsistent. Various convolutional graph neural networks for 3D shapes have been developed using isotropic filters or using anisotropic filters with predefined local coordinate systems to overcome the node inconsistency on graphs. However, isotropic filters or predefined local coordinate systems limit the representation power. In this paper, we propose a permutable anisotropic convolutional operation (PAI Conv) that learns adaptive soft-permutation matrices for each node according to the geometric shape of its neighbors and performs shared anisotropic filters as CNN does. Comprehensive experiments demonstrate that our model produces significant improvement in 3D shape reconstruction compared to state-of-the art methods.Point cloud is a principal data structure adopted for 3D geometric information encoding. Unlike other conventional visual data, such as images and videos, these irregular points describe the complex shape features of 3D objects, which makes shape feature learning an essential component of point cloud analysis. To this end, a shape-oriented message passing scheme dubbed ShapeConv is proposed to focus on the representation learning of the underlying shape formed by each local neighboring point. Despite this intra shape relationship learning, ShapeConv is also designed to incorporate the contextual effects from the inter-shape relationship through capturing the long ranged dependencies between local underlying shapes. This shape-oriented operator is stacked into our hierarchical learning architecture, namely Shape Oriented Convolutional Neural Network (SOCNN), developed for point cloud analysis. Extensive experiments have been performed to evaluate its significance in the tasks of point cloud classification and part segmentation.

![PAIGCNfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/PAIGCNfig2.png)

------

##### [2] Towards Generalization of 3D Human Pose Estimation In The Wild

- University of Luxembourg: Renato Baptista, et al. 
- <https://arxiv.org/pdf/2004.09989.pdf>

In this paper, we propose 3DBodyTex.Pose, a dataset that addresses the task of 3D human pose estimation in-the-wild. Generalization to in-the-wild images remains limited due to the lack of adequate datasets. Existent ones are usually collected in indoor controlled environments where motion capture systems are used to obtain the 3D ground-truth annotations of humans. 3DBodyTex.Pose offers high quality and rich data containing 405 different real subjects in various clothing and poses, and 81k image samples with ground-truth 2D and 3D pose annotations. These images are generated from 200 viewpoints among which 70 challenging extreme viewpoints. This data was created starting from high resolution textured 3D body scans and by incorporating various realistic backgrounds. Retraining a state-of-the-art 3D pose estimation approach using data augmented with 3DBodyTex.Pose showed promising improvement in the overall performance, and a sensible decrease in the per joint position error when testing on challenging viewpoints. The 3DBodyTex.Pose is expected to offer the research community with new possibilities for generalizing 3D pose estimation from monocular in-the-wild images.

![TGfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/TGfig2.png)

------

##### [3] Fast and Robust Registration of Aerial Images and LiDAR data Based on Structrual Features and 3D Phase Correlation

- Southwest Jiaotong University: Bai Zhu, et al.
- <https://arxiv.org/ftp/arxiv/papers/2004/2004.09811.pdf>

Co-Registration of aerial imagery and Light Detection and Ranging (LiDAR) data is quilt challenging because the different imaging mechanism causes significant geometric and radiometric distortions between such data. To tackle the problem, this paper proposes an automatic registration method based on structural features and three-dimension (3D) phase correlation. In the proposed method, the LiDAR point cloud data is first transformed into the intensity map, which is used as the reference image. Then, we employ the Fast operator to extract uniformly distributed interest points in the aerial image by a partition strategy and perform a local geometric correction by using the collinearity equation to eliminate scale and rotation difference between images. Subsequently, a robust structural feature descriptor is build based on dense gradient features, and the 3D phase correlation is used to detect control points (CPs) between aerial images and LiDAR data in the frequency domain, where the image matching is accelerated by the 3D Fast Fourier Transform (FFT). Finally, the obtained CPs are employed to correct the exterior orientation elements, which is used to achieve co-registration of aerial images and LiDAR data. Experiments with two datasets of aerial images and LiDAR data show that the proposed method is much faster and more robust than state of the art methodsEgocentric gestures are the most natural form of communication for humans to interact with wearable devices such as VR/AR helmets and glasses. A major issue in such scenarios for real-world applications is that may easily become necessary to add new gestures to the system e.g., a proper VR system should allow users to customize gestures incrementally. Traditional deep learning methods require storing all previous class samples in the system and training the model again from scratch by incorporating previous samples and new samples, which costs humongous memory and significantly increases computation over time. In this work, we demonstrate a lifelong 3D convolutional framework -- c(C)la(a)ss increment(t)al net(Net)work (CatNet), which considers temporal information in videos and enables lifelong learning for egocentric gesture video recognition by learning the feature representation of an exemplar set selected from previous class samples. Importantly, we propose a two-stream CatNet, which deploys RGB and depth modalities to train two separate networks. We evaluate CatNets on a publicly available dataset -- EgoGesture dataset, and show that CatNets can learn many classes incrementally over a long period of time. Results also demonstrate that the two-stream architecture achieves the best performance on both joint training and class incremental training compared to 3 other one-stream architectures. 

![FRRfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/FRRfig1.png)

------

##### [4] Unsupervised Domain Adaptation through Inter-modal Rotation for RGB-D Object Recognition

- ACIN: Mohammad Reza Loghmani, et al.
- <https://arxiv.org/pdf/2004.10016.pdf>

Unsupervised Domain Adaptation (DA) exploits the supervision of a label-rich source dataset to make predictions on an unlabeled target dataset by aligning the two data distributions. In robotics, DA is used to take advantage of automatically generated synthetic data, that come with "free" annotation, to make effective predictions on real data. However, existing DA methods are not designed to cope with the multi-modal nature of RGB-D data, which are widely used in robotic vision. We propose a novel RGB-D DA method that reduces the synthetic-to-real domain shift by exploiting the inter-modal relation between the RGB and depth image. Our method consists of training a convolutional neural network to solve, in addition to the main recognition task, the pretext task of predicting the relative rotation between the RGB and depth image. To evaluate our method and encourage further research in this area, we define two benchmark datasets for object categorization and instance recognition. With extensive experiments, we show the benefits of leveraging the inter-modal relations for RGB-D DA.

![UDAfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/UDAfig2.png)

------

##### [5] Self-Supervised Feature Extraction for 3D Axon Segmentation

- [CVPRW 2020] MIT: Tzofi Klinghoffer, et al.
- <https://arxiv.org/pdf/2004.09629.pdf> 

Existing learning-based methods to automatically trace axons in 3D brain imagery often rely on manually annotated segmentation labels. Labeling is a labor-intensive process and is not scalable to whole-brain analysis, which is needed for improved understanding of brain function. We propose a self-supervised auxiliary task that utilizes the tube-like structure of axons to build a feature extractor from unlabeled data. The proposed auxiliary task constrains a 3D convolutional neural network (CNN) to predict the order of permuted slices in an input 3D volume. By solving this task, the 3D CNN is able to learn features without ground-truth labels that are useful for downstream segmentation with the 3D U-Net model. To the best of our knowledge, our model is the first to perform automated segmentation of axons imaged at subcellular resolution with the SHIELD technique. We demonstrate improved segmentation performance over the 3D U-Net model on both the SHIELD PVGPe dataset and the BigNeuron Project, single neuron Janelia dataset.

![SSFEfig2](https://github.com/Pan3D/Daily_Paper/blob/master/images/SSFEfig2.png)

------

##### [6] 4D Spatio-Temporal Deep Learning with 4D fMRI Data for Autism Spectrum Disorder Classification

- [MIDL 2019] Hamburg University of Technology: Marcel Bengs, Nils Gessert, et al.
- <https://arxiv.org/pdf/2004.10165.pdf> 

Autism spectrum disorder (ASD) is associated with behavioral and communication problems. Often, functional magnetic resonance imaging (fMRI) is used to detect and characterize brain changes related to the disorder. Recently, machine learning methods have been employed to reveal new patterns by trying to classify ASD from spatio-temporal fMRI images. Typically, these methods have either focused on temporal or spatial information processing. Instead, we propose a 4D spatio-temporal deep learning approach for ASD classification where we jointly learn from spatial and temporal data. We employ 4D convolutional neural networks and convolutional-recurrent models which outperform a previous approach with an F1-score of 0.71 compared to an F1-score of 0.65.

![4DSTfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/4DSTfig1.png)

------

##### [7] A Deep Learning Approach for Motion Forecasting Using 4D OCT Data

- [MIDL 2020] Hamburg University of Technology: Marcel Bengs, Nils Gessert, et al.
- <https://arxiv.org/pdf/2004.10121.pdf> 

Forecasting motion of a specific target object is a common problem for surgical interventions, e.g. for localization of a target region, guidance for surgical interventions, or motion compensation. Optical coherence tomography (OCT) is an imaging modality with a high spatial and temporal resolution. Recently, deep learning methods have shown promising performance for OCT-based motion estimation based on two volumetric images. We extend this approach and investigate whether using a time series of volumes enables motion forecasting. We propose 4D spatio-temporal deep learning for end-to-end motion forecasting and estimation using a stream of OCT volumes. We design and evaluate five different 3D and 4D deep learning methods using a tissue data set. Our best performing 4D method achieves motion forecasting with an overall average correlation coefficient of 97.41%, while also improving motion estimation performance by a factor of 2.5 compared to a previous 3D approach.

![ADLfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/ADLfig1.png)

------

##### [8] Spatio-Temporal Deep Learning Methods for Motion Estimation Using 4D OCT Image Data

- [IJCARS] Hamburg University of Technolog: Marcel Bengs, Nils Gessert, et al.
- <https://arxiv.org/pdf/2004.10114.pdf> 

Purpose. Localizing structures and estimating the motion of a specific target region are common problems for navigation during surgical interventions. Optical coherence tomography (OCT) is an imaging modality with a high spatial and temporal resolution that has been used for intraoperative imaging and also for motion estimation, for example, in the context of ophthalmic surgery or cochleostomy. Recently, motion estimation between a template and a moving OCT image has been studied with deep learning methods to overcome the shortcomings of conventional, feature-based methods.
Methods. We investigate whether using a temporal stream of OCT image volumes can improve deep learning-based motion estimation performance. For this purpose, we design and evaluate several 3D and 4D deep learning methods and we propose a new deep learning approach. Also, we propose a temporal regularization strategy at the model output.
Results. Using a tissue dataset without additional markers, our deep learning methods using 4D data outperform previous approaches. The best performing 4D architecture achieves an correlation coefficient (aCC) of 98.58% compared to 85.0% of a previous 3D deep learning method. Also, our temporal regularization strategy at the output further improves 4D model performance to an aCC of 99.06%. In particular, our 4D method works well for larger motion and is robust towards image rotations and motion distortions.
Conclusions. We propose 4D spatio-temporal deep learning for OCT-based motion estimation. On a tissue dataset, we find that using 4D information for the model input improves performance while maintaining reasonable inference times. Our regularization strategy demonstrates that additional temporal information is also beneficial at the model output.

![STDfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/STDfig1.png)

------

##### [9] Deep variational network for rapid 4D flow MRI reconstruction

- ETH Zurich: Valery Vishnevskiy, Jonas Walheim, et al.
- <https://arxiv.org/pdf/2004.09610.pdf> 

Phase-contrast magnetic resonance imaging (MRI) provides time-resolved quantification of blood flow dynamics that can aid clinical diagnosis. Long in vivo scan times due to repeated three-dimensional (3D) volume sampling over cardiac phases and breathing cycles necessitate accelerated imaging techniques that leverage data correlations. Standard compressed sensing reconstruction methods require tuning of hyperparameters and are computationally expensive, which diminishes the potential reduction of examination times. We propose an efficient model-based deep neural reconstruction network and evaluate its performance on clinical aortic flow data. The network is shown to reconstruct undersampled 4D flow MRI data in under a minute on standard consumer hardware. Remarkably, the relatively low amounts of tunable parameters allowed the network to be trained on images from 11 reference scans while generalizing well to retrospective and prospective undersampled data for various acceleration factors and anatomies.

![DVNfig1](https://github.com/Pan3D/Daily_Paper/blob/master/images/DVNfig1.png)

------

##### [10] Facial Action Unit Intensity Estimation via Semantic Correspondence Learning with Dynamic Graph Convolution

- [AAAI20] The University of Hong Kong: Yingruo Fan, et al.
- <https://arxiv.org/pdf/2004.09681.pdf> 

The intensity estimation of facial action units (AUs) is challenging due to subtle changes in the person's facial appearance. Previous approaches mainly rely on probabilistic models or predefined rules for modeling co-occurrence relationships among AUs, leading to limited generalization. In contrast, we present a new learning framework that automatically learns the latent relationships of AUs via establishing semantic correspondences between feature maps. In the heatmap regression-based network, feature maps preserve rich semantic information associated with AU intensities and locations. Moreover, the AU co-occurring pattern can be reflected by activating a set of feature channels, where each channel encodes a specific visual pattern of AU. This motivates us to model the correlation among feature channels, which implicitly represents the co-occurrence relationship of AU intensity levels. Specifically, we introduce a semantic correspondence convolution (SCC) module to dynamically compute the correspondences from deep and low resolution feature maps, and thus enhancing the discriminability of features. The experimental results demonstrate the effectiveness and the superior performance of our method on two benchmark datasets.

![FAUfig3](https://github.com/Pan3D/Daily_Paper/blob/master/images/FAUfig3.png)

---

