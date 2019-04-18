## 一、Classic detection model

### 1.Proposal or not

#### （1）One-stage

**YOLOv1——>SSD——>DSSD——>YOLOv2——>RetinaNet——>DSOD——>YOLOv3——>RefineDet——>RFBNet——>M2Det——>Consistent Optimization(11)**

#### （2）Two-stage

**RCNN——>SppNet——>Fast RCNN——>Faster RCNN——>OHEM——>R-FCN——>FPN——>DCN——>Mask RCNN——>Soft-NMS——>Cascade R-CNN——>iounet——>TrindentNet(13)**

#### （3）One-Two Combination

### 2.Improvement of detection modules

#### （1）based RPN

[R-CNN for Small Object Detection]

#### （2）based NMS

[Soft-NMS]

[Softer-NMS]

[ConvNMS]

[Pure NMS Network]

[Fitness NMS]

#### （3）based anchor

### 3.Improvement to solve problems

#### （1）small object

[FPN]

[DSSD]

[SNIP]

[SNIPPER]

[R-SSD](<https://arxiv.org/abs/1705.09587>)

[R-CNN for Small Object Detection]

[Relation Network for Object Detection]

[M2Det]

#### （2）scale variation
[image pyramid/multi-scale testing]

[feature pyramid]

[anchor box]

[M2Det]

#### （3）shelter

[Repulsion Loss]

[Occlusion-aware R-CNN]

[Soft-NMS]

[Bi-box]

[R-DAD] 

#### （4）Imbalance Of Positive&Negative

#### （5）Mobile or Light weight

## 二、Classic classification/detection backbone

### 1.deepen

**（1）resnet**

### 2.widen

**（1）Inception**

### 3.smaller

**（1）mobilenet**

**（2）shufflenet**

**（3）pelee**

### 4.feature 

**（1）DenseNet**

**（2）SeNet**

### 5.detection specific

**（1）darknet**

## 三、Detection modules

### 1.Selective Search&&RPN

### 2.ROI pooling&&ROI align

### 3.[IoU](<https://blog.csdn.net/weixin_41278720/article/details/88770034>)

### 4.NMS

### 5.[Generic metrics](<https://zhuanlan.zhihu.com/p/60218684>)

### 6.[mAP](<https://zhuanlan.zhihu.com/p/60319755>)

## 四、Reference

[1]**(YOLOv1)** J. Redmon, S. Divvala, R. Girshick, and A. Farhadi. You only look once: Unified, real-time object detection. In CVPR, 2016.

[2]**(SSD)** W. Liu, D. Anguelov, D. Erhan, C. Szegedy, S. Reed, C.-Y. Fu, and A. C. Berg. SSD: Single shot multibox detector. In ECCV, 2016.

[3]**(DSSD)** C.-Y. Fu, W. Liu, A. Ranga, A. Tyagi, and A. C. Berg. DSSD:Deconvolutional single shot detector. In arXiv,2017.

[4]**(YOLOv2)** J. Redmon and A. Farhadi. YOLO9000: Better, faster, stronger. In CVPR, 2017. 

[5]**(RetinaNet)** T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar. Focal loss for dense object detection. In ICCV, 2017. 

[6]**(DSOD)** Shen Z., Liu Z., Li J., Jiang Y., Chen Y., Xue X. DSOD: Learning deeply supervised object detectors from scratch. In ICCV, 2017

[7] **(YOLOv3)** J. Redmon and A. Farhadi. YOLOv3: An incremental im- provement. In arXiv, 2018. 

[8]**(RefineDet)** S. Zhang, L. Wen, X. Bian, Z. Lei, and S. Z. Li. Single-shot refinement neural network for object detection. In CVPR, 2018.

[9]**(RFBNet)** Songtao Liu, Di Huang⋆, and Yunhong Wang. Receptive Field Block Net for Accurate and Fast Object Detection. In ECCV ,2018.

[10]**(M2Det)** Qijie Zhao, Tao Sheng, Yongtao Wang, Zhi Tang, Ying Chen, Ling Cai and Haibin Ling. M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network. In AAAI,2019.

[11]**(Consistent Optimization)** Tao Kong,Fuchun Sun,Huaping Liu,Yuning Jiang and Jianbo Shi. Consistent Optimization for Single-Shot Object Detection. In arXiv, 2019.

[12]**(R-CNN)** R. Girshick, J. Donahue, T. Darrell, and J. Malik. Rich feature hierarchies for accurate object detection and semantic segmentation. In CVPR, 2014. 

[13]**(SppNet)** K.He,X.Zhang,S.Ren,andJ.Sun.Spatialpyramidpooling in deep convolutional networks for visual recognition. In ECCV,2014.

[14]**(Fast R-CNN)** R. Girshick. Fast R-CNN. In ICCV, 2015.

[15]**(Faster R-CNN)** S. Ren, K. He, R. Girshick, and J. Sun. Faster R-CNN: Towards real-time object detection with region proposal net-works. In NIPS, 2015.

[16]**(OHEM)** Abhinav Shrivastava,Abhinav Gupta and Ross Girshick. Training Region-based Object Detectors with Online Hard Example Mining.In CVPR, 2016.

[17] **(R-FCN)** J.Dai,Y.Li,K.He,andJ.Sun.R-FCN:Object detection via region-based fully convolutional networks. In NIPS, 2016. 

[18]**(FPN)** T.-Y. Lin, P. Dolla ́r, R. B. Girshick, K. He, B. Hariharan, and S. J. Belongie. Feature pyramid networks for object detection. In CVPR, 2017. 

[19]**(DCN)** J. Dai, H. Qi, Y. Xiong, Y. Li, G. Zhang, H. Hu, and Y. Wei. Deformable convolutional networks. In ICCV, 2017. 

[20]**(Mask R-CNN)** K.He,G.Gkioxari,P.Dolla ́r,and R.Girshick.MaskR-CNN. In ICCV, 2017.

[21]**(Soft- NMS)** N. Bodla, B. Singh, R. Chellappa, and L. S. Davis. Soft-NMS-improving object detection with one line of code. In ICCV, 2017. 

[22]**(Cascade R-CNN)** Z. Cai and N. Vasconcelos. Cascade R-CNN: Delving into high quality object detection. In CVPR, 2018. 

[23]**(IoUNet)** Borui Jiang,Ruixuan Luo,Jiayuan Mao,Tete Xiao,and Yuning Jiang.Acquisition of Localization Confidence for Accurate Object Detection.In ECCV 2018.

[24]**(TridentNet)** Yanghao Li,Yuntao Chen,Naiyan Wang,Zhaoxiang Zhang.Scale-Aware Trident Networks for Object Detection.In arXiv,2019.

[25]**(ResNet)** K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.

[26]**(DenseNet)** Gao Huang,Zhuang Liu,Laurens van der Maaten.Densely Connected Convolutional Networks. In CVPR,2017.
