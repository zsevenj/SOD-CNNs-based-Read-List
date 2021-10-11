<div align = 'center'><h1><b>SOD CNNs-based Read List</div>

Sort out some saliency methods (**2D RGB, 2D RGB-D, 360 RGB, Video SOD**) and summarize (**Code and Paper**). 

## Keywords

**`obd.`** : Object Detection    **`sod.`** : SOD    **`seg.`** : segmentation    **`depe.`** : Depth Estimation    **`rgbd.`** : RGB-D     **`360. `** : 360° image    **`suv.`** : survey

## Papers

### 2015

| NO.  |  Keyword   | Title                                                        |                          Paper                           |                         Code                         |
| :--: | :--------: | :------------------------------------------------------------ | :------------------------------------------------------: | :--------------------------------------------------: |
|  01  | **`suv.`** | What is a Salient Object? A Dataset and a Baseline Model for Salient Object Detection | [IEEE TIP](http://ieeexplore.ieee.org/document/6990522/) |                          -                           |
|  02  | **`suv.`** | Salient Object Detection: A Benchmark                        | [IEEE TIP](http://ieeexplore.ieee.org/document/7293665/) | [C++ & Matlab]( http://mmcheng.net/salobjbenchmark/) |

### 2017

| NO.  |  Keywords  | Title                                                        |                        Paper                         |                             Code                             |
| :--: | :--------: | :----------------------------------------------------------- | :--------------------------------------------------: | :----------------------------------------------------------: |
|  01  | **`obd.`** | Feature Pyramid Networks for Object Detection                | [CVPR](https://ieeexplore.ieee.org/document/8099589) |                              -                               |
|  02  | **`seg.`** | DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs | [IEEE](https://ieeexplore.ieee.org/document/7913730) | [Tensorflow](http://liangchiehchen.com/projects/DeepLab.html) |
|  03  |     -      | Look around you: Saliency maps for omnidirectional images in VR applications | [IEEE](http://ieeexplore.ieee.org/document/7965634/) |                              -                               |

### 2018

| NO.  |     Keywords      | Title                                        |                            Paper                             |                             Code                             |
| :--: | :---------------: | :-------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  01  | **`obd.` `360.`** | Object Detection in Equirectangular Panorama |           [CVPR](https://arxiv.org/abs/1805.08009)           |                              -                               |
|  02  |    **`360.`**     | Saliency Detection in 360° Videos            | [ECCV](http://link.springer.com/10.1007/978-3-030-01234-2_30) | [PyTorch](https://github.com/xuyanyu-shh/Saliency-detection-in-360-video) |

### 2019

| NO.  |  Keywords  | Title                                                        |                  Paper                  |                        Code                         |
| :--: | :--------: | :----------------------------------------------------------- | :-------------------------------------: | :-------------------------------------------------: |
|  01  | **`sod.`** | A Simple Pooling-Based Design for Real-Time Salient Object Detection | [CVPR](http://arxiv.org/abs/1904.09569) |  [PyTorch](https://github.com/backseason/PoolNet)   |
|  02  | **`sod.`** | Cascaded Partial Decoder for Fast and Accurate Salient Object Detection | [CVPR](http://arxiv.org/abs/1904.08739) |      [PyTorch](https://github.com/wuzhe71/CPD)      |
|  03  | **`360.`** | HorizonNet: Learning Room Layout with 1D Representation and Pano Stretch Data Augmentation | [CVPR](http://arxiv.org/abs/1901.03861) | [PyTorch](https://github.com/sunset1995/HorizonNet) |

### 2020

| NO.  |      Keywords      | Title                                                        |                          Paper                           |                           Code                           |
| :--: | :----------------: | :------------------------------------------------------------ | :------------------------------------------------------: | :------------------------------------------------------: |
|  01  | **`depe.` `360.`** | BiFuse: Monocular 360 Depth Estimation via Bi-Projection Fusion |  [CVPR](https://ieeexplore.ieee.org/document/9157424/)   |    [PyTorch](https://github.com/Yeh-yu-hsuan/BiFuse)     |
|  02  |     **`360.`**     | SalBiNet360: Saliency Prediction on 360° Images with Local-Global Bifurcated Deep Network | [IEEE VR](https://ieeexplore.ieee.org/document/9089519/) |                            -                             |
|  03  | **`sod.` `360.`**  | Distortion-Adaptive Salient Object Detection in 360∘ Omnidirectional Images |  [IEEE](https://ieeexplore.ieee.org/document/8926489/)   | [Caffe](http://cvteam.net/projects/JSTSP20_DDS/DDS.html) |

### 2021

| NO.  |        Keywords        | Title                                                        |                            Paper                             |                            Code                             |
| :--: | :--------------------: | :------------------------------------------------------------ | :----------------------------------------------------------: | :---------------------------------------------------------: |
|  01  | **`rgbd.`** **`sod.`** | Calibrated RGB-D Salient Object Detection                    | [CVPR](https://openaccess.thecvf.com/content/CVPR2021/html/Ji_Calibrated_RGB-D_Salient_Object_Detection_CVPR_2021_paper.html) |         [PyTorch](https://github.com/jiwei0921/DCF)         |
|  02  |      **`rgbd.`**       | Deep RGB-D Saliency Detection with Depth-Sensitive Attention and Automatic Multi-Modal Fusion |           [CVPR](http://arxiv.org/abs/2103.11832)            |                              -                              |
|  03  |      **`rgbd.`**       | Uncertainty Inspired RGB-D Saliency Detection                |     [CVPR](https://ieeexplore.ieee.org/document/9405467)     |      [PyTorch](https://github.com/JingZhang617/UCNet)       |
|  04  |   **`depe.` `360.`**   | HoHoNet: 360 Indoor Holistic Understanding with Latent Horizontal Features |           [CVPR](https://arxiv.org/abs/2011.11498)           |      [PyTorch](https://github.com/sunset1995/HoHoNet)       |
|  05  |   **`depe.` `360.`**   | UniFuse: Unidirectional Fusion for 360$^{\circ}$ Panorama Depth Estimation |           [CVPR](http://arxiv.org/abs/2102.03550)            |                           [PyTorch](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)                            |
|  06  |       **`seg.`**       | Fully Convolutional Networks for Panoptic Segmentation       |           [CVPR](https://arxiv.org/abs/2012.00720)           | [Detectron2](https://github.com/dvlab-research/PanopticFCN) |
|  07  |       **`360.`**       | 一种立体全景图像显著性检测模型                               | [激光与光电子学进展](http://www.opticsjournal.net/Articles/Abstract?aid=OJ1c8876d8937c381e) |                              -                              |

## Datasets

* 360°
  * Fixation Prediction
    * [[Salient!360](https://salient360.ls2n.fr/datasets/toolbox/)] : indoor & outdoor, 85 images. ([Details](https://hal.archives-ouvertes.fr/hal-01994923/document))
    * [[Stanford360](https://vsitzmann.github.io/vr-saliency/ )] : indoor & outdoor, 12 images. ([Details](https://ieeexplore.ieee.org/document/8269807))
  * Salient Object Detection
    * [[360-SOD](http://cvteam.net/projects/JSTSP20_DDS/DDS.html)] : indoor & outdoor, 400 training images and 100 test images. ([Details](https://ieeexplore.ieee.org/document/8926489/))
    * [[F-360iSOD](https://github.com/PanoAsh/F-360iSOD)] : The F-360iSOD is a small-scale 360◦ dataset with totally 107 panoramic images collected from Stanford360 and Salient!360 which contain 85 and 22 equirectangular images, respectively. ([Details](https://arxiv.org/abs/2001.07960))
  * (pending)
    * Matterport3D : ([Details](https://arxiv.org/abs/1709.06158))
    * Stanford2D3D : ([Details](https://arxiv.org/abs/1702.01105))
    * PanoSUNCG : ([Details](https://arxiv.org/abs/1811.05304))
    * 360D : ([Details](https://arxiv.org/abs/1807.09620))
* 2D（pending）
  * [[MSRA10K]( https://mmcheng.net/msra10k/)] : formally named as THUS10000; 195MB: images + binary masks. Pixel accurate salient object labeling for 10000 images from MSRA dataset.
  * [[THUR15K](https://mmcheng.net/code-data/)] : 15000 images.
  * [[ECSSD](https://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/dataset.html)] : 1000 images.
  * [ Judd ] : 900 images.
  * [[SED1/2](http://www.wisdom.weizmann.ac.il/~vision/Seg_Evaluation_DB/dl.html)] : 200个灰度图像以及真实标注分割.
  * [[DUT-OMRON](http://saliencydetection.net/dut-omron/#outline-container-org0e04792)] : 5168 images.
  * [[DUTS](http://saliencydetection.net/duts/)] : 10,553 training images and 5,019 test images.
  * [[HKU-IS](https://i.cs.hku.hk/~gbli/deep_saliency.html)] : 4447个具有显着对象的像素注释的图像.
  * [[PASCAL-S](https://pan.baidu.com/s/1DZcfwCYdeMW4EGawhXQyig)] : 850 images.
