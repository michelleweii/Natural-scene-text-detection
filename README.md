# Text Detection
## Abstract
  a new model based on instance segmentation is designed for text detection in a document and natural image. Consisting of convolutional and recurrent neural networks, it focuses on segmenting the close text instances and detecting long text to improve the practicability in real applications. The input images are encoded by their grid locations related to the four quadrants of an object and the background. The bidirectional long short term memory (BiLSTM) networks are used to combine the left-right and up-down contexts. Only one output classification branch is designed to predict the accurate location of each pixel, namely quadrant perception. Without bounding box regression, simple post-processing is employed to find text locations naturally. The experiments are implemented on several benchmark datasets, and the results show the proposed method has excellent performances and is competitive in the existing models.

## 1 Grid encoding （网格编码）

<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/网格编码.PNG" width="1000" alt="grid encoding">

## 2 Quadrant perception network （象限感知网络）


### 2.1 Network architecture

coming soon ~

### 2.2 Loss function

We use focal loss to solve the sample imbalance problem.

## 3 Post-processing
<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/后处理.PNG" width="1000" alt="">

## 4 Experiments

###  4.1 Experiments on InftyCDB-2 and Harvard
### 4.1.1 infty-2训练过程

<img src="<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/infty-2训练过程.PNG" width="1000" alt="">
                                                                                                                       
### 4.1.2 infty-2 检测结果

<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/intfy-2检测结果.PNG" width="1000" alt="">

### 4.1.3 Harvard 检测结果

<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/harvard检测结果.PNG" width="1000" alt="">

### 4.1.4 The Results Compared with CTPN

<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/与CTPN对比.PNG" width="1000" alt="">

Table 1 Comparison of experimental results on Infty-2(%).

methods| CTPN |	QPNet(ours)	
--|--|--
Precision |	84.3 | 93.9				
Recall |	76.5 | 91.8			
F1 |80.2 | 92.9

### 4.2 Experiments on ICDAR2013 and SVT
### 4.2.1 ICDAR2013和SVT的检测结果
<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/ICDAR2013_svt结果图.PNG" width="1000" alt="">

### 4.2.2 More ICDAR2013 

<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/icdar13更多预测结果/img_32.jpg" width="500" alt="">
<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/icdar13更多预测结果/img_68.jpg" width="500" alt="">
<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/icdar13更多预测结果/img_99.jpg" width="500" alt="">

Table 2 Comparison of experimental results on the ICDAR2013 dataset(%).

methods| CTPN |	Jaderberg et al.	| FCRN	| FCN	| SegLink | TextBoxes++ MS | Tang et al. | PixelLink MS| QPNet(ours) 
--|--|--|--|--|--|--|--|--|--
Precision|93.0|88.5|92.0|88.0|87.7|91.0|91.9|88.6|91.7|				
Recall|83.0|67.8|75.5|78.0|83.0|84.0|87.1|87.5|82.1|
F1 |87.7|	76.8|83.0|83.0|85.3|88.0|89.5|88.1|86.6|

### 4.3 Other

#### 4.3.1 自然场景数学公式检测
The result of CTPN (green) vs. The result of QPNet(blue)
<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/自然场景数学公式检测.png" width="1000" alt="">

### 4.3.2 不同尺度运行时间比较
<img src="https://github.com/michelleweii/QPNet/blob/master/QPNet_images/运行时间.PNG" width="1000" alt="">

