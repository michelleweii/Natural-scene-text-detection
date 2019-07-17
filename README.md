# Text Detection
  a new model based on instance segmentation is designed for text detection in a document and natural image. Consisting of convolutional and recurrent neural networks, it focuses on segmenting the close text instances and detecting long text to improve the practicability in real applications. The input images are encoded by their grid locations related to the four quadrants of an object and the background. The bidirectional long short term memory (BiLSTM) networks are used to combine the left-right and up-down contexts. Only one output classification branch is designed to predict the accurate location of each pixel, namely quadrant perception. Without bounding box regression, simple post-processing is employed to find text locations naturally. The experiments are implemented on several benchmark datasets, and the results show the proposed method has excellent performances and is competitive in the existing models.

## 1 Grid encoding （网格编码）


## 2 Quadrant perception network （象限感知网络）


### 2.1 Network architecture

coming soon ~

### 2.2 Loss function

We use focal loss to solve the sample imbalance problem.

## 3 Post-processing
<img src="https://github.com/michelleweii/QPNet/blob/master/images/exp02-1.eps" width="400" alt="">


## 4 Experiments

###  4.1 Experiments on InftyCDB-2 and Harvard

Table

methods| CTPN |	QPNet(ours)	
--|--|--
Precision |	84.3 | 93.9				
Recall |	76.5 | 91.8			
F1 |80.2 | 92.9

### 4.2 Experiments on ICDAR2013 and SVT

Table 

methods| CTPN |	Jaderberg et al.	| FCRN	| FCN	| SegLink | TextBoxes++ MS | Tang et al. | PixelLink MS| QPNet(ours) 
--|--|--|--|--|--|--|--|--|--
Precision|					
Recall|					
F1 |					
