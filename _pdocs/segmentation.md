---
title: "Semantic segmentation"
permalink: /pdocs/segmentation/
excerpt: "segmentation"
sitemap: true

sidebar:
  nav: "docs"

---
##  Deep learning for interferogram segmentation using [FCN](#fcn) and [ICnet](#icnet)
## FCN
### Implementation   
- Model : VGG16/resnet18 + fcn
    - VGG16 FCN layer : 
        ```python  
        class fcn32(nn.Module):
            def __init__(self):
                super().__init__()
                self.features_map=VGG16(num_classes=2)
                self.conv=nn.Sequential(nn.Conv2d(512,4096,7),
                                nn.ReLU(inplace=True),
                                nn.Dropout(),
                                nn.Conv2d(4096,4096,1),
                                nn.ReLU(inplace=True),
                                nn.Dropout()
                                )
                self.score_fr=nn.Conv2d(4096,21,1) 
                self.upscore=nn.ConvTranspose2d(21,2,64,32)
            def forward(self,x):
                x_size=x.size()
                pool=self.conv(self.features_map(x))
                score_fr=self.score_fr(pool)
                upscore=self.upscore(score_fr)
                return upscore[:,:,16:(16+x_size[2]),16:(16+x_size[3])]
        ```
    - Resnet FCN layer : 
        ```python  
        class fcn(nn.Module):
            def __init__(self):
                super().__init__()
                self.backnone = resnet18()
                self.conv1 = nn.Conv2d(512,64,3,1,1)
                self.conv2 = nn.Conv2d(64,2,1,1)
                self.convtrans1 = nn.ConvTranspose2d(2,2, 16, 8, 4)
                self.convtrans1.weight.data = bilinear_kernel(2,2, 16)
    
        def forward(self, x):
            x = self.backbone(x)
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.convtrans1(x)
            return x
        ```
- loss :  Cross entropy Loss and asymmertric  loss
- optimizer : Adam (lr = 0.001)
- environment : pytorch 1.6

### Input data
- 3072 x 512 x 1 interferogram crop fom 3072 x 3072 image randomly
- Data labeled using opencv selectROI 
  - to make it simple, use bbox to label data
- Then turn bbox data into binary mask (sample and background)

### Result
```
In this project I assume asymmertric loss would perfomed 
better since sp and bg pixel considerable disparity 
```

- Use IoU and pixel accuracy to scored the performence  
- After training, score for testing image can be up to : 
  -  ASL Loss : 
     - Mean IoU : 0.764
  - Cross entropy loss :
     - Mean IoU : 0.801
     - Pixel accuracy : 0.995
- Change backbone to resnet18
  - Cross entropy loss :
     - Mean IoU : 0.832

![](https://i.imgur.com/qvivuOH.png)
```
Above image shows the process of pridiction using trained 
model. the whole precell cost around 0.158 sec
```


### Reference
- vgg : https://github.com/chongwar/vgg16-pytorch
- ASL : https://github.com/Alibaba-MIIL/ASL
- fcn32 : https://github.com/sairin1202/fcn32-pytorch

___
---

## ICnet

inorder to build a [faster](#few-tips-to-improve-frame-rate) sementic segmentation program, I choose ICnet(image cascade network) to implement.

### Model
- Basic ICnet model and loss function from  https://github.com/liminn/ICNet-pytorch
- Slightly modified the input size, and parameters of different resolution for loss calculation 

### Data augumentation 
- To enhance the target region, multiple image had been average then subtract by every image 
- Use opencv clahe do the histogram equalization 
- flip image to enlarge data size

### TODO
- [ ] Change pyrimid pooling structure to RNN for better sequence image segmentation 
    - in this project, the removal of pyrimid pooling layer only influnce the time consume to converge

### Result
- input data : 3072x3072x1 (around 800 image), then downsize to 1024x1024 for training and testing.
- Data augumentation : flipud, fliplr
- MeanIOU : ~0.82
- Frame rate:
    - In HDD : 11 fps
    - In SSD : 23 fps 

![](https://github.com/yohschang/minimal-mistakes/blob/master/image/cell.gif)
![](https://i.imgur.com/orjWSl0.gif)


### Reference
- ICnet : https://arxiv.org/pdf/1704.08545.pdf

## Few tips to improve frame rate
```
Since most time cost by loading data, following are 
few ways to reduce time consume of data loading
```
- Data should be **storage in SSD** or the hard drive that runs pythn. Data in ssd can lead to 2x faster then in HDD

- **Use OPENCV2** instead of **PIL** ~~(pretty slow)~~  for image reading and simple transformation (easy to implement while loading test data which has no need to do complex transform)

- **set num_workers > 0**, since data will preloaded to RAM while GPU training/testing. However if image data were storage *in HDD, larger num_worker might take even slower than num_workers=0*

- Use ```for data , target in data loader : ``` , instead of using **"iter"** function
