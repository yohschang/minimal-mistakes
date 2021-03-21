---
title: "QPI project overview"
permalink: /pdocs/qpioverview/
excerpt: "QPI project overview"
sitemap: true

sidebar:
  nav: "docs"

---


This part describe my research project which I use quantitative phase image method to record the motion of red blood cell flow in side microfludic device

```!
Here I briefly shows the process and the result, check out the sidebar on the left for detail and the links to source code 
```

## Optical setup
![](https://i.imgur.com/2KVijgN.png)

## [Segmentation](https://yohschang.github.io/minimal-mistakes/pdocs/segmentation/#icnet)
- Using ICnet to detect to pisition of red blood cell for each frame
- Track the bounding box and seperate each cell

![](https://i.imgur.com/NkAdAmS.gif)


## Phase retrival, Refocus and Realign
- In QPI we use **interferogram** to store the phase delay cause by different sample. Therefore, an algorithm is needed to retrive the quantitative information

- After retrival, an numerical refocus algorithm is using to calibrate the focus distance change during cell motion

- Then I align all frames belong to each cell using center of mass, also unified the size and orientation of every frames


## Angle calculation
- In order to do the three dimentional reconstruction, the representation angle of each frame should be estimate
- With uniform distribution, RBC can be seen as **biolens** and estimate its angle using **Zernike polynomials** which are used to describe wavefront aberrations


## 3D reconstruction
- We use Fourier diffraction theory to do the reconstruction by mapping the frequency domain of each frame to 3d fourier space
- TVmin and constraint are add to the result to remove artifact 
 
 ![](https://i.imgur.com/7IaPPoS.jpg)![](https://i.imgur.com/B7WTnLo.png)
