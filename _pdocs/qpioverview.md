---
title: "QPI project overview"
permalink: /pdocs/qpioverview/
excerpt: "QPI project overview"
sitemap: true

sidebar:
  nav: "docs"

---


This part describe my research project which I use quantitative phase image method to record the motion of red blood cell flow in side microfludic device

```
Here I briefly shows the process and the result, 
check out the sidebar on the left for detail and the links to source code
```

## Optical setup
![](https://i.imgur.com/2KVijgN.png)


## [Segmentation](https://yohschang.github.io/minimal-mistakes/pdocs/segmentation/#icnet)
- Using ICnet to detect to pisition of red blood cell for each frame
- Track the bounding box and seperate each cell

     ![](https://i.imgur.com/NkAdAmS.gif)


## [Phase retrieval](https://yohschang.github.io/minimal-mistakes/pdocs/qpiprocess/#phase-retrieval), [Refocus](https://yohschang.github.io/minimal-mistakes/pdocs/qpiprocess/#refocus) and [Realign](https://yohschang.github.io/minimal-mistakes/pdocs/qpiprocess/#realign)
- In QPI we use **interferogram** to store the phase delay cause by different sample. Therefore, an algorithm is needed to retrieve the quantitative information

![](https://i.imgur.com/Jl3BfmY.png)

- After retrieval, an numerical refocus algorithm is using to calibrate the focus distance change during cell motion

![](https://i.imgur.com/AaodwHP.png)

- Then I align all frames belong to each cell using center of mass, also unified the size and orientation of every frames

   ![](https://i.imgur.com/j04thjr.gif)


## Angle calculation
- In order to do the three dimentional reconstruction, the representation angle of each frame should be estimate
- With uniform distribution, RBC can be seen as **biolens** and estimate its angle using **Zernike polynomials** which are used to describe wavefront aberrations
- Here I use differnt order to discribe different angle
    - Horizontal secondary astigmatism for 90 degree estimation
    - Defocus + Primary spherical abberation for 0 degree estimation
    - Defocus + Horizontal for other frame between each section of 0~90

![](https://i.imgur.com/powLKGx.png)


## 3D reconstruction
- We use Fourier diffraction theory to do the reconstruction by mapping the frequency domain of each frame to 3d fourier space
- TVmin and constraint are add to the result to remove artifact 

![](https://i.imgur.com/EB5H3kP.png)

