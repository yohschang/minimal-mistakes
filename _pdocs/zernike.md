---
title: "Zernike fitting"
permalink: /pdocs/zernike/
excerpt: "Zernike fitting"
sitemap: true

sidebar:
  nav: "docs"

---

Zernike polynomial play important roles in various optics branches
![](https://i.imgur.com/A1H7zGi.png)
Where C is the coefficient for each order Z. Therefore, we can get C for every W we measure by intergration. 
```python
Defocus = np.sqrt(3)*(2*r**2-1)  
HPA = np.sqrt(6)*r**2*np.cos(2*u)  # horizontal primary astigmatism
PSA = np.sqrt(5)*(6*r**4-6*r**2 + 1)  # primary spherical abberation
HSA = np.sqrt(10)*(4*r**2-3)*r**2*np.cos(2*u)  #  horizontal secondary astigmatism

def fitting(Z,n):
    fitlist = []
    l = len(Z)
    x2 = np.linspace(-1, 1, l)
    y2 = np.linspace(-1, 1, l)
    [X2,Y2] = np.meshgrid(x2,y2)
    r = np.sqrt(X2**2 + Y2**2)  
    u = np.arctan2(Y2, X2)   
    for q in range(n):
            ZF = "Formula of different order Z"
            Z_tot = Z * ZF
            Z_tot[r > 1] = 0
            C = sum(sum(Z_tot))*2*2/l/l/np.pi   
            fitlist.append(np.around(a,3))
    for i in range(l):
        for j in range(l):
            if x2[i]**2+y2[j]**2>1:
                Z_new[i][j]=0

    return fitlist , Z_new
```

## Angle calculation

In order to do the three dimentional reconstruction, the representation angle of each frame should be estimate
- With uniform distribution, RBC can be seen as **biolens** and estimate its angle using **Zernike polynomials** which are used to describe wavefront aberrations
- Here I use differnt order to discribe different angle
    - Horizontal secondary astigmatism for 90 degree estimation
    - Defocus + Primary spherical abberation for 0 degree estimation
    - Defocus + Horizontal for other frame between each section of 0~90

![](https://i.imgur.com/powLKGx.png)

- For 90 degree and 0 degree, we find out local maximum of C of different Z to define 
- For other angle
    - From simulation, we find out the Coefficient of the combination of Defocus and Horizontal primary astigmatism is related to cos square curve by rotating ideal discocyte shape
    - The formula to transform C value to angle would be  
    ```
    np.abs(np.rad2deg(np.arccos(np.sqrt(C)))
    ```
    ![](https://i.imgur.com/iBAx3cb.png)
- After anle transformation of each 0~90 section, combine them and caliberate to 0~360 cycle

- To accerlerate, Multiprocessing calculation is using
```python
def run(self):
    fitting_data , phimap_stack = self.read_rbc(self.isdash)
    pool = mp.Pool(6)
    zc_list = [pool.starmap(self.zernikeFitting,fitting_data)]
    pool.close()
    pool.join()
```
