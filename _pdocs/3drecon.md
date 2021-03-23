---
title: "3D reconstruction"
permalink: /pdocs/3drecon/
excerpt: "3D reconstruction"
sitemap: true

sidebar:
  nav: "docs"

---

For the last step of cell prepocessing, we use fourier diffraction theorem to do the three dimentional reconstruction
![](https://i.imgur.com/sMl0RU5.png)

## Basic concept
**For this part we use CUDA to accelerate whole process**
- For each 3D object its frequency domain is also 3D, hence, each projection would be a slice form this domain
- However, cell is too small that it will cause diffraction while projection, so further estimation above should be considered before mapping each 2d slice back to 3d slice
- After estimation scatter field flatten 2d slice will become half spherical, with size related to the numerical aperture of objective lens
- By 3d retotion(rotaion matrix), each projection can be map to origin 3d frquency space 
```c    
    #include <cupy/complex.cuh>
    extern "C" {
        __global__ void fillEwaldSphere(float2* u_sp, float2* F, int* C, 
                                        float fx0, float fy0, float fz0, float angX, float angY, float fm0,
                                        float df, int sizeX, int sizeY, int sizeZ) {
            bool Fz_err;
            float Fx, Fy, Fz, fx, fy, fz, tmp_fz;
            int ii, jj, Nx, Ny, Nz, Nx2, Ny2, Nz2, idx;
            float fm02 = fm0 * fm0;
            int i = blockDim.x * blockIdx.x + threadIdx.x;
            int j = blockDim.y * blockIdx.y + threadIdx.y;

            if (i < sizeX && j < sizeY)
            {
                ii = i - (sizeX / 2.0f);
                jj = j - (sizeY / 2.0f);
                Fx = ii * df;
                Fy = jj * df;
                fx = Fx + fx0;
                fy = Fy + fy0;
                tmp_fz = fm02 - (fx*fx + fy*fy);

                if (tmp_fz<0)
                    Fz_err = true;
                else {
                    Fz_err = false;
                    fz = sqrt(tmp_fz);
                    Fz = fz-fz0;
                    
                    if( fm02 * 2 < fx*fx+fy*fy) Fz_err=true;
                }                        

                Nx = cuRound(Fx/df);
                Ny = cuRound(Fy/df);
                

                if ( !Fz_err && Nx >= -sizeX / 2 && Nx<sizeX / 2 && Ny >= -sizeY / 2 && Ny<sizeY / 2)
                {
                    Nz = cuRound(Fz/df);
                    Nx2 = Nx*cos(angX)-Nz*sin(angX);
                    Nz2 = Nx*sin(angX)+Nz*cos(angX);
                    
                    if ( Nx2 >= -sizeX / 2 && Nx2<sizeX / 2 && Nz2 >= -sizeY / 2 && Nz2<sizeY / 2) 
                    {
                        Nx = cuMod(Nx , sizeX);
                        Ny = cuMod(Ny , sizeY);
                        Nx2 = cuMod(Nx2 , sizeX);
                        Nz2 = cuMod(Nz2 , sizeY);
                        idx = Nx2 + Nz2*sizeX + Ny*sizeX*sizeY;
                        F[idx].x += (-fz * 2 * M_PI * u_sp[Nx + Ny*sizeX].y);
                        F[idx].y += ( fz * 2 * M_PI * u_sp[Nx + Ny*sizeX].x);
                        atomicAdd(&C[idx], 1);
                    }
                }
            }        
        }
    }
```
## Optimization
- After recontruction some artifact still remain, cause by **missing angle problem**(yellow part) which is the missing part duting rotation

![](https://i.imgur.com/u46JBEC.png)

- To minimize this problem, we use 
    - Total variation minimization
        - https://github.com/gokhangg
    - Negative constraint
        ```python
        dn_3D[cupy.less(cupy.real(dn_3D),n_med)] = n_med+1j*cupy.imag(dn_3D[cupy.less(cupy.real(dn_3D),n_med)])
        dn_3D[cupy.less(cupy.imag(dn_3D),0)]     = cupy.real(dn_3D[cupy.less(cupy.imag(dn_3D), 0)])
        ```
    - Spatial constraint
        ```python
        n_3D = cupy.asnumpy(self.resultImage).astype(float)
        otsu_val = filters.threshold_otsu(n_3D)
        n_3D_bi = np.zeros_like(n_3D)
        n_3D_bi[n_3D >= otsu_val]= 1
        self.dn_3D_bi = cupy.asarray(n_3D_bi) 
        ```
## Result
![](https://i.imgur.com/EB5H3kP.png)

![](https://i.imgur.com/sD3cYTz.gif)
