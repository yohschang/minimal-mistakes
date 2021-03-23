---
title: "CUDA numreical refocus"
permalink: /odocs/cuda/
excerpt: "QPI project overview"
sitemap: true

sidebar:
  nav: "odocs"

---

This project is same as [python version](https://yohschang.github.io/minimal-mistakes/pdocs/qpiprocess/#refocus), but using CUDA to accelerate the calculation process.
The process can speed up 4 times, compare to python version
check out my [github](https://github.com/yohschang/refocus_cuda) for more detial about source code

## kernel
```c
__global__ void fft_propogate_cu(cufftComplex*p_in, cufftComplex*p_out, double d, float nm, float res, int sizex, int sizey)
{
	int i = threadIdx.x + blockIdx.x*blockDim.x;
	int j = threadIdx.y + blockIdx.y*blockDim.y;
	double km = (2 * PI*nm) / res;
	double kx_o, ky_o, kx, ky;
	cufftComplex I;

	
	if (i < sizex && j < sizey)
	{
		if (sizex % 2 != 0)
		{
			if (i < (sizex + 1) / 2)
				kx_o = i;
			else if (i >= (sizex + 1) / 2)
				kx_o = -(sizex - i);
		}
		else if (sizex % 2 == 0)
		{
			if (i < (sizex / 2))
				kx_o = i;
			else if (i >= (sizex / 2))
				kx_o = -(sizex - i);
		}

		if (sizey % 2 != 0)
		{
			if (j < (sizey + 1) / 2)
				ky_o = j;
			else if (j >= (sizey + 1) / 2)
				ky_o = -(sizey - j);
		}
		else if (sizey % 2 == 0)
		{
			if (j < (sizey / 2))
				ky_o = j;
			else if (j >= (sizey / 2))
				ky_o = -(sizey - j);
		}

		kx = (kx_o / sizex) * 2 * PI;
		ky = (ky_o / sizey) * 2 * PI;
		double root_km = km * km - kx * kx - ky * ky;
		bool rt0 = root_km > 0;

		if (root_km > 0)
		{
			I.x = 0;
			I.y = (sqrt(root_km * rt0) - km)*d;
			p_out[i*sizex + j] = com_mul(p_in[i*sizex + j], com_exp(I));
		}
		else
		{
			p_out[i*sizex + j].x = 0.0;
			p_out[i*sizex + j].y = 0.0;
		}
	}

}
```
