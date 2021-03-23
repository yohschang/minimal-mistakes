---
title: "Phase retrival, Refocus and Realign"
permalink: /pdocs/qpiprocess/
excerpt: "QPI project process"
sitemap: true

sidebar:
  nav: "docs"

---


## Phase retrieval
- In QPI we use **interferogram** to store the phase delay cause by different sample. Therefore, an algorithm is needed to retrieve the quantitative information

- The process shown as below
    1. Interferogram take Fourier Transform 
    2. crop high frequency region(imformation from high frequency strip) and set pixel with max value as center
    3. optional padding, then do inverse FFT
    4. calculate wrap phase by arctan(real part / imag part) 
    5. Use unwrap algorithm to recover phase image

```python
def imgft(img):     # x = 0 for bg x = 1 for sp
    out = np.fft.fft2(img)
    shift = np.fft.fftshift(out)
    shift1 = shift.copy()

    shift1[(1*row)*2//5:row,0:col] = 0.01 #set a crop regeion approximatly to find max position

    newcenter = shift[int(i) - row // 8 :int(i) + row // 8 ,int(j) - col // 8 :int(j) + col // 8 ]

    inv_fshift = np.pad(newcenter, ((384,384),(384,384)), 'constant',constant_values=0.1)  # reverse fourier
    img_amp = np.fft.ifft2(inv_fshift)**2  #calculate amplitude
    img_phase = np.arctan2(np.imag(np.fft.ifft2(inv_fshift)),np.real(np.fft.ifft2(inv_fshift)))   #calculate phase difference as wrapped image
    return img_phase,img_amp, 
```

![](https://i.imgur.com/Jl3BfmY.png)

## Refocus
- After retrieval, an numerical refocus algorithm is using to calibrate the focus distance change during cell motion
- Numerical refocus is base on following formula
![](https://i.imgur.com/gIP4zmh.png)
- In order to conform which position(z) is the best focus position, we use Tamura Coefficient which is the proper method to discribe te sparsity of edge
- Since the iteration takes lot of time, I use CUDA to accelerate this process 

```python
def fft_propagate_3d(self,phimap ,d):
    km = (2 * np.pi * self.n_med) / self.wavelen
    kx = (cupy.fft.fftfreq(phimap.shape[0]) * 2 * np.pi).reshape(-1, 1)
    ky = (cupy.fft.fftfreq(phimap.shape[1]) * 2 * np.pi).reshape(1, -1)
    root_km = km**2 - kx**2 - ky**2
    rt0 = (root_km > 0)
    fstemp = cupy.exp(1j * (np.sqrt(root_km * rt0) - km) * d) * rt0
    return cupy.fft.ifft2(phimap* fstemp)
```

## Realign
- Then I align all frames belong to each cell using center of mass, also unified the size and orientation of every frames
```python
if np.nansum(phimap.real[phase_img == 255]) > 0:
x_ct_mass = int(round(np.nansum(phi_med.real[phase_img == 255]*row[phase_img == 255])/np.nansum(phi_med.real[phase_img == 255]))) #calculate x coodinate of center of mass
y_ct_mass = int(round(np.nansum(phi_med.real[phase_img == 255]*column[phase_img == 255])/np.nansum(phi_med.real[phase_img == 255]))) #calculate y coodinate of center of mass
```
- Also unified the size of each frame
```python
def size_uniform(self, frame_num):
    sizelist = []
    frame_num = frame_num.astype(int)
    for i in frame_num :
            sizelist.append(self.phistack[i].real.shape[0])
    max_size = max(sizelist)
        
    newphimaplist = np.zeros((len(frame_num), max_size , max_size))
    for count, i in enumerate(frame_num):
        img = self.phistack[i].real
        i_shape = img.shape[0]
        size_diff = max_size - i_shape
        resize_real = np.pad(img.real, (size_diff // 2, size_diff // 2), "constant", constant_values=(0, 0))
        resize_imag = np.pad(img.imag, (size_diff // 2, size_diff // 2), "constant", constant_values=(0, 0))
        newphimaplist[count] = resize_real + 1j * resize_imag
    return newphimaplist
```
