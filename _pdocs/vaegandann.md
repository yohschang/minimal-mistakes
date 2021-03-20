---
title: "VAE, GAN and DANN"
permalink: /pdocs/vaegandann/
excerpt: "Implementation of VAE, GAN and DANN"
sitemap: true

sidebar:
  nav: "docs"

---
In this project I implement (Variational Autoencoder), GAN(Generative adversarial networks) and DANN(domain-adversarial neural network) using public database.

## VAE
A variational autoencoder (VAE) is a type of neural network that learns to reproduce its input, and also map data to latent space.

### Model
- The VAE contains two modules:
    - Encoder: Learn to predict the mean and std of the input images in the latent space.
    - Decoder: Reconstruct an image from a latent vector sampled from the latent space.
```python   
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder(label = True)
        self.decoder = Decoder()
    
    def forward(self, x):
        latent_mu, latent_logvar = self.encoder(x)
        latent = self.latent_sample(latent_mu, latent_logvar)
        x_recon = self.decoder(latent.view(latent.size()+(1,1)))
        return x_recon, latent_mu, latent_logvar
    
    def latent_sample(self, mu, logvar):
        if self.training:
            # the reparameterization trick
            std = logvar.mul(0.5).exp_()
            eps = torch.empty_like(std).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu
```
- Loss function
    - Mean square loss + Kdivergence
```python
def vae_loss(recon_x, x, mu, logvar):
    recon_loss = F.mse_loss(recon_x.view(-1, 4096), x.view(-1, 4096), reduction='mean')
    kldivergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kldivergence /= (400 * 3 * 64 * 64) 
    return recon_loss + kldivergence , kldivergence
```
### Result
- Here I use a subset of human face dataset **CelebA**
- Learning curve
![](https://i.imgur.com/Ze3aZdD.png)
- Reconstructed image
![](https://i.imgur.com/1nYvHGG.png)
- Randomly generate images
![](https://i.imgur.com/BmSRM3Y.png)
- tSNE (Dimention reduction)
![](https://i.imgur.com/SeGdCGK.png)


## GAN
A generative adversarial network (GAN) is a deep learning method in which two neural networks (Generator and discriminator) compete with each other to become more accurate in their predictions.
### Model
- Discriminator : The discriminator learns to detect fake image inputs.
```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(3,64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),    # 64 x 32 x 32
            nn.Conv2d(64,128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),    # 128 x 16 x 16
            nn.Conv2d(128,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),    # 256 x 8 x 8
            nn.Conv2d(256,512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),    # 512 x 4 x 4
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)
```
- Generator : The generator learns to fool the discriminator.
```python
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d( 64,512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),   # 512 x 4 x 4
            nn.ConvTranspose2d(512,256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),   # 256 x 8 x 8
            nn.ConvTranspose2d(256,128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),   # 128 x 16 x 16
            nn.ConvTranspose2d(128,64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),   # 64 x 32 x 32
            nn.ConvTranspose2d(64,3, 4, 2, 1, bias=False),
            nn.Tanh()        # 3 x 64 x 64
        )

    def forward(self, input):
        return self.main(input)
```
- Using BCEloss for both loss calculation

### Result
- Here I use a subset of human face dataset **CelebA**
![](https://i.imgur.com/6GqdTGp.png)
- Because using colorjitter, rotation and other data augmentation method, the contrast and angle perfomed a little bit wierd.


## DANN
As a domain-adversarial learning method, DANN has the ability to train on different dataset which has similar feature compare to target dataset

### Model
The model of DANN contain three part. Include feature extractor, target classification model, and data domain discriminator
- Feature extractor
```python
class FeatureExtractor(nn.Module):
    def __init__(self, in_channel=3, hidden_dims=512):
        super(FeatureExtractor, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, hidden_dims, 3, padding=1),
            nn.BatchNorm2d(hidden_dims),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1,1)),
        )
        
    def forward(self, x):
        h = self.conv(x).squeeze() # (N, hidden_dims)
        return h
```

- Classifier
```python
class Classifier(nn.Module):
    def __init__(self, input_size=512, num_classes=10):
        super(Classifier, self).__init__()

        self.linear1 = nn.Linear(input_size, 256)
        self.relu1 = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(256, num_classes)
        
    def forward(self, h):
        h = self.relu1(self.linear1(h))
        c = self.linear2(h)
        return h ,c
```
- Discriminator
```python
class Discriminator(nn.Module):
    def __init__(self, input_size=512, num_classes=1):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, num_classes),
            nn.Sigmoid(),
        )
    def forward(self, h):
        y = self.layer(h)
        return  y
```
- BCEloss and Crossentropyloss for discriminator and classifier loss calculation, respectively

### Result
- Here I three hand written number database. USPS, MNIST-M and SVHN

|                 | USPS -> MNIST-M | MNIST-M -> SVHN | SVHN -> USPS |
|:---------------:|:---------------:|:---------------:|:------------:|
| Train on target |       13.67%        |     19.99%      |  40.51%       |
|      DANN       |   28.73%      |       48.25%   |    51.67%    |
| Train on source |     91.28%       |     91.49%      | 96.91%     |

---
## Reference
1. https://colab.research.google.com/github/smartgeometry-ucl/dl4g/blob/master/variational_autoencoder.ipynb#scrollTo=mZaVrj0hX1ry
2. https://github.com/atinghosh/VAE-pytorch/blob/master/VAE_celeba.py
3. https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
4. https://github.com/Yangyangii/DANN-pytorch/blob/master/DANN.ipynb
