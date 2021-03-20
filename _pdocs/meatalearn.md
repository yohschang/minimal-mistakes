---
title: "Meta learning"
permalink: /pdocs/metalearn/
excerpt: "Meta learning"
sitemap: true

sidebar:
  nav: "docs"

---

In this project I implement few type of metalearning method include [prototypical network](#prototypical-network) and [Data Hallucination](#data-hallucination) for Few-shot Learning

## Prototypical network
Metalearning use small set _(N-ways K-shots)_ of data seprate from full dataset to learn how to learn
### Model
- Feature extractor
```python
def conv_block(in_channel , out_channel):
    bn = nn.BatchNorm2d(out_channel)
    return nn.Sequential(
        nn.Conv2d(in_channel , out_channel ,3 ,padding=1),
        bn,
        nn.ReLU(),
        nn.MaxPool2d(2)
        )
class convnet(nn.Module):
    def __init__(self , in_channel = 3 , hid_channel = 64 , out_channel = 64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(in_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            conv_block(hid_channel , hid_channel),
            )
    def forward(self,x):
        x = self.encoder(x)
        return x.view(x.size(0),-1)
```
- Loss function _(Distances calculation)_
    - l2 norm
    ```python
        distances = (
                x.unsqueeze(1).expand(x_shape, y_shape, -1) -
                y.unsqueeze(0).expand(x_shape, y_shape, -1)
        ).pow(2).sum(dim=2)  #calculate each embadding query with 5 ways
    ```
    - cosine
    ```python
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + 1e-8)

        expanded_x = normalised_x.unsqueeze(1).expand(x_shape, y_shape, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(x_shape, y_shape, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        distance = 1 - cosine_similarities
    ```
    - dot 
    ```python
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)
        distance =  -(expanded_x * expanded_y).sum(dim=2)
    ```
### Result
- Here I use Mini-ImageNet Dataset for training and testing with different combination of shots
    **Accuracy**
    - 5 ways 1 shot :　43.65 ± 2.10%
    - 5 ways 5 shot :　45.55 ± 0.82%
    - 5 ways 10 shot :　44.01 ± 0.92%

## Data Hallucination
In order to enlarge the data which use for learning, a concept of generator has been added to the above model

### Model
- Generator
```python
class generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1600*2,2400)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout()
        self.linear2 = nn.Linear(2400,2400)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout()
        self.linear3 = nn.Linear(2400,1600)
        self.relu3 = nn.ReLU()

    def forward(self,x_in,M,seed):
        H_stack = torch.tensor([]).to("cuda")
        for _ in range(M):
            if seed :
                torch.manual_seed(1234)
                noise = torch.randn(x_in.size()).to("cuda")
            else :
                noise = torch.randn(x_in.size()).to("cuda")
            x = torch.cat((x_in,noise)) 
            x = self.drop1(self.relu1(self.linear1(x)))
            x = self.drop2(self.relu2(self.linear2(x)))
            x = self.relu3(self.linear3(x))
            H_stack = torch.cat((H_stack , x)) 
        return H_stack.view(-1,1600)
```
Combine the output of generator to origin data 
```python
output = model(data)    
support_output = output[:ways * N_shot]
rand_sample = np.random.randint(0,N_shot,(ways,)) + np.array([i for i in range(0,N_shot*ways , N_shot)])
new_support = torch.tensor([]).to("cuda")
for index , r in enumerate(rand_sample):
    hallucinate = generator(support_output[r,...] , M , False)
    hallucinate = torch.cat((support_output[index*N_shot:(index+1)*N_shot,...] , hallucinate))
    new_support = torch.cat((new_support,hallucinate) , dim = 0)
```

### Result
- tSNE visualize the real and hallucinated data in the latent space
![](https://i.imgur.com/l9BThM9.png)


- Accuracy of different amount **(M)** of generation data  
    - M = 10 :　44.1 ± 1.1%
    - M = 50 :　44.26 ± 0.62%
    - M = 100 :47.59 ± 0.79%

