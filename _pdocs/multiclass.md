---
title: "Multiclass classification"
permalink: /pdocs/multiclass/
excerpt: "Multiclass classification on brain CT"
sitemap: true

sidebar:
  nav: "docs"

---

In this project I use Multiclass classification to predict multiple type of brain injury for each patient with a set of slice CT image. 

## Data augmentation
- Contrast Limited Adaptive Histogram Equalization (CLAHE)
    - CLAHE limiting the slope of the cumulative distribution function (CDF), which is equivalent to limiting the amplitude of the histogram.
    - If the bins in the histogram exceed the upper limit of contrast, the pixels in the histogram will be evenly dispersed into other bins.
- Other data augmentation method
    - In this project, we also use several common augmentation techniques to avoid training data overfitting
    - For instance, flip, rotation, scale and color jittering
    - During training and testing, the image data has been resized to 224x224

## Model
### CNN backbone
- VGG16 : extract feature of different slice and conbine them to 2D array
### RNN 
**Here we use two types of rnn model to learn different relationship in this case**
- LSTM : Learn the correlation between different trauma or disease
    - This model runs 5 time-steps, each time producing a single scalar which is thescore for one particular class, or produce a feature matrix construct by each class
    - The CNN feature act as an initial state of LSTM and the input is set to zero
    - since LSTM seperate hidden state and cell state, the cell state can be update independently, which led to higher performance then GRU in the predition of trauma relationship
    ```python
    class DecoderBinaryRNN(nn.Module):
        def __init__(self, hidden_size, cnn_output_size, num_labels,vgg = None,mode = "lstm"):
            super(DecoderBinaryRNN, self).__init__()

            self.mode = mode
            self.num_labels = num_labels
            self.linear_img_to_lstm = nn.Linear(cnn_output_size, hidden_size)
            if self.mode == "lstm":
                self.lstm = nn.LSTM(1, hidden_size, 1, batch_first=True, bidirectional=True)
                self.linear_final = nn.Linear(hidden_size*2, 1)
            elif self.mode == "gru":
                self.gru = nn.GRU(1, hidden_size, 1, batch_first=True , bidirectional=True)
                self.linear_final = nn.Linear(hidden_size*2, 1)

        def forward(self, cnn_features):

            h0 = torch.unsqueeze(self.linear_img_to_lstm(cnn_features), 0).to("cuda")
            c0 = torch.autograd.Variable(torch.zeros(h0.size(0), h0.size(1), h0.size(2)), requires_grad = False).to("cuda")
            zero_input = torch.autograd.Variable(torch.zeros(cnn_features.size(0), self.num_labels, 1), requires_grad = False).to("cuda")

            if self.mode == "lstm":
                hiddens, _ = self.lstm(zero_input, (h0.repeat(2,1,1), c0.repeat(2,1,1)))
            elif self.mode == "gru":
                hiddens, _ = self.gru(zero_input, h0.repeat(2,1,1))

            return hiddens
    ```
 
- GRU: Intuitively, the relationship between different slices within one patient should be considered
```python
class SliceRNN(nn.Module):
    def __init__(self,inputsize=512,hiddensize=512,outputsize=5,rnntype='GRU'):
        super(SliceRNN, self).__init__()
        self.type=rnntype
        self.outputsize=outputsize
        self.hiddensize=hiddensize
        
        if self.type=='GRU':
            self.RNN=nn.GRU(inputsize,self.hiddensize,1,batch_first=True,bidirectional=True)
        elif self.type=='LSTM':
            self.RNN=nn.LSTM(inputsize,self.hiddensize,1,batch_first=True,bidirectional=True)
        self.linear=nn.Linear(hiddensize*2,outputsize)
        
        
    def forward(self, x):
        x=x.unsqueeze(0)
        if self.type=='GRU':
            h0=torch.zeros(2,1,self.hiddensize)
            if torch.cuda.is_available():
                h0=h0.cuda()
            output,hn=self.RNN(x,h0)
        elif self.type=='LSTM':
            output,hn=self.RNN(x)
        print(output.size())
        output=torch.squeeze(output,0)
        return output
```

**With different combination I find out that the best architecture would be CNN -> LSTM -> GRU**
![](https://i.imgur.com/BQPQfNz.png)

### Loss function:
- Asymmetric loss function[^first]
    - To deal with imbalance data, positive samples is much more important than the negative one
    - ASL is a variation of Binary Cross-Entropy and Focal Loss
    - By setting r->r+ it’s easy to emphasize positive samples

## Result
**Because of the privacy, here only shows the score and the result of the course competition on kaggle**

- F2 score : 0.783
- ![](https://i.imgur.com/olmSxDx.jpg)


[^first]: https://arxiv.org/abs/2009.14119
