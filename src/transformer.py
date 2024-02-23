#https://www.youtube.com/watch?v=ISNdQcPhsts&t=9595s for the transformers
# https://www.youtube.com/watch?v=kCc8FmEb1nY For the GPT

'''Start with the input embedding. The original input is converted into
   a vocabulary with size being the number of words.
   The Each word will then reflect a position in the vocabulary and will 
   get the corresponding input ID. Then each number corresponds to an 
   embedding which is a vector of size 512. In the paper this is a learned embedding.
'''
import torch 
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):
    
   def __init__(self, d_model: int, vocab_size: int):
        super().__init()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        
   def forward(self, x):
       return self.embedding(x) * math.sqrt(self.d_model)
   
'''Now we do the positional encoding. We want to inject the information about the position of each 
   word in the sequence. We do this by adding a vector of similar dimension to the input.
   The position embedding of d = 512 will be added.
'''
    
class PositionalEncoding(nn.Module):
    
   def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
      super().__init__()
      self.d_model = d_model
      self.seq_len = seq_len
      self.dropout = nn.Dropout(dropout)

      # Create a matrix of shape (seq_len, d_model)
      pe = torch.zeros(seq_len, d_model)
      '''
      For each word we create a 512-dimensional vector, where for the even
      positions we apply the even formula, and for the odd, the odd ones.
      We will use log, same thing'''
      # Create a vector of shape (seq_len, 1)
      position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
      div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
      # Apply the sin to even positions
      pe[:, 0::2] = torch.sin(position * div_term)
      pe[:, 1::2] = torch.co(position * div_term)

      pe = pe.unsqueeze(0) # tensor of (1, seq_len, d_model)

      # This way the tensor will be saved with the file 
      self.register_buffer('pe', pe)

   def forward(self, x):
      x = x + (self.pe[:, :x.shape[1], :]).requires_grad(False) #False is for not learned
      return self.dropout(x)

'''
   Now Layer normalization. This stabilizes the learning process by computing the mean
   and variance used for normalization across the features (not across the batch as in
   batch normalization). The network can put more emphasis on one unit this way.'''
# https://levelup.gitconnected.com/understanding-transformers-from-start-to-end-a-step-by-step-math-example-16d4e64e6eb1


class LayerNormalization(nn.Module):
   
   def __init__(self, eps: float = 10**-6) -> None:
      super().__init__()
      self.eps = eps
      self.alpha = nn.Parameter(torch.ones(1)) # Multiplitied
      self.bias = nn.Parameter(torch.zeros(1)) # Additive

   def forward(self, x):
      mean = x.mean(dim = -1, keepdim=True)
      std = x.std(dim = -1, keepdim=True)
      return self.alpha * (x - mean) / (std + self.eps) + self.bias
   
'''
Now the feed forward layer. 
'''

class FeedForwardBlock(nn.Module):

   def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
      super().__init__()
      self.linear_1 = nn.Linear(d_model, d_ff) # W1 and b1
      self.dropout = nn.Dropout(dropout)
      self.linear_2 = nn.Linear(d_ff, d_model) # W2 and b2

   def forward(self, x):
      # (Batch, seq_len, d_model) --> (batch, seq_len, d_ff) --> (batch, seq_len, d_model)
      return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))

'''
Now the multi-head attention'''
   
   
   
   
'''1. Create a small dataset of words to use for the Transformer.
2. Find the vocab size (number of unique words in the dataset)
3. We assign a number to each word (encoding)
4. Calculating the embedding of the sentence. The paper uses a 512-dimensional embedding vector
for each word. However, for visualization we will use six dimensions.
The values in the vectors will be initialized randomly between 0 and 1.
Later, they will be updated as the transformer starts understanding the meaning of the words.
5. Positional encoding will now be done, as it is essential that the model knows
   information about the order of the input sequence.'''