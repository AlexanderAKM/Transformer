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
        super().__init__()
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
      # Apply the cos to odd positions
      pe[:, 1::2] = torch.cos(position * div_term)

      pe = pe.unsqueeze(0) # tensor of (1, seq_len, d_model)

      # This way the tensor will be saved with the file 
      self.register_buffer('pe', pe)

   def forward(self, x):
      x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False) #False is for not learned
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

class MultiHeadAttentionBlock(nn.Module):

   def __init__(self, d_model: int, h: int, dropout: float) -> None:
      super().__init__()
      self.d_model = d_model
      self.h = h
      assert d_model % h == 0, "d_model is not divisible by h"

      self.d_k = d_model // h
      self.w_q = nn.Linear(d_model, d_model) # Wq
      self.w_k = nn.Linear(d_model, d_model) # Wk
      self.w_v = nn.Linear(d_model, d_model) # Wv

      self.w_o = nn.Linear(d_model, d_model) #Wo
      self.dropout = nn.Dropout(dropout)

   @staticmethod # can call without having an instance
   def attention(query, key, value, mask, dropout: nn.Dropout):
      d_k = query.shape[-1]

      # Here we do the Scaled dot-product attention. 
      # (batch, h, seq_len, d_k) --> (batch, h, seq_len, seq_len)
      attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
      if mask is not None:
         attention_scores.masked_fill(mask == 0, -1e9)
      attention_scores = attention_scores.softmax(dim = -1) # (batch, h, seq_len, seq_len)
      if dropout is not None:
         attention_scores = dropout(attention_scores)
      
      return (attention_scores @ value), attention_scores
   
   def forward(self, q, k, v, mask):
      query = self.w_q(q) #(Batch, seq_len, d_model) --> (batch, seq_len, d_model)
      key = self.w_k(k) #(Batch, seq_len, d_model) --> (batch, seq_len, d_model)
      value = self.w_v(v) #(Batch, seq_len, d_model) --> (batch, seq_len, d_model)

      # (Batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
      # Every head will still see the whole sentence just with d_model/h embedding
      query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(1, 2)
      key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
      value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(1, 2)

      x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
      
      # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k) --> (batch, seq_len, d_model)
      x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)

      # (batch, seq_len, d_model) --> (batch, seq_len, d_model)
      return self.w_o(x)
   
class ResidualConnection(nn.Module):

   def __init__(self, dropout: float) -> None:
      super().__init__()
      self.dropout = nn.Dropout(dropout)
      self.norm = LayerNormalization()

   def forward(self, x, sublayer):
      # Normally first the normalization is done on the sublayer, but this seems to be a better implementation
      return x + self.dropout(sublayer(self.norm(x))) 

class EncoderBlock(nn.Module):

   def __init__(self, self_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
      super().__init__()
      self.self_attention_block = self_attention_block
      self.feed_forward_block = feed_forward_block
      self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)])

   # src_mask for encoder for the padding of the sequence
   def forward(self, x, src_mask):
      x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
      x = self.residual_connection[1](x, self.feed_forward_block)
      return x
   
class Encoder(nn.Module):

   def __init__(self, layers: nn.ModuleList):
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization()

   def forward(self, x, mask):
      for layer in self.layers:
         x = layer(x, mask)
      
      return self.norm(x)

'''
For Decoder The output embedding is same as input embedding and positional encoding is also already done
'''

class DecoderBlock(nn.Module):
   
   def __init__(self, self_attention_block: MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
      super().__init__()
      self.self_attention_block = self_attention_block
      self.cross_attention_block = cross_attention_block
      self.feed_forward_block = feed_forward_block
      self.residual_connections = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

# src_mask is the mask for the input so the original language, tgt from target language (decoder).
   def forward(self, x, encoder_output, src_mask, tgt_mask):
      x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
      x = self.residual_connections[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
      x = self.residual_connections[2](x, self.feed_forward_block)
      return x
   
class Decoder(nn.Module):

   def __init__(self, layers: nn.ModuleList) -> None:
      super().__init__()
      self.layers = layers
      self.norm = LayerNormalization()

   def forward(self, x, encoder_output, src_mask, tgt_mask):
      for layer in self.layers:
         x = layer(x, encoder_output, src_mask, tgt_mask)
      return self.norm(x)
   
class ProjectionLayer(nn.Module):
   
   def __init__(self, d_model: int, vocab_size: int) -> None:
      super().__init__()
      self.proj = nn.Linear(d_model, vocab_size)

   def forward(self, x):
      # (batch, seq_len, d_model) --> (batch, seq_len, vocab_size)
      return torch.log_softmax(self.proj(x), dim = -1)
   
class Transformer(nn.Module):
   def __init__(self, encoder: Encoder, decoder: Decoder, src_embed: InputEmbeddings, tgt_embed: InputEmbeddings, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding, projection_layer: ProjectionLayer) -> None:
      super().__init__()
      self.encoder = encoder
      self.decoder = decoder
      self.src_embed = src_embed
      self.tgt_embed = tgt_embed
      self.src_pos = src_pos
      self.tgt_pos = tgt_pos
      self.projection_layer = projection_layer

   def encode(self, src, src_mask):
      src = self.src_embed(src)
      src = self.src_pos(src)
      return self.encoder(src, src_mask)
   
   def decode(self, encoder_output, src_mask, tgt, tgt_mask):
      tgt = self.tgt_embed(tgt)
      tgt = self.tgt_pos(tgt)
      return self.decoder(tgt, encoder_output, src_mask, tgt_mask)
   
   def project(self, x):
      return self.projection_layer(x)
   
def build_transformer(src_vocab_size: int, tgt_vocab_size: int, src_seq_len: int, tgt_seq_len: int, d_model: 512, N: int = 6, h: int = 8, dropout: float = 0.1, d_ff: int = 2048) -> Transformer:
   # Create the embedding layers
   src_embed = InputEmbeddings(d_model, src_vocab_size)
   tgt_embed = InputEmbeddings(d_model, tgt_vocab_size)

   # Create the positional encoding layers
   src_pos = PositionalEncoding(d_model, src_seq_len, dropout)
   tgt_pos = PositionalEncoding(d_model, tgt_seq_len, dropout)

   # Create the encoder blocks
   encoder_blocks = []
   for _ in range(N):
      encoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      encoder_block = EncoderBlock(encoder_self_attention_block, feed_forward_block, dropout)
      encoder_blocks.append(encoder_block)

   # Create the decoder blocks
   decoder_blocks = []
   for _ in range(N):
      decoder_self_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      decoder_cross_attention_block = MultiHeadAttentionBlock(d_model, h, dropout)
      feed_forward_block = FeedForwardBlock(d_model, d_ff, dropout)
      decoder_block = DecoderBlock(decoder_self_attention_block, decoder_cross_attention_block, feed_forward_block, dropout)
      decoder_blocks.append(decoder_block)

   encoder = Encoder(nn.ModuleList(encoder_blocks))
   decoder = Decoder(nn.ModuleList(decoder_blocks))

   projection_layer = ProjectionLayer(d_model, tgt_vocab_size)

   transformer = Transformer(encoder, decoder, src_embed, tgt_embed, src_pos, tgt_pos, projection_layer)

   # Initialize the parameters for faster training.
   for p in transformer.parameters():
      if p.dim() > 1:
         nn.init.xavier_uniform_(p)
   
   return transformer





'''1. Create a small dataset of words to use for the Transformer.
2. Find the vocab size (number of unique words in the dataset)
3. We assign a number to each word (encoding)
4. Calculating the embedding of the sentence. The paper uses a 512-dimensional embedding vector
for each word. However, for visualization we will use six dimensions.
The values in the vectors will be initialized randomly between 0 and 1.
Later, they will be updated as the transformer starts understanding the meaning of the words.
5. Positional encoding will now be done, as it is essential that the model knows
   information about the order of the input sequence.'''