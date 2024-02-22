


'''Start with the input embedding. The original input is converted into
   a vocabulary with size being the number of words.
   The Each word will then reflect a position in the vocabulary and will 
   get the corresponding input ID. Then each number corresponds to an 
   embedding which is a vector of size 512. In the paper this is a learned embedding.
'''
import torch 
import torch.nn as nn

class InputEmbeddings(nn.Module):
    
    def __init__(self, d_model: int, vocab_size: int):
        super().__init()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        




# https://levelup.gitconnected.com/understanding-transformers-from-start-to-end-a-step-by-step-math-example-16d4e64e6eb1
'''
1. Create a small dataset of words to use for the Transformer.
2. Find the vocab size (number of unique words in the dataset)
3. We assign a number to each word (encoding)
4. Calculating the embedding of the sentence. The paper uses a 512-dimensional embedding vector
for each word. However, for visualization we will use six dimensions.
The values in the vectors will be initialized randomly between 0 and 1.
Later, they will be updated as the transformer starts understanding the meaning of the words.
5. Positional encoding will now be done, as it is essential that the model knows
   information about the order of the input sequence. 
'''

