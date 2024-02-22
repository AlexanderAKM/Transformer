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

