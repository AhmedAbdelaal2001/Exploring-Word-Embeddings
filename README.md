# Exploring Word Embeddings
An inspired attempt to explore the properties of different word embedding techniques, motivated by Stanford's CS224n class.  
## Count Based Word Vectors:
The first folder outlines the process of extracting word vectors from Reuter's corpus (obtained from the NLTK) using a count-based method; a co-occurence matrix of word counts
was constructed from the corpus, and truncated singular value decomposition was then applied on the matrix to obtain a 2-dimensional version of the word vectors, which were
then plotted on a scatter diagram for comparison with GloVe vectors.  
## Prediciton Based Word Vectors:
The second folder applies truncated SVD to GloVe word vectors (dimension: 200) to obtain 2 dimensional versions, which are then plotted and compared with the previous ones
as shown below. The vectors are then evaluated in terms of their capabilities to model analogies, polysemes, synonyms, antonyms, and bias. Results can be seen in the notebooks.  
![image](https://user-images.githubusercontent.com/101427765/217091645-c2baf817-c2bc-4ed9-8986-e05c6f286b0c.png)

