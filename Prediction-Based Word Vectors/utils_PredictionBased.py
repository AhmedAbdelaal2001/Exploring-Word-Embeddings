import numpy as np
import matplotlib.pyplot as plt
import gensim.downloader as api
import random
from sklearn.decomposition import TruncatedSVD

#------------------------------------------------------------------------------

def load_embedding_model():
    """ Loads GloVe Vectors using the Gensim library. Loaded vectors are trained on
        2B tweets, 27B tokens, 1.2M vocab.
        
        Returns:
            wv_from_bin: All 400000 embeddings, each of lengh 200
    """
    wv_from_bin = api.load("glove-wiki-gigaword-200")
    return wv_from_bin

#------------------------------------------------------------------------------

def get_matrix_of_vectors(wv_from_bin, required_words=['tonnes', 'grain', 'wheat',  'agriculture', 'corn', 'maize', 'export', 'department', 'barley', 'grains', 'soybeans', 'sorghum']):
    """ Takes the loaded GloVe vectors and places them in a matrix M.
        Note: to avoid memory issues, only 10k vectors will be placed in the matrix.
        
        Params:
            wv_from_bin (KeyedVectors object); the 400000 GloVe vectors loaded from file
        Returns:
            M (numpy matrix): contains the GloVe vectors
            word2ind(dictionary): maps each word to its row number in M
    """
    words = list(wv_from_bin.index_to_key)
    random.seed(10)
    random.shuffle(words)
    words = words[:10000]
    word2ind = {}
    M = []
    curInd = 0
    for w in words:
        M.append(wv_from_bin.get_vector(w))
        word2ind[w] = curInd
        curInd += 1
    
    for w in required_words:
        if w in words:
            continue
        M.append(wv_from_bin.get_vector(w))
        word2ind[w] = curInd
        curInd += 1
    
    M = np.stack(M)
    return M, word2ind

#------------------------------------------------------------------------------

def reduce_to_k_dim(M, k=2):
    """ Reduces a matrix of word vectors so that each vector would have a
        dimension of k using Truncated Singular Value Decomposition.
    
        Params:
            M (numpy matrix): matrix of word vectors.
            k (int): embedding size of each word after dimensionality reduction
        Return:
            M_reduced (numpy matrix): matrix of k-dimensioal word embeddings.
    """    
    
    svd = TruncatedSVD(n_components = k, n_iter=10)
    M_reduced = svd.fit_transform(M)

    return M_reduced

#------------------------------------------------------------------------------

def plot_embeddings(M_reduced, word2ind, words):
    """ Plots the word embeddings using a 2 dimensional scatter plot.
        Beside each plotted point, include a label of the word.
        
        Params:
            M_reduced (numpy matrix): reduced matrix of word vectors (2 dimensional)
            word2ind (dictionary): maps each unique word in the corpus to its index within M_reduced.
            words (list of strings): contains all the words.
    """

    for word in words:
        x = M_reduced[word2ind[word]][0]
        y = M_reduced[word2ind[word]][1]
        
        plt.scatter(x, y, marker = 'x', color = 'red')
        plt.text(x + 0.0001, y + 0.0001, word, fontsize = 20)
        
    plt.show()