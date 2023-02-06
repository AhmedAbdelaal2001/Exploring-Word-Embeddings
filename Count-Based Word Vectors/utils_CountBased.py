import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import TruncatedSVD
from nltk.corpus import reuters

#------------------------------------------------------------------------------

START_TOKEN = '<START>'
END_TOKEN = '<END>'

#------------------------------------------------------------------------------

def read_corpus(category="grain"):
    """
    Reads files from the Reuter's corpus under the specified category.
    All words within said files are returned in a list of lists.
    
    Params:
        category (string): the name of the desired category
    Returns:
        list of lists of strings containing all the words from each file.
    """
    
    files = reuters.fileids(category)
    return [[START_TOKEN] + [w.lower() for w in list(reuters.words(f))] + [END_TOKEN] for f in files]

#------------------------------------------------------------------------------

def distinct_words(corpus):
    """ Determines all the unique words that exist within the corpus of interest.
        
        Params:
            corpus (list of list of strings): corpus of documents
        Return:
            corpus_words (list of strings): sorted list of all the unique words across the corpus
            n_corpus_words (integer): number of distinct words across the corpus
    """
    
    corpus_words = []
    n_corpus_words = -1
    
    corpus_words = sorted(set([w for file in corpus for w in file]))
    n_corpus_words = len(corpus_words)


    return corpus_words, n_corpus_words

#------------------------------------------------------------------------------

def compute_co_occurrence_matrix(corpus, window_size=4):
    """ Constructs the co-occurrence matrix for the given corpus using the specified window size.
    
        Params:
            corpus (list of list of strings): corpus of documents
            window_size (int): size of context window
        Return:
            M (numpy matrix): symmetric Co-occurence matrix of word counts.
            word2ind (dictionary): maps each unique word in the corpus to its index within M.
    """
    words, n_words = distinct_words(corpus)
    M = None
    word2ind = {}
    
    for i in range(n_words):
        word2ind[words[i]] = i
        
    M = np.zeros((n_words, n_words))
    for file in  corpus:
        for word_index in range(len(file)):
            neighbours = file[max(0, word_index - window_size) : min(len(words), word_index + window_size + 1)]
            
            current_word = file[word_index]
            for word in neighbours:
                M[word2ind[current_word]][word2ind[word]] += 1
            
            M[word2ind[current_word]][word2ind[current_word]] -= 1


    return M, word2ind

#------------------------------------------------------------------------------

def reduce_to_k_dim(M, k=2):
    """ Reduces a co-occurence matrix of word counts to a matrix of dimension (num_corpus_words, k)
        using Truncated Singular Value Decomposition.
    
        Params:
            M (numpy matrix): co-occurence matrix of word counts
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
            M_reduced (numpy matrix): reduced Co-occurence matrix (2 dimensional)
            word2ind (dictionary): maps each unique word in the corpus to its index within M_reduced.
            words (list of strings): contains all the words.
    """

    for word in words:
        x = M_reduced[word2ind[word]][0]
        y = M_reduced[word2ind[word]][1]
        
        plt.scatter(x, y, marker = 'x', color = 'red')
        plt.text(x + 0.0001, y + 0.0001, word, fontsize = 20)
        
    plt.show()
