from typing import List

import numpy as np
from gensim.models.word2vec import Word2Vec


def vectorizer(
    corpus: List[List[str]], model: Word2Vec, num_features: int = 100
) -> np.ndarray:
    """
    This function takes a list of tokenized text documents (corpus) and a pre-trained
    Word2Vec model as input, and returns a matrix where each row represents the
    vectorized form of a document.

    Args:
        corpus : list
            A list of text documents that needs to be vectorized.

        model : Word2Vec | KeyedVectors
            A pre-trained Word2Vec model that will be used to vectorize the corpus.

        num_features : int
            The size of the vector representation of each word. Default is 100.

    Returns:
        corpus_vectors : numpy.ndarray
            A 2D numpy array where each row represents the vectorized form of a
            document in the corpus.
    """

    corpus_vectors = []

    for sentence in corpus:
        vectors = [model.wv[word] for word in sentence if word in model.wv]
        if vectors:
            sentence_vectors = np.mean(vectors, axis = 0)
        else:
            sentence_vectors = np.zeros(num_features)

        corpus_vectors.append(sentence_vectors)

    return corpus_vectors

