import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer

#TODO: gensim.downloader only needed if we need pretrained word embedding
import gensim.downloader as api

DEFAULT_PUNCTUATIONS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')
DEFAULT_STOPWORDS = set(stopwords.words("english")).union(DEFAULT_PUNCTUATIONS)

def clean_tokens(corpus, ignore=DEFAULT_STOPWORDS):
    """
    Helper function to remove punctuations, tokenize words, and remove stopwords from corpus

    Parameters
    ----------
    corpus : str
        A str representing a corpus
    
    ignore : set of str, optional
        stopwords to ignore (default: nltk.corpus.stopwords.words("english") for list of common English words and punctuations)

    Returns
    -------
    list of str
        List of clean word tokens

    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> corpysprofiling.clean_tokens("How many species of animals are there in Russia?")
    ["How", many", "species", "animals", "Russia"] 
    >>> corpysprofiling.clean_tokens("How many species of animals are there in Russia?", ignore=set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    ["How", many", "species", "of", "animals", "are", "there", "in", "Russia"]      
    """
    all_tokens = word_tokenize(corpus)
    # Remove stopwords
    clean_tokens = [t for t in all_tokens if t not in set(ignore)]
    return clean_tokens

def corpus_analysis(corpus):
    """
    Generate descriptive statistics on corpus

    Parameters
    ----------
    corpus : str
        A str representing a corpus
    

    Returns
    -------
    pandas.DataFrame
        Summary statistics of the corpus will be generated:
            word_total: The number of total words in the corpus
            word_unique: The number of unique words in the corpus
            avg_wd_len: The average length of words
            avg_sens_len: The average length of sentences
            topic_analysis: Related topics of the corpus


    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> corpysprofiling.corpus_analysis("How many species of animals are there in Russia?")
    
    word_total       9
    word_unique      9
    avg_wd_len       4
    avg_sens_len     1
    topic_analysis   ["animal", "country"]


    >>> corpysprofiling.corpus_analysis([2, 3, 4])
    TypeError: Input must be a string
    """
    return

def corpus_viz(corpus, stats):
    """
    Generate visualizations for the key statistics on the input corpus

    Parameters
    ----------
    corpus : str
        A str representing a corpus
    stats : str
        A str representing the interested aspect to the corpus such as words within the documents or topic_analysis; The parameter can take values such as 'words', 'topic', and etc.
    
    Returns
    -------
    Chart object
        To present a visualization for words summary or topic summary

    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> corpysprofiling.corpus_viz("How many species of animals are there in Russia?", 'words')
    >>> corpysprofiling.corpus_viz("How many species of animals are there in Russia?", 'topic')
    >>> corpysprofiling.corpus_viz("How many species of animals are there in Russia?", 15)
    TypeError: Input must be a string
    """
    return

def corpora_compare(corpus1, corpus2, metric="cosine_similarity"):
    """
    Calculate similarity score between two corpora.  The closer the score is to zero, the more similar the corpora are. 

    Parameters
    ----------
    corpus1 : str
        A str representing a corpus
    corpus2 : list of str
        A str representing a corpus
    metric : str, optional
        metric used to determine corpora similarity (default: "cosine_similarity")

    Returns
    -------
    float
        Similarity score 

    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> corpysprofiling.corpora_compare("My friend loves cats so much, she is obsessed!", "My friend adores all animals equally.")
    0.09288773
    >>> corpysprofiling.corpora_compare([2, 3, 4], [2, 3, 4])
    TypeError: Input must be a string
    """
    embedder=SentenceTransformer("paraphrase-distilroberta-base-v1")
    emb1 = embedder.encode(corpus1)
    emb2 = embedder.encode(corpus2)

    if metric == "cosine_similarity":
        score = 1-(np.dot(emb1, emb2)/(np.linalg.norm(emb1)*np.linalg.norm(emb2)))
    
    if metric == "euclidean":
        score = np.linalg.norm(emb1-emb2)
    
    # Return absolute value to avoid small negative value due to rounding 
    return np.abs(score)

def corpora_best_match(refDoc, corpora, metric="cosine_similarity"):
    """
    Rank a list of corpora in the order of most relevance to the reference document.

    Parameters
    ----------
    refDoc : str
        A str for reference document
    corpora : list of str
        A list of str, each representing a corpus

    metric : str, optional
        metric used to determine corpora similarity (default: "cosine_similarity")

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with a column of corpora and a column of distances from the reference document, sorted from closest to furthest.

    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    # TODO: modify the toy examples with correct outputs after the function is implemented
    >>> corpysprofiling.corpora_best_match("kitten meows", ["ice cream is yummy", "cat meowed", "dog barks", "The Hitchhiker's Guide to the Galaxy has become an international multi-media phenomenon"])
                                                corpora  metric
    0                                         cat meowed     0.5
    1                                          dog barks     1.0
    2                                 ice cream is yummy    10.0
    3  The Hitchhiker's Guide to the Galaxy has becom...    42.0
    >>> corpysprofiling.corpora_best_match("kitten meows", [], metric="cosine_similarity")
    Empty DataFrame
    Columns: [corpora, metric]
    Index: []
    >>> corpysprofiling.corpora_best_match(None, None, metric="cosine_similarity")
    TypeError: unsupported operand type(s) for 'corpora_best_match': 'NoneType' and 'NoneType'
    """

    # Naive implementation
    try:
        dist_df = pd.DataFrame(
            ([corpus, corpora_compare(refDoc, corpus, metric=metric)] for corpus in corpora),
            columns = ["corpora", "metric"]
        )
    except TypeError as error:
        raise TypeError("TypeError raised while calling corpora_compare:\n" + error)
    return dist_df.sort_values(by="metric")