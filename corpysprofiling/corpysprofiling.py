import pandas as pd
import numpy as np
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud

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
            word_total: The number of total words after removing punctuations from the corpus 
            token_total: The number of total tokens after removing stopwords from the corpus
            token_unique: The number of unique tokens in the corpus
            token_avg_len: The average length of total tokens in the corpus
            sent_count: The number of sentences in the corpus
            sens_avg_token: The average number of tokens of all sentencense


    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> text = "How many species of animals are there in Russia? and how many in US?"
    >>> corpysprofiling.corpus_analysis(text)
                    value
    word_total       14.0
    token_total       7.0
    token_unique      6.0
    token_avg_len     4.7
    sent_count        2.0
    sens_avg_token    3.5

    {'word_total': 14,
     'token_total': 7,
     'token_unique': 6,
     'token_avg_len': 4.7,
     'sent_count': 2,
     'sens_avg_token': 3.5}

    >>> corpysprofiling.corpus_analysis([2, 3, 4])
    TypeError: Input must be a string
    """
    # create empty dictionary to store outputs
    analysis_dict = {}
    
    # statistics on words and tokens
    token = clean_tokens(text, ignore=DEFAULT_PUNCTUATIONS)
    token_clean = clean_tokens(text)
    token_len = [len(t) for t in df_clean]

    analysis_dict['word_total'] = len(df)
    analysis_dict['token_total'] = len(df_clean)
    analysis_dict['token_unique'] = len(set(df_clean))
    analysis_dict['token_avg_len'] = round(np.mean(token_len),1)

    # statistics on sentences of the corpus
    sents = sent_tokenize(text)
    sents_tokenize = [clean_tokens(sent) for sent in sents]
    sens_avg_token = [len(sent) for sent in sents_tokenize]

    analysis_dict['sent_count'] = len(sents)
    analysis_dict['sens_avg_token'] = round(np.mean(sens_avg_token),1)
    
    # organize distionary into pandas dataframe
    output_df = pd.DataFrame.from_dict(analysis_dict, orient = 'index', columns = ['value'])
    
    # print output as dataframe 
    print(output_df)
    
    # return dictionary of statitics
    return analysis_dict


def corpus_viz(corpus, display=True):
    """
    Generate visualizations for words from the input corpus

    Parameters
    ----------
    corpus : str
        A str representing a corpus
    display: boolean (optional)
        If display is False, the plots will be hidden from the output
    
    Returns
    -------
    dictionary
        contains a wordcloud.WordCloud, which can be used to present a word cloud,
        and a data frame, which can be used to draw a bar chart for words and word lengths

    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> corpysprofiling.corpus_viz("How many species of animals are there in Russia?")
    >>> corpysprofiling.corpus_viz("How many species of animals are there in Russia?")['word cloud']
    >>> plt.figure()
    >>> plt.imshow(wordcloud, interpolation="bilinear")
    >>> plt.axis("off")
    >>> corpysprofiling.corpus_viz("How many species of animals are there in Russia?")['df used for bar']
    >>> df.plot.bar(rot=0, x='words')
    >>> plt.xticks(rotation=90)
    >>> plt.xlabel("Words")
    >>> corpysprofiling.corpus_viz("How many species of animals are there in Russia?", 15)
    TypeError: Input must be a string
    """
    # Step 1. To get a word cloud
    wordcloud = WordCloud().generate(corpus.lower())
    plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    
    # Step 2. To get a bar chart to visualize the words and the length of words
    # To get a list of words from the input text
    clean_corpus = clean_tokens(corpus)
    # To get a data frame summary of the words and length of the words
    df = pd.DataFrame({'corpus': clean_corpus})
    df = pd.DataFrame(df['corpus'].value_counts())
    df.reset_index(level=0, inplace=True)
    df = df.rename(columns={'index': 'words', 'corpus': 'length'})
    # To limit the number of words to display in the plot
    if len(df) < 30:
        df = df
    else: df = df.head(30)
    # To make a bar chart
    df.plot.bar(rot=0, x='words')
    plt.xticks(rotation=90)
    plt.xlabel("Words")
    
    if display==False:
        plt.close()
        plt.close(1)
    
    return {'word cloud': wordcloud, 'df used for bar': df}

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
