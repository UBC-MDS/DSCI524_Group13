from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt
import pandas as pd
import numpy as np
import nltk

nltk.download("stopwords")

DEFAULT_PUNCTUATIONS = set("!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")
DEFAULT_STOPWORDS = set(stopwords.words("english")).union(DEFAULT_PUNCTUATIONS)


def clean_tokens(corpus, ignore=DEFAULT_STOPWORDS):
    """
    Helper function to remove punctuations, tokenize words, and remove
    stopwords from corpus

    Parameters
    ----------
    corpus : str
        A str representing a corpus

    ignore : set of str, optional
        stopwords to ignore (default: nltk.corpus.stopwords.words("english")
        for list of common English words and punctuations)

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
    >>> corpysprofiling.clean_tokens("How many species of animals are there
    >>> in Russia?")
    ["many", "species", "animals", "russia"]
    >>> corpysprofiling.clean_tokens("How many species of animals are there
    >>> in Russia?", ignore=set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'))
    ["how", "many", "species", "of", "animals", "are", "there", "in", "russia"]
    """
    if type(corpus) != str:
        raise TypeError("Input must be a string")
    all_tokens = word_tokenize(corpus.lower())
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
            word_total: The number of total words after removing punctuations
            from the corpus
            token_total: The number of total tokens after removing stopwords
            from the corpus
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
    >>> text = "How many species of animals are there in Russia?
    >>> and how many in US?"
    >>> corpysprofiling.corpus_analysis(text)
                    value
    word_total       14.0
    token_total       6.0
    token_unique      5.0
    token_avg_len     5.0
    sent_count        2.0
    sens_avg_token    3.0

    >>> corpysprofiling.corpus_analysis([2, 3, 4])
    TypeError: Input must be a string
    """

    if not isinstance(corpus, str):
        raise TypeError("Input must be a string")

    # create empty dictionary to store outputs
    analysis_dict = {}

    # statistics on words and tokens
    token = clean_tokens(corpus, ignore=DEFAULT_PUNCTUATIONS)
    token_clean = clean_tokens(corpus)
    token_len = [len(t) for t in token_clean]

    analysis_dict["word_total"] = len(token)
    analysis_dict["token_total"] = len(token_clean)
    analysis_dict["token_unique"] = len(set(token_clean))
    analysis_dict["token_avg_len"] = round(np.mean(token_len), 1)

    # statistics on sentences of the corpus
    sents = sent_tokenize(corpus)
    sents_tokenize = [clean_tokens(sent) for sent in sents]
    sens_avg_token = [len(sent) for sent in sents_tokenize]

    analysis_dict["sent_count"] = len(sents)
    analysis_dict["sens_avg_token"] = round(np.mean(sens_avg_token), 1)

    # organize distionary into pandas dataframe
    output_df = pd.DataFrame.from_dict(
        analysis_dict, orient="index", columns=["value"]
    )

    # return dataframe of statitics
    return output_df


def corpus_viz(corpus):
    """
    Generate visualizations for words from the input corpus

    Parameters
    ----------
    corpus : str
        A str representing a corpus

    Returns
    -------
    dictionary
        contains a word cloud, a histogram of word length frequencies,
        and a histogram of word frequencies

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> corpysprofiling.corpus_viz("How many species of animals are there
    >>> in Russia?")
    >>> corpysprofiling.corpus_viz("How many species of animals are there
    >>> in Russia?")['word cloud']
    >>> corpysprofiling.corpus_viz("How many species of animals are there
    >>> in Russia?")['word freq bar chart']
    >>> corpysprofiling.corpus_viz("How many species of animals are there
    >>> in Russia?")['word length bar chart']

    """
    if not isinstance(corpus, str):
        raise TypeError("Input must be a string")

    # Step 1. To get a word cloud
    wordcloud = WordCloud().generate(corpus.lower())
    wordcloud_fig = plt.figure()
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.close(1)

    # Step 2. To get a bar chart to visualize the words and the length of words

    # To get a list of words from the input text
    clean_corpus = clean_tokens(corpus.lower())

    # To get a data frame summary of the words and length of the words
    df = pd.DataFrame({"word": clean_corpus})
    df["length"] = df["word"].str.len()

    df_length = df.sort_values(by="length", ascending=False).head(30)

    df_freq = df.value_counts().reset_index().rename(columns={0: "freq"})
    # To limit the number of words to display in the plot
    # Select top 30 most frequent words to display
    df_freq = df_freq.sort_values(by="freq", ascending=False).head(30)
    df_freq = df_freq.reindex(
        df_freq.index.repeat(df_freq["freq"])
    ).reset_index()[["word"]]

    # To make a bar chart
    bar_length = (
        alt.Chart(df_length)
        .encode(
            x=alt.X("length", bin=True, title="Word Length"),
            y=alt.Y("count()", title="Frequency"),
        )
        .mark_bar()
        .properties(title="Frequency of Words by Length")
    )

    bar_freq = (
        alt.Chart(df_freq)
        .encode(
            x=alt.X("word", sort="-y", title="Word"),
            y=alt.Y("count()", title="Frequency"),
        )
        .mark_bar()
        .properties(title="Frequency of Words")
    )

    return {
        "word cloud": wordcloud_fig,
        "word freq bar chart": bar_freq,
        "word length bar chart": bar_length,
    }


def corpora_compare(corpus1, corpus2, metric="cosine_similarity"):
    """
    Calculate similarity score between two corpora.  The closer the score is to
    zero, the more similar the corpora are.

    Parameters
    ----------
    corpus1 : str
        A str representing a corpus
    corpus2 : list of str
        A str representing a corpus
    metric : {'cosine_simiarity', 'euclidean'}, default 'cosine_simiarlty'
        metric used to determine corpora similarity

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
    >>> corpysprofiling.corpora_compare("My friend loves cats so much, she is
    >>> obsessed!", "My friend adores all animals equally.")
    0.3874828815460205
    >>> corpysprofiling.corpora_compare([2, 3, 4], [2, 3, 4])
    TypeError: Inputs must be a string
    """
    if not isinstance(corpus1, str) or not isinstance(corpus2, str):
        raise TypeError("Inputs must be a string")

    if metric not in ["euclidean", "cosine_similarity"]:
        raise ValueError("metric must be cosine_similarity or euclidean")

    embedder = SentenceTransformer("paraphrase-distilroberta-base-v1")
    emb1 = embedder.encode(corpus1)
    emb2 = embedder.encode(corpus2)

    if metric == "cosine_similarity":
        score = 1 - (
            np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        )

    if metric == "euclidean":
        score = np.linalg.norm(emb1 - emb2)

    # Return absolute value to avoid small negative value due to rounding
    return np.abs(score)


def corpora_best_match(refDoc, corpora, metric="cosine_similarity"):
    """
    Rank a list of corpora in the order of most relevance to the reference
    document.

    Parameters
    ----------
    refDoc : str
        A str for reference document
    corpora : list of str
        A list of str, each representing a corpus

    metric : str, optional
        metric used to determine corpora similarity
        (default: "cosine_similarity")

    Returns
    -------
    pandas.DataFrame
        A pandas DataFrame with a column of corpora and a column of distances
        from the reference document, sorted from closest to furthest.

    Raises
    -------
    TypeError
        If argument passed is of wrong type

    Examples
    --------
    >>> from corpysprofiling import corpysprofiling
    >>> corpysprofiling.corpora_best_match("kitten meows",
    >>> ["ice cream is yummy", "cat meowed", "dog barks",
    >>> "The Hitchhiker's Guide to the Galaxy has become an international
    >>> multi-media phenomenon"])
                                                corpora  metric
    0                                         cat meowed     0.534642
    1                                          dog barks     0.725425
    2  The Hitchhiker's Guide to the Galaxy has becom...     0.888661
    3                                 ice cream is yummy     0.915435
    >>> corpysprofiling.corpora_best_match("kitten meows",
    >>> ["ice cream is yummy", "cat meowed", "dog barks",
    >>> "The Hitchhiker's Guide to the Galaxy has become an international
    >>> multi-media phenomenon"],
    >>>  metric="euclidean")
                                                corpora  metric
    0                                         cat meowed     9.335772
    1                                          dog barks    10.302269
    2  The Hitchhiker's Guide to the Galaxy has becom...    11.293064
    3                                 ice cream is yummy    11.397785
    >>> corpysprofiling.corpora_best_match("kitten meows", [],
    >>> metric="cosine_similarity")
    Empty DataFrame
    Columns: [corpora, metric]
    Index: []
    >>> corpysprofiling.corpora_best_match(None, None,
    >>> metric="cosine_similarity")
    TypeError: TypeError raised while calling corpora_compare:
    'NoneType' object is not iterable
    """

    # Naive implementation
    try:
        dist_df = pd.DataFrame(
            (
                [corpus, corpora_compare(refDoc, corpus, metric=metric)]
                for corpus in corpora
            ),
            columns=["corpora", "metric"],
        )
    except TypeError as error:
        raise TypeError(
            "TypeError raised while calling corpora_compare:\n" + str(error)
        )
    except ValueError as error:
        raise ValueError(
            "ValueError raised while calling corpora_compare:\n" + str(error)
        )
    return dist_df.sort_values(by="metric")
