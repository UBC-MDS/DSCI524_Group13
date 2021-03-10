from corpysprofiling import __version__
from corpysprofiling import corpysprofiling
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import altair as alt
def test_version():
    assert __version__ == '0.1.0'

DEFAULT_PUNCTUATIONS = set('!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~')

def test_clean_tokens():
    corpus = "How many species of animals are there in Russia?"
    
    testCase1 = corpysprofiling.clean_tokens(corpus)
    testCase2 = corpysprofiling.clean_tokens(corpus, ignore=DEFAULT_PUNCTUATIONS)

    assert isinstance(testCase1, list), "Return type is not list"
    assert all(isinstance(token, str) for token in testCase1), "Return type is not list of str"
    assert testCase1 == ["many", "species", "animals", "russia"], "Output does not match expected result." 
    assert isinstance(testCase2, list), "Return type is not list"
    assert all(isinstance(token, str) for token in testCase2), "Return type is not list of str"
    assert testCase2 == ["how", "many", "species", "of", "animals", "are", "there", "in", "russia"], "Output does not match expected result."

    try:
        corpysprofiling.clean_tokens(123)
        # TypeError not raised
        assert False, "TypeError not raised. corpus_analysis should not accept non-string inputs"
    except TypeError:
        # TypeError raised as expected
        pass
    
def test_corpus_analysis():
    """ Test corpus_analysis function"""
    
    corpus = "How many species of animals are there in Russia? and how many in US?"

    testCase1 = corpysprofiling.corpus_analysis(corpus)

    assert isinstance(testCase1, pd.DataFrame), "Return type is not pd.DataFrame"
    assert testCase1.shape == (6, 1), "Shape returned does not match output dataframe"
    
    try:
        corpysprofiling.corpus_analysis(123)
        # TypeError not raised
        assert False, "TypeError not raised. corpus_analysis should not accept non-string inputs"
    except TypeError:
        # TypeError raised as expected
        pass

def test_corpus_viz():
    """ Test corpus_viz function"""
    
    corpus1 = "How many species of animals are there in Russia?"
    corpus2 = "Let's assume that this unsupervised model is used to assist human experts to identify fraud transaction. So instead of making humans examine all 284,807 transactions for fraud transactions, this model would extract transactions which look suspicious and pass them to humans for examination. So our goal is to come up with a list of transactions which look suspicious.We said before that PCA won't be able to capture characteristic features of fraud transactions because they are like outliers (occur very rarely) in our dataset, and so the reconstruction error would be higher for them compared to non-fraud transactions. But what do we mean by high reconstruction error? What should be threshold which makes a transaction suspicious?"

    testCase1 = corpysprofiling.corpus_viz(corpus1)
    testCase2 = corpysprofiling.corpus_viz(corpus2)

    # Test whether the corpus_viz returns a dictionary
    assert isinstance(testCase1, dict), "Return type is not a dict"
    assert isinstance(testCase2, dict), "Return type is not a dict"

    # Tests for the length bar chart
    assert isinstance(testCase1['word length bar chart'], alt.Chart), "Return type is not an altair Chart"
    assert testCase1['word length bar chart'].encoding.x.shorthand == "length"
    assert testCase1['word length bar chart'].encoding.y.shorthand== "count()"
    assert isinstance(testCase2['word length bar chart'], alt.Chart), "Return type is not an altair Chart"
    assert testCase2['word length bar chart'].encoding.x.shorthand == "length"
    assert testCase2['word length bar chart'].encoding.y.shorthand == "count()"

    # Tests for the word cloud 
    assert isinstance(corpysprofiling.corpus_viz(corpus1)['word cloud'], plt.Figure), "Return type is not a word cloud"
    assert isinstance(corpysprofiling.corpus_viz(corpus2)['word cloud'], plt.Figure), "Return type is not a word cloud"

    # Tests for the frequency bar chart
    assert isinstance(corpysprofiling.corpus_viz(corpus1)['word freq bar chart'], alt.Chart), "Word frequency histogram is not an altair Chart"
    assert testCase1['word freq bar chart'].data['index'].unique().shape[0] == 4, "Number of words does not match number of unique tokens in corpus."
    assert testCase1['word freq bar chart'].encoding.x.shorthand == "index"
    assert testCase1['word freq bar chart'].encoding.y.shorthand== "count()"
    assert isinstance(corpysprofiling.corpus_viz(corpus2)['word freq bar chart'], alt.Chart), "Word frequency histogram is not an altair Chart"   
    assert testCase2['word freq bar chart'].data['index'].unique().shape[0] <= 30, "Too many words"
    assert testCase2['word freq bar chart'].encoding.x.shorthand == "index"
    assert testCase2['word freq bar chart'].encoding.y.shorthand== "count()"

    try:
        corpysprofiling.clean_tokens(123)
        # TypeError not raised
        assert False, "TypeError not raised. corpus_analysis should not accept non-string inputs"
    except TypeError:
        # TypeError raised as expected
        pass
    
def test_corpora_compare():
    """ Test corpora_compare function"""
    corpus1 = "kitten meows"
    corpus2 = "kitten meows"

    testCase1 = corpysprofiling.corpora_compare(corpus1, corpus2)
    testCase2 = corpysprofiling.corpora_compare(corpus1, corpus2, metric="euclidean")

    assert isinstance(testCase1, np.float64), "Return type is not np.float"
    assert np.isclose(testCase1, 0.0, atol=1e-06), "Identical corpora should return score of 0.0"
    assert np.isclose(testCase2, 0.0, atol=1e-06), "Identical corpora should return score of 0.0"
    assert (testCase1 >= 0), "Distances should be between 0 and 1 inclusive for cosine_similarity"
    assert (testCase1 <= 1), "Distances should be between 0 and 1 inclusive for cosine_similarity"
    assert (testCase2 >= 0), "Distances should be greater than 0 for euclidean"

    try:
        corpysprofiling.corpora_compare(corpus1, 123)
        # TypeError not raised
        assert False, "TypeError not raised. corpora_compare should not accept non-string inputs"
    except TypeError:
        # TypeError raised as expected
        pass
    try:
        corpysprofiling.corpora_compare(123, corpus1)
        # TypeError not raised
        assert False, "TypeError not raised. corpora_compare should not accept non-string inputs"
    except TypeError:
        # TypeError raised as expected
        pass
    try:
        corpysprofiling.corpora_compare(123, [1, 2, 3])
        # TypeError not raised
        assert False, "TypeError not raised. corpora_compare should not accept non-string inputs"
    except TypeError:
        # TypeError raised as expected
        pass
    try:
        corpysprofiling.corpora_compare(corpus1, corpus2, metric="other")
        # ValueError not raised
        assert False, "ValueError not raised. metric must be cosine_similarity or euclidean"
    except ValueError:
        # ValueError raised as expected
        pass

def test_corpora_best_match():
    """ Test corpora_best_match function"""
    
    refDoc = "kitten meows"
    corpora = ["kitten meows", "ice cream is yummy", "cat meowed", "dog barks"]
    testCase1 = corpysprofiling.corpora_best_match(refDoc, corpora)

    assert isinstance(testCase1, pd.DataFrame), "Return type is not pd.DataFrame"
    assert testCase1.shape == (len(corpora), 2), "Shape returned does not match input length of corpora"
    assert Counter(testCase1.corpora.to_list()) == Counter(corpora), "Rows in the DataFrame returned do not match elements of corpora"
    assert testCase1.metric.dtype == np.float64, "Distances are not of type np.float64"
    # Make sure all distances are between 0 and 1, inclusive (True for cosine_similarity)
    assert testCase1.metric.between(0, 1).all(), "Distances should be between 0 and 1 inclusive for cosine_similarity"
    # Make sure that distances are sorted in ascending order
    assert testCase1.metric.is_monotonic_increasing, "Distances should be sorted in ascending order"

    testCase2 = corpysprofiling.corpora_best_match(refDoc, corpora, metric="euclidean")

    assert isinstance(testCase2, pd.DataFrame), "Return type is not pd.DataFrame"
    assert testCase2.shape == (len(corpora), 2), "Shape returned does not match input length of corpora"
    assert Counter(testCase2.corpora.to_list()) == Counter(corpora), "Rows in the DataFrame returned do not match elements of corpora"
    assert testCase2.metric.dtype == np.float64, "Distances are not of type np.float64"
    # Make sure all distances are greater than or equal to 0 (True for euclidean)
    assert (testCase2.metric >= 0).all(), "Distances should be greater than or equal to 0 for euclidean"
    # Make sure that distances are sorted in ascending order
    assert testCase2.metric.is_monotonic_increasing, "Distances should be sorted in ascending order"
    
    testCase3 = corpysprofiling.corpora_best_match(refDoc, list())
    assert isinstance(testCase3, pd.DataFrame), "Return type is not pd.DataFrame"
    assert testCase3.shape == (0, 2), "There should be 0 rows when corpora is empty list"

    try:
        corpysprofiling.corpora_best_match(None, None)
        # TypeError not raised
        assert False, "TypeError not raied. corpora_best_match should not accept NoneType inputs"
    except TypeError:
        # TypeError raised as expected
        pass

    try:
        corpysprofiling.corpora_best_match(refDoc, corpora, metric="other")
        # ValueError not raised
        assert False, "ValueError not raied. corpora_best_match should not accept NoneType inputs"
    except ValueError:
        # ValueError raised as expected
        pass
    
