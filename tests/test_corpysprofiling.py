from corpysprofiling import __version__
from corpysprofiling import corpysprofiling
import numpy as np
import pandas as pd
from collections import Counter

def test_version():
    assert __version__ == '0.1.0'

def test_corpora_compare():
    """ Test corpora_compare function"""
    corpus1 = "kitten meows"
    corpus2 = "kitten meows"

    testCase1 = corpysprofiling.corpora_compare(corpus1, corpus2, metric="cosine_similarity")
    testCase2 = corpysprofiling.corpora_compare(corpus1, corpus2, metric="euclidean")

    assert isinstance(testCase1, np.float64), "Return type is not np.float"
    assert np.isclose(testCase1, 0.0, atol=1e-06), "Identical corpora should return score of 0.0"
    assert np.isclose(testCase2, 0.0, atol=1e-06), "Identical corpora should return score of 0.0"
    assert (testCase1 >= 0), "Distances should be between 0 and 1 inclusive for cosine_similarity"
    assert (testCase1 <= 1), "Distances should be between 0 and 1 inclusive for cosine_similarity"

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

    