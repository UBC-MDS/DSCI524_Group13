from corpysprofiling import __version__
from corpysprofiling import corpysprofiling
import numpy as np
import pandas as pd

def test_version():
    assert __version__ == '0.1.0'

def test_corpora_compare():
    """ Test corpora_compare function"""
    corpus1 = "Test1"
    corpus2 = "Test1"

    # test output type is float
    assert (isinstance(corpysprofiling.corpora_compare(corpus1, corpus2, metric="cosine_similarity"), np.float64) == True)

    # test identical corpora return a score of 0.0 - cosine_similarity
    assert (corpysprofiling.corpora_compare(corpus1, corpus2, metric="cosine_similarity") == 0.0)

    # test identical corpora return a score of 0.0 - euclidean
    assert (corpysprofiling.corpora_compare(corpus1, corpus2, metric="euclidean") == 0.0)

def test_corpora_best_match():
    """ Test corpora_best_match function"""
    
    refDoc1 = "kitten meows"
    corpora1 = ["kitten meows", "ice cream is yummy", "cat meowed", "dog barks"]
    testCase1 = corpysprofiling.corpora_best_match(refDoc1, corpora1)

    assert isinstance(testCase1, pd.DataFrame), "Return type is not pd.DataFrame"
    assert testCase1.shape == (len(corpora1), 2), "Shape returned does not match input length of corpora"
    assert testCase1.corpora.to_list() == corpora1, "Rows in the DataFrame returned do not match elements of corpora"
    assert (testCase1.metric.dtype == np.float64).all(), "Distances are not of type np.float64"
    # Make sure all distances are between 0 and 1, inclusive (True for cosine_similarity)
    assert testCase1.metric.between(0, 1).all(), "Distances should be between 0 and 1 inclusive for cosine_similarity"
    # Make sure that distances are sorted in ascending order
    assert testCase1.metric.is_monotonic_increasing(), "Distances should be sorted in ascending order"

    testCase2 = corpysprofiling.corpora_best_match(refDoc1, corpora1, metric="euclidean")

    assert isinstance(testCase1, pd.DataFrame), "Return type is not pd.DataFrame"
    assert testCase1.shape == (len(corpora1), 2), "Shape returned does not match input length of corpora"
    assert testCase1.corpora.to_list() == corpora1, "Rows in the DataFrame returned do not match elements of corpora"
    assert (testCase1.metric.dtype == np.float64).all(), "Distances are not of type np.float64"
    # Make sure all distances are greater than or equal to 0 (True for euclidean)
    assert (testCase1.metric >= 0).all(), "Distances should be greater than or equal to 0 for euclidean"
    # Make sure that distances are sorted in ascending order
    assert testCase1.metric.is_monotonic_increasing(), "Distances should be sorted in ascending order"
    
    testCase3 = corpysprofiling.corpora_best_match(refDoc1, list())
    assert isinstance(testCase1, pd.DataFrame), "Return type is not pd.DataFrame"
    assert testCase1.shape == (0, 2), "There should be 0 rows when corpora is empty list"

    try:
        corpysprofiling.corpora_best_match(None, None)
        # ValueError not raised
        assert False, "ValueError not raied. corpora_best_match should not accept NoneType inputs"
    except ValueError:
        # ValueError raised as expected
        pass

    