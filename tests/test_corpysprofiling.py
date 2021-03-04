from corpysprofiling import __version__
from corpysprofiling import corpysprofiling
import numpy as np

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