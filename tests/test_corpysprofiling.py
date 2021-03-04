from corpysprofiling import __version__
from corpysprofiling import corpysprofiling
import numpy as np

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