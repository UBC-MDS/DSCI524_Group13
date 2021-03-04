from corpysprofiling import __version__
from corpysprofiling import corpysprofiling

def test_version():
    assert __version__ == '0.1.0'

def test_corpora_compare():
    """ Test corpora_compare function"""
    corpus1 = "Test1"
    corpus2 = "Test1"

    # test output type is float
    assertTrue (is_float(corpysprofiling.corpora_compare(corpus1, corpus2)))

    # test identical corpora return a score of 0.0
    assert (corpysprofiling.corpora_compare(corpus1, corpus2) == 0.0)