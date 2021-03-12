# coRPysprofiling 

![](https://github.com/UBC-MDS/DSCI524_Grp13_coRPysprofiling/workflows/build/badge.svg) [![codecov](https://codecov.io/gh/UBC-MDS/corpysprofiling/branch/main/graph/badge.svg)](https://codecov.io/gh/UBC-MDS/coRPysprofiling) [![Deploy](https://github.com/UBC-MDS/coRPysprofiling/actions/workflows/deploy.yml/badge.svg)](https://github.com/UBC-MDS/coRPysprofiling/actions/workflows/deploy.yml) [![Documentation Status](https://readthedocs.org/projects/corpysprofiling/badge/?version=latest)](https://corpysprofiling.readthedocs.io/en/latest/?badge=latest)

Package  for EDA and EDV on text

## Summary

coRPysprofiling is an open-source library designed to bring exploratory data analysis and visualization to the domain of natural language processing. Functions in the package will be used to provide some elementary statistics and visualizations for a single text corpus or provide functions to compare multiple corpora with each other.

## Installation

```bash
$ pip install -i https://test.pypi.org/simple/ corpysprofiling
```

## Features

Some specific functions include:

- `corpus_analysis`: `corpus analysis` will generate a statistical report about the characteristics of a single corpus (e.g. unique word count, average word/sentence length, top words used, topic analysis).
- `corpus_viz`: `corpus_viz` will generate relevant visualizations of a single corpus (e.g. word cloud, histograms for average word/sentence length, top words used).
- `corpora_compare`: Given two or more corpora, `corpora_compare` will find similarity (e.g, Euclidean distance or cosine similarity) between each pair of corpora.
- `corpora_best_match`: Given a reference document and two or more corpora, `corpora_best_match` will rank the corpora in the order of most relevance to the reference document.

## Relevance to the Python Ecosystem

To our knowledge, while [`wordcloud`](https://pypi.org/project/wordcloud/) library generates wordcloud visualization for a given corpus, there is no general-purpose library for exploratory analysis and visualization of a text corpus in the Python ecosystem. There are several advanced libraries for comparing similarities between different corpora: most notably, [`gensim`](https://pypi.org/project/gensim/) provides similarity comparison between large corpora using word embeddings. We believe that coRPysprofiling will provide some useful functionality for exploratory analysis and visualization and help bridge the gap between elementary text analysis to more sophisticated approaches utilizing word embeddings.

## Dependencies
- python = "^3.8"
- pandas = "^1.2.3"
- nltk = "^3.5"
- sentence-transformers = "^0.4.1"
- numpy = "^1.20.1"
- matplotlib = "^3.3.4"
- wordcloud = "^1.8.1"
- altair = "^4.1.0"
## Usage

- TODO

## Documentation

The official documentation is hosted on Read the Docs: https://corpysprofiling.readthedocs.io/en/latest/

## Contributors

We welcome and recognize all contributions. You can see a list of current contributors in the [contributors tab](https://github.com/UBC-MDS/DSCI524_Grp13_coRPysprofiling/graphs/contributors).

### Development Team
| Name  | GitHub ID |
| ----- | ----- |
| Anita Li | [AnitaLi-0371](https://github.com/AnitaLi-0371) |
| Elanor Boyle-Stanley | [eboylestanley](https://github.com/eboylestanley) |
| Ivy Zhang | [ssyayayy](https://github.com/ssyayayy) |
| Junghoo Kim | [jkim222383](https://github.com/jkim222383) |

### Credits

This package was created with Cookiecutter and the UBC-MDS/cookiecutter-ubc-mds project template, modified from the [pyOpenSci/cookiecutter-pyopensci](https://github.com/pyOpenSci/cookiecutter-pyopensci) project template and the [audreyr/cookiecutter-pypackage](https://github.com/audreyr/cookiecutter-pypackage).
