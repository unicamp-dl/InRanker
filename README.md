# InRanker

This library provides a simple way for testing the InRanker models, which are also directly accessible at [Hugging Face](https://huggingface.co/unicamp-dl)

## Setup

This library was tested using python 3.10 and can be installed using pip.

```bash
pip install inranker
```

## Usage

The library provides a simple interface to rank documents given a query. The following example shows how to rank documents using the InRanker-small model.
There are three sizes available:

- [unicamp-dl/InRanker-small](https://huggingface.co/unicamp-dl/InRanker-small)
- [unicamp-dl/InRanker-base](https://huggingface.co/unicamp-dl/InRanker-base)
- [unicamp-dl/InRanker-3B](https://huggingface.co/unicamp-dl/InRanker-3B)

```python
from inranker import T5Ranker

model = T5Ranker(model_name_or_path="unicamp-dl/InRanker-small")

docs = [
    "The capital of France is Paris",
    "Learn deep learning with InRanker and transformers"
]
scores = model.get_scores(
    query="What is the best way to learn deep learning?",
    docs=docs
)
# Scores are sorted in descending order (most relevant to least)
# scores -> [0, 1]
sorted_scores = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

""" InRanker-small:
sorted_scores = [
    (0.4844, 'Learn deep learning with InRanker and transformers'),
    (7.83e-06, 'The capital of France is Paris')
]
"""
```

## Docs

You can find more information about the library, including details on how to train the model and generate soft labels for custom datasets, in the [docs](docs/) folder.

## How to Cite

```
@misc{laitz2024inranker,
      title={InRanker: Distilled Rankers for Zero-shot Information Retrieval},
      author={Thiago Laitz and Konstantinos Papakostas and Roberto Lotufo and Rodrigo Nogueira},
      year={2024},
      eprint={2401.06910},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```
