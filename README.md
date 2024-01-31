# DistillRanker

This library provides a simple way for testing the distillranker models, which are also directly accessible at Hugging Face

## Setup

This library was tested using python 3.10 and can be installed using pip.

```bash
pip install distillranker (anonymous name)
```

## Usage

The library provides a simple interface to rank documents given a query. The following example shows how to rank documents using the distillranker-small model.
There are three sizes available:

- [anonymous/distillranker-small](https://huggingface.co/anonymous/distillranker-small)
- [anonymous/distillranker-base](https://huggingface.co/anonymous/distillranker-base)
- [anonymous/distillranker-3B](https://huggingface.co/anonymous/distillranker-3B)

```python
from distillranker import T5Ranker

model = T5Ranker(model_name_or_path="anonymous/distillranker-small")

docs = [
    "The capital of France is Paris",
    "Learn deep learning with distillranker and transformers"
]
scores = model.get_scores(
    query="What is the best way to learn deep learning?",
    docs=docs
)
# Scores are sorted in descending order (most relevant to least)
# scores -> [0, 1]
sorted_scores = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

""" distillranker-small:
sorted_scores = [
    (0.4844, 'Learn deep learning with distillranker and transformers'),
    (7.83e-06, 'The capital of France is Paris')
]
"""
```

## Docs

You can find more information about the library, including details on how to train the model and generate soft labels for custom datasets, in the [docs](docs/) folder.
