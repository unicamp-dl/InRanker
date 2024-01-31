# Training

This guide provides the steps for training the InRanker model using the library. Details about the model and datasets are found in the [paper](https://arxiv.org/abs/2401.06910).

## Datasets

Training the InRanker models involves two distillation steps:

- First, we start by training a T5 model using the MS Marco soft labels: [link](https://huggingface.co/datasets/unicamp-dl/InRanker-msmarco)
- Then, we finetune the model using the BEIR soft labels: [link](https://huggingface.co/datasets/unicamp-dl/InRanker-beir)

## Training Steps

- Create a conda environment with python 3.10: `conda create -n inranker python=3.10`
- Install the library: `pip install inranker`
- Distill the model on MS Marco:

```python
from inranker import InRankerTrainer

trainer = InRankerTrainer(
    model="t5-small",
    batch_size=32,
    gradient_accumulation_steps=1,
    bf16=True, # If you have a GPU with BF16 support
    output_dir="trained_model",
    save_steps=40000,
)

train_dataset = trainer.load_msmarco_dataset()

trainer.train(train_dataset=train_dataset)
```

- Finetune the model on BEIR:

```python
from inranker import InRankerTrainer

trainer = InRankerTrainer(
    model="my-msmarco-trained-model",
    batch_size=32,
    gradient_accumulation_steps=1,
    bf16=True, # If you have a GPU with BF16 support
    output_dir="trained_model",
    save_steps=40000,
)

# If you don't pass the dataset, the trainer will automatically download it
train_dataset = trainer.load_custom_dataset(
    distill_file=None # None to use the default BEIR distillation file
)

trainer.train(train_dataset=train_dataset)
```

## Custom Dataset

If you want to finetune the model using a custom dataset, you can use the `load_custom_dataset` method. The dataset must be a .jsonl file with the following format for each line:

```json
{
  "query": "your query",
  "contents": "your document",
  "true_logit": 23.2,
  "false_logit": 5.0
}
```

In the paper, we used a dataset with 10.5M query-documents using 1.05M synthetic queries generated using [InPars-v1](https://github.com/zetaalphavector/InPars). We also included 9 "negative" documents for each query which were sampled from the same collection from the top-1000 documents retrieved by BM25 (using the [pyserini implementation](https://github.com/castorini/pyserini)).

For generating the teacher soft labels, simply set a flag return_logits on Inranker, and it will return the scores and the logits for each query-document pair (negative_logit, positive_logit):

```python
from inranker import T5Ranker

model = T5Ranker(
    model_name_or_path="castorini/monot5-3b-msmarco-10k",
    fp8=True
)

docs = [
    "The capital of France is Paris",
    "Learn deep learning with InRanker and transformers"
]

scores, logits = model.get_scores(
    query="What is the best way to learn deep learning?",
    docs=docs,
    return_logits=True
)

# Logits are a list with [false_logit, true_logit]
```
