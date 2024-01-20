import json
import os

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import (
    AutoConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments,
)

from .utils import tokenize


class InRankerTrainer:
    def __init__(
        self,
        model: str = "t5-small",
        batch_size: int = 32,
        warmup_steps: int = 20000,
        num_train_epochs: float = 1.0,
        logging_steps: int = 1000,
        save_steps: int = 30000,
        learning_rate: float = 7e-5,
        gradient_accumulation_steps: int = 1,
        bf16: bool = False,
        gradient_checkpointing: bool = False,
        device: str = None,
        **kwargs,
    ):
        """
        Initialize the trainer.
        Args:
            model: Pretrained T5 model.
            batch_size: Batch size.
            warmup_steps: Warmup steps used in the optimizer.
            num_train_epochs: Number of training epochs.
            logging_steps: Logging steps.
            save_steps: Save steps.
            learning_rate: Learning rate.
            gradient_accumulation_steps: Gradient accumulation steps.
            bf16: Whether to use bfloat16 (might not be available for all GPUs).
            gradient_checkpointing: Whether to use gradient checkpointing.
            device: Device to use (e.g. "cuda:0", "cpu").
            **kwargs: Additional arguments to be passed to the Huggingface Trainer.
        """
        # Model
        self.model = model
        # Training Arguments
        self.training_arguments = TrainingArguments()
        self.training_arguments.warmup_steps = warmup_steps
        self.training_arguments.num_train_epochs = num_train_epochs
        self.training_arguments.logging_steps = logging_steps
        self.training_arguments.save_steps = save_steps
        self.training_arguments.learning_rate = learning_rate
        self.training_arguments.per_device_train_batch_size = batch_size
        self.training_arguments.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.training_arguments.bf16 = bf16
        self.training_arguments.gradient_checkpointing = gradient_checkpointing
        # This is required to allow on-the-fly transformation on the dataset
        self.training_arguments.remove_unused_columns = False

        if device is not None:
            self.training_arguments.device = device
        else:
            self.training_arguments.device = (
                "cuda:0" if torch.cuda.is_available() else "cpu"
            )
        print(f"Using device: {self.training_arguments.device}")

        for key, value in kwargs.items():
            setattr(self.training_arguments, key, value)

        print(f"Loading tokenizer: {self.model}...")
        self.tokenizer = T5Tokenizer.from_pretrained(self.model)
        print("Tokenizer loaded.")
        self.config = AutoConfig.from_pretrained(self.model)
        self.config.num_labels = 1
        self.config.problem_type = "regression"

        print(f"Loading model: {self.model}...")
        model = T5ForConditionalGeneration.from_pretrained(
            self.model, config=self.config
        )
        print("Model loaded.")
        self.total_training_e

    def load_msmarco_dataset(
        self,
        msmarco_tsv_file: str,
        language: str = "english",
        max_length: int = 512,
    ):
        """
        Load the MS-MARCO dataset from the distillation file.
        Args:
            msmarco_tsv_file: Path to the MS-MARCO TSV file.
            language: Language of the dataset.
            max_length: Maximum length of the input sequence (tokens).
        Expected format:
            Each line is a tab-separated list containing the following fields:
                - positive_score: float containing the positive score.
                - negative_score: float containing the negative score.
                - query_id: int containing the query id.
                - positive_id: int containing the positive document id.
                - negative_id: int containing the negative document id.
        """
        assert os.path.isfile(
            msmarco_tsv_file
        ), f"Distill file {msmarco_tsv_file} does not exist."
        assert msmarco_tsv_file.endswith(
            ".tsv"
        ), f"Distill file {msmarco_tsv_file} is not a tsv file."

        corpus = load_dataset("unicamp-dl/mmarco", f"collection-{language}")[
            "collection"
        ]

        queries_dataset = load_dataset("unicamp-dl/mmarco", f"queries-{language}")[
            "train"
        ]
        queries = {item["id"]: item["text"] for item in queries_dataset}

        training_data = {"query": [], "text": [], "label": []}
        with open(msmarco_tsv_file, "r", encoding="utf8") as fin:
            for line in fin:
                (
                    positive_score,
                    negative_score,
                    query_id,
                    positive_id,
                    negative_id,
                ) = line.strip().split("\t")
            query_id, positive_id, negative_id = (
                int(query_id),
                int(positive_id),
                int(negative_id),
            )
            positive_label = float(positive_score)
            negative_label = float(negative_score)
            training_data["query"].append(query_id)
            training_data["text"].append(positive_id)
            training_data["label"].append(positive_label)
            training_data["query"].append(query_id)
            training_data["text"].append(negative_id)
            training_data["label"].append(negative_label)

        train_dataset = Dataset.from_dict(training_data)
        train_dataset.set_transform(
            lambda batch: tokenize(
                batch,
                self.tokenizer,
                max_length=max_length,
                from_msmarco=True,
                msmarco_queries=queries,
                msmarco_corpus=corpus,
            )
        )
        return train_dataset

    def load_custom_dataset(
        self,
        distill_file: str,
        max_length: int = 512,
    ):
        """
        Load the dataset from the distillation file.
        Args:
            distill_file: Path to the distillation file.
        Expected format:
            Each line is a JSON object with the following fields:
                - query: Query text.
                - contents: string containing the contents of the document.
                - true_logit: float containing the true logit.
                - false_logit: float containing the false logit.
        """
        assert os.path.isfile(
            distill_file
        ), f"Distill file {distill_file} does not exist."
        assert distill_file.endswith(
            ".jsonl"
        ), f"Distill file {distill_file} is not a jsonl file."

        training_data = {"query": [], "text": [], "label": []}
        with open("distill_file", "r", encoding="utf8") as fin:
            for line in fin:
                try:
                    data = json.loads(line)
                except Exception as e:
                    print(f"Error while parsing line: {line}. Error: {e}")
                    continue
                training_data["query"].append(data["query"])
                training_data["text"].append(data["contents"])
                # Zero-center the logits
                false_logit = float(data["false_logit"])
                true_logit = float(data["true_logit"])
                average = (false_logit + true_logit) / 2
                false_logit -= average
                true_logit -= average
                training_data["label"].append([false_logit, true_logit])

        train_dataset = Dataset.from_dict(training_data)
        train_dataset.set_transform(
            lambda batch: tokenize(batch, self.tokenizer, max_length=max_length)
        )
        return train_dataset

    def train(self, train_dataset, output_dir: str, resume_training=False):
        """
        Train the model.
        Args:
            train_dataset: Training dataset.
            output_dir: Output directory.
            resume_training: Whether to resume training from a checkpoint.
        """
        trainer = T5Trainer(
            self.model,
            self.training_arguments,
            train_dataset=train_dataset,
            tokenizer=self.tokenizer,
        )
        train_result = trainer.train(resume_from_checkpoint=resume_training)
        trainer.save_model(output_dir)
        trainer.save_state(output_dir)
        return train_result

    def tokenize(
        batch,
        tokenizer,
        max_length: int = 512,
        from_msmarco: bool = False,
        msmarco_queries=None,
        msmarco_corpus=None,
    ):
        """
        Tokenize a batch of data (queries and documents) using a tokenizer.
        Args:
            batch: A batch of data.
            tokenizer: A HuggingFace tokenizer.
            max_length: The maximum length of the input sequence (tokens).
            from_msmarco:
            msmarco_queries:
            msmarco_corpus:
        """
        queries_documents = []
        for query, document in zip(batch["query"], batch["text"]):
            # msmarco queries and documents are passed as an ID
            query = msmarco_queries[query] if from_msmarco else query
            document = msmarco_corpus[document] if from_msmarco else document
            queries_documents.append(f"Query: {query} Document: {document} Relevant:")

        tokenized = tokenizer(
            queries_documents,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            max_length=max_length,
        )
        tokenized["labels"] = [[label] for label in batch["label"]]
        return tokenized


# Huggingface's trainer
class T5Trainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        token_false, token_true = ["▁false", "▁true"]
        self.token_false_id = self.tokenizer.get_vocab()[token_false]
        self.token_true_id = self.tokenizer.get_vocab()[token_true]
        print(f"False token: {token_false} ({self.token_false_id})")
        print(f"True token: {token_true} ({self.token_true_id})")

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        labels = np.squeeze(inputs.pop("labels"))
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        model_inputs = model.prepare_inputs_for_generation(
            decode_ids,
            encoder_outputs=encoder_outputs,
            past=None,
            attention_mask=attention_mask,
            use_cache=True,
        )
        outputs = model(**model_inputs)
        logits = outputs[0][:, :, [self.token_false_id, self.token_true_id]][:, 0]
        loss = torch.nn.MSELoss()(logits, labels)

        return (loss, outputs) if return_outputs else loss