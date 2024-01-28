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

from .utils import InRankerDatasets, download_file_from_huggingface


class InRankerTrainer:
    def __init__(
        self,
        model: str = "t5-small",
        batch_size: int = 32,
        warmup_steps: int = 20000,
        num_train_epochs: float = 1.0,
        logging_steps: int = 1000,
        save_steps: int = 40000,
        learning_rate: float = 7e-5,
        gradient_accumulation_steps: int = 1,
        bf16: bool = False,
        gradient_checkpointing: bool = False,
        device: str = "",
        output_dir: str = "trained_model",
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
        # Training Arguments
        self.training_arguments = TrainingArguments(output_dir=output_dir)
        self.training_arguments.warmup_steps = warmup_steps
        self.training_arguments.num_train_epochs = num_train_epochs
        self.training_arguments.logging_steps = logging_steps
        self.training_arguments.save_steps = save_steps
        self.training_arguments.learning_rate = learning_rate
        self.training_arguments.per_device_train_batch_size = batch_size
        self.training_arguments._n_gpu = 1 if "cuda" in device else 0
        self.training_arguments.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.training_arguments.dataloader_num_workers = 4

        self.training_arguments.bf16 = bf16
        self.training_arguments.gradient_checkpointing = gradient_checkpointing
        # This is required to allow on-the-fly transformation on the dataset
        self.training_arguments.remove_unused_columns = False
        self.training_arguments.do_eval = False

        print(f"Using device: {self.training_arguments.device}")

        for key, value in kwargs.items():
            setattr(self.training_arguments, key, value)

        print(f"Loading tokenizer: {model}...")
        self.tokenizer = T5Tokenizer.from_pretrained(model)
        print("Tokenizer loaded.")
        self.config = AutoConfig.from_pretrained(model)
        self.config.num_labels = 1
        self.config.problem_type = "regression"

        print(f"Loading model: {model}...")
        self.model = T5ForConditionalGeneration.from_pretrained(
            model, config=self.config
        )
        print("Model loaded.")

    def load_msmarco_dataset(
        self,
        msmarco_tsv_file: str = None,
        language: str = "english",
        max_length: int = 512,
    ):
        """
        Load the MS-MARCO dataset from the distillation file.
        If the file is not provided, it will be downloaded from HuggingFace.
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
        if msmarco_tsv_file:
            assert os.path.isfile(
                msmarco_tsv_file
            ), f"Distill file {msmarco_tsv_file} does not exist."
            assert msmarco_tsv_file.endswith(
                ".tsv"
            ), f"Distill file {msmarco_tsv_file} is not a tsv file."
        else:
            download_file_from_huggingface(
                url=InRankerDatasets.msmarco_url.value,
                destination="msmarco.tsv",
                checksum=InRankerDatasets.msmarco_md5.value,
                desc="Downloading MS MARCO soft labels dataset...",
            )
            msmarco_tsv_file = "msmarco.tsv"

        corpus = load_dataset("unicamp-dl/mmarco", f"collection-{language}")[
            "collection"
        ]

        queries_dataset = load_dataset("unicamp-dl/mmarco", f"queries-{language}")[
            "train"
        ]
        queries = {item["id"]: item["text"] for item in queries_dataset}

        total_examples = 0
        training_data = {"query": [], "text": [], "label": []}
        with open(msmarco_tsv_file, "r", encoding="utf8") as fin:
            for line in fin:
                positive_score, negative_score, query_id, doc_id = line.strip().split(
                    "\t"
                )
                training_data["query"].append(int(query_id))
                training_data["text"].append(int(doc_id))
                training_data["label"].append(
                    [float(negative_score), float(positive_score)]
                )

                total_examples += 1
                if total_examples >= 12800000:
                    break

        train_dataset = Dataset.from_dict(training_data)
        train_dataset.set_transform(
            lambda batch: self.tokenize(
                batch,
                tokenizer=self.tokenizer,
                max_length=max_length,
                from_msmarco=True,
                msmarco_queries=queries,
                msmarco_corpus=corpus,
            )
        )
        return train_dataset

    def load_custom_dataset(
        self,
        distill_file: str = None,
        max_length: int = 512,
    ):
        """
        Load the dataset from the distillation file.
        if the file is not provided, it will be downloaded from HuggingFace.
        Args:
            distill_file: Path to the distillation file.
        Expected format:
            Each line is a JSON object with the following fields:
                - query: Query text.
                - contents: string containing the contents of the document.
                - true_logit: float containing the true logit.
                - false_logit: float containing the false logit.
        """
        if distill_file:
            assert os.path.isfile(
                distill_file
            ), f"Distill file {distill_file} does not exist."
            assert distill_file.endswith(
                ".jsonl"
            ), f"Distill file {distill_file} is not a jsonl file."
        else:
            download_file_from_huggingface(
                url=InRankerDatasets.beir_url.value,
                destination="beir_logits.jsonl",
                checksum=InRankerDatasets.beir_md5.value,
                desc="Downloading BEIR soft labels dataset...",
            )
            distill_file = "beir_logits.jsonl"

        training_data = {"query": [], "text": [], "label": []}
        with open(distill_file, "r", encoding="utf8") as fin:
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
            lambda batch: self.tokenize(batch, self.tokenizer, max_length=max_length)
        )
        return train_dataset

    def train(self, train_dataset, resume_training=False):
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
        trainer.save_model()
        trainer.save_state()
        return train_result

    @staticmethod
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
            from_msmarco: Boolean indicating if the data is from MS MARCO dataset.
            msmarco_queries: The queries from the MS MARCO dataset.
            msmarco_corpus: The corpus from the MS MARCO dataset.
        """
        queries_documents = []
        for query, document in zip(batch["query"], batch["text"]):
            # msmarco queries and documents are passed as an ID
            query = msmarco_queries[query] if from_msmarco else query
            document = msmarco_corpus[document] if from_msmarco else document

            # Tokenize query and document separately
            tokenized_query = tokenizer.encode(query, add_special_tokens=False)
            tokenized_document = tokenizer.encode(document, add_special_tokens=False)

            # Calculate space for "Relevant" tag and adjust max_length accordingly
            relevant_tag_space = (
                10  # for "Relevant", special tokens, and extra safe space
            )
            adjusted_max_length = max_length - relevant_tag_space - len(tokenized_query)

            # Truncate only the document to fit within adjusted_max_length
            if len(tokenized_document) > adjusted_max_length:
                tokenized_document = tokenized_document[:adjusted_max_length]

            # Convert tokens back to text and add "Relevant" tag
            text_query = tokenizer.decode(tokenized_query, skip_special_tokens=True)
            text_document = tokenizer.decode(
                tokenized_document, skip_special_tokens=True
            )
            queries_documents.append(
                f"Query: {text_query} Document: {text_document} Relevant:"
            )

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
