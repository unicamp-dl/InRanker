from math import ceil
from typing import List

import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .base import BaseRanker


class T5Ranker(BaseRanker):
    def __init__(self, model_name_or_path: str = "unicamp-dl/InRanker-base", **kwargs):
        """
        MonoT5Ranker is a wrapper for the MonoT5 model for ranking.
        Args:
            model_name_or_path: Path to the MonoT5 model.
        """
        super().__init__(**kwargs)
        model_args = {}

        model_args["torch_dtype"] = self.precision

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name_or_path, **model_args
        ).to(self.device)
        self.model.eval()

        # Get tokenizer and relevance token IDs
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        token_false, token_true = ["▁false", "▁true"]
        self.token_false_id = self.tokenizer.get_vocab()[token_false]
        self.token_true_id = self.tokenizer.get_vocab()[token_true]

        if not self.silent:
            print("Using the model:", model_name_or_path)

    @torch.no_grad()
    def get_scores(
        self,
        query: str,
        docs: List[str],
        max_length: int = 512,
        return_logits: bool = False,
    ) -> List[float]:
        """
        Given a query and a list of documents, return a list of scores.
        Args:
            query: The query string.
            docs: A list of document strings.
            max_length: The maximum length of the input sequence.
            return_logits: Whether to return the logits (false_logit, true_logit) for each document.
        """
        scores = []
        logits = []
        for batch in tqdm(
            self.chunks(docs, self.batch_size),
            disable=self.silent,
            desc="Rescoring",
            total=ceil(len(docs) / self.batch_size),
        ):
            queries_documents = [
                f"Query: {query} Document: {text} Relevant:" for text in batch
            ]
            tokenized = self.tokenizer(
                queries_documents,
                padding=True,
                truncation="longest_first",
                return_tensors="pt",
                max_length=max_length,
            ).to(self.device)
            input_ids = tokenized["input_ids"].to(self.device)
            attention_mask = tokenized["attention_mask"].to(self.device)
            _, batch_scores = self.greedy_decode(
                model=self.model,
                input_ids=input_ids,
                length=1,
                attention_mask=attention_mask,
                return_last_logits=True,
            )
            batch_scores = batch_scores[:, [self.token_false_id, self.token_true_id]]
            logits.extend(batch_scores.tolist())
            batch_scores = torch.log_softmax(batch_scores, dim=-1)
            batch_scores = torch.exp(batch_scores[:, 1])
            batch_scores = batch_scores.tolist()
            scores.extend(batch_scores)
        if return_logits:
            return scores, logits
        return scores

    @torch.no_grad()
    def greedy_decode(
        self,
        model,
        input_ids: torch.Tensor,
        length: int,
        attention_mask: torch.Tensor = None,
        return_last_logits: bool = True,
    ):
        decode_ids = torch.full(
            (input_ids.size(0), 1),
            model.config.decoder_start_token_id,
            dtype=torch.long,
        ).to(input_ids.device)
        encoder_outputs = model.get_encoder()(input_ids, attention_mask=attention_mask)
        next_token_logits = None
        for _ in range(length):
            model_inputs = model.prepare_inputs_for_generation(
                decode_ids,
                encoder_outputs=encoder_outputs,
                past=None,
                attention_mask=attention_mask,
                use_cache=True,
            )
            outputs = model(**model_inputs)  # (batch_size, cur_len, vocab_size)
            next_token_logits = outputs[0][:, -1, :]  # (batch_size, vocab_size)
            decode_ids = torch.cat(
                [decode_ids, next_token_logits.max(1)[1].unsqueeze(-1)], dim=-1
            )
        if return_last_logits:
            return decode_ids, next_token_logits
        return decode_ids
