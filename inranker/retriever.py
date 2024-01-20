from abc import ABC, abstractmethod
from typing import List

import torch


# This abstract should be used for all retrievers
class Retriever(ABC):
    def __init__(
        self,
        fp16: bool = False,
        bf16: bool = False,
        batch_size: int = 32,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        silent: bool = True,
    ):
        """
        Retriever is an abstract class for a retriever model.
        Args:
            fp16: Whether to use fp16 precision.
            bf16: Whether to use bf16 precision.
            batch_size: The batch size to use when encoding.
            device: The device to use for inference ("cpu" or "cuda").
            silent: Whether to show progress bars.
        """
        assert device in ["cpu", "cuda"], "Device must be 'cpu' or 'cuda'."
        assert not (fp16 and bf16), "Cannot use both fp16 and bf16."
        assert not (fp16 and device == "cpu"), "Cannot use fp16 on CPU."
        assert not (bf16 and device == "cpu"), "Cannot use bf16 on CPU."
        self.device = device
        if bf16:
            self.precision = torch.bfloat16
            self.info("WARNING: Be aware that bfloat16 is not supported on all GPUs.")
        elif fp16:
            self.precision = torch.float16
        else:
            self.precision = torch.float32

        if not silent:
            self.info(f"INFO: Using {self.device} with {self.precision} precision.")

        self.batch_size = batch_size
        self.silent = silent

    @abstractmethod
    def get_scores(self, query: str, docs: List[str]) -> List[float]:
        """
        This method is called to get a score for each document given a query.
        """
        pass

    @staticmethod
    def chunks(l, n):  # noqa: E741
        """
        This method is used to split a list l into chunks of batch size n.
        """
        for i in range(0, len(l), n):
            yield l[i : i + n]

    def info(self, message):
        """
        This method is used to show warnings.
        """
        if not self.silent:
            print(message)
