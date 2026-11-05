"""A large language model, as experiment E8 uses it: probabilities of texts and greedy answers.

Three questions are asked of the model, always in batches:

- ``text_logprobs``: ln P(text | prefix), with the end of the text, for rescoring a list;
- ``choice_logprobs``: ln P of each number as the first token of the answer, for choosing;
- ``generate``: the greedy answer to a conversation, for correcting.

The tests use a fake with the same methods; ``TransformersLLM`` needs the extra ``hf`` and a GPU in
practice.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from asr_lab.correction.prompts import Message


class LanguageModel(Protocol):
    name: str
    config: dict[str, Any]

    def text_logprobs(self, prefix: str, texts: Sequence[str]) -> list[float]:
        """ln P(text, end | prefix) of each text."""

    def choice_logprobs(self, conversations: Sequence[list[Message]], n: int) -> list[list[float]]:
        """ln P of the answers "1" to ``str(n)`` after each conversation."""

    def generate(self, conversations: Sequence[list[Message]], max_new_tokens: int) -> list[str]:
        """The greedy answer to each conversation."""

    def count_tokens(self, text: str) -> int:
        """The number of tokens of a text."""


def batches(indices: Sequence[int], size: int) -> list[list[int]]:
    """Consecutive groups of at most ``size`` indices.

    >>> batches([3, 1, 2], 2)
    [[3, 1], [2]]
    """
    return [list(indices[i : i + size]) for i in range(0, len(indices), size)]


class TransformersLLM:
    """A causal language model of Hugging Face Transformers, at a fixed revision."""

    def __init__(
        self,
        model_id: str,
        revision: str,
        *,
        dtype: str = "bfloat16",
        batch_size: int = 16,
        device: str = "cuda",
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.name = model_id
        self.config = {"revision": revision, "dtype": dtype, "batch_size": batch_size}
        self.batch_size = batch_size
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, revision=revision, dtype=getattr(torch, dtype)
        ).to(device)
        self.model.eval()
        self._end = self.tokenizer.eos_token_id

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer(text, add_special_tokens=False)["input_ids"])

    def text_logprobs(self, prefix: str, texts: Sequence[str]) -> list[float]:
        torch = self._torch
        start = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
        sequences = [
            start + self.tokenizer(text, add_special_tokens=False)["input_ids"] + [self._end]
            for text in texts
        ]
        result = [0.0] * len(texts)
        for batch in batches(_longest_first(sequences), self.batch_size):
            width = max(len(sequences[i]) for i in batch)
            ids = torch.full((len(batch), width), self._end, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for row, i in enumerate(batch):
                ids[row, : len(sequences[i])] = torch.tensor(sequences[i])
                mask[row, : len(sequences[i])] = 1
            with torch.inference_mode():
                logits = self.model(
                    input_ids=ids.to(self.device), attention_mask=mask.to(self.device)
                ).logits.float()
            logp = torch.log_softmax(logits[:, :-1], dim=-1)
            target = logp.gather(2, ids[:, 1:].to(self.device).unsqueeze(-1)).squeeze(-1)
            for row, i in enumerate(batch):
                # The tokens after the prefix, up to and including the end of the text
                result[i] = target[row, len(start) - 1 : len(sequences[i]) - 1].sum().item()
        return result

    def choice_logprobs(self, conversations: Sequence[list[Message]], n: int) -> list[list[float]]:
        torch = self._torch
        choices = [self.tokenizer.convert_tokens_to_ids(str(i)) for i in range(1, n + 1)]
        result: list[list[float]] = [[] for _ in conversations]
        for batch, inputs in self._encoded(conversations):
            with torch.inference_mode():
                # Only the logits of the last position: those of the whole prompt take gigabytes
                logits = self.model(**inputs, logits_to_keep=1).logits[:, -1].float()
            logp = torch.log_softmax(logits, dim=-1)[:, choices]
            for row, i in enumerate(batch):
                result[i] = logp[row].tolist()
        return result

    def generate(self, conversations: Sequence[list[Message]], max_new_tokens: int) -> list[str]:
        torch = self._torch
        result = [""] * len(conversations)
        for batch, inputs in self._encoded(conversations):
            with torch.inference_mode():
                output = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            answers = output[:, inputs["input_ids"].shape[1] :]
            for row, i in enumerate(batch):
                result[i] = self.tokenizer.decode(answers[row], skip_special_tokens=True)
        return result

    def _encoded(self, conversations: Sequence[list[Message]]) -> list[tuple[list[int], Any]]:
        """Batches of conversations in the chat template, padded on the left, longest first."""
        texts = [
            self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            for messages in conversations
        ]
        lengths = [self.count_tokens(text) for text in texts]
        self.tokenizer.padding_side = "left"
        encoded = []
        for batch in batches(_longest_first(lengths), self.batch_size):
            inputs = self.tokenizer(
                [texts[i] for i in batch],
                return_tensors="pt",
                padding=True,
                add_special_tokens=False,
            ).to(self.device)
            encoded.append((batch, inputs))
        return encoded


def _longest_first(items: Sequence[Any]) -> list[int]:
    """Indices from the longest item to the shortest, so that a batch wastes little padding."""
    sizes = [item if isinstance(item, int) else len(item) for item in items]
    return sorted(range(len(items)), key=lambda i: -sizes[i])
