"""Generate oulipo-style texts with LLM."""

from __future__ import annotations

import itertools
import os
import sys
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import override

import torch
import typer
from transformers import AutoModelForCausalLM, AutoTokenizer

Constraint = Callable[[Sequence[str], str], bool]


@dataclass(frozen=True)
class TokenBranch[Node]:
    """Branch of a token tree."""

    prob: float
    token: str
    node: Node

    def replace[B](self, new_node: B) -> TokenBranch[B]:
        """Return a new branch where node is replaced by another value."""
        return TokenBranch(self.prob, self.token, new_node)


class LLM:
    """Host a large language model."""

    def __init__(
        self,
        model_name: str,
        token: str | None,
        temperature: float = 0.8,
    ) -> None:
        """Initialize a large language model."""
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
        )
        self._temperature = temperature

    def prompt_to_tokens(self, prompt: str) -> torch.Tensor:
        """Return the state after reading the given prompt."""
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        return inputs["input_ids"]

    def best_next_tokens(
        self,
        prefix_tokens: Sequence[str],
        prefix_tensor: torch.Tensor,
        constraint: Constraint,
    ) -> Iterator[TokenBranch[torch.Tensor]]:
        """Return the next best tokens, in descending order."""
        # Forward pass
        with torch.no_grad():
            outputs = self._model(input_ids=prefix_tensor)
            logits = outputs.logits  # shape: [batch, seq_len, vocab_size]

        # Get logits for the last token position
        next_token_logits = logits[0, -1, :]

        # Apply temperature
        next_token_logits = next_token_logits / self._temperature

        # Convert to probabilities
        probs = torch.softmax(next_token_logits, dim=-1)

        sorted_vals, sorted_indices = torch.sort(probs, descending=True)

        for prob, token_id in zip(sorted_vals, sorted_indices):
            token_str = self._tokenizer.decode([token_id])
            if constraint(prefix_tokens, token_str):
                chosen_id = token_id.unsqueeze(0).unsqueeze(0)
                generated = torch.cat((prefix_tensor, chosen_id), dim=1)
                yield TokenBranch(prob, token_str, generated)


class ChoiceNode(ABC):
    """Node of a token tree."""

    @abstractmethod
    def grow(
        self,
        llm: LLM,
        lookup_count: int,
        constraint: Constraint,
    ) -> ChoiceInternal:
        """
        Return a tree that is one level higher.

        Every leaf is replaced by an additionnal level of branches.
        """


@dataclass
class ChoiceInternal(ChoiceNode):
    """Internal node of a token tree."""

    children: Sequence[TokenBranch[ChoiceNode]]

    @override
    def grow(
        self,
        llm: LLM,
        lookup_count: int,
        constraint: Constraint,
    ) -> ChoiceInternal:
        children = [
            branch.replace(branch.node.grow(llm, lookup_count, constraint))
            for branch in self.children
        ]
        return ChoiceInternal(children)

    def best_branch(self) -> tuple[float, TokenBranch[ChoiceNode]]:
        """Return the best branch."""
        result = None
        for branch in self.children:
            prob = (
                branch.prob * branch.node.best_branch()[0]
                if isinstance(branch.node, ChoiceInternal)
                else branch.prob
            )
            if result is None or result[0] < prob:
                result = (prob, branch)
        if result is None:
            msg = "No such branch"
            raise ValueError(msg)
        return result


@dataclass
class ChoiceLeaf(ChoiceNode):
    """Leaf of a token tree."""

    tokens: list[str]
    tensor: torch.Tensor

    @override
    def grow(
        self,
        llm: LLM,
        lookup_count: int,
        constraint: Constraint,
    ) -> ChoiceInternal:
        next_tokens = llm.best_next_tokens(self.tokens, self.tensor, constraint)
        list_next_tokens = list(itertools.islice(next_tokens, lookup_count))
        children = [
            branch.replace(ChoiceLeaf([*self.tokens, branch.token], branch.node))
            for branch in list_next_tokens
        ]
        return ChoiceInternal(children)


def answer_prompt(
    llm: LLM,
    prompt: str,
    lookup_count: int,
    lookup_depth: int,
    constraint: Constraint,
) -> Iterator[str]:
    """Return the answer of a given prompt."""
    generated = llm.prompt_to_tokens(prompt)
    tree: ChoiceNode = ChoiceLeaf([], generated)
    for _ in range(lookup_depth):
        tree = tree.grow(llm, lookup_count, constraint)
    while True:
        if not isinstance(tree, ChoiceInternal):
            msg = "Internal node expected"
            raise TypeError(msg)
        _prob, branch = tree.best_branch()
        yield branch.token
        tree = branch.node.grow(llm, lookup_count, constraint)


def constraint_no_e(_sequence: Sequence[str], token: str) -> bool:
    """Check that a token contains no 'e'."""
    return not any(letter in token for letter in "eéèëEÉÈêÊË")


def main(prompt: str, model_name: str, lookup_count: int, lookup_depth: int) -> None:
    """Entry-point."""
    token = os.environ.get("HF_TOKEN")
    llm = LLM(model_name, token)
    answer = answer_prompt(llm, prompt, lookup_count, lookup_depth, constraint_no_e)
    while True:
        sys.stdout.write(next(answer))
        sys.stdout.flush()


if __name__ == "__main__":
    typer.run(main)
