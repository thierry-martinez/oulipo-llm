"""Test for oulipo-style text generator."""

import itertools

import torch

from oulipo_llm import (
    LLM,
    ChoiceInternal,
    ChoiceLeaf,
    ChoiceNode,
    answer_prompt,
    constraint_no_e,
)


def test_prompt_to_tokens(fx_token: str, fx_model_name: str) -> None:
    """Test LLM.prompt_to_tokens."""
    llm = LLM(fx_model_name, fx_token)
    prompt = "Be or not to be."
    generated = llm.prompt_to_tokens(prompt)
    assert isinstance(generated, torch.Tensor)


def test_best_next_tokens(fx_token: str, fx_model_name: str) -> None:
    """Test LLM.best_next_tokens."""
    llm = LLM(fx_model_name, fx_token)
    prompt = "Be or not to be."
    generated = llm.prompt_to_tokens(prompt)
    next_tokens = llm.best_next_tokens([], generated, constraint_no_e)
    token_count = 3
    three_next_tokens = list(itertools.islice(next_tokens, token_count))
    assert len(three_next_tokens) == token_count
    assert all(constraint_no_e([], branch.token) for branch in three_next_tokens)


def test_choice_tree(fx_token: str, fx_model_name: str) -> None:
    """Test ChoiceNode.grow."""
    llm = LLM(fx_model_name, fx_token)
    prompt = "Be or not to be."
    generated = llm.prompt_to_tokens(prompt)
    tree: ChoiceNode = ChoiceLeaf([], generated)
    for _ in range(2):
        tree = tree.grow(llm, 2, constraint_no_e)
    assert isinstance(tree, ChoiceInternal)
    assert all(
        isinstance(branch.node, ChoiceInternal)
        and all(
            isinstance(subbranch.node, ChoiceLeaf) for subbranch in branch.node.children
        )
        for branch in tree.children
    )


def test_best_branch(fx_token: str, fx_model_name: str) -> None:
    """Test ChoiceInternal.best_branch."""
    llm = LLM(fx_model_name, fx_token)
    prompt = "Be or not to be."
    generated = llm.prompt_to_tokens(prompt)
    tree: ChoiceNode = ChoiceLeaf([], generated)
    for _ in range(2):
        tree = tree.grow(llm, 2, constraint_no_e)
    for _ in range(2):
        assert isinstance(tree, ChoiceInternal)
        _prob, branch = tree.best_branch()
        assert constraint_no_e([], branch.token)
        tree = branch.node.grow(llm, 2, constraint_no_e)


def test_answer_prompt(fx_token: str, fx_model_name: str) -> None:
    """Test answer_prompt."""
    llm = LLM(fx_model_name, fx_token)
    prompt = "Be or not to be."
    answer = answer_prompt(llm, prompt, 2, 2, constraint_no_e)
    token_count = 3
    answer_list = list(itertools.islice(answer, token_count))
    assert len(answer_list) == token_count
    for token in answer_list:
        assert constraint_no_e([], token)
