"""Test configuration for oulipo-style text generator."""

from __future__ import annotations

import os

import pytest

token = os.environ.get("HF_TOKEN")
model_name = "gpt2"


@pytest.fixture
def fx_token() -> str | None:
    """Return the HuggingFace token from the environment."""
    return token


@pytest.fixture
def fx_model_name() -> str:
    """Return a model name."""
    return model_name
