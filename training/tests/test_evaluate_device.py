"""`python -m training.evaluate eval|calibrate --device mlx` is refused."""

import pytest
import typer

from training.evaluate import _device


def test_device_refuses_mlx():
    with pytest.raises(typer.BadParameter):
        _device("mlx")


def test_device_passes_an_explicit_choice_through():
    assert _device("cpu") == "cpu"
