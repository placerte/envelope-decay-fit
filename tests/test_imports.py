"""Import smoke tests for package modules."""

import importlib


def test_cli_module_import() -> None:
    """Ensure CLI module import does not raise."""
    importlib.import_module("envelope_decay_fit.cli")
