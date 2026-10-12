"""Test gpt2.main module."""

import pytest

import gpt2.main


def test_main_logs_startup_message(caplog: pytest.LogCaptureFixture) -> None:
    """Test that main() logs a startup message including the module's file path."""
    with caplog.at_level("INFO"):
        gpt2.main.main()

    assert gpt2.main.__file__ in caplog.text
    assert "Starting the main function" in caplog.text
