"""Test the example methods"""

from gpt2.example import echo


def test_echo_returns_same_message() -> None:
    """Test that echo returns the same message"""
    msg = "Hello, World!"
    assert echo(msg) == msg

def test_echo_empty_string() -> None:
    """Test that echo returns an empty string when given an empty string"""
    assert echo("") == ""

def test_echo_preserves_whitespace_and_unicode() -> None:
    msg = "  héllo wörld  \n"
    assert echo(msg) == msg
