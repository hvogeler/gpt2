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


def test_check_prime_with_prime_number() -> None:
    """Test that check_prime returns True for a prime number"""
    from gpt2.example import check_prime

    assert check_prime(223) is True


def test_check_prime_with_non_prime_number() -> None:
    """Test that check_prime returns False for a non-prime number"""
    from gpt2.example import check_prime

    assert check_prime(100) is False


def test_check_prime_with_edge_cases() -> None:
    """Test that check_prime returns False for edge cases"""
    from gpt2.example import check_prime

    assert check_prime(1) is False
    assert check_prime(0) is False
    assert check_prime(-5) is False
