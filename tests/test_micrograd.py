"""Test Micrograd methods"""

from gpt2.micrograd.micrograd import Value


def test_value_initialization() -> None:
    """Test that Value initializes correctly with a float."""
    val: Value = Value(3.14)
    assert val.data == 3.14
    assert repr(val) == "Value(data=3.14)"


def test_value_add() -> None:
    """Test that Value addition works correctly."""
    a: Value = Value(1.0)
    b: Value = Value(2.0)
    result: Value = a + b
    assert isinstance(result, Value)
    assert result.data == a.data + b.data


def test_value_mul() -> None:
    """Test that Value multiplication works correctly."""
    a: Value = Value(7.0)
    b: Value = Value(-3.0)
    result: Value = a * b
    assert isinstance(result, Value)
    assert result.data == a.data * b.data


def test_value_prev() -> None:
    """Test that Value keeps track of its children correctly."""
    a: Value = Value(4.0)
    b: Value = Value(5.0)
    result: Value = a + b
    assert a in result.prev
    assert b in result.prev

    c = Value(6.0)

    d = (a * b) + c
    assert {v.data for v in d.prev} == {6.0, 20.0}
    assert d.op == "+"

    d = a * (b + c)
    assert {v.data for v in d.prev} == {4.0, 11.0}
    assert d.op == "*"
