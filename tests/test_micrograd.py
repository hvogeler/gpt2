"""Test Micrograd methods"""

import math

# type: ignore[import-untyped]
from gpt2.micrograd.micrograd import Value, trace


def test_value_initialization() -> None:
    """Test that Value initializes correctly with a float."""
    val: Value = Value(3.14, label="pi")
    assert val.data == 3.14
    assert repr(val) == "Value(data=3.14, label=pi, grad=0.0)"


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


def test_backward() -> None:
    """Test that backward propagation computes gradients correctly."""
    a: Value = Value(2.0, label="a")
    b: Value = Value(3.0, label="b")
    c: Value = a * b  # c = 2 * 3
    d: Value = c + a  # d = (2 * 3) + 2
    d.grad = 1.0  # Set the gradient of the output to 1

    d.backward()  # Perform backward propagation

    assert a.grad == 4.0  # ∂d/∂a = ∂((a*b)+a)/∂a = b + 1 = 3 + 1 = 4
    assert b.grad == 2.0  # ∂d/∂b = ∂((a*b)+a)/∂b = a = 2


def test_backward_multi() -> None:
    """Test that gradients accumulate when the same Value is used multiple times as an input."""
    a: Value = Value(3.0, label="a")
    b = a + a
    b.label = "b"
    b.grad = 1.0  # Set the gradient of the output to 1
    b.backward()  # Perform backward propagation

    c: Value = Value(3.0, label="c")
    d = c * c
    d.label = "d"
    d.grad = 1.0  # Set the gradient of the output to 1
    d.backward()  # Perform backward propagation

    assert a.grad == 2.0  # ∂b/∂a = ∂(a+a)/∂a = 1 + 1 = 2
    assert c.grad == 6.0  # ∂d/∂c = ∂(c*c)/∂c = 2 * c = 2 * 3 = 6


def test_tanh1() -> None:
    """Test that the tanh activation function works correctly."""
    a: Value = Value(0.5, label="a")
    b: Value = a.tanh()
    b.label = "b"
    b.grad = 1.0  # Set the gradient of the output to 1
    b.backward()  # Perform backward propagation

    expected_tanh = (math.exp(2 * 0.5) - 1) / (math.exp(2 * 0.5) + 1)
    expected_grad = (1 - expected_tanh**2) * 1.0

    assert abs(b.data - expected_tanh) < 1e-6
    assert abs(a.grad - expected_grad) < 1e-6


def test_tanh2() -> None:
    a: Value = Value(0.0, label="a")
    b: Value = Value(1.0, label="b")

    assert a.tanh().data == 0.0
    assert b.tanh().data == 0.7615941559557649


def test_trace() -> None:
    """Test that trace() collects all nodes and edges of the computation graph."""
    a: Value = Value(2.0, label="a")
    b: Value = Value(3.0, label="b")
    c: Value = a + b
    c.label = "c"
    d: Value = c * a
    d.label = "d"

    nodes, edges = trace(d)

    assert nodes == {a, b, c, d}
    assert edges == {(a, c), (b, c), (c, d), (a, d)}
