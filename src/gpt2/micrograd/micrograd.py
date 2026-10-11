from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from graphviz import Digraph  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class Value:
    """A simple class to represent a value in a computational graph."""

    def __init__(
        self,
        data: float,
        _children: tuple[Value, ...] = (),
        _op: str = "",
        label: str = "",
    ) -> None:
        self.data = data
        self.grad = 0.0  # the gradient of this value
        self._prev = set(_children)
        self._op = _op  # the operation that produced this node, for graphviz / debugging / etc
        self.label = label  # optional label for graphviz / debugging / etc
        self._backward: Callable[[], None] = lambda: (
            None
        )  # function to compute the gradient of this value

    @property
    def prev(self) -> set[Value]:
        """Return the set of previous Value instances."""
        return self._prev

    @property
    def op(self) -> str:
        """Return the operation that produced this node."""
        return self._op

    def __repr__(self) -> str:
        return f"Value(data={self.data}, label={self.label}, grad={self.grad})"

    def __add__(self, other: Value) -> Value:
        """Add two Value instances."""

        out = Value(self.data + other.data, _children=(self, other), _op="+")

        def _backward() -> None:
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward

        return out

    def __mul__(self, other: Value) -> Value:
        """Multiply two Value instances."""
        out = Value(self.data * other.data, _children=(self, other), _op="*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def tanh(self) -> Value:
        """Apply the tanh activation function to this Value instance."""
        x = self.data
        t = (math.exp(2 * x) - 1) / (math.exp(2 * x) + 1)  # tanh formula

        out = Value(t, _children=(self,), _op="tanh")

        def _backward() -> None:
            self.grad = (1 - t**2) * out.grad

        out._backward = _backward

        return out

    def backward(self) -> None:
        """Perform backpropagation to compute gradients for all nodes in the computation graph."""

        ordered_nodes: list[Value] = []
        visited: set[Value] = set()

        def sort_nodes(v: Value) -> None:
            """Build a topological ordering of the computation graph."""
            if v not in visited:
                visited.add(v)
                for child in v.prev:
                    sort_nodes(child)
                ordered_nodes.append(v)

        sort_nodes(self)
        ordered_nodes.reverse()

        for n in ordered_nodes:
            n._backward()


def trace(root: Value) -> tuple[set[Value], set[tuple[Value, Value]]]:
    """Perform a depth-first search to trace the computational graph."""
    nodes: set[Value] = set()
    edges: set[tuple[Value, Value]] = set()

    def build(v: Value) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v.prev:
                edges.add((child, v))
                build(child)
            nodes.add(v)

    build(root)
    return nodes, edges


def draw_dot(root: Value) -> Digraph:
    """Draw the computational graph using Graphviz."""
    from graphviz import Digraph  # type: ignore[import-untyped]

    dot: Any = Digraph(format="svg", graph_attr={"rankdir": "LR"})  # left to right

    nodes, edges = trace(root)

    for n in nodes:
        uid = str(id(n))
        dot.node(name=uid, label=f"{n.label} | v: {n.data:.4f} | g: {n.grad:.4f}", shape="record")
        if n.op:
            dot.node(name=uid + n.op, label=n.op)
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
