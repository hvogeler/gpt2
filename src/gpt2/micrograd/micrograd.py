from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from graphviz import Digraph  # type: ignore[import-untyped]

logger = logging.getLogger(__name__)


class Value:
    """A simple class to represent a value in a computational graph."""

    def __init__(self, data: float, _children: tuple[Value, ...] = (), _op: str = "") -> None:
        self.data = data
        self._prev = set(_children)
        self._op = _op  # the operation that produced this node, for graphviz / debugging / etc

    @property
    def prev(self) -> set[Value]:
        """Return the set of previous Value instances."""
        return self._prev

    @property
    def op(self) -> str:
        """Return the operation that produced this node."""
        return self._op

    def __repr__(self) -> str:
        return f"Value(data={self.data})"

    def __add__(self, other: Value) -> Value:
        """Add two Value instances."""

        return Value(self.data + other.data, _children=(self, other), _op="+")

    def __mul__(self, other: Value) -> Value:
        """Multiply two Value instances."""

        return Value(self.data * other.data, _children=(self, other), _op="*")


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
        dot.node(name=uid, label=f"{n.data:.4f}", shape="record")
        if n.op:
            dot.node(name=uid + n.op, label=n.op)
            dot.edge(uid + n.op, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), str(id(n2)) + n2.op)

    return dot
