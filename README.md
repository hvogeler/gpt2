# gpt2

[![CI](https://git.home.vogeler.cc/hvo/gpt2/badges/workflows/ci.yml/badge.svg?branch=main)](https://git.home.vogeler.cc/hvo/gpt2/actions?workflow=ci.yml)

Working through Andrej Karpathy's [Neural Networks: Zero to Hero](https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ) as a typed Python application — no notebooks.

## Lectures

| # | Topic | Module |
|---|-------|--------|
| 0 | Micrograd — backprop & autograd engine | `gpt2.micrograd` |
| 1 | Makemore I — bigram language model | `gpt2.bigram` |
| 2 | Makemore II — MLP | `gpt2.mlp` |
| 3 | Makemore III — BatchNorm | `gpt2.batchnorm` |
| 4 | Makemore IV — Backprop ninja | `gpt2.backprop` |
| 5 | Makemore V — WaveNet | `gpt2.wavenet` |
| 6 | GPT from scratch | `gpt2.transformer` |

## Setup

```bash
uv sync --dev
```

## Running checks

```bash
uv run ruff check .
uv run ruff format .
uv run ty check src/gpt2
uv run pytest tests/ -v
```
