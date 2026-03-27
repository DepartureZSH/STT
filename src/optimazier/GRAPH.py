from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional


class GraphOptimizer:
    """v1 scaffold: keeps interfaces stable for future GNN implementation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}

    def train(self, graph_instances: List[Any], solutions: Dict[str, List[dict]], output_dir: str | Path):
        raise NotImplementedError("GRAPH training is scaffolded in v1; implement in phase 2.")

    def optimize(self, solution: dict) -> dict:
        # Baseline pass-through for pipeline compatibility.
        return solution


def train(graph_instances, solutions, output_dir, config=None):
    return GraphOptimizer(config=config).train(graph_instances, solutions, output_dir)


def optimize(solution, config=None):
    return GraphOptimizer(config=config).optimize(solution)
