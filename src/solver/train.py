from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List

from src.data.DataReader import DataReader
from src.data.GraphMapping import GraphMapping
from src.optimazier.GRAPH import GraphOptimizer
from src.optimazier.MARL import MARLOptimizer


def _collect_instances(base_dir: Path, split: str) -> List[Path]:
    split_dir = base_dir / split
    return sorted(split_dir.glob("*.xml"))


def run(config: Dict[str, Any]) -> Dict[str, Any]:
    logger = logging.getLogger("stt.train")

    data_root = Path(config["paths"]["data_source"])
    output_dir = Path(config["output"]["dir"]) / "training"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training scaffolding does not call MIP directly; matrix=False is lighter.
    reader = DataReader(matrix=False)
    train_cfg = config.get("train", {})
    max_per_split = int(train_cfg.get("max_instances_per_split", 2))

    early_paths = _collect_instances(data_root, "early")[:max_per_split]
    middle_paths = _collect_instances(data_root, "middle")[:max_per_split]
    train_paths = early_paths + middle_paths
    val_paths = _collect_instances(data_root, "late")[:max_per_split]
    test_paths = _collect_instances(data_root, "test")

    train_instances = [reader.read(p) for p in train_paths]
    val_instances = [reader.read(p) for p in val_paths]

    graph_train = [GraphMapping(inst).build() for inst in train_instances]
    graph_val = [GraphMapping(inst).build() for inst in val_instances]

    marl = MARLOptimizer(config=config.get("optimizer", {}).get("marl", {}))
    graph_opt = GraphOptimizer(config=config.get("optimizer", {}).get("graph", {}))

    artifacts = {
        "train_instances": len(train_instances),
        "val_instances": len(val_instances),
        "test_instances": len(test_paths),
        "graph_train_samples": len(graph_train),
        "graph_val_samples": len(graph_val),
        "marl": "scaffold_only",
        "graph": "scaffold_only",
    }

    # Keep scaffolding explicit: call interfaces and capture deferred status.
    try:
        marl.train(train_instances, solutions={}, output_dir=output_dir)
        artifacts["marl"] = "trained"
    except NotImplementedError as exc:
        logger.info("MARL train skipped: %s", exc)

    try:
        graph_opt.train(graph_train, solutions={}, output_dir=output_dir)
        artifacts["graph"] = "trained"
    except NotImplementedError as exc:
        logger.info("GRAPH train skipped: %s", exc)

    artifacts_path = output_dir / "artifacts.json"
    artifacts_path.write_text(json.dumps(artifacts, indent=2), encoding="utf-8")
    return artifacts


# Backward-compatible stubs preserved for legacy imports.
def MIP2Step_Solver(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Use main.py pipeline/batch mode for v1 execution.")


def MIP3Step_Solver(*args, **kwargs):  # pragma: no cover
    raise NotImplementedError("Use main.py pipeline/batch mode for v1 execution.")
