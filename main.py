from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.data.DataReader import DataReader
from src.data.GraphMapping import GraphMapping
from src.data.Segmentation import Segmentation
from src.optimazier.SOTA import optimize as sota_optimize
from src.solver.mip_adapter import MIPSolverAdapter
from src.solver.train import run as run_train
from src.utils.solutionWriter import export_solution_xml
from src.utils.validator import solution as SolutionValidator


def load_config(path: str | Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("stt")


class PipelineRunner:
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger("stt.pipeline")
        self.data_reader = DataReader(matrix=True)
        self.solver = MIPSolverAdapter(config=config, logger=self.logger)

        seed = int(config.get("repro", {}).get("seed", 42))
        random.seed(seed)

    def run(self, xml_path: str | Path) -> Dict[str, Any]:
        started = time.time()
        xml_path = Path(xml_path)
        instance = self.data_reader.read(xml_path)

        class_count = len(instance.classes)
        threshold = int(self.config["pipeline"]["segmentation_threshold_n"])
        use_seg = class_count >= threshold

        if use_seg:
            graph_bundle = GraphMapping(instance).build()
            segmenter = Segmentation(instance, graph_bundle)
            subinstances = segmenter.partition(max_segment_size=threshold)
            path_used = "segmented" if len(subinstances) > 1 else "direct_fallback"
            assignments = self._solve_segmented(subinstances)
        else:
            path_used = "direct"
            assignments = self._solve_single(instance)

        if assignments is None:
            report = {
                "instance": instance.problem_name,
                "source": str(xml_path),
                "valid": False,
                "invalid_reason": "No assignments produced",
                "path": path_used,
                "runtime_sec": time.time() - started,
            }
            self._write_report(report)
            return report

        if self.config.get("optimizer", {}).get("sota", {}).get("enabled", True):
            budget = float(self.config.get("optimizer", {}).get("sota", {}).get("budget_sec", 2.0))
            seed = int(self.config.get("repro", {}).get("seed", 42))
            improved, opt_log = sota_optimize(instance, [assignments], budget_sec=budget, seed=seed)
            if improved is not None:
                assignments = improved
            self.logger.info("SOTA optimization log: %s", opt_log)

        result = self._write_and_validate(instance, assignments, xml_path)
        result["path"] = path_used
        result["runtime_sec"] = time.time() - started
        self._write_report(result)
        return result

    def _solve_single(self, instance) -> Optional[dict]:
        candidates = self.solver.solve(instance)
        valid = [c for c in candidates if c.valid and c.assignments is not None]
        if valid:
            return valid[0].assignments
        if candidates and candidates[0].assignments is not None:
            self.logger.warning("No valid MIP candidate; using first candidate for diagnostics.")
            return candidates[0].assignments
        return None

    def _solve_segmented(self, subinstances) -> Optional[dict]:
        merged = {}
        for sub in subinstances:
            self.logger.info("Solving subinstance %s with %d classes", sub.name, len(sub.instance.classes))
            assignments = self._solve_single(sub.instance)
            if assignments is None:
                self.logger.error("Subinstance %s had no assignments", sub.name)
                return None
            merged.update(assignments)
        return merged

    def _write_and_validate(self, instance, assignments: dict, src_xml: Path) -> Dict[str, Any]:
        output_root = Path(self.config["output"]["dir"])
        output_root.mkdir(parents=True, exist_ok=True)
        inst_out_dir = output_root / instance.problem_name
        inst_out_dir.mkdir(parents=True, exist_ok=True)

        out_xml = inst_out_dir / f"solution_{instance.problem_name}.xml"
        export_solution_xml(
            assignments=assignments,
            out_path=str(out_xml),
            name=instance.problem_name,
            runtime_sec=0.0,
            cores=int(self.config["mip"].get("threads", 1)),
            technique=self.config["output"].get("technique", "MIP+SOTA"),
            author=self.config["output"].get("author", "STT"),
            institution=self.config["output"].get("institution", "N/A"),
            country=self.config["output"].get("country", "N/A"),
            include_students=bool(self.config["output"].get("include_students", False)),
        )

        summary = SolutionValidator(instance, str(out_xml)).total_penalty()
        return {
            "instance": instance.problem_name,
            "source": str(src_xml),
            "solution_xml": str(out_xml),
            "valid": bool(summary.get("valid", False)),
            "invalid_reason": None if summary.get("valid", False) else "Validator failed",
            "total_cost": summary.get("Total_cost"),
            "time_penalty": summary.get("Time penalty"),
            "room_penalty": summary.get("Room penalty"),
            "distribution_penalty": summary.get("Distribution penalty"),
            "unassigned": len(summary.get("not assignment", [])),
        }

    def _write_report(self, report: Dict[str, Any]) -> None:
        output_root = Path(self.config["output"]["dir"])
        output_root.mkdir(parents=True, exist_ok=True)

        report_json = output_root / "report.jsonl"
        with report_json.open("a", encoding="utf-8") as f:
            f.write(json.dumps(report, ensure_ascii=False) + "\n")

        csv_path = output_root / "report.csv"
        row = {
            "instance": report.get("instance"),
            "valid": report.get("valid"),
            "path": report.get("path"),
            "total_cost": report.get("total_cost"),
            "time_penalty": report.get("time_penalty"),
            "room_penalty": report.get("room_penalty"),
            "distribution_penalty": report.get("distribution_penalty"),
            "unassigned": report.get("unassigned"),
            "runtime_sec": report.get("runtime_sec"),
            "solution_xml": report.get("solution_xml"),
            "invalid_reason": report.get("invalid_reason"),
        }
        headers = list(row.keys())
        write_header = not csv_path.exists()
        with csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            if write_header:
                writer.writeheader()
            writer.writerow(row)


def run_pipeline_mode(config: Dict[str, Any], instance_path: str) -> Dict[str, Any]:
    runner = PipelineRunner(config=config)
    return runner.run(instance_path)


def run_batch_mode(config: Dict[str, Any], input_dir: str) -> List[Dict[str, Any]]:
    runner = PipelineRunner(config=config)
    xml_files = sorted(Path(input_dir).glob("*.xml"))
    results = []
    for path in xml_files:
        results.append(runner.run(path))
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="STT v1 pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    parser.add_argument("--mode", choices=["pipeline", "batch", "train"], default="pipeline")
    parser.add_argument("--instance", help="Input instance XML path (pipeline mode)")
    parser.add_argument("--input-dir", help="Input directory with XML files (batch mode)")
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    logger = setup_logging(config.get("logging", {}).get("level", "INFO"))

    if args.mode == "pipeline":
        instance_path = args.instance or config["paths"].get("default_instance")
        if not instance_path:
            raise ValueError("Pipeline mode requires --instance or paths.default_instance in config")
        result = run_pipeline_mode(config, instance_path)
        logger.info("Pipeline result: %s", json.dumps(result, ensure_ascii=False))

    elif args.mode == "batch":
        input_dir = args.input_dir or config["paths"].get("batch_input_dir")
        if not input_dir:
            raise ValueError("Batch mode requires --input-dir or paths.batch_input_dir in config")
        results = run_batch_mode(config, input_dir)
        logger.info("Batch completed: %d instances", len(results))

    elif args.mode == "train":
        artifacts = run_train(config)
        logger.info("Training scaffold completed: %s", json.dumps(artifacts, ensure_ascii=False))


if __name__ == "__main__":
    main()
