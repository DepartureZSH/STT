from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class CandidateSolution:
    assignments: Optional[dict]
    valid: bool
    invalid_reason: Optional[str]
    objective: Optional[float]
    runtime: float
    details: Dict[str, Any]


class MIPSolverAdapter:
    """Adapter exposing stable solve API around legacy MIPSolver."""

    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)

    def solve(self, instance) -> List[CandidateSolution]:
        seed = int(self.config.get("repro", {}).get("seed", 42))
        random.seed(seed)

        legacy_cfg = self._to_legacy_config(self.config)

        from src.solver.gurobi import MIPSolver

        solver = MIPSolver(instance, self.logger, legacy_cfg)

        # Optional deterministic seed for Gurobi.
        try:
            solver.model.setParam("Seed", seed)
        except Exception:
            pass

        start = time.time()
        solver.build_model()
        assignments_list = solver.solve()
        runtime = time.time() - start

        if not assignments_list:
            return [
                CandidateSolution(
                    assignments=None,
                    valid=False,
                    invalid_reason="No feasible solution returned by MIP solver",
                    objective=None,
                    runtime=runtime,
                    details={"status": "infeasible_or_no_solution"},
                )
            ]

        candidates: List[CandidateSolution] = []
        for assignments in assignments_list:
            objective = None
            try:
                objective = float(solver.model.ObjVal)
            except Exception:
                objective = None

            # Lightweight completeness check before full validation.
            unassigned = [cid for cid, data in assignments.items() if data[0] is None]
            valid = len(unassigned) == 0
            invalid_reason = None if valid else f"{len(unassigned)} classes unassigned"

            candidates.append(
                CandidateSolution(
                    assignments=assignments,
                    valid=valid,
                    invalid_reason=invalid_reason,
                    objective=objective,
                    runtime=runtime,
                    details={
                        "unassigned_count": len(unassigned),
                        "model_solutions": len(assignments_list),
                    },
                )
            )

        return candidates

    def _to_legacy_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "train": {
                "MIP": {
                    "time_limit": config["mip"]["time_limit"],
                    "Threads": config["mip"]["threads"],
                    "MIPGap": config["mip"]["mip_gap"],
                    "PoolSolutions": config["mip"]["pool_solutions"],
                }
            },
            "config": {
                "technique": config["output"].get("technique", "MIP+SOTA"),
                "author": config["output"].get("author", "STT"),
                "institution": config["output"].get("institution", "N/A"),
                "country": config["output"].get("country", "N/A"),
                "include_students": bool(config["output"].get("include_students", False)),
            },
            "method": {
                "epoch": 1,
                "reproduction": False,
            },
        }
