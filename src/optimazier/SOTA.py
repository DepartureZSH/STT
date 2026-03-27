from __future__ import annotations

import copy
import math
import random
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from src.utils.solutionWriter import export_solution_xml
from src.utils.validator import solution as SolutionValidator


@dataclass
class SOTAResult:
    solution: Optional[dict]
    log: Dict[str, Any]


class SOTAOptimizer:
    """Hard-safe SA baseline using validator feedback as objective."""

    def __init__(self, instance, budget_sec: float = 5.0, seed: int = 42):
        self.instance = instance
        self.budget_sec = budget_sec
        self.rng = random.Random(seed)

    def optimize(self, seed_solutions: List[dict]) -> Tuple[Optional[dict], Dict[str, Any]]:
        start = time.time()
        if not seed_solutions:
            return None, {"accepted": 0, "rejected": 0, "reason": "no_seed_solutions"}

        current = copy.deepcopy(seed_solutions[0])
        current_valid, current_cost = self._evaluate(current)
        if not current_valid:
            # Try to find first valid seed solution.
            for candidate in seed_solutions[1:]:
                valid, cost = self._evaluate(candidate)
                if valid:
                    current = copy.deepcopy(candidate)
                    current_valid, current_cost = valid, cost
                    break

        if not current_valid:
            return seed_solutions[0], {
                "accepted": 0,
                "rejected": 0,
                "reason": "no_valid_seed_solution",
            }

        best = copy.deepcopy(current)
        best_cost = current_cost

        temp = 5.0
        min_temp = 0.1
        cooling = 0.995
        accepted = 0
        rejected = 0

        while time.time() - start < self.budget_sec and temp > min_temp:
            candidate = self._neighbor(current)
            valid, cost = self._evaluate(candidate)
            if not valid:
                rejected += 1
                temp *= cooling
                continue

            delta = cost - current_cost
            if delta <= 0:
                current = candidate
                current_cost = cost
                accepted += 1
            else:
                p = math.exp(-delta / max(temp, 1e-6))
                if self.rng.random() < p:
                    current = candidate
                    current_cost = cost
                    accepted += 1
                else:
                    rejected += 1

            if current_cost < best_cost:
                best = copy.deepcopy(current)
                best_cost = current_cost

            temp *= cooling

        return best, {
            "accepted": accepted,
            "rejected": rejected,
            "best_cost": best_cost,
            "runtime_sec": time.time() - start,
        }

    def _neighbor(self, assignments: dict) -> dict:
        cand = copy.deepcopy(assignments)
        class_ids = list(self.instance.classes.keys())
        cid = self.rng.choice(class_ids)
        class_data = self.instance.classes[cid]

        time_options = class_data.get("time_options", [])
        if not time_options:
            return cand

        new_time = self.rng.choice(time_options)

        room_required = class_data.get("room_required", True)
        room_id = None
        if room_required:
            room_options = class_data.get("room_options", [])
            if room_options:
                room_id = self.rng.choice(room_options)["id"]

        cand[cid] = (new_time, room_required, room_id, [])
        return cand

    def _evaluate(self, assignments: dict) -> Tuple[bool, float]:
        # Delegate hard/soft checking to project validator by writing temp XML.
        with tempfile.TemporaryDirectory(prefix="stt_sota_") as tmpdir:
            out = Path(tmpdir) / "candidate.xml"
            export_solution_xml(
                assignments=assignments,
                out_path=str(out),
                name=self.instance.problem_name,
                runtime_sec=0.0,
                cores=1,
                technique="SOTA-SA",
                author="STT",
                institution="N/A",
                country="N/A",
                include_students=False,
            )
            result = SolutionValidator(self.instance, str(out)).total_penalty()
            return bool(result.get("valid", False)), float(result.get("Total_cost", 1e18))


def optimize(instance, seed_solutions: List[dict], budget_sec: float, seed: int):
    optimizer = SOTAOptimizer(instance=instance, budget_sec=budget_sec, seed=seed)
    return optimizer.optimize(seed_solutions)
