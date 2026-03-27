from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

import networkx as nx


@dataclass
class SubInstance:
    name: str
    classes: Set[str]
    instance: object


class Segmentation:
    """Partition large graphs while protecting hard constraints."""

    def __init__(self, instance, graph_bundle):
        self.instance = instance
        self.graph_bundle = graph_bundle

    def partition(self, max_segment_size: int) -> List[SubInstance]:
        graph = self.graph_bundle.graph
        if graph.number_of_nodes() <= max_segment_size:
            return [SubInstance(name=f"{self.instance.problem_name}_all", classes=set(graph.nodes()), instance=self.instance)]

        communities = self._detect_communities(graph)
        communities = self._enforce_size_cap(communities, max_segment_size)
        communities = self._enforce_hard_constraint_safety(communities)

        if len(communities) <= 1:
            return [SubInstance(name=f"{self.instance.problem_name}_all", classes=set(graph.nodes()), instance=self.instance)]

        return [
            SubInstance(name=f"{self.instance.problem_name}_seg_{idx}", classes=comm, instance=self._slice_instance(comm))
            for idx, comm in enumerate(communities)
        ]

    def _detect_communities(self, graph: nx.Graph) -> List[Set[str]]:
        try:
            import community as community_louvain

            partition = community_louvain.best_partition(graph, weight="weight")
            grouped: Dict[int, Set[str]] = {}
            for node, gid in partition.items():
                grouped.setdefault(gid, set()).add(node)
            return list(grouped.values())
        except Exception:
            # Fallback without python-louvain.
            groups = nx.algorithms.community.greedy_modularity_communities(graph, weight="weight")
            return [set(g) for g in groups]

    def _enforce_size_cap(self, communities: List[Set[str]], max_segment_size: int) -> List[Set[str]]:
        capped: List[Set[str]] = []
        for comm in communities:
            if len(comm) <= max_segment_size:
                capped.append(comm)
                continue
            comm_sorted = sorted(comm)
            for i in range(0, len(comm_sorted), max_segment_size):
                capped.append(set(comm_sorted[i : i + max_segment_size]))
        return capped

    def _enforce_hard_constraint_safety(self, communities: List[Set[str]]) -> List[Set[str]]:
        # If required hard constraints span multiple communities, merge those communities.
        comms = [set(c) for c in communities]
        hard_constraints = self.instance.distributions.get("hard_constraints", [])

        changed = True
        while changed:
            changed = False
            for hc in hard_constraints:
                classes = {str(cid) for cid in hc.get("classes", [])}
                touched = [idx for idx, comm in enumerate(comms) if comm & classes]
                if len(touched) <= 1:
                    continue
                merged = set()
                for idx in sorted(touched, reverse=True):
                    merged |= comms.pop(idx)
                comms.append(merged)
                changed = True
                break
        return comms

    def _slice_instance(self, selected_classes: Set[str]):
        """Create a shallow class-filtered view compatible with solver/validator."""

        class SlicedInstance:
            pass

        sliced = SlicedInstance()
        sliced.problem_name = self.instance.problem_name
        sliced.nrDays = self.instance.nrDays
        sliced.nrWeeks = self.instance.nrWeeks
        sliced.slotsPerDay = self.instance.slotsPerDay
        sliced.optimization = self.instance.optimization
        sliced.rooms = self.instance.rooms
        sliced.travel = self.instance.travel
        sliced.students = self.instance.students
        sliced.rid_to_idx = self.instance.rid_to_idx
        sliced.sid_to_idx = self.instance.sid_to_idx
        sliced.path = self.instance.path

        selected = {str(c) for c in selected_classes}
        sliced.classes = {cid: cdata for cid, cdata in self.instance.classes.items() if str(cid) in selected}
        sliced.cid_to_idx = {cid: i for i, cid in enumerate(sliced.classes.keys())}

        # Keep full courses for compatibility; solver only relies on class subset.
        sliced.courses = self.instance.courses

        def _filter_constraints(constraints):
            filtered = []
            for cons in constraints:
                cls = [cid for cid in cons.get("classes", []) if str(cid) in selected]
                if len(cls) < 2:
                    continue
                filtered.append({**cons, "classes": cls})
            return filtered

        sliced.distributions = {
            "hard_constraints": _filter_constraints(self.instance.distributions.get("hard_constraints", [])),
            "soft_constraints": _filter_constraints(self.instance.distributions.get("soft_constraints", [])),
        }

        return sliced
