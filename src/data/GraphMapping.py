from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set, Tuple

import networkx as nx


@dataclass
class GraphBundle:
    graph: nx.Graph
    node_features: Dict[str, Dict[str, float]]
    edge_features: Dict[Tuple[str, str], Dict[str, float]]


class GraphMapping:
    """Build conflict graph and aligned features from an instance."""

    def __init__(self, instance):
        self.instance = instance
        self._class_course_map = self._build_class_to_course_map()
        self._course_students = self._build_course_students_map()

    def build(self) -> GraphBundle:
        graph = nx.Graph()
        node_features: Dict[str, Dict[str, float]] = {}
        edge_features: Dict[Tuple[str, str], Dict[str, float]] = {}

        for cid, class_data in self.instance.classes.items():
            cid_str = str(cid)
            students = self._students_for_class(cid_str)
            node_features[cid_str] = {
                "enrollment": float(len(students)),
                "num_time_options": float(len(class_data.get("time_options", []))),
                "num_room_options": float(len(class_data.get("room_options", []))),
                "room_required": float(1 if class_data.get("room_required", True) else 0),
                "hard_constraint_degree": float(self._class_constraint_count(cid_str, hard=True)),
                "soft_constraint_degree": float(self._class_constraint_count(cid_str, hard=False)),
            }
            graph.add_node(cid_str, **node_features[cid_str])

        class_ids = list(self.instance.classes.keys())
        for i, c1 in enumerate(class_ids):
            for c2 in class_ids[i + 1 :]:
                c1s = str(c1)
                c2s = str(c2)
                shared_students = self._shared_students(c1s, c2s)
                room_conflict = self._room_option_overlap(c1s, c2s)
                time_conflict = self._time_option_overlap(c1s, c2s)

                # Edge existence: strong student overlap or structural resource conflict.
                has_edge = shared_students > 0 or (room_conflict and time_conflict)
                if not has_edge:
                    continue

                weight = float(shared_students) + (1.0 if room_conflict else 0.0) + (1.0 if time_conflict else 0.0)
                graph.add_edge(c1s, c2s, weight=weight)
                edge_key = tuple(sorted((c1s, c2s)))
                edge_features[edge_key] = {
                    "shared_students": float(shared_students),
                    "room_conflict": float(1 if room_conflict else 0),
                    "time_overlap": float(1 if time_conflict else 0),
                    "weight": weight,
                }

        return GraphBundle(graph=graph, node_features=node_features, edge_features=edge_features)

    def to_pyg_data(self, bundle: GraphBundle):
        """Optional conversion helper if torch_geometric is available."""
        try:
            import torch
            from torch_geometric.data import Data
        except Exception as exc:  # pragma: no cover
            raise ImportError("torch_geometric is not installed.") from exc

        node_ids = sorted(bundle.graph.nodes())
        idx_map = {nid: i for i, nid in enumerate(node_ids)}
        x = []
        for nid in node_ids:
            feat = bundle.node_features[nid]
            x.append([
                feat["enrollment"],
                feat["num_time_options"],
                feat["num_room_options"],
                feat["room_required"],
                feat["hard_constraint_degree"],
                feat["soft_constraint_degree"],
            ])
        edge_index = [[], []]
        edge_attr = []
        for u, v in bundle.graph.edges():
            key = tuple(sorted((u, v)))
            ef = bundle.edge_features[key]
            ui, vi = idx_map[u], idx_map[v]
            edge_index[0].extend([ui, vi])
            edge_index[1].extend([vi, ui])
            attr = [ef["shared_students"], ef["room_conflict"], ef["time_overlap"], ef["weight"]]
            edge_attr.extend([attr, attr])

        return Data(
            x=torch.tensor(x, dtype=torch.float),
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            edge_attr=torch.tensor(edge_attr, dtype=torch.float),
        )

    def _build_class_to_course_map(self) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for course_id, course_data in self.instance.courses.items():
            for cfg in course_data.get("configs", {}).values():
                for subpart in cfg.get("subparts", {}).values():
                    for class_id in subpart.get("classes", {}).keys():
                        mapping[str(class_id)] = str(course_id)
        return mapping

    def _build_course_students_map(self) -> Dict[str, Set[str]]:
        mapping: Dict[str, Set[str]] = {}
        for sid, student in self.instance.students.items():
            sid_str = str(sid)
            for course_id in student.get("courses", []):
                mapping.setdefault(str(course_id), set()).add(sid_str)
        return mapping

    def _students_for_class(self, class_id: str) -> Set[str]:
        course_id = self._class_course_map.get(class_id)
        if not course_id:
            return set()
        return self._course_students.get(course_id, set())

    def _shared_students(self, c1: str, c2: str) -> int:
        return len(self._students_for_class(c1) & self._students_for_class(c2))

    def _room_option_overlap(self, c1: str, c2: str) -> bool:
        rooms1 = {str(r["id"]) for r in self.instance.classes[c1].get("room_options", [])}
        rooms2 = {str(r["id"]) for r in self.instance.classes[c2].get("room_options", [])}

        # If one class does not require room, room overlap is irrelevant.
        if not self.instance.classes[c1].get("room_required", True):
            return False
        if not self.instance.classes[c2].get("room_required", True):
            return False

        return len(rooms1 & rooms2) > 0

    def _time_option_overlap(self, c1: str, c2: str) -> bool:
        opts1 = [t["optional_time_bits"] for t in self.instance.classes[c1].get("time_options", [])]
        opts2 = [t["optional_time_bits"] for t in self.instance.classes[c2].get("time_options", [])]
        for b1 in opts1:
            for b2 in opts2:
                if self._bits_overlap(b1, b2):
                    return True
        return False

    def _bits_overlap(self, b1, b2) -> bool:
        w1, d1, s1, l1 = b1
        w2, d2, s2, l2 = b2
        if (int(w1, 2) & int(w2, 2)) == 0:
            return False
        if (int(d1, 2) & int(d2, 2)) == 0:
            return False
        return (s1 < s2 + l2) and (s2 < s1 + l1)

    def _class_constraint_count(self, class_id: str, hard: bool) -> int:
        key = "hard_constraints" if hard else "soft_constraints"
        return sum(1 for cons in self.instance.distributions.get(key, []) if class_id in cons.get("classes", []))
