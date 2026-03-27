from pathlib import Path

from src.data.DataReader import DataReader
from src.data.GraphMapping import GraphMapping
from src.data.Segmentation import Segmentation
from src.utils.solutionReader import PSTTReader as SolutionReader
from src.utils.solutionWriter import export_solution_xml


def _instance(path: str):
    return DataReader(matrix=True).read(path)


def test_xml_parse_invariants_early_and_late():
    early = _instance("data/source/early/bet-fal17.xml")
    late = _instance("data/source/late/bet-spr18.xml")

    assert len(early.classes) > 0
    assert len(early.rooms) > 0
    assert len(early.students) > 0

    assert len(late.classes) > 0
    assert len(late.rooms) > 0
    assert len(late.students) > 0


def test_graph_mapping_builds_conflict_graph():
    inst = _instance("data/source/early/bet-fal17.xml")
    bundle = GraphMapping(inst).build()

    assert bundle.graph.number_of_nodes() == len(inst.classes)
    assert bundle.graph.number_of_edges() > 0


def test_segmentation_safety_or_fallback():
    inst = _instance("data/source/late/bet-spr18.xml")
    bundle = GraphMapping(inst).build()
    segments = Segmentation(inst, bundle).partition(max_segment_size=80)

    assert len(segments) >= 1
    # If segmented, every subinstance should contain classes and filtered constraints only.
    for seg in segments:
        assert len(seg.instance.classes) > 0
        for hc in seg.instance.distributions["hard_constraints"]:
            assert len(hc["classes"]) >= 2


def test_solution_writer_reader_roundtrip(tmp_path: Path):
    inst = _instance("data/source/early/bet-fal17.xml")

    assignments = {}
    for cid, cdata in inst.classes.items():
        time_option = cdata["time_options"][0] if cdata["time_options"] else None
        room_required = cdata.get("room_required", True)
        room_id = None
        if room_required and cdata.get("room_options"):
            room_id = cdata["room_options"][0]["id"]
        assignments[cid] = (time_option, room_required, room_id, [])

    out = tmp_path / "solution.xml"
    export_solution_xml(
        assignments=assignments,
        out_path=str(out),
        name=inst.problem_name,
        runtime_sec=0.0,
        cores=1,
        technique="test",
        author="test",
        institution="test",
        country="test",
        include_students=False,
    )

    parsed = SolutionReader(str(out))
    assert len(parsed.classes) == len(assignments)
