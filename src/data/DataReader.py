from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from src.utils.dataReader import PSTTReader


@dataclass
class InstanceBundle:
    """Thin wrapper around the parsed instance object."""

    instance: PSTTReader
    source_path: Path


class DataReader:
    """Project-level adapter over the low-level XML parser."""

    def __init__(self, matrix: bool = True):
        # MIP solver expects matrix-backed time options.
        self.matrix = matrix

    def read(self, xml_path: str | Path) -> PSTTReader:
        return PSTTReader(str(xml_path), matrix=self.matrix)

    def read_group(self, xml_paths: Iterable[str | Path]) -> List[PSTTReader]:
        return [self.read(path) for path in xml_paths]

    def data_maker(self, xml_path: str | Path) -> InstanceBundle:
        path = Path(xml_path)
        return InstanceBundle(instance=self.read(path), source_path=path)


# Backward-compatible legacy function names used by previous skeleton code.
def ReadFromInstance(xml_path: str | Path, matrix: bool = True) -> PSTTReader:
    return DataReader(matrix=matrix).read(xml_path)


def ReadFromGroup(xml_paths: Iterable[str | Path], matrix: bool = True) -> List[PSTTReader]:
    return DataReader(matrix=matrix).read_group(xml_paths)


def DataMaker(xml_path: str | Path, matrix: bool = True) -> InstanceBundle:
    return DataReader(matrix=matrix).data_maker(xml_path)
