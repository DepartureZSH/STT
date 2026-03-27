from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET


class PSTTReader:
    """Reader for ITC-2019 solution XML."""

    def __init__(self, xml_path):
        self.path = pathlib.Path(xml_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)

        self.tree = ET.parse(str(self.path))
        self.root = self.tree.getroot()
        self.meta = self._parse_meta()
        self.classes = self._parse_classes()

    def _parse_meta(self):
        return {
            "name": self.root.attrib.get("name"),
            "runtime": self._to_float(self.root.attrib.get("runtime")),
            "cores": self._to_int(self.root.attrib.get("cores")),
            "technique": self.root.attrib.get("technique"),
            "author": self.root.attrib.get("author"),
            "institution": self.root.attrib.get("institution"),
            "country": self.root.attrib.get("country"),
        }

    def _parse_classes(self):
        classes = {}
        for c in self.root.findall("class"):
            cid = c.attrib.get("id")
            weeks = c.attrib.get("weeks")
            days = c.attrib.get("days")
            start = self._to_int(c.attrib.get("start"))
            optional_time = None
            if weeks is not None and days is not None and start is not None:
                optional_time = (weeks, days, start)

            classes[cid] = {
                "id": cid,
                "optional_time": optional_time,
                "room": c.attrib.get("room"),
                "students": [s.attrib.get("id") for s in c.findall("student") if s.attrib.get("id") is not None],
            }
        return classes

    @staticmethod
    def _to_int(x, default=None):
        if x is None:
            return default
        try:
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _to_float(x, default=None):
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default
