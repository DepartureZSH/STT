import pathlib
import torch
import xml.etree.ElementTree as ET

class PSTTReader:
    def __init__(self, xml_path, matrix=False):
        self.path = pathlib.Path(xml_path)
        if not self.path.exists():
            raise FileNotFoundError(self.path)
        
        self.tree = ET.parse(str(xml_path))
        self.root = self.tree.getroot()
        
        # print(f"root : {self.root}")
        self.matrix = matrix

        # 公共元信息
        self.problem_name = None
        self.nrDays = None
        self.nrWeeks = None
        self.slotsPerDay = None
        
        self.timeTable_matrix = None

        # 各模块数据
        self.optimization = None
        self.rooms = {}
        self.rid_to_idx = {}
        self.travel = None
        self.courses = {}
        self.classes = {}
        self.cid_to_idx = {}
        self.students = {}
        self.sid_to_idx = {}
        self.distributions = []
        self.solution = None

        self._parse()

    # ---------- 顶层调度 ----------
    def _parse(self):
        self._parse_problem(self.root)
    
    # ---------- Problem ----------
    def _parse_problem(self, problem: ET.Element):
        # 根属性：name / nrDays / nrWeeks / slotsPerDay
        self.problem_name = problem.attrib.get("name")
        self.nrDays = self._to_int(problem.attrib.get("nrDays"))
        self.nrWeeks = self._to_int(problem.attrib.get("nrWeeks"))
        self.slotsPerDay = self._to_int(problem.attrib.get("slotsPerDay"))

        print(f"Problem Name: {self.problem_name}, Days: {self.nrDays}, Weeks: {self.nrWeeks}, Slots/Day: {self.slotsPerDay}")
        
        # optimization
        opt = problem.find("optimization")
        if opt is not None:
            self.optimization = {
                "time": self._to_int(opt.attrib.get("time"), 0),
                "room": self._to_int(opt.attrib.get("room"), 0),
                "distribution": self._to_int(opt.attrib.get("distribution"), 0),
                "student": self._to_int(opt.attrib.get("student"), 0),
            }

        # rooms
        rooms_node = problem.find("rooms")
        if rooms_node is not None:
            self.rooms, self.travel, self.rid_to_idx = self._parse_rooms(rooms_node)

        # courses
        courses_node = problem.find("courses")
        if courses_node is not None:
            self.courses, self.classes, self.cid_to_idx = self._parse_courses(courses_node)

        # distributions
        dist_node = problem.find("distributions")
        if dist_node is not None:
            self.distributions = self._parse_distributions(dist_node)

        # students
        students_node = problem.find("students")
        if students_node is not None:
            self.students, self.sid_to_idx = self._parse_students(students_node)

    # ---------- Rooms ----------
    def _parse_rooms(self, rooms_node):
        result = {}
        travel = {}
        rid_to_idx = {}
        for i, r in enumerate(rooms_node.findall("room")):
            rid = self._to_int(r.attrib["id"])
            rid_to_idx[rid] = i
            cap = self._to_int(r.attrib.get("capacity"), 0)
            unavailables = []
            unavailable_zip = torch.zeros((self.nrWeeks, self.nrDays, self.slotsPerDay), dtype=int)
            unavailables_bits = []

            # travel
            for t in r.findall("travel"):
                other = t.attrib["room"]
                value = self._to_int(t.attrib.get("value"), 0)
                if not travel.get(r.attrib["id"], 0): travel[r.attrib["id"]] = {}
                travel[r.attrib["id"]].update({other: value})
                if not travel.get(other, 0): travel[other] = {}
                travel[other].update({r.attrib["id"]: value})

            # unavailable
            for u in r.findall("unavailable"):
                if self.matrix:
                    unavailable = torch.zeros((self.nrWeeks, self.nrDays, self.slotsPerDay), dtype=int)
                weeks_bits = u.attrib.get("weeks")
                if weeks_bits is not None:
                    weeks_list = self.bits_to_list(weeks_bits)
                days_bits = u.attrib.get("days")
                if days_bits is not None:
                    days_list = self.bits_to_list(days_bits)
                start = self._to_int(u.attrib.get("start"))
                length = self._to_int(u.attrib.get("length"))
                if start is not None and length is not None:
                    w_idx = torch.tensor(weeks_list, dtype=torch.long)
                    d_idx = torch.tensor(days_list, dtype=torch.long)
                    t_idx = torch.arange(start, start + length, dtype=torch.long)
                    if self.matrix:
                        W, D, T = torch.meshgrid(w_idx, d_idx, t_idx, indexing='ij')
                        unavailable[W, D, T] = 1
                        # unavailable[weeks_list, days_list, start: start + length] = 1
                if self.matrix:
                    unavailable_zip = torch.logical_or(unavailable_zip, unavailable)
                    unavailables.append(unavailable)
                    unavailables_bits.append((weeks_bits, days_bits, start, length))
            if self.matrix:
                room = {
                    "id": rid, 
                    "capacity": cap,
                    "unavailables_bits": unavailables_bits,
                    "unavailables": unavailables,
                    "unavailable_zip": unavailable_zip,
                    "ocupied": [] # (cid, time_bits, value)
                }
            else:
                room = {
                    "id": rid, 
                    "capacity": cap,
                    "unavailables_bits": unavailables_bits,
                    "ocupied": [] # (cid, time_bits, value)
                }
            result[r.attrib["id"]] = room
        return result, travel, rid_to_idx

    # # ---------- Courses / Config / Subpart / Class ----------
    def _parse_courses(self, courses_node):
        result = {}
        classes = {}
        cid_to_idx = {}
        for i, c in enumerate(courses_node.findall("course")):
            cid = self._to_int(c.attrib["id"])
            cid_to_idx[c.attrib["id"]] = i
            course = {
                "id": cid,
                "configs": {}
            }

            for cfg in c.findall("config"):
                cfg_id = cfg.attrib["id"]
                config = {
                    "id": cfg_id,
                    "subparts": {}
                }

                for sp in cfg.findall("subpart"):
                    sp_id = sp.attrib["id"]
                    subpart = {
                        "id": sp_id,
                        "classes": {}
                    }

                    for cl in sp.findall("class"):
                        cl_id = cl.attrib["id"]
                        limit = self._to_int(cl.attrib.get("limit")) if "limit" in cl.attrib else None
                        parent = cl.attrib.get("parent")
                        room_required = True
                        if "room" in cl.attrib and cl.attrib["room"].lower() == "false":
                            room_required = False

                        cdef = {
                            "id": cl_id,
                            "limit": limit,
                            "parent": parent,
                            "room_required": room_required,
                            "room_options": [],
                            "time_options": []
                        }

                        # 可选房间（含 penalty）
                        for rnode in cl.findall("room"):
                            cdef["room_options"].append({
                                "id":rnode.attrib["id"],
                                "penalty":self._to_int(rnode.attrib.get("penalty"), 0)
                            })


                        # 可选时间（含 penalty）
                        for tnode in cl.findall("time"):
                            if self.matrix:
                                optional_time = torch.zeros((self.nrWeeks, self.nrDays, self.slotsPerDay), dtype=int)
                            weeks_bits = tnode.attrib.get("weeks")
                            if weeks_bits is not None:
                                weeks_list = self.bits_to_list(weeks_bits)
                            days_bits = tnode.attrib.get("days")
                            if days_bits is not None:
                                days_list = self.bits_to_list(days_bits)
                            start = self._to_int(tnode.attrib.get("start"))
                            length = self._to_int(tnode.attrib.get("length"))
                            if start is not None and length is not None:
                                w_idx = torch.tensor(weeks_list, dtype=torch.long)
                                d_idx = torch.tensor(days_list, dtype=torch.long)
                                t_idx = torch.arange(start, start + length, dtype=torch.long)
                                if self.matrix:
                                    W, D, T = torch.meshgrid(w_idx, d_idx, t_idx, indexing='ij')
                                    optional_time[W, D, T] = 1
                            if self.matrix:
                                cdef["time_options"].append({
                                    "optional_time_bits": (weeks_bits, days_bits, start, length),
                                    "optional_time":optional_time,
                                    "penalty":self._to_int(tnode.attrib.get("penalty"), 0)
                                })
                            else:
                                cdef["time_options"].append({
                                    "optional_time_bits": (weeks_bits, days_bits, start, length),
                                    "penalty":self._to_int(tnode.attrib.get("penalty"), 0)
                                })
                        # Sort time_options by penalty
                        cdef["time_options"].sort(key=lambda x: x["penalty"])
                        subpart["classes"][cl_id] = cdef
                        classes[cl_id] = cdef
                    config["subparts"][sp_id] = subpart
                course["configs"][cfg_id] = config

            result[cid] = course
        return result, classes, cid_to_idx

    # # ---------- Distributions ----------
    def _parse_distributions(self, dist_node):
        results = []
        hard_constraints = []
        soft_constraints = []
        for d in dist_node.findall("distribution"):
            dtype = d.attrib["type"]
            required = d.attrib.get("required", "false").lower() == "true"
            penalty = self._to_int(d.attrib.get("penalty")) if "penalty" in d.attrib else None
            classes = [c.attrib["id"] for c in d.findall("class") if "id" in c.attrib]
            if required:
                hard_constraints.append({
                    "type": dtype, 
                    "required": required, 
                    "penalty": penalty,
                    "classes": classes
                })
            else:
                soft_constraints.append({
                    "type": dtype, 
                    "required": required, 
                    "penalty": penalty,
                    "classes": classes
                })
        return {
            "hard_constraints": hard_constraints, 
            "soft_constraints": soft_constraints
        }

    # # ---------- Students ----------
    def _parse_students(self, students_node):
        results = {}
        sid_to_idx = {}
        for i, s in enumerate(students_node.findall("student")):
            sid = self._to_int(s.attrib["id"])
            sid_to_idx[sid] = i
            courses = [c.attrib["id"] for c in s.findall("course") if "id" in c.attrib]
            results[sid] = {
                "id": sid, 
                "courses": courses
            }
        return results, sid_to_idx

    # ---------- Solution（可作为根，或 problem 子节点） ----------
    def _parse_solution(self, node):
        meta = {
            "name": node.attrib.get("name"),
            "runtime": self._to_float(node.attrib.get("runtime")),
            "cores": self._to_int(node.attrib.get("cores")) if "cores" in node.attrib else None,
            "technique": node.attrib.get("technique"),
            "author": node.attrib.get("author"),
            "institution": node.attrib.get("institution"),
            "country": node.attrib.get("country"),
        }
        classes = {}
        for c in node.findall("class"):
            cid = c.attrib["id"]
            weeks_bits = c.attrib.get("weeks", "")
            if weeks_bits is not None:
                weeks_list = self.bits_to_list(weeks_bits)
            days_bits = c.attrib.get("days", "")
            if days_bits is not None:
                days_list = self.bits_to_list(days_bits)
            sc = {
                "id": cid,
                "days_bits": days_list,
                "start": self._to_int(c.attrib.get("start"), 0),
                "weeks_bits": weeks_list,
                "room": c.attrib.get("room"),
                "students": [s.attrib["id"] for s in c.findall("student") if "id" in s.attrib],
            }
            classes[cid] = sc
        print({
            "meta": meta, 
            "classes": classes
        })
        return {
            "meta": meta, 
            "classes": classes
        }

    # ---------- 工具 ----------
    @staticmethod
    def _to_int(x, default = None):
        if x is None:
            return default
        try:
            return int(x)
        except Exception:
            return default

    @staticmethod
    def _to_float(x, default = None):
        if x is None:
            return default
        try:
            return float(x)
        except Exception:
            return default
        
    @staticmethod
    def bits_to_list(bits):
        return [i for i, bit in enumerate(list(bits)) if bit == "1"]