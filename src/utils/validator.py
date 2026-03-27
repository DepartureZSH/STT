import os
from src.utils.dataReader import PSTTReader
import pathlib
import yaml
import copy
try:
    import torch
except Exception:  # pragma: no cover
    from src.utils.torch_compat import torch
import json
import numpy as np
from src.utils.constraints import HardConstraints, SoftConstraints
from src.utils.solutionReader import PSTTReader as SolutionReader
from src.utils.dataReader import PSTTReader
from math import inf
from tqdm import tqdm
folder = pathlib.Path(__file__).parent.resolve()

def load_cfg(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

class Agent:
    def __init__(self, class_info):
        self.id = class_info['id']
        self.limit = class_info['limit']
        self.parent = class_info['parent']
        self.room_required = class_info['room_required']
        self.room_options = class_info['room_options']
        self.time_options = class_info['time_options']
        self.action_space, self.action_map = self._actions()
        self.value = len(self.action_space)
        self.candidate = None
        self.action = None
        self.include_students = False
        self.penalty = inf

    def _actions(self):
        action_map = {}
        actions = []
        if self.room_required:
            for i in range(len(self.room_options)):
                p1 = self.room_options[i]['penalty']
                for j in range(len(self.time_options)):
                    p2 = self.time_options[j]['penalty']
                    actions.append((i, j, p1 + p2))
        else:
            for j in range(len(self.time_options)):
                p = self.time_options[j]['penalty']
                actions.append((-1, j, p))
        actions = sorted(actions, key=lambda k:k[2])
        for action in actions:
            weeks_bits, days_bits, start = self.time_options[action[1]]["optional_time_bits"][:3]
            if self.room_required:
                rid = self.room_options[action[0]]["id"]
                action_map[weeks_bits, days_bits, start, rid] = actions.index(action)
            else:
                action_map[weeks_bits, days_bits, start] = actions.index(action)
        return actions, action_map
    
    def result(self):
        if self.action == None:
            room_id = None
            topt = self.time_options[0]
        elif self.room_required==False:
            room_id = None
            time_option_idx = self.action[1]
            topt = self.time_options[time_option_idx]
            # return f"""<class id="{self.id}" days="{time_option[1]}" start="{time_option[2]}" weeks="{time_option[0]}"></class>"""
        else:
            oid = self.action[0]
            room_id = self.room_options[oid]['id']
            time_option_idx = self.action[1]
            topt = self.time_options[time_option_idx]
            # return f"""<class id="{self.id}" days="{time_option[1]}" start="{time_option[2]}" weeks="{time_option[0]}" room="{room_id}"></class>"""
        return {self.id: (topt, self.room_required, room_id, None)}
    
    def get_action_idx(self, weeks_bits, days_bits, start, rid):
        if self.room_required:
            idx = self.action_map[weeks_bits, days_bits, start, rid]
        else:
            idx = self.action_map[weeks_bits, days_bits, start]
        return idx
        
class solution:
    def __init__(self, reader, solution_file):
        self.reader = reader

        solution_reader = SolutionReader(solution_file)

        self.optimization = self.reader.optimization
        self.hard_constrains = self.reader.distributions['hard_constraints']
        self.soft_constrains = self.reader.distributions['soft_constraints']
        self.Hard_validator = HardConstraints()
        self.Soft_validator = SoftConstraints()
        self.rooms = copy.deepcopy(self.reader.rooms)

        self.agents = [] # classes
        self.cid2ind = {}
        i = 0
        for key, each in self.reader.classes.items():
            agent = Agent(each)
            self.agents.append(agent)
            self.cid2ind[key] = i
            i += 1
        
        self.not_assignment = []

        for i, (cid, solu) in enumerate(solution_reader.classes.items()):
            ind = self.cid2ind[cid]
            agent = self.agents[ind]
            optional_time = solu['optional_time']
            if optional_time:
                weeks_bits, days_bits, start = optional_time
            else:
                self.not_assignment.append(cid)
                continue
            rid = solu['room']
            if agent.room_required and rid == None:
                self.not_assignment.append(cid)
                continue
            aidx = agent.get_action_idx(weeks_bits, days_bits, start, rid)
            action = agent.action_space[aidx]
            room_option_ind, time_option_ind, penalty = action
            agent.action = action
            time_option = agent.time_options[time_option_ind]
            if room_option_ind != -1:
                room_option = agent.room_options[room_option_ind]
                self.rooms[room_option['id']]['ocupied'].append((cid, time_option['optional_time_bits'], action))

        self.travel = self.reader.travel
        self.Hard_validator.setTravel(self.travel)
        self.Hard_validator.sefnrDays(self.reader.nrDays)
        self.Hard_validator.sefnrWeeks(self.reader.nrWeeks)
        self.Hard_validator.setCid2ind(self.cid2ind)
        self.Soft_validator.setTravel(self.travel)
        self.Soft_validator.sefnrDays(self.reader.nrDays)
        self.Soft_validator.sefnrWeeks(self.reader.nrWeeks)
        self.Soft_validator.setCid2ind(self.cid2ind)
        
        # self.timeTable_matrix = self.reader.timeTable_matrix
        # self.rooms = solutions['rooms'] # {'id': {'id', 'capacity', 'unavailables_bits', 'unavailables', 'occupid'# (cid, time_bits, value)}}

    def total_penalty(self):
        penalty = 0
        Room_penalty = 0
        Time_penalty = 0
        Times = []
        Rooms = []
        for agent in self.agents:
            if agent.id not in self.not_assignment:
                penalty += agent.penalty
                if agent.action == None:
                    # print(f"agent {agent.id} not assigned!")
                    continue
                rid, tid, _ = agent.action
                if agent.room_required:
                    Room_penalty += agent.room_options[rid]['penalty']
                    if agent.room_options[rid]['penalty']:
                        Rooms.append((agent.id, f"Room {agent.room_options[rid]['id']}", agent.room_options[rid]['penalty']))
                Time_penalty += agent.time_options[tid]['penalty']
                if agent.time_options[tid]['penalty']:
                    Times.append((agent.id, agent.time_options[tid]['optional_time_bits'], agent.time_options[tid]['penalty']))
        self.Hard_validator.setClasses(self.agents)
        Hard_Distributions = []
        for hard_constrain in self.hard_constrains:
            violate = self.Hard_validator._violation_rate(hard_constrain)
            if violate:
                Hard_Distributions.append(hard_constrain)
        if len(Hard_Distributions) > 0:
            print("Solution Infeasiable!")
            for hard_constrain in Hard_Distributions:
                print("Hard Distributions violate: ", hard_constrain)

        self.Soft_validator.setClasses(self.agents)
        Distribution_penalty = 0
        Student_penalty = 0
        Distributions = []
        for soft_constrain in self.soft_constrains:
            # TODO
            violation_rate = self.Soft_validator._violation_rate(soft_constrain)
            if violation_rate:
                Distributions.append((soft_constrain, violation_rate, soft_constrain['penalty'], violation_rate * soft_constrain['penalty']))
                Distribution_penalty += violation_rate * soft_constrain['penalty']
                penalty += violation_rate * soft_constrain['penalty']
            Total_cost = self.optimization["time"] * Time_penalty + \
                        self.optimization["room"] * Room_penalty + \
                        self.optimization["distribution"] * Distribution_penalty + \
                        self.optimization["student"] * Student_penalty
        valid = True
        if len(self.not_assignment) > 0 or len(Hard_Distributions) > 0:
            valid = False
        return {
            "not assignment": self.not_assignment,
            "valid": valid,
            "penalty": penalty,
            "Total_cost": Total_cost,
            "Student conflicts": "TODO",
            "Time penalty": Time_penalty,
            "Room penalty": Room_penalty,
            "Time details": Times,
            "Room details": Rooms,
            "Distribution penalty": Distribution_penalty,
            "Distributions": Distributions
        }

    def check_assignment(self, cid):
        i = self.cid2ind[cid]
        action = self.agents[i].action
        time_option = self.agents[i].time_options[action[1]]['optional_time_bits'] if action else None
        room_option = self.agents[i].room_options[action[0]] if action and self.agents[i].room_required else None
        # print(f"check assignment for agent {cid}, time_option: {time_option} room_option: {room_option}")

    def check(self, type_name="SameRoom"):
        for constraint in self.hard_constrains:
            if constraint['type'] == type_name:
                # print(type_name)
                if type_name in ["SameRoom", "DifferentRoom"]:
                    for cid in constraint['classes']:
                        i = self.cid2ind[cid]
                        # print(f"agent {cid}: Room {self.agents[i].action[0]}", end=" | ")
                if type_name in ["SameStart", "SameTime", "DifferentTime", "Precedence"]:
                    for cid in constraint['classes']:
                        i = self.cid2ind[cid]
                        if self.agents[i].action:
                            oid = self.agents[i].action[1]
                            # print(f"agent {cid}: time {self.agents[i].time_options[oid]['optional_time_bits']}", end=" | ")
                        else:
                            pass
                            # print(f"agent {cid}: time not assign", end=" | ")
                # print("")

    def results(self):
        assignment = {}
        for agent in self.agents:
            result = agent.result()
            assignment.update(result)
        return assignment

def solus_validate(pname, xml_path, solu_path):
    file = f'{xml_path}/{pname}.xml'
    reader = PSTTReader(file)
    solution_list = []
    invalid_solutions = []
    for solu_file in os.listdir(solu_path):
        if solu_file.endswith('.xml') and solu_file.startswith('solution'):
            valid, result = solu_validate(pname, xml_path, solu_path, solu_file, reader)
            if valid:
                solution_list.append((solu_file, result["Total_cost"], result["Time penalty"], result["Room penalty"], result["Distribution penalty"]))
            else:
                invalid_solutions.append(result)
    if len(solution_list) > 0:
        solution_list = sorted(solution_list, key=lambda k: k[1])
        print("The best solution: ")
        print("Solution: ", solution_list[0][0])
        print("Total_cost: ", solution_list[0][1])
        print("Time penalty: ", solution_list[0][2])
        print("Room penalty: ", solution_list[0][3])
        print("Distribution penalty: ", solution_list[0][4])
        return solution_list
    else:
        not_assigned_count = {}
        for result in invalid_solutions:
            for cid in result['not assignment']:
                if not_assigned_count.get(cid, 0) == 0: not_assigned_count[cid] = 1
                else: not_assigned_count[cid] += 1
        print(not_assigned_count)
        print("relative hard constraints:")
        for cid in not_assigned_count.keys():
            for hc in reader.distributions['hard_constraints']:
                if cid in hc['classes']:
                    print(hc)
        return []

def solu_validate(pname, xml_path, solu_path, solu_file, reader=None):
    if reader == None: 
        file = f'{xml_path}/{pname}.xml'
        reader = PSTTReader(file)
    solution_file = f"{solu_path}/{solu_file}"
    solu = solution(reader, solution_file)
    result = solu.total_penalty()
    # print()
    if result['valid']:
        # print(f"valid solution: {solution_file}")
        # print("Total_cost", result["Total_cost"])
        # print("Time penalty", result["Time penalty"])
        # print("Room penalty", result["Room penalty"])
        # print("Distribution penalty", result["Distribution penalty"])
        return True, result
    else:
        pass
        # print(f"invalid solution: {solution_file}")
        # print(f"not assigned classes:", result['not assignment'])
        # exit(1)
    return False, result