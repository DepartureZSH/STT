import itertools
import numpy as np
try:
    import torch
except Exception:  # pragma: no cover
    from src.utils.torch_compat import torch

class ConstraintBase:
    def __init__(self):
        self.nrDays = 7
        self.nrWeeks = 16
        self.travel = {}
        self.classes = []
        self.cid2ind = {}
    
    def sefnrDays(self, nrDays):
        self.nrDays = nrDays

    def sefnrWeeks(self, nrWeeks):
        self.nrWeeks = nrWeeks

    def setTravel(self, travel):
        self.travel = travel

    def setClasses(self, classes):
        self.classes = classes
    
    def setCid2ind(self, cid2ind):
        self.cid2ind = cid2ind

    ##############################################################
    # Tools
    ##############################################################
    def getOptions(self, ind, isCandidate=False):
        if isCandidate:
            rop = self.classes[ind].candidate[0]
            top = self.classes[ind].candidate[1]
            time_option = self.classes[ind].time_options[top]["optional_time_bits"]
            if rop == -1: room_option = None
            else: room_option = self.classes[ind].room_options[rop]
        else:
            if self.classes[ind].action == None:
                return None, None
            rop = self.classes[ind].action[0]
            top = self.classes[ind].action[1]
            time_option = self.classes[ind].time_options[top]["optional_time_bits"]
            if rop == -1: room_option = None
            else: room_option = self.classes[ind].room_options[rop]
        return time_option, room_option

    def merge_slots(self, class_time_slots, S):
        merge_time_slots = []
        merge_time_len = []
        breaks = -1
        class_time_slots = sorted(class_time_slots, key=lambda x: x[0])
        max_slots = 0
        for i, time_slot in enumerate(class_time_slots):
            if breaks == -1:
                merge_time_slots.append(time_slot)
                merge_time_len.append(1)
                breaks += 1
                continue
            start1, end1 = merge_time_slots[breaks]
            start2, end2 = time_slot
            if start1 + end1 + S >= start2:
                merge_time_slots[breaks][1] = max(start2 + end2, start1 + end1) - start1
                merge_time_len[breaks] += 1 
            else:
                merge_time_slots.append(time_slot)
                merge_time_len.append(1)
                breaks += 1
        return breaks, merge_time_slots, merge_time_len
        
# ================================================================
#                      Hard Constraints
# ================================================================
class HardConstraints(ConstraintBase):
    """违反即失败：返回 True=违反，False=满足"""

    def _violation_rate(self, cons, cid=None):
        ctype = cons["type"]
        if "(" in ctype and ")" in ctype:
            base, attr = ctype.split("(")[0], ctype.split("(")[1].split(")")[0]
            if cid:
                return getattr(self, base)(cons, attr, cid)
            return getattr(self, base)(cons, attr)
        if cid:
            return getattr(self, ctype)(cons, cid)
        return getattr(self, ctype)(cons)

    def RoomConflicts(self, cid, room_assignments):
        ind = self.cid2ind[cid]
        time_option, _ = self.getOptions(ind, isCandidate=True)
        week_bits1, day_bits1, start1, end1 = time_option
        for assignment in room_assignments:
            week_bits2, day_bits2, start2, end2 = assignment[1]
            days_int1 = int(day_bits1, 2)
            days_int2 = int(day_bits2, 2)
            and_days = days_int1 & days_int2
            week_int1 = int(week_bits1, 2)
            week_int2 = int(week_bits2, 2)
            and_week = week_int1 & week_int2
            if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0):
                return True
        return False
    
    def RoomUnavailable(self, cid, unavailables_bits):
        ind = self.cid2ind[cid]
        time_option, _ = self.getOptions(ind, isCandidate=True)
        week_bits1, day_bits1, start1, end1 = time_option
        for unavailables_bit in unavailables_bits:
            week_bits2, day_bits2, start2, end2 = unavailables_bit
            days_int1 = int(day_bits1, 2)
            days_int2 = int(day_bits2, 2)
            and_days = days_int1 & days_int2
            week_int1 = int(week_bits1, 2)
            week_int2 = int(week_bits2, 2)
            and_week = week_int1 & week_int2
            if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0):
                return True
        return False

    # ---- Pair-wise 类型 ----

    def SameRoom(self, hc, cid=None):
        cids=hc["classes"]
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action == None:
                        continue
                    _, room_option1 = self.getOptions(ind1, isCandidate=True)
                    _, room_option2 = self.getOptions(ind2, isCandidate=False)
                    if room_option1 and room_option2 and room_option1['id'] != room_option2['id']:
                        return True
        return False

    def DifferentRoom(self, hc, cid=None):
        cids=hc["classes"]
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action == None:
                        continue
                    _, room_option1 = self.getOptions(ind1, isCandidate=True)
                    _, room_option2 = self.getOptions(ind2, isCandidate=False)
                    if room_option1 and room_option2 and room_option1['id'] == room_option2['id']:
                        return True
        return False
    
    def SameStart(self, hc, cid=None):
        cids = hc["classes"]; 
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        _, _, start1, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if start1 != start2:
                            return True
        return False

    def SameTime(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        _, _, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if start1 <= start2 and start2 + end2 <= start1 + end1:
                            continue
                        elif start2 <= start1 and start1 + end1 <= start2 + end2:
                            continue
                        else:
                            return True
        return False

    def DifferentTime(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        _, _, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if (start1 + end1 <= start2) or (start2 + end2 <= start1):
                            continue
                        else:
                            return True
        return False

    def SameDays(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        _, day_bits2, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        or_ = days_int1 | days_int2
                        if not (or_ == days_int1 or or_ == day_bits2):
                            return True
        return False

    def DifferentDays(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        _, day_bits2, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        and_ = days_int1 & days_int2
                        if not and_ == 0:
                            return True
        return False

    def SameWeeks(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, _, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, _, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        week_int2 = int(week_bits2, 2)
                        or_ = week_int1 | week_int2
                        if not (or_ == week_int1 or or_ == week_int2):
                            return True
        return False

    def DifferentWeeks(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, _, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, _, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        week_int2 = int(week_bits2, 2)
                        and_ = week_int1 & week_int2
                        if not and_ == 0:
                            return True
        return False

    def Overlap(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int1 = int(day_bits1, 2)
                        days_int2 = int(day_bits2, 2)
                        and_days = days_int1 & days_int2
                        week_int1 = int(week_bits1, 2)
                        week_int2 = int(week_bits2, 2)
                        and_week = week_int1 & week_int2
                        if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0):
                            continue
                        else:
                            return True
        return False

    def NotOverlap(self, hc, cid=None):
        cids = hc["classes"]
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int1 = int(day_bits1, 2)
                        days_int2 = int(day_bits2, 2)
                        and_days = days_int1 & days_int2
                        week_int1 = int(week_bits1, 2)
                        week_int2 = int(week_bits2, 2)
                        and_week = week_int1 & week_int2
                        if (start1 + end1 <= start2) or (start2 + end2 <= start1) or (and_days == 0) or (and_week == 0):
                            continue
                        else:
                            return True
        
        return False

    def SameAttendees(self, hc, cid=None):
        # (Ci.end + travel(Ci.room→Cj.room) ≤ Cj.start) ∨ (Cj.end + travel(Cj.room→Ci.room) ≤ Ci.start)
        # 或 天/周不重叠即满足
        cids = hc["classes"]
        if cid:
            ind1 = self.cid2ind[cid]
            room_required1 = self.classes[ind1].room_required
            if room_required1:
                rid1 = self.classes[ind1].candidate[0]
                room_id1 = self.classes[ind1].room_options[rid1]['id']
            else:
                room_id1 = -1
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            for i in cids:
                if i != cid:
                    ind2 = self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    room_required2 = self.classes[ind2].room_required
                    if room_required2:
                        rid2 = self.classes[ind2].action[0]
                        room_id2 = self.classes[ind2].room_options[rid2]['id']
                    else:
                        room_id2 = -1
                    travel1 = self.travel.get(room_id1, {}).get(room_id2, 0)
                    travel2 = self.travel.get(room_id2, {}).get(room_id1, 0)
                    oid2 = self.classes[ind2].action[1]
                    week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0): # Overlap
                        return True
                    elif (start1 + end1 + travel1 <= start2) or (start2 + end2 + travel2 <= start1) or (and_days == 0) or (and_week == 0):
                        continue
                    else:
                        return True
        return False

    def Precedence(self, hc, cid=None):
        # 列表顺序：C1 在 C2 之前... 按“first(week)->first(day)->end<=start”
        cids = hc["classes"]
        if cid:
            i = cids.index(cid)
            for j in range(len(cids)):
                if i != j:
                    cid2 = cids[j]
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        first_day1 = day_bits1.find('1')
                        first_day2 = day_bits2.find('1')
                        first_week1 = week_bits1.find('1')
                        first_week2 = week_bits2.find('1')
                        if i < j:
                            w_pre, d_pre, s_pre, e_pre = first_week1, first_day1, start1, end1
                            w_sub, d_sub, s_sub, e_sub = first_week2, first_day2, start2, end2
                        else:
                            w_pre, d_pre, s_pre, e_pre = first_week2, first_day2, start2, end2
                            w_sub, d_sub, s_sub, e_sub = first_week1, first_day1, start1, end1
                        if (w_pre < w_sub) or ( # first(week_i) < first(week_j) or
                            (w_pre == w_sub) and (
                                (d_pre < d_sub ) or ( # first(day_i) < first(day_j) or
                                    (d_pre == d_sub) and (s_pre+e_pre <= s_sub) # end_i <= start_j
                                )
                            )
                        ):
                            continue
                        else:
                            return True
        return False

    def WorkDay(self, hc, S, cid=None):
        # 同天同周：max(end)-min(start) ≤ S
        cids = hc["classes"]
        S = int(S)
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int1 = int(day_bits1, 2)
                        days_int2 = int(day_bits2, 2)
                        and_days = days_int1 & days_int2
                        week_int1 = int(week_bits1, 2)
                        week_int2 = int(week_bits2, 2)
                        and_week = week_int1 & week_int2
                        # if cid == "1087":
                        #     print("WorkDay cid: ", cid, " other classes ids: ",cids)
                        #     print(cid, " time slots: ", (week_bits1, day_bits1, start1, end1))
                        #     print(i, " time slots: ", (week_bits2, day_bits2, start2, end2))
                        if (and_days == 0) or (and_week == 0) or ((max(start1 + end1, start2 + end2) - min(start1, start2)) <= S):
                            continue
                        else:
                            return True
        return False

    def MinGap(self, hc, G, cid=None):
        # 同天同周：要求 end+G ≤ start（任意顺序其中之一）
        cids = hc["classes"]
        G = int(G)
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int1 = int(day_bits1, 2)
                        days_int2 = int(day_bits2, 2)
                        and_days = days_int1 & days_int2
                        week_int1 = int(week_bits1, 2)
                        week_int2 = int(week_bits2, 2)
                        and_week = week_int1 & week_int2
                        if (and_days == 0) or (and_week == 0) or (start1 + end1 + G <= start2) or (start2 + end2 + G <= start1):
                            continue
                        else:
                            return True
        return False

    def MaxDays(self, hc, D, cid=None):
        # countNonzeroBits( OR_i days_i ) ≤ D
        cids = hc["classes"]
        days_all_ints = 0
        # ind = self.cid2ind[cid]
        # oid = self.classes[ind].candidate[1]
        # _, day_bits, _, _ = self.classes[ind].time_options[oid]["optional_time_bits"]
        # day_ints = int(day_bits, 2)
        # days_all_ints = day_ints
        D = int(D)
        for i in cids:
            ind1 = self.cid2ind[i]
            if self.classes[ind1].action == None:
                continue
            oid1 = self.classes[ind1].action[1]
            _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            day_ints1 = int(day_bits1, 2)
            days_all_ints = days_all_ints | day_ints1
        days_all = bin(days_all_ints)[2:]
        if days_all.count("1") > D:
            return True
        return False

    def MaxDayLoad(self, hc, S, cid=None):
        # 对每个 (w,d)：DayLoad(d,w) = sum(length of classes covering该 day/week) ≤ S
        cids = hc["classes"]
        S = int(S)
        time_options = []
        total_load = 0
        for i in cids:
            ind = self.cid2ind[i]
            time_option, _ = self.getOptions(ind, isCandidate=(i==cid))
            if time_option: 
                total_load += time_option[3]
                time_options.append(time_option)
        if total_load <= S:
            return False
        for w in range(self.nrWeeks):
            for d in range(self.nrDays):
                dayloads = 0
                for time_option in time_options:
                    if time_option[0][w] == '1' and time_option[1][d] == '1':
                        dayloads += time_option[3]
                        if dayloads > S:
                            return True
        return False

    def MaxBreaks(self, hc, RS, cid=None):
        # RS = "R,S"：每天最多 R 个 break（gap > S 才算 break）
        cids = hc["classes"]
        R, S = map(int, RS.split(","))
        time_options = []
        for i in cids:
            ind = self.cid2ind[i]
            time_option, _ = self.getOptions(ind, isCandidate=(i==cid))
            if time_option: time_options.append(time_option)
        for w in range(self.nrWeeks):
            for d in range(self.nrDays):
                count = 0
                valid_top = []
                for time_option in time_options:
                    if time_option[0][w] == '1' and time_option[1][d] == '1':
                        count +=1
                        valid_top.append([time_option[2], time_option[3]])
                if count > R:
                    breaks, _, _ = self.merge_slots(valid_top, S)
                    if breaks > R:
                        return True
        return False

    def MaxBlock(self, hc, MS, cid=None):
        # MS = "M,S"：合并间隔≤S 的块，每块长度 ≤ M，且仅考虑含≥2门课的块
        M, S = map(int, MS.split(","))
        cids = hc["classes"]
        time_options = []
        for i in cids:
            ind = self.cid2ind[i]
            time_option, _ = self.getOptions(ind, isCandidate=(i==cid))
            if time_option: time_options.append(time_option)
        if len(time_options) <= 1:
            return False 
        for w in range(self.nrWeeks):
            for d in range(self.nrDays):
                valid_top = []
                for time_option in time_options:
                    if time_option[0][w] == '1' and time_option[1][d] == '1':
                        valid_top.append([time_option[2], time_option[3]])
                if len(valid_top) <= 1:
                    continue
                _, merge_time_slots, merge_time_len = self.merge_slots(valid_top, S)
                for slots_len, slots in zip(merge_time_len, merge_time_slots):
                    if slots_len > 1 and slots[1] > M: 
                        return True
        return False

# ================================================================
#                      Soft Constraints
# ================================================================
class SoftConstraints(ConstraintBase):
    """
    返回违反率 (0~1) 或违反次数；外层乘 penalty。
    """
    def _violation_rate(self, cons, cid=None):
        ctype = cons["type"]
        if "(" in ctype and ")" in ctype:
            base, attr = ctype.split("(")[0], ctype.split("(")[1].split(")")[0]
            if cid:
                return getattr(self, base)(cons, attr, cid)
            return getattr(self, base)(cons, attr)
        if cid:
            return getattr(self, ctype)(cons, cid)
        return getattr(self, ctype)(cons)
    
    def SameRoom(self, sc, cid=None):
        cids=sc["classes"]; viol = 0
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action == None:
                        continue
                    _, room_option1 = self.getOptions(ind1, isCandidate=True)
                    _, room_option2 = self.getOptions(ind2, isCandidate=False)
                    if room_option1 and room_option2 and room_option1['id'] != room_option2['id']:
                        viol += 1
        else:
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    ind1, ind2 = self.cid2ind[cids[i]], self.cid2ind[cids[j]]
                    if self.classes[ind2].action == None:
                        continue
                    _, room_option1 = self.getOptions(ind1, isCandidate=False)
                    _, room_option2 = self.getOptions(ind2, isCandidate=False)
                    if room_option1 and room_option2 and room_option1['id'] != room_option2['id']:
                        viol += 1
        return viol

    def DifferentRoom(self, sc, cid=None):
        cids=sc["classes"]; viol = 0
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action == None:
                        continue
                    _, room_option1 = self.getOptions(ind1, isCandidate=True)
                    _, room_option2 = self.getOptions(ind2, isCandidate=False)
                    if room_option1 and room_option2 and room_option1['id'] == room_option2['id']:
                        viol += 1
        else:
            for i in range(len(cids)):
                for j in range(i + 1, len(cids)):
                    ind1, ind2 = self.cid2ind[cids[i]], self.cid2ind[cids[j]]
                    if self.classes[ind2].action == None:
                        continue
                    _, room_option1 = self.getOptions(ind1, isCandidate=False)
                    _, room_option2 = self.getOptions(ind2, isCandidate=False)
                    if room_option1 and room_option2 and room_option1['id'] == room_option2['id']:
                        viol += 1
        return viol
    
    def SameStart(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        _, _, start1, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if start1 != start2:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind1, ind2 = self.cid2ind[cid1], self.cid2ind[cid2]
                    if self.classes[ind2].action==None or self.classes[ind1].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].action[1]
                        _, _, start1, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if start1 != start2:
                            viol += 1
        return viol

    def SameTime(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        _, _, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if start1 <= start2 and end2 <= end1:
                            continue
                        elif start2 <= start1 and end1 <= end2:
                            continue
                        else:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind1, ind2 = self.cid2ind[cid1], self.cid2ind[cid2]
                    if self.classes[ind2].action==None or self.classes[ind1].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].action[1]
                        _, _, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if start1 <= start2 and start2 + end2 <= start1 + end1:
                            continue
                        elif start2 <= start1 and start1 + end1 <= start2 + end2:
                            continue
                        else:
                            viol += 1
        return viol

    def DifferentTime(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            for i in cids:
                if i != cid:
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        _, _, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if (start1 + end1 <= start2) or (start2 + end2 <= start1):
                            continue
                        else:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind1, ind2 = self.cid2ind[cid1], self.cid2ind[cid2]
                    if self.classes[ind2].action==None or self.classes[ind1].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].action[1]
                        _, _, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        _, _, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        if (start1 + end1 <= start2) or (start2 + end2 <= start1):
                            continue
                        else:
                            viol += 1
        return viol

    def SameDays(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        _, day_bits2, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        or_ = days_int1 | days_int2
                        if not (or_ == days_int1 or or_ == days_int2):
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                days_int1 = int(day_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        _, day_bits2, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        or_ = days_int1 | days_int2
                        if not (or_ == days_int1 or or_ == days_int2):
                            viol += 1
        return viol

    def DifferentDays(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        _, day_bits2, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        and_ = days_int1 & days_int2
                        if not and_ == 0:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                days_int1 = int(day_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        _, day_bits2, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        and_ = days_int1 & days_int2
                        if not and_ == 0:
                            viol += 1
        return viol

    def SameWeeks(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, _, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, _, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        week_int2 = int(week_bits2, 2)
                        or_ = week_int1 | week_int2
                        if not (or_ == week_int1 or or_ == week_int2):
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                week_bits1, _, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                week_int1 = int(week_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, _, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        week_int2 = int(week_bits2, 2)
                        or_ = week_int1 | week_int2
                        if not (or_ == week_int1 or or_ == week_int2):
                            viol += 1
        return viol

    def DifferentWeeks(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, _, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, _, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        week_int2 = int(week_bits2, 2)
                        and_ = week_int1 & week_int2
                        if not and_ == 0:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                week_bits1, _, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                week_int1 = int(week_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, _, _, _ = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        week_int2 = int(week_bits2, 2)
                        and_ = week_int1 & week_int2
                        if not and_ == 0:
                            viol += 1
        return viol

    def Overlap(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0):
                            continue
                        else:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                days_int1 = int(day_bits1, 2)
                week_int1 = int(week_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0):
                            continue
                        else:
                            viol += 1
        return viol
    
    def NotOverlap(self, sc, cid=None):
        cids = sc["classes"]; viol=0
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (start1 + end1 <= start2) or (start2 + end2 <= start1) or (and_days == 0) or (and_week == 0):
                            continue
                        else:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                days_int1 = int(day_bits1, 2)
                week_int1 = int(week_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (start1 + end1 <= start2) or (start2 + end2 <= start1) or (and_days == 0) or (and_week == 0):
                            continue
                        else:
                            viol += 1
        return viol
        
    def SameAttendees(self, sc, cid=None):
        cids = sc["classes"]; viol = 0
        if cid:
            ind1 = self.cid2ind[cid]
            room_required1 = self.classes[ind1].room_required
            if room_required1:
                rid1 = self.classes[ind1].candidate[0]
                room_id1 = self.classes[ind1].room_options[rid1]['id']
            else:
                room_id1 = -1
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            for i in cids:
                if i != cid:
                    ind2 = self.cid2ind[i]
                    if self.classes[ind2].action==None:
                        continue
                    room_required2 = self.classes[ind2].room_required
                    if room_required2:
                        rid2 = self.classes[ind2].action[0]
                        room_id2 = self.classes[ind2].room_options[rid2]['id']
                    else:
                        room_id2 = -1
                    travel1 = self.travel.get(room_id1, {}).get(room_id2, 0)
                    travel2 = self.travel.get(room_id2, {}).get(room_id1, 0)
                    oid2 = self.classes[ind2].action[1]
                    week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0): # Overlap
                        viol += 1
                    elif (start1 + end1 + travel1 <= start2) or (start2 + end2 + travel2 <= start1) or (and_days == 0) or (and_week == 0):
                        continue
                    else:
                        viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                room_required1 = self.classes[ind1].room_required
                if room_required1:
                    if self.classes[ind1].action==None:
                        continue
                    rid1 = self.classes[ind1].action[0]
                    room_id1 = self.classes[ind1].room_options[rid1]['id']
                else:
                    room_id1 = -1
                oid1 = self.classes[ind1].action[1]
                week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    room_required2 = self.classes[ind2].room_required
                    if room_required2:
                        rid2 = self.classes[ind2].action[0]
                        room_id2 = self.classes[ind2].room_options[rid2]['id']
                    else:
                        room_id2 = -1
                    travel1 = self.travel.get(room_id1, {}).get(room_id2, 0)
                    travel2 = self.travel.get(room_id2, {}).get(room_id1, 0)
                    oid2 = self.classes[ind2].action[1]
                    week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                    days_int1 = int(day_bits1, 2)
                    days_int2 = int(day_bits2, 2)
                    and_days = days_int1 & days_int2
                    week_int1 = int(week_bits1, 2)
                    week_int2 = int(week_bits2, 2)
                    and_week = week_int1 & week_int2
                    if (start1 < start2 + end2) and (start2 < start1 + end1) and (not and_days == 0) and (not and_week == 0): # Overlap
                        viol += 1
                    elif (start1 + end1 + travel1 <= start2) or (start2 + end2 + travel2 <= start1) or (and_days == 0) or (and_week == 0):
                        continue
                    else:
                        viol += 1
        return viol

    def Precedence(self, sc, cid=None):
        # 对 i<j 的对进行评估，违反对数 / (N-1)
        cids = sc["classes"]; viol=0
        if cid:
            i = cids.index(cid)
            for j in range(len(cids)):
                if i != j:
                    cid2 = cids[j]
                    ind1, ind2 = self.cid2ind[cid], self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid1 = self.classes[ind1].candidate[1]
                        week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        first_day1 = day_bits1.find('1')
                        first_day2 = day_bits2.find('1')
                        first_week1 = week_bits1.find('1')
                        first_week2 = week_bits2.find('1')
                        if i < j:
                            w_pre, d_pre, s_pre, e_pre = first_week1, first_day1, start1, end1
                            w_sub, d_sub, s_sub, e_sub = first_week2, first_day2, start2, end2
                        else:
                            w_pre, d_pre, s_pre, e_pre = first_week2, first_day2, start2, end2
                            w_sub, d_sub, s_sub, e_sub = first_week1, first_day1, start1, end1
                        if (w_pre < w_sub) or ( # first(week_i) < first(week_j) or
                            (w_pre == w_sub) and (
                                (d_pre < d_sub ) or ( # first(day_i) < first(day_j) or
                                    (d_pre == d_sub) and (s_pre+e_pre <= s_sub) # end_i <= start_j
                                )
                            )
                        ):
                            continue
                        else:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                first_day1 = day_bits1.find('1')
                first_week1 = week_bits1.find('1')
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        first_day2 = day_bits2.find('1')
                        first_week2 = week_bits2.find('1')
                        w_pre, d_pre, s_pre, e_pre = first_week1, first_day1, start1, end1
                        w_sub, d_sub, s_sub, e_sub = first_week2, first_day2, start2, end2
                        if (w_pre < w_sub) or ( # first(week_i) < first(week_j) or
                            (w_pre == w_sub) and (
                                (d_pre < d_sub ) or ( # first(day_i) < first(day_j) or
                                    (d_pre == d_sub) and (s_pre+e_pre <= s_sub) # end_i <= start_j
                                )
                            )
                        ):
                            continue
                        else:
                            viol += 1
            return viol

    def WorkDay(self, sc, S, cid=None):
        cids = sc["classes"]; viol=0
        S = int(S)
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (and_days == 0) or (and_week == 0) or ((max(start1 + end1, start2 + end2) - min(start1, start2)) <= S):
                            continue
                        else:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                days_int1 = int(day_bits1, 2)
                week_int1 = int(week_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (and_days == 0) or (and_week == 0) or ((max(start1 + end1, start2 + end2) - min(start1, start2)) <= S):
                            continue
                        else:
                            viol += 1
        return viol
        

    def MinGap(self, sc, G, cid=None):
        cids = sc["classes"]; viol=0
        G = int(G)
        if cid:
            ind1 = self.cid2ind[cid]
            oid1 = self.classes[ind1].candidate[1]
            week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            days_int1 = int(day_bits1, 2)
            week_int1 = int(week_bits1, 2)
            for j in cids:
                if j != cid:
                    ind2 = self.cid2ind[j]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (and_days == 0) or (and_week == 0) or (start1 + end1 + G <= start2) or (start2 + end2 + G <= start1):
                            continue
                        else:
                            viol += 1
        else:
            for i in range(len(cids)):
                cid1 = cids[i]
                ind1 = self.cid2ind[cid1]
                if self.classes[ind1].action==None:
                    continue
                oid1 = self.classes[ind1].action[1]
                week_bits1, day_bits1, start1, end1 = self.classes[ind1].time_options[oid1]["optional_time_bits"]
                days_int1 = int(day_bits1, 2)
                week_int1 = int(week_bits1, 2)
                for j in range(i + 1, len(cids)):
                    cid2 = cids[j]
                    ind2 = self.cid2ind[cid2]
                    if self.classes[ind2].action==None:
                        continue
                    else:
                        oid2 = self.classes[ind2].action[1]
                        week_bits2, day_bits2, start2, end2 = self.classes[ind2].time_options[oid2]["optional_time_bits"]
                        days_int2 = int(day_bits2, 2)
                        week_int2 = int(week_bits2, 2)
                        and_days = days_int1 & days_int2
                        and_week = week_int1 & week_int2
                        if (and_days == 0) or (and_week == 0) or (start1 + end1 + G <= start2) or (start2 + end2 + G <= start1):
                            continue
                        else:
                            viol += 1
        return viol
        

    def MaxDays(self, sc, D, cid=None):
        # 超过 D 的天数个数 / 可能的最大超额（这里直接返回“超额天数”作为违反度的一种度量）
        cids = sc["classes"]; viol = 0
        D = int(D)
        days_all_ints = 0
        for i in cids:
            ind1 = self.cid2ind[i]
            if cid and i == cid:
                oid1 = self.classes[ind1].candidate[1]
            else:
                if self.classes[ind1].action == None:
                    continue
                oid1 = self.classes[ind1].action[1]
            _, day_bits1, _, _ = self.classes[ind1].time_options[oid1]["optional_time_bits"]
            day_ints1 = int(day_bits1, 2)
            days_all_ints = days_all_ints | day_ints1
        days_all = bin(days_all_ints)[2:]
        work_days = days_all.count("1")
        if work_days > D:
            viol = work_days - D
        return viol

    def MaxDayLoad(self, sc, S, cid=None):
        # (∑_w,d max(DayLoad(d,w)-S,0)) / nrWeeks
        cids = sc["classes"]; viol = 0
        S = int(S)
        time_options = []
        total_load = 0
        for i in cids:
            ind = self.cid2ind[i]
            time_option, _ = self.getOptions(ind, isCandidate=(i==cid))
            if time_option: 
                total_load += time_option[3]
                time_options.append(time_option)
        if total_load <= S:
            return 0
        for w in range(self.nrWeeks):
            for d in range(self.nrDays):
                dayloads = 0
                for time_option in time_options:
                    if time_option[0][w] == '1' and time_option[1][d] == '1':
                        dayloads += time_option[3]
                        if dayloads > S:
                            viol += dayloads - S
                        # if time_option[1][d] == '1':
                        #     print(f"W{w+1}: {dayloads}/{S} slots")
        return viol / max(self.nrWeeks, 1)

    def MaxBreaks(self, sc, RS, cid=None):
        # ∑_w,d max(breaks - R, 0) / nrWeeks，break: gap > S
        R, S = map(int, RS.split(","))
        cids = sc["classes"]
        time_options = []
        total_extra_breaks = 0
        for i in cids:
            ind = self.cid2ind[i]
            time_option, _ = self.getOptions(ind, isCandidate=(i==cid))
            if time_option: time_options.append(time_option)
        for w in range(self.nrWeeks):
            for d in range(self.nrDays):
                count = 0
                valid_top = []
                for time_option in time_options:
                    if time_option[0][w] == '1' and time_option[1][d] == '1':
                        count +=1
                        valid_top.append([time_option[2], time_option[3]])
                if count > R:
                    breaks, _, _ = self.merge_slots(valid_top, S)
                    total_extra_breaks += max(breaks - R, 0)
        return int(total_extra_breaks / max(self.nrWeeks, 1))

    def MaxBlock(self, sc, MS, cid=None):
        # 统计合并块（间隙≤S），若某块含≥2门课且长度>M → 记 1；返回 (超限块数 / nrWeeks)
        M, S = map(int, MS.split(","))
        cids = sc["classes"]
        overM_blocks = 0
        time_options = []
        for i in cids:
            ind = self.cid2ind[i]
            time_option, _ = self.getOptions(ind, isCandidate=(i==cid))
            if time_option: time_options.append(time_option)
        for w in range(self.nrWeeks):
            for d in range(self.nrDays):
                valid_top = []
                for time_option in time_options:
                    if time_option[0][w] == '1' and time_option[1][d] == '1':
                        valid_top.append([time_option[2], time_option[3]])
                _, merge_time_slots, merge_time_len = self.merge_slots(valid_top, S)
                for slots_len, slots in zip(merge_time_len, merge_time_slots):
                    if slots_len > 1 and slots[1] > M:
                        overM_blocks += 1
        return int(overM_blocks / max(self.nrWeeks, 1))
    