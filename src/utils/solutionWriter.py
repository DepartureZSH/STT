import os
import xml.etree.ElementTree as ET

def _order_class_ids(class_ids):
    # 尝试按数字排序，否则按字符串
    def _key(x):
        return (0, int(x)) if x.isdigit() else (1, x)
    return sorted(class_ids, key=_key)

def export_solution_xml(
        assignments, 
        out_path: str,
        name,
        runtime_sec,
        cores,
        technique,
        author,
        institution,
        country,
        include_students = False
):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    attrib = {}
    attrib["name"] = name
    if runtime_sec is not None:
        attrib["runtime"] = f"{runtime_sec:.3f}"
    if cores is not None:
        attrib["cores"] = str(int(cores))
    if technique:
        attrib["technique"] = technique
    if author:
        attrib["author"] = author
    if institution:
        attrib["institution"] = institution
    if country:
        attrib["country"] = country

    root = ET.Element("solution", attrib=attrib)

    # 以稳定顺序输出 <class>，便于 diff
    for cid in _order_class_ids(list(assignments.keys())):
        time_option, room_required, room_id, student_ids = assignments[cid]
        if time_option is None:
            c_attr = {"id": str(cid)}
            c_elem = ET.SubElement(root, "class", c_attr)
            continue
        
        topt = time_option['optional_time_bits']
        c_attr = {
            "id": str(cid),
            "days": topt[1],
            "start": str(topt[2]),
            "weeks": topt[0],
        }
        if room_id is not None and room_required:
            c_attr["room"] = str(room_id)

        c_elem = ET.SubElement(root, "class", c_attr)

        if include_students:
            for sid in student_ids:
                ET.SubElement(c_elem, "student", {"id": str(sid)})
        
    # 缩进美化（Python 3.9+ 提供 ET.indent）
    tree = ET.ElementTree(root)
    try:
        ET.indent(tree, space="\t", level=0)  # type: ignore[attr-defined]
    except Exception:
        pass

    # 写入：手动写 XML 头 + DOCTYPE，再写 ElementTree 字节
    with open(out_path, "wb") as f:
        f.write(b'<?xml version="1.0" encoding="UTF-8"?> \n')
        f.write(b'<!DOCTYPE solution PUBLIC \n')
        f.write(b'\t"-//ITC 2019//DTD Problem Format/EN" \n')
        f.write(b'\t"http://www.itc2019.org/competition-format.dtd"> \n')
        tree.write(f, encoding="utf-8")

    return out_path