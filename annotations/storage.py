import json
import os
import xml.etree.ElementTree as ET
from typing import Any, Dict, List
from xml.dom import minidom

from annotations.models import annotation_extension


def save_annotations(
    project_path: str,
    image_path: str,
    annotations: List[Dict[str, Any]],
    topologia: str,
    image_size: tuple[int, int],
) -> str | None:
    annotations_path = os.path.join(project_path, "annotations")
    os.makedirs(annotations_path, exist_ok=True)
    image_name = os.path.basename(image_path)
    base_name = os.path.splitext(image_name)[0]

    if not annotations:
        for extension in (".xml", ".json"):
            old_path = os.path.join(annotations_path, f"{base_name}{extension}")
            if os.path.exists(old_path):
                os.remove(old_path)
        return None

    width, height = image_size
    if topologia == "Bounding Boxes":
        output_path = os.path.join(annotations_path, f"{base_name}.xml")
        root = ET.Element("annotation")
        ET.SubElement(root, "folder").text = os.path.basename(project_path)
        ET.SubElement(root, "filename").text = image_name
        ET.SubElement(root, "path").text = os.path.abspath(image_path)
        size_node = ET.SubElement(root, "size")
        ET.SubElement(size_node, "width").text = str(width)
        ET.SubElement(size_node, "height").text = str(height)
        ET.SubElement(size_node, "depth").text = "3"

        for annotation in annotations:
            if annotation.get("type") != "bbox":
                continue
            object_node = ET.SubElement(root, "object")
            ET.SubElement(object_node, "name").text = str(annotation["label"])
            ET.SubElement(object_node, "pose").text = "Unspecified"
            ET.SubElement(object_node, "truncated").text = "0"
            ET.SubElement(object_node, "difficult").text = "0"
            box_node = ET.SubElement(object_node, "bndbox")
            for name in ("xmin", "ymin", "xmax", "ymax"):
                ET.SubElement(box_node, name).text = str(annotation[name])
    else:
        output_path = os.path.join(annotations_path, f"{base_name}.json")
        data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": image_name,
            "imageData": None,
            "imageHeight": height,
            "imageWidth": width,
        }
        for annotation in annotations:
            if annotation.get("type") == "polygon":
                data["shapes"].append({
                    "label": annotation["label"],
                    "points": annotation["points"],
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                })
        with open(output_path, "w", encoding="utf-8") as arquivo:
            json.dump(data, arquivo, indent=2, ensure_ascii=False)
        return output_path

    xml_string = minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")
    with open(output_path, "w", encoding="utf-8") as arquivo:
        arquivo.write(xml_string)
    return output_path


def load_polygon_annotations(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as arquivo:
        data = json.load(arquivo)
    return [
        {"type": "polygon", "label": shape.get("label", "Objeto"), "points": shape.get("points", [])}
        for shape in data.get("shapes", [])
    ]


def load_bbox_annotations(path: str) -> List[Dict[str, Any]]:
    root = ET.parse(path).getroot()
    annotations = []
    for object_node in root.findall("object"):
        box_node = object_node.find("bndbox")
        if box_node is None:
            continue
        annotations.append({
            "type": "bbox",
            "label": object_node.findtext("name", default="Objeto"),
            "xmin": int(float(box_node.findtext("xmin", default="0"))),
            "ymin": int(float(box_node.findtext("ymin", default="0"))),
            "xmax": int(float(box_node.findtext("xmax", default="0"))),
            "ymax": int(float(box_node.findtext("ymax", default="0"))),
        })
    return annotations


def load_annotations(path: str, topologia: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    return load_bbox_annotations(path) if topologia == "Bounding Boxes" else load_polygon_annotations(path)
