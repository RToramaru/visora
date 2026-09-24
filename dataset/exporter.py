import json
import os
import shutil
import xml.etree.ElementTree as ET
from typing import Any, Dict, Iterable, List, Tuple
from xml.dom import minidom

from PIL import Image


EXTENSOES_IMAGEM = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")
EXTENSOES_ORIGEM = (".xml", ".json")
EXTENSOES_DESTINO = {"YOLOv8": ".txt", "COCO JSON": ".json", "Pascal VOC XML": ".xml"}


def carregar_anotacoes(rotulo_path: str, img_path: str) -> Tuple[List[Dict[str, Any]], int, int]:
    largura, altura = Image.open(img_path).size
    anotacoes = []
    if rotulo_path.lower().endswith(".xml"):
        root = ET.parse(rotulo_path).getroot()
        for objeto in root.findall("object"):
            caixa = objeto.find("bndbox")
            if caixa is None:
                continue
            anotacoes.append({
                "label": objeto.findtext("name", default="Objeto"),
                "type": "bbox",
                "bbox": [
                    float(caixa.findtext("xmin", default="0")),
                    float(caixa.findtext("ymin", default="0")),
                    float(caixa.findtext("xmax", default="0")),
                    float(caixa.findtext("ymax", default="0")),
                ],
            })
    else:
        with open(rotulo_path, "r", encoding="utf-8") as arquivo:
            dados = json.load(arquivo)
        for forma in dados.get("shapes", []):
            pontos = [[float(x), float(y)] for x, y in forma.get("points", [])]
            if not pontos:
                continue
            anotacoes.append({
                "label": forma.get("label", "Objeto"),
                "type": "polygon",
                "points": pontos,
                "bbox": [
                    min(ponto[0] for ponto in pontos), min(ponto[1] for ponto in pontos),
                    max(ponto[0] for ponto in pontos), max(ponto[1] for ponto in pontos),
                ],
            })
    return anotacoes, largura, altura


def serializar_anotacao(formato: str, anotacoes: List[Dict[str, Any]], nome_imagem: str,
                        largura: int, altura: int, classes: Dict[str, int]) -> str:
    if formato == "YOLOv8":
        linhas = []
        for anotacao in anotacoes:
            classe_id = classes[anotacao["label"]]
            if anotacao["type"] == "polygon":
                coordenadas = " ".join(
                    f"{max(0, min(1, x / largura)):.6f} {max(0, min(1, y / altura)):.6f}"
                    for x, y in anotacao["points"]
                )
                linhas.append(f"{classe_id} {coordenadas}")
            else:
                xmin, ymin, xmax, ymax = anotacao["bbox"]
                linhas.append(
                    f"{classe_id} {((xmin + xmax) / 2) / largura:.6f} "
                    f"{((ymin + ymax) / 2) / altura:.6f} "
                    f"{(xmax - xmin) / largura:.6f} {(ymax - ymin) / altura:.6f}"
                )
        return "\n".join(linhas) + ("\n" if linhas else "")

    if formato == "COCO JSON":
        categorias = [{"id": indice, "name": nome} for nome, indice in classes.items()]
        objetos = []
        for indice, anotacao in enumerate(anotacoes, start=1):
            xmin, ymin, xmax, ymax = anotacao["bbox"]
            objeto = {
                "id": indice,
                "image_id": 1,
                "category_id": classes[anotacao["label"]],
                "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                "area": max(0, xmax - xmin) * max(0, ymax - ymin),
                "iscrowd": 0,
            }
            if anotacao["type"] == "polygon":
                objeto["segmentation"] = [[coordenada for ponto in anotacao["points"] for coordenada in ponto]]
            objetos.append(objeto)
        return json.dumps({
            "images": [{"id": 1, "file_name": nome_imagem, "width": largura, "height": altura}],
            "annotations": objetos,
            "categories": categorias,
        }, indent=2, ensure_ascii=False)

    root = ET.Element("annotation")
    ET.SubElement(root, "filename").text = nome_imagem
    size_node = ET.SubElement(root, "size")
    ET.SubElement(size_node, "width").text = str(largura)
    ET.SubElement(size_node, "height").text = str(altura)
    ET.SubElement(size_node, "depth").text = "3"
    for anotacao in anotacoes:
        xmin, ymin, xmax, ymax = anotacao["bbox"]
        objeto = ET.SubElement(root, "object")
        ET.SubElement(objeto, "name").text = anotacao["label"]
        caixa = ET.SubElement(objeto, "bndbox")
        for nome, valor in (("xmin", xmin), ("ymin", ymin), ("xmax", xmax), ("ymax", ymax)):
            ET.SubElement(caixa, nome).text = str(int(valor))
    return minidom.parseString(ET.tostring(root)).toprettyxml(indent="  ")


def export_dataset(project_path: str, export_path: str, formato: str) -> Dict[str, Any]:
    if formato not in EXTENSOES_DESTINO:
        raise ValueError(f"Formato de exportação não suportado: {formato}")
    frames_path = os.path.join(project_path, "frames")
    image_path = frames_path if os.path.isdir(frames_path) else project_path
    annotations_path = os.path.join(project_path, "annotations")
    pairs = []
    for filename in os.listdir(image_path):
        if not filename.lower().endswith(EXTENSOES_IMAGEM):
            continue
        base_name = os.path.splitext(filename)[0]
        label_path = next(
            (os.path.join(annotations_path, f"{base_name}{extension}") for extension in EXTENSOES_ORIGEM
             if os.path.exists(os.path.join(annotations_path, f"{base_name}{extension}"))),
            None,
        )
        if label_path:
            pairs.append((os.path.join(image_path, filename), label_path, filename))
    if not pairs:
        raise ValueError("Nenhum frame rotulado foi encontrado para exportação!")

    samples = []
    class_names = set()
    for image, label, filename in pairs:
        annotations, width, height = carregar_anotacoes(label, image)
        samples.append((image, filename, annotations, width, height))
        class_names.update(annotation["label"] for annotation in annotations)
    classes = {name: index for index, name in enumerate(sorted(class_names))}

    destinations = {
        "train_images": os.path.join(export_path, "train", "images"),
        "train_labels": os.path.join(export_path, "train", "labels"),
        "val_images": os.path.join(export_path, "val", "images"),
        "val_labels": os.path.join(export_path, "val", "labels"),
    }
    for destination in destinations.values():
        os.makedirs(destination, exist_ok=True)

    split_index = int(len(samples) * 0.8)
    if split_index == 0:
        split_index = len(samples)
    for index, (image, filename, annotations, width, height) in enumerate(samples):
        split = "train" if index < split_index else "val"
        image_destination = destinations[f"{split}_images"]
        label_destination = destinations[f"{split}_labels"]
        shutil.copy2(image, os.path.join(image_destination, filename))
        label_name = f"{os.path.splitext(filename)[0]}{EXTENSOES_DESTINO[formato]}"
        content = serializar_anotacao(formato, annotations, filename, width, height, classes)
        with open(os.path.join(label_destination, label_name), "w", encoding="utf-8") as arquivo:
            arquivo.write(content)

    train_count = min(split_index, len(samples))
    return {
        "total": len(samples),
        "train": train_count,
        "val": len(samples) - train_count,
        "format": formato,
        "path": export_path,
    }
