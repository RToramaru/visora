import json
import os
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

from annotations.models import annotation_extension
from annotations.storage import load_annotations, save_annotations
from dataset.exporter import export_dataset, serializar_anotacao
from domain.project import ProjectData, normalizar_project_data, validar_project_data
from media.media_service import extract_frames
from ml import mobile_sam_training, model_io, yolo_training
from services.project_discovery import inferir_topologia, validar_pasta_projeto
from services.recent_projects import RecentProjectsStore


def create_image(path, size=(100, 80)):
    Image.new("RGB", size, "white").save(path)


def create_bbox_xml(path):
    path.write_text(
        "<annotation><object><name>obj</name><bndbox>"
        "<xmin>10</xmin><ymin>20</ymin><xmax>50</xmax><ymax>60</ymax>"
        "</bndbox></object></annotation>", encoding="utf-8"
    )


def create_polygon_json(path, image_name="sample.png"):
    path.write_text(json.dumps({
        "imagePath": image_name,
        "imageHeight": 80,
        "imageWidth": 100,
        "shapes": [{"label": "obj", "points": [[10, 10], [50, 10], [30, 40]]}],
    }), encoding="utf-8")


def test_project_data_and_validation(tmp_path):
    project = ProjectData("demo", str(tmp_path), "Instance Mask", "1 item")
    assert project.to_dict()["topologia"] == "Instance Mask"
    assert normalizar_project_data({"nome": "demo", "caminho": str(tmp_path)})["volume"] == "0 itens"
    validar_project_data({"nome": "demo", "caminho": str(tmp_path), "topologia": "Bounding Boxes"})
    with pytest.raises(ValueError):
        validar_project_data({"nome": "", "caminho": str(tmp_path), "topologia": "Bounding Boxes"})
    with pytest.raises(ValueError):
        validar_project_data({"nome": "demo", "caminho": str(tmp_path), "topologia": "Unknown"})


def test_recent_projects_store_round_trip_and_limit(tmp_path):
    store = RecentProjectsStore(str(tmp_path / "recentes.json"), limit=2)
    store.upsert("one", str(tmp_path), "Bounding Boxes")
    store.upsert("two", str(tmp_path), "Instance Mask")
    store.upsert("three", str(tmp_path), "Semantic Mask")
    projects = store.load()
    assert [project["nome"] for project in projects] == ["three", "two"]
    (tmp_path / "recentes.json").write_text("invalid", encoding="utf-8")
    assert store.load() == []


def test_project_discovery(tmp_path):
    assert inferir_topologia(str(tmp_path)) == "Bounding Boxes"
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    create_polygon_json(annotations / "sample.json")
    assert inferir_topologia(str(tmp_path)) == "Instance Mask"
    assert validar_pasta_projeto(str(tmp_path))
    assert not validar_pasta_projeto(str(tmp_path / "missing"))


def test_annotation_storage_bbox_and_polygon(tmp_path):
    image = tmp_path / "sample.png"
    create_image(image)
    bbox_path = save_annotations(str(tmp_path), str(image), [
        {"type": "bbox", "label": "obj", "xmin": 10, "ymin": 20, "xmax": 50, "ymax": 60}
    ], "Bounding Boxes", (100, 80))
    assert bbox_path.endswith(".xml")
    assert load_annotations(bbox_path, "Bounding Boxes")[0]["label"] == "obj"
    polygon_path = save_annotations(str(tmp_path), str(image), [
        {"type": "polygon", "label": "obj", "points": [[10, 10], [50, 10], [30, 40]]}
    ], "Instance Mask", (100, 80))
    assert load_annotations(polygon_path, "Instance Mask")[0]["points"][1] == [50, 10]
    assert annotation_extension("Instance Mask") == ".json"
    save_annotations(str(tmp_path), str(image), [], "Instance Mask", (100, 80))
    assert not os.path.exists(polygon_path)


def test_annotation_storage_ignores_invalid_objects(tmp_path):
    xml_path = tmp_path / "invalid.xml"
    xml_path.write_text("<annotation><object><name>ignored</name></object></annotation>", encoding="utf-8")
    assert load_annotations(str(xml_path), "Bounding Boxes") == []
    assert load_annotations(str(tmp_path / "missing.xml"), "Bounding Boxes") == []


def test_dataset_export_all_formats(tmp_path):
    project = tmp_path / "project"
    frames = project / "frames"
    annotations = project / "annotations"
    frames.mkdir(parents=True)
    annotations.mkdir()
    image = frames / "sample.png"
    create_image(image)
    create_bbox_xml(annotations / "sample.xml")
    for format_name, extension in (("YOLOv8", ".txt"), ("COCO JSON", ".json"), ("Pascal VOC XML", ".xml")):
        output = tmp_path / format_name.replace(" ", "_")
        summary = export_dataset(str(project), str(output), format_name)
        assert summary["total"] == 1
        assert (output / "train" / "labels" / f"sample{extension}").exists()


def test_dataset_export_rejects_invalid_format_and_empty_project(tmp_path):
    with pytest.raises(ValueError):
        export_dataset(str(tmp_path), str(tmp_path / "out"), "invalid")
    (tmp_path / "frames").mkdir()
    (tmp_path / "annotations").mkdir()
    create_image(tmp_path / "frames" / "sample.png")
    with pytest.raises(ValueError):
        export_dataset(str(tmp_path), str(tmp_path / "out"), "YOLOv8")


def test_dataset_serialization_for_polygon():
    annotations = [{"label": "obj", "type": "polygon", "points": [[0, 0], [50, 0], [25, 40]], "bbox": [0, 0, 50, 40]}]
    yolo = serializar_anotacao("YOLOv8", annotations, "sample.png", 100, 80, {"obj": 0})
    coco = json.loads(serializar_anotacao("COCO JSON", annotations, "sample.png", 100, 80, {"obj": 0}))
    assert yolo.startswith("0 ")
    assert coco["annotations"][0]["segmentation"]


def test_extract_frames(tmp_path):
    video_path = str(tmp_path / "sample.avi")
    writer = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"MJPG"), 5, (32, 32))
    for _ in range(3):
        writer.write(np.zeros((32, 32, 3), dtype="uint8"))
    writer.release()
    count = extract_frames(video_path, str(tmp_path / "frames"), 1, 3)
    assert count == 3
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 3


def test_model_export_checkpoints(tmp_path):
    source = tmp_path / "model.pt"
    source.write_bytes(b"checkpoint")
    destination = tmp_path / "out"
    destination.mkdir()
    assert model_io.export_model(str(source), str(destination), "pth", "MobileSAM").endswith(".pth")
    with pytest.raises(ValueError):
        model_io.export_model(str(source), str(destination), "engine", "MobileSAM")


def test_model_io_yolo_paths(tmp_path, monkeypatch):
    image = tmp_path / "sample.png"
    create_image(image)

    class FakeResult:
        def plot(self):
            return np.zeros((80, 100, 3), dtype="uint8")

    class FakeModel:
        def __call__(self, *_args, **_kwargs):
            return [FakeResult()]
        def export(self, **_kwargs):
            output = tmp_path / "converted.onnx"
            output.write_bytes(b"onnx")
            return output

    monkeypatch.setattr(model_io, "YOLO", lambda *_args: FakeModel())
    monkeypatch.setattr(model_io, "ULTRALYTICS_DISPONIVEL", True)
    result = model_io.infer_image("model.pt", str(image), "YOLO")
    assert result.size == (100, 80)
    destination = tmp_path / "exported"
    destination.mkdir()
    assert model_io.export_model("model.pt", str(destination), "onnx", "YOLO").endswith(".onnx")


def test_model_io_mobile_sam_inference_and_onnx_dispatch(tmp_path, monkeypatch):
    image = tmp_path / "sample.png"
    create_image(image)

    class FakeModel:
        def to(self, _device):
            return self
        def eval(self):
            return self

    class FakeGenerator:
        def __init__(self, _model, **_options):
            pass
        def generate(self, _image):
            mask = np.zeros((80, 100), dtype=bool)
            mask[10:20, 10:20] = True
            return [{"segmentation": mask}]

    monkeypatch.setattr(model_io, "build_sam_vit_t", lambda **_kwargs: FakeModel())
    monkeypatch.setattr(model_io, "SamAutomaticMaskGenerator", FakeGenerator)
    monkeypatch.setattr(model_io, "MOBILE_SAM_DISPONIVEL", True)
    result = model_io.infer_image("sam.pt", str(image), "MobileSAM")
    assert result.size == (100, 80)
    monkeypatch.setattr(model_io, "_export_mobile_sam_onnx", lambda *_args: "sam.onnx")
    assert model_io.export_model("sam.pt", str(tmp_path), "onnx", "MobileSAM") == "sam.onnx"


def test_mobile_sam_sample_discovery(tmp_path):
    frames = tmp_path / "frames"
    annotations = tmp_path / "annotations"
    frames.mkdir()
    annotations.mkdir()
    create_image(frames / "sample.png")
    create_polygon_json(annotations / "sample.json")
    samples = mobile_sam_training.find_samples(str(tmp_path))
    assert len(samples) == 1
    assert samples[0][1][0] == [10.0, 10.0]


def test_mobile_sam_checkpoint_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(mobile_sam_training.urllib.request, "urlretrieve",
                        lambda url, path: (calls.append(url), Path(path).write_bytes(b"weights")))
    checkpoint = mobile_sam_training.ensure_base_checkpoint(str(tmp_path), lambda _message: None)
    assert os.path.exists(checkpoint)
    assert calls


def test_mobile_sam_training_dependency_and_dataset_errors(tmp_path, monkeypatch):
    monkeypatch.setattr(mobile_sam_training, "MOBILE_SAM_DISPONIVEL", False)
    with pytest.raises(RuntimeError, match="dependências"):
        mobile_sam_training.train_mobile_sam(str(tmp_path), 1, None, lambda: False, lambda _model: None, lambda _message: None)
    monkeypatch.setattr(mobile_sam_training, "MOBILE_SAM_DISPONIVEL", True)
    with pytest.raises(ValueError, match="polígonos"):
        mobile_sam_training.train_mobile_sam(str(tmp_path), 1, None, lambda: False, lambda _model: None, lambda _message: None)


def test_yolo_training_with_fake_model(tmp_path, monkeypatch):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    create_bbox_xml(annotations / "sample.xml")
    calls = []

    class FakeModel:
        names = {0: "obj"}
        def add_callback(self, event, callback):
            calls.append(event)
        def train(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr(yolo_training, "YOLO", lambda _: FakeModel())
    monkeypatch.setattr(yolo_training, "ULTRALYTICS_DISPONIVEL", True)
    output, paused = yolo_training.train_yolo(
        str(tmp_path), 1, False, None, lambda: False, lambda model: None, lambda message: None
    )
    assert output.endswith("yolo_train_results")
    assert paused is False
    assert os.path.exists(tmp_path / "dataset_yolo.yaml")
    assert "on_train_epoch_end" in calls


def test_yolo_training_requires_checkpoint_when_continuing(tmp_path, monkeypatch):
    monkeypatch.setattr(yolo_training, "ULTRALYTICS_DISPONIVEL", True)
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    create_bbox_xml(annotations / "sample.xml")
    with pytest.raises(ValueError, match="last.pt"):
        yolo_training.train_yolo(str(tmp_path), 1, True, None, lambda: False, lambda _model: None, lambda _message: None)
