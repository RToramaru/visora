import os
import xml.etree.ElementTree as ET

try:
    from ultralytics import YOLO
    ULTRALYTICS_DISPONIVEL = True
except ImportError:
    ULTRALYTICS_DISPONIVEL = False


def train_yolo(project_path, epochs, continue_training, checkpoint_path, should_pause, on_model, log):
    if not ULTRALYTICS_DISPONIVEL:
        raise RuntimeError("A biblioteca Ultralytics (YOLO) não está instalada no ambiente.")

    annotations_path = os.path.join(project_path, "annotations")
    if not os.path.isdir(annotations_path) or not os.listdir(annotations_path):
        raise ValueError("Nenhuma anotação foi encontrada para o treinamento.")

    if continue_training:
        if not checkpoint_path:
            raise ValueError("Nenhum arquivo last.pt foi encontrado na pasta selecionada.")
        log(f"[YOLO] Continuando a partir de: {checkpoint_path}")
        model = YOLO(checkpoint_path)
    else:
        log("[YOLO] Carregando modelo pré-treinado yolov8n.pt...")
        model = YOLO("yolov8n.pt")
    on_model(model)

    def check_pause(trainer):
        if should_pause():
            trainer.stop = True

    model.add_callback("on_train_epoch_end", check_pause)
    classes = set()
    for filename in os.listdir(annotations_path):
        if not filename.lower().endswith(".xml"):
            continue
        try:
            root = ET.parse(os.path.join(annotations_path, filename)).getroot()
            classes.update(object_node.findtext("name") for object_node in root.findall("object"))
        except (OSError, ET.ParseError):
            continue

    names = sorted(name for name in classes if name) or ["Objeto"]
    frames_path = os.path.join(project_path, "frames")
    train_path = frames_path if os.path.isdir(frames_path) else project_path
    yaml_path = os.path.join(project_path, "dataset_yolo.yaml")
    with open(yaml_path, "w", encoding="utf-8") as dataset_file:
        dataset_file.write(f"path: {os.path.abspath(project_path)}\n")
        dataset_file.write(f"train: {os.path.abspath(train_path)}\n")
        dataset_file.write(f"val: {os.path.abspath(train_path)}\n")
        dataset_file.write("names:\n")
        for index, name in enumerate(names):
            dataset_file.write(f"  {index}: '{name}'\n")

    output_path = os.path.join(project_path, "runs", "yolo_train_results")
    log(f"[YOLO] Iniciando fit com {epochs} épocas...")
    parameters = {
        "data": yaml_path, "epochs": epochs, "imgsz": 640, "workers": 0,
        "project": os.path.join(project_path, "runs"), "name": "yolo_train_results",
        "verbose": False,
    }
    if continue_training:
        parameters["resume"] = True
    model.train(**parameters)
    return output_path, should_pause()
