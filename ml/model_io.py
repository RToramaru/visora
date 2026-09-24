import os
import shutil
import warnings
from typing import Any

import cv2
import numpy as np
from PIL import Image

try:
    warnings.filterwarnings(
        "ignore", message="Importing from timm.models.layers is deprecated.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message="Importing from timm.models.registry is deprecated.*", category=FutureWarning
    )
    warnings.filterwarnings(
        "ignore", message="Overwriting tiny_vit_.* in registry with mobile_sam.*", category=UserWarning
    )
    import torch
    from mobile_sam import SamAutomaticMaskGenerator, build_sam_vit_t
    MOBILE_SAM_DISPONIVEL = True
except ImportError:
    MOBILE_SAM_DISPONIVEL = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_DISPONIVEL = True
except ImportError:
    ULTRALYTICS_DISPONIVEL = False


def infer_image(model_path: str, image_path: str, model_type: str) -> Image.Image:
    if model_type == "MobileSAM":
        if not MOBILE_SAM_DISPONIVEL:
            raise RuntimeError("MobileSAM não está instalado no ambiente.")
        image_bgr = cv2.imread(image_path)
        if image_bgr is None:
            raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        model = build_sam_vit_t(checkpoint=model_path)
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        with torch.inference_mode():
            masks = SamAutomaticMaskGenerator(
                model, points_per_side=8, points_per_batch=8,
                crop_n_layers=0, min_mask_region_area=100
            ).generate(image_rgb)
        output = image_rgb.copy()
        for index, mask in enumerate(masks):
            color = np.array(((37 * (index + 3)) % 255, (97 * (index + 5)) % 255,
                              (173 * (index + 7)) % 255), dtype=np.uint8)
            area = mask.get("segmentation")
            if area is not None:
                output[area] = (output[area].astype(np.float32) * 0.45 + color * 0.55).astype(np.uint8)
        return Image.fromarray(output)

    if not ULTRALYTICS_DISPONIVEL:
        raise RuntimeError("Ultralytics não está instalado no ambiente.")
    result = YOLO(model_path)(image_path, verbose=False)[0]
    return Image.fromarray(cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB))


def export_model(model_path: str, destination: str, model_format: str, model_type: str = "YOLO") -> str:
    base_name = os.path.splitext(os.path.basename(model_path))[0]
    if model_format in ("pt", "pth"):
        output_path = os.path.join(destination, f"{base_name}.{model_format}")
        shutil.copy2(model_path, output_path)
        return output_path
    if model_type == "MobileSAM":
        if model_format != "onnx":
            raise ValueError("MobileSAM suporta exportação ONNX ou cópia do checkpoint em .pt/.pth.")
        return _export_mobile_sam_onnx(model_path, destination)
    if not ULTRALYTICS_DISPONIVEL:
        raise RuntimeError("Ultralytics não está instalado no ambiente.")
    result = YOLO(model_path).export(format=model_format)
    source = os.fspath(result) if result else os.path.splitext(model_path)[0] + f".{model_format}"
    if not os.path.exists(source):
        raise FileNotFoundError(f"O arquivo exportado não foi encontrado para {model_format}.")
    output_path = os.path.join(destination, f"{base_name}.{model_format}")
    if os.path.isdir(source):
        shutil.copytree(source, output_path, dirs_exist_ok=True)
    else:
        shutil.copy2(source, output_path)
    return output_path


def _export_mobile_sam_onnx(model_path: str, destination: str) -> str:
    if not MOBILE_SAM_DISPONIVEL:
        raise RuntimeError("MobileSAM não está instalado no ambiente.")

    class MobileSAMOnnxWrapper(torch.nn.Module):
        def __init__(self, sam_model):
            super().__init__()
            self.sam_model = sam_model

        def forward(self, image, boxes):
            image_embeddings = self.sam_model.image_encoder(self.sam_model.preprocess(image))
            sparse_embeddings, dense_embeddings = self.sam_model.prompt_encoder(
                points=None, boxes=boxes, masks=None
            )
            low_res_masks, iou_predictions = self.sam_model.mask_decoder(
                image_embeddings=image_embeddings,
                image_pe=self.sam_model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse_embeddings,
                dense_prompt_embeddings=dense_embeddings,
                multimask_output=False,
            )
            return low_res_masks, iou_predictions

    model = build_sam_vit_t(checkpoint=model_path).cpu().eval()
    wrapper = MobileSAMOnnxWrapper(model).eval()
    output_path = os.path.join(destination, f"{os.path.splitext(os.path.basename(model_path))[0]}.onnx")
    dummy_image = torch.zeros((1, 3, model.image_encoder.img_size, model.image_encoder.img_size))
    dummy_boxes = torch.tensor([[0.0, 0.0, float(model.image_encoder.img_size), float(model.image_encoder.img_size)]])
    torch.onnx.export(
        wrapper,
        (dummy_image, dummy_boxes),
        output_path,
        opset_version=18,
        input_names=["image", "boxes"],
        output_names=["masks", "iou_predictions"],
        dynamic_shapes={"boxes": {0: torch.export.Dim("num_boxes")}},
    )
    return output_path
