import json
import os
import urllib.request
import warnings

import cv2
import numpy as np

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
    import torch.nn.functional as F
    from mobile_sam import build_sam_vit_t
    from mobile_sam.utils.transforms import ResizeLongestSide
    MOBILE_SAM_DISPONIVEL = True
except ImportError:
    MOBILE_SAM_DISPONIVEL = False


CHECKPOINT_URL = "https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt"
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".webp")


def find_samples(project_path):
    annotations_path = os.path.join(project_path, "annotations")
    images_path = os.path.join(project_path, "frames")
    if not os.path.isdir(images_path):
        images_path = project_path
    samples = []
    if not os.path.isdir(annotations_path):
        return samples

    for label_name in os.listdir(annotations_path):
        if not label_name.lower().endswith(".json"):
            continue
        base_name = os.path.splitext(label_name)[0]
        image_name = next(
            (base_name + extension for extension in IMAGE_EXTENSIONS
             if os.path.exists(os.path.join(images_path, base_name + extension))), None
        )
        if not image_name:
            continue
        try:
            with open(os.path.join(annotations_path, label_name), "r", encoding="utf-8") as label_file:
                data = json.load(label_file)
            for shape in data.get("shapes", []):
                points = shape.get("points", [])
                if len(points) >= 3:
                    samples.append((os.path.join(images_path, image_name),
                                    [[float(x), float(y)] for x, y in points]))
        except (OSError, ValueError, TypeError):
            continue
    return samples


def ensure_base_checkpoint(project_path, log):
    checkpoint_path = os.path.join(project_path, "mobile_sam.pt")
    if not os.path.exists(checkpoint_path):
        log("[MOBILE SAM] Baixando checkpoint pré-treinado...")
        urllib.request.urlretrieve(CHECKPOINT_URL, checkpoint_path)
    return checkpoint_path


def train_mobile_sam(project_path, epochs, checkpoint_path, should_pause, on_model, log):
    if not MOBILE_SAM_DISPONIVEL:
        raise RuntimeError("MobileSAM e suas dependências não estão instalados no ambiente.")
    samples = find_samples(project_path)
    if not samples:
        raise ValueError("Nenhum par imagem/JSON com polígonos válidos foi encontrado para o MobileSAM.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_path = os.path.join(project_path, "runs", "mobile_sam_train")
    os.makedirs(output_path, exist_ok=True)
    checkpoint_path = checkpoint_path or ensure_base_checkpoint(project_path, log)
    model = build_sam_vit_t(checkpoint=checkpoint_path).to(device)
    on_model(model)
    model.image_encoder.eval()
    for parameter in model.image_encoder.parameters():
        parameter.requires_grad = False
    model.prompt_encoder.train()
    model.mask_decoder.train()
    optimizer = torch.optim.AdamW(
        list(model.prompt_encoder.parameters()) + list(model.mask_decoder.parameters()),
        lr=1e-5, weight_decay=1e-4
    )
    transform = ResizeLongestSide(model.image_encoder.img_size)
    best_loss = float("inf")
    last_path = os.path.join(output_path, "last.pt")
    best_path = os.path.join(output_path, "best.pt")
    log(f"[MOBILE SAM] Fine-tuning real em {device}: {len(samples)} polígonos, {epochs} épocas.")

    for epoch in range(epochs):
        if should_pause():
            break
        epoch_loss = 0.0
        for image_path, points in samples:
            if should_pause():
                break
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                raise FileNotFoundError(f"Não foi possível carregar a imagem: {image_path}")
            image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            height, width = image.shape[:2]
            target = np.zeros((height, width), dtype="uint8")
            cv2.fillPoly(target, [np.array(points, dtype="int32")], 1)
            transformed = transform.apply_image(image)
            image_tensor = torch.as_tensor(transformed, device=device).permute(2, 0, 1).float()
            target = cv2.resize(target, (transformed.shape[1], transformed.shape[0]), interpolation=cv2.INTER_NEAREST)
            target_tensor = torch.as_tensor(target, device=device).float()
            target_tensor = F.pad(target_tensor, (0, model.image_encoder.img_size - target_tensor.shape[1],
                                                   0, model.image_encoder.img_size - target_tensor.shape[0]))[None, None]
            target_tensor = F.interpolate(target_tensor, (256, 256), mode="nearest")
            box = [min(x for x, _ in points), min(y for _, y in points), max(x for x, _ in points), max(y for _, y in points)]
            box = transform.apply_boxes(np.array([box]), (height, width))
            box_tensor = torch.as_tensor(box, dtype=torch.float32, device=device)

            optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                embedding = model.image_encoder(model.preprocess(image_tensor)[None])
            sparse, dense = model.prompt_encoder(points=None, boxes=box_tensor, masks=None)
            logits, _ = model.mask_decoder(
                image_embeddings=embedding, image_pe=model.prompt_encoder.get_dense_pe(),
                sparse_prompt_embeddings=sparse, dense_prompt_embeddings=dense, multimask_output=False
            )
            target_logits = F.interpolate(target_tensor, logits.shape[-2:], mode="nearest")
            bce = F.binary_cross_entropy_with_logits(logits, target_logits)
            probabilities = torch.sigmoid(logits)
            intersection = (probabilities * target_logits).sum()
            dice = 1 - (2 * intersection + 1) / (probabilities.sum() + target_logits.sum() + 1)
            loss = bce + dice
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        mean_loss = epoch_loss / max(1, len(samples))
        torch.save(model.state_dict(), last_path)
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(model.state_dict(), best_path)
        log(f"[MOBILE SAM] Época {epoch + 1}/{epochs} - loss: {mean_loss:.5f}")

    return (last_path if should_pause() else best_path), should_pause()
