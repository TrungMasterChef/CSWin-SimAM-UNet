from pathlib import Path
from typing import Dict, Iterable, List, Optional

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image

from train_unet_fusion import UNet
from train_resUNet_fusion import ResUNet


# Base paths
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "Result" / "ResUNet" / "best_resunet_fusion.pth"
IMAGE_DIR = BASE_DIR / "data_ir" / "03-Fusion(50IRT) images"
OUTPUT_DIR = BASE_DIR / "prediction_all"
MODEL_ARCH = "resunet"  # 'unet' or 'resunet'

# Inference configuration
IMG_SIZE = 224
OVERLAY_COLOR = (0, 255, 0)
OVERLAY_OPACITY = 0.5
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

# Shared preprocessing pipeline (match training pipeline for IR images)
TRANSFORM = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
    ]
)


def _extract_model_config(checkpoint: Dict) -> Optional[Dict]:
    """Return stored model hyperparameters if the checkpoint provides them."""
    if not isinstance(checkpoint, dict):
        return None
    for key in ("model_config", "config", "hyper_parameters"):
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    return None


def _infer_unet_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict:
    """Infer UNet constructor arguments directly from a lightweight checkpoint."""
    inc_weight = state_dict.get("inc.block.0.weight")
    if inc_weight is None:
        raise ValueError("Checkpoint missing 'inc.block.0.weight'; cannot infer model structure.")

    base_channels = inc_weight.shape[0]
    input_channels = inc_weight.shape[1]

    down_indices = set()
    up_indices = set()
    for key in state_dict.keys():
        if key.startswith("downs."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                down_indices.add(int(parts[1]))
        elif key.startswith("ups."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                up_indices.add(int(parts[1]))

    if down_indices:
        num_stages = max(down_indices) + 1
    elif up_indices:
        num_stages = max(up_indices) + 1
    else:
        num_stages = 4
    num_stages = max(3, min(4, num_stages))  # constrained by UNet implementation

    conv_keys = [
        key
        for key, tensor in state_dict.items()
        if key.startswith("inc.block") and key.endswith(".weight") and tensor.ndim == 4
    ]
    block_depth = max(1, len(conv_keys))

    def _deduce_use_skip() -> bool:
        chs = [base_channels * (2 ** i) for i in range(num_stages + 1)]
        reversed_indices = list(reversed(range(num_stages)))
        for up_idx, enc_idx in enumerate(reversed_indices):
            key = f"ups.{up_idx}.conv.block.0.weight"
            tensor = state_dict.get(key)
            if tensor is None or tensor.ndim != 4:
                continue
            out_channels, in_channels, _, _ = tensor.shape
            expected_out = chs[enc_idx]
            if out_channels != expected_out:
                continue
            reduced = chs[enc_idx + 1] // 2 if enc_idx + 1 < len(chs) else chs[-1] // 2
            with_skip = reduced + chs[enc_idx]
            without_skip = reduced
            if in_channels == with_skip:
                return True
            if in_channels == without_skip:
                return False
        return True

    config = {
        "num_classes": 1,
        "input_channels": input_channels,
        "base_channels": base_channels,
        "num_stages": num_stages,
        "depth": block_depth,
        "use_skip": _deduce_use_skip(),
    }
    print(f"Inferred UNet config from checkpoint: {config}")
    return config


def _infer_resunet_config_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Dict:
    """Infer ResUNet constructor arguments from checkpoint weights."""
    inc_weight = state_dict.get("inc.0.weight")
    if inc_weight is None:
        raise ValueError("Checkpoint missing 'inc.0.weight'; cannot infer ResUNet structure.")

    base_channels = inc_weight.shape[0]
    input_channels = inc_weight.shape[1]

    encoder_indices = set()
    for key in state_dict.keys():
        if key.startswith("encoder_layers."):
            parts = key.split(".")
            if len(parts) > 1 and parts[1].isdigit():
                encoder_indices.add(int(parts[1]))

    num_stages = max(encoder_indices) + 1 if encoder_indices else 4
    num_stages = max(3, min(4, num_stages))

    config = {
        "n_channels": input_channels,
        "n_classes": 1,
        "base_channels": base_channels,
        "num_stages": num_stages,
    }
    print(f"Inferred ResUNet config from checkpoint: {config}")
    return config


def collect_image_paths(root: Path) -> List[Path]:
    """Return a sorted list of image paths inside the given directory."""
    return sorted(
        path
        for path in root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def load_model(model_path: Path, device: torch.device, arch: str = MODEL_ARCH) -> torch.nn.Module:
    """Load the trained checkpoint with automatic architecture detection."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    print(f"Loading model from {model_path} ...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise ValueError("Unsupported checkpoint format. Expected a state_dict or checkpoint dict.")

    if any(key.startswith("module.") for key in state_dict.keys()):
        state_dict = {key.replace("module.", "", 1): value for key, value in state_dict.items()}

    config = _extract_model_config(checkpoint)

    if arch == "resunet":
        if config is None:
            config = _infer_resunet_config_from_state_dict(state_dict)
        model = ResUNet(
            n_channels=config.get("n_channels", config.get("input_channels", 1)),
            n_classes=config.get("n_classes", 1),
            base_channels=config.get("base_channels", 64),
            num_stages=config.get("num_stages", 4),
        )
    elif arch == "unet":
        if config is None:
            config = _infer_unet_config_from_state_dict(state_dict)
        model = UNet(
            num_classes=config.get("num_classes", 1),
            input_channels=config.get("input_channels", 1),
            base_channels=config.get("base_channels", 64),
            num_stages=config.get("num_stages", 4),
            depth=config.get("depth", 2),
            use_skip=config.get("use_skip", True),
        )
    else:
        raise ValueError(f"Unsupported architecture '{arch}'.")
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    return model


def preprocess_image(image_path: Path) -> torch.Tensor:
    """Load an IR image, convert to tensor, and add batch dimension."""
    image = Image.open(image_path).convert("L")
    tensor = TRANSFORM(image).unsqueeze(0)  # Shape: [1, 1, IMG_SIZE, IMG_SIZE]
    return tensor


def postprocess_mask(prediction: torch.Tensor) -> np.ndarray:
    """Convert model output to a binary uint8 mask."""
    if isinstance(prediction, dict):
        if "out" in prediction:
            prediction = prediction["out"]
        else:
            raise ValueError("Prediction dict does not contain 'out' key.")

    prediction = torch.sigmoid(prediction)
    mask = (prediction > 0.5).float().squeeze()
    mask_np = mask.cpu().numpy()
    return (mask_np * 255).astype(np.uint8)


def create_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Overlay a binary mask on top of the original BGR image."""
    if image_bgr.ndim == 2:
        image_bgr = cv2.cvtColor(image_bgr, cv2.COLOR_GRAY2BGR)

    color_mask = np.zeros_like(image_bgr)
    color_mask[mask == 255] = OVERLAY_COLOR

    return cv2.addWeighted(image_bgr, 1 - OVERLAY_OPACITY, color_mask, OVERLAY_OPACITY, 0)


def write_image(path: Path, image: np.ndarray) -> None:
    """Persist an image to disk, handling Unicode paths gracefully."""
    path.parent.mkdir(parents=True, exist_ok=True)

    success = cv2.imwrite(str(path), image)
    if success:
        return

    # Fallback for cases where OpenCV cannot handle Unicode paths on Windows.
    ext = path.suffix.lower() or ".png"
    if not ext.startswith("."):
        ext = f".{ext}"

    encode_ext = ext if ext in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"} else ".png"
    _, buffer = cv2.imencode(encode_ext, image)
    path.with_suffix(encode_ext).write_bytes(buffer.tobytes())


def save_predictions(image_path: Path, mask: np.ndarray) -> None:
    """Save resized input, binary mask, and overlay images to OUTPUT_DIR."""
    pil_image = Image.open(image_path).convert("RGB")
    original = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    resized = cv2.resize(original, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
    overlay = create_overlay(resized, mask)

    base_name = image_path.stem
    input_path = OUTPUT_DIR / f"{base_name}_input.png"
    mask_path = OUTPUT_DIR / f"{base_name}_mask.png"
    overlay_path = OUTPUT_DIR / f"{base_name}_overlay.png"

    write_image(input_path, resized)
    write_image(mask_path, mask)
    write_image(overlay_path, overlay)


def predict_directory(model: torch.nn.Module, image_paths: Iterable[Path], device: torch.device) -> None:
    """Run segmentation on each image and persist the outputs."""
    if not image_paths:
        print(f"No images found in {IMAGE_DIR}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for path in image_paths:
        print(f"Processing {path.name} ...")
        tensor = preprocess_image(path).to(device)

        with torch.no_grad():
            prediction = model(tensor)

        mask = postprocess_mask(prediction)
        save_predictions(path, mask)

    print(f"\nPredictions saved to: {OUTPUT_DIR}")


def main() -> None:
    if not IMAGE_DIR.exists():
        raise FileNotFoundError(f"Image directory not found: {IMAGE_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = load_model(MODEL_PATH, device)
    image_paths = collect_image_paths(IMAGE_DIR)
    predict_directory(model, image_paths, device)


if __name__ == "__main__":
    main()
