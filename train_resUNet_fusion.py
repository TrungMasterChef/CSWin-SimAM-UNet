#!/usr/bin/env python3
"""
Lightweight ResUNet training script that mirrors the data pipeline used by train_unet_fusion.py.
Adds configurable width/depth, stratified train/val splits, and dataset introspection for fusion/RGB inputs.
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from train_unet_fusion import FusionDataset


LOG_FILE = Path(__file__).resolve().with_name("training_log.text")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8'),
    ]
)
logger = logging.getLogger(__name__)


# --------------------------- MODEL COMPONENTS --------------------------- #

class ResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return F.relu(out)


class ResUNet(nn.Module):
    def __init__(self, n_channels: int = 1, n_classes: int = 1, base_channels: int = 64, num_stages: int = 4):
        super().__init__()
        assert num_stages in (3, 4), "num_stages must be 3 or 4 for ResUNet"
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.base_channels = base_channels
        self.num_stages = num_stages

        self.inc = nn.Sequential(
            nn.Conv2d(n_channels, base_channels, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )

        encoder_channels = [base_channels * (2 ** i) for i in range(num_stages)]
        self.encoder_layers = nn.ModuleList()
        in_ch = base_channels
        for idx, out_ch in enumerate(encoder_channels):
            stride = 1 if idx == 0 else 2
            self.encoder_layers.append(self._make_layer(in_ch, out_ch, blocks=2, stride=stride))
            in_ch = out_ch

        bridge_channels = base_channels * (2 ** num_stages)
        self.bridge = self._make_layer(encoder_channels[-1], bridge_channels, blocks=2, stride=2)

        self.up_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        decoder_in = bridge_channels
        for out_ch in reversed(encoder_channels):
            self.up_layers.append(nn.ConvTranspose2d(decoder_in, out_ch, kernel_size=2, stride=2))
            self.decoder_layers.append(self._make_layer(out_ch * 2, out_ch, blocks=2, stride=1))
            decoder_in = out_ch

        self.outc = nn.Conv2d(base_channels, n_classes, kernel_size=1)
        self.apply(self._init_weights)

    @staticmethod
    def _make_layer(in_channels: int, out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [ResidualBlock(in_channels, out_channels, stride)]
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.inc(x)
        encoder_outputs: List[torch.Tensor] = []
        current = x
        for layer in self.encoder_layers:
            current = layer(current)
            encoder_outputs.append(current)

        bottleneck = self.bridge(encoder_outputs[-1])
        skip_features = list(reversed(encoder_outputs))

        x = bottleneck
        for up, dec, skip in zip(self.up_layers, self.decoder_layers, skip_features):
            x = up(x)
            if x.shape[-2:] != skip.shape[-2:]:
                diff_y = skip.size(2) - x.size(2)
                diff_x = skip.size(3) - x.size(3)
                x = F.pad(x, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([skip, x], dim=1)
            x = dec(x)

        return self.outc(x)


# --------------------------- METRICS & HELPERS --------------------------- #

def calculate_metrics(pred_logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> Dict[str, float]:
    pred = torch.sigmoid(pred_logits)
    if pred.dim() == 4:
        pred = pred.squeeze(1)
    if target.dim() == 4:
        target = target.squeeze(1)

    # FIX: Đảm bảo target là float để tính toán metrics
    target = target.float()

    pred_bin = (pred > threshold).float()
    pred_flat = pred_bin.view(pred_bin.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    eps = 1e-6
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = (intersection + eps) / (union + eps)
    dice = (2 * intersection + eps) / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + eps)
    precision = (intersection + eps) / (pred_flat.sum(dim=1) + eps)
    recall = (intersection + eps) / (target_flat.sum(dim=1) + eps)

    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }


def _describe_tensor(tensor) -> Tuple[str, str]:
    if tensor is None:
        return "None", "None"
    if hasattr(tensor, "shape"):
        shape = tuple(tensor.shape)
    elif hasattr(tensor, "size"):
        shape = tensor.size() if callable(tensor.size) else tensor.size
    else:
        shape = "Unknown"
    dtype = getattr(tensor, "dtype", type(tensor).__name__)
    return str(shape), str(dtype)


def log_dataset_overview(logger: logging.Logger, name: str, dataset: torch.utils.data.Dataset) -> None:
    size = len(dataset)
    logger.info("%s dataset size: %d samples", name, size)
    if size == 0:
        return

    try:
        sample = dataset[0]
    except Exception as exc:  # pragma: no cover - informational only
        logger.warning("Unable to inspect %s dataset sample: %s", name, exc)
        return

    if isinstance(sample, dict):
        image = sample.get("image")
        label = sample.get("label")
    elif isinstance(sample, (list, tuple)):
        image = sample[0]
        label = sample[1] if len(sample) > 1 else None
    else:
        image, label = sample, None

    img_shape, img_dtype = _describe_tensor(image)
    lbl_shape, lbl_dtype = _describe_tensor(label)
    logger.info(
        "%s sample shapes -> image=%s (%s), label=%s (%s)",
        name,
        img_shape,
        img_dtype,
        lbl_shape,
        lbl_dtype,
    )


def stratified_split_indices(dataset: FusionDataset, val_ratio: float = 0.2, fg_threshold: float = 0.05) -> Dict[str, np.ndarray]:
    fg_ratios = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        label = sample['label']
        fg_ratio = (label == 1).float().mean().item()
        fg_ratios.append(fg_ratio)

    fg_ratios = np.array(fg_ratios)
    high_fg_indices = np.where(fg_ratios > fg_threshold)[0]
    low_fg_indices = np.where(fg_ratios <= fg_threshold)[0]

    rng = np.random.default_rng(42)
    rng.shuffle(high_fg_indices)
    rng.shuffle(low_fg_indices)

    def _split(indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        split_point = int((1 - val_ratio) * len(indices))
        return indices[:split_point], indices[split_point:]

    train_high, val_high = _split(high_fg_indices)
    train_low, val_low = _split(low_fg_indices)

    train_indices = np.concatenate([train_high, train_low]) if len(train_high) + len(train_low) > 0 else np.array([], dtype=int)
    val_indices = np.concatenate([val_high, val_low]) if len(val_high) + len(val_low) > 0 else np.array([], dtype=int)

    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    return {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "train_high": train_high,
        "val_high": val_high,
        "train_low": train_low,
        "val_low": val_low,
    }


def build_datasets(data_dir: str, dataset_type: str, val_ratio: float, fg_threshold: float) -> Tuple[Subset, Subset, Dict[str, np.ndarray]]:
    """
    FIX DATA LEAKAGE: Tạo train/val datasets với kiểm tra overlap
    """
    # Tạo dataset với augment_factor=1 để tính foreground ratio trên ảnh gốc
    full_dataset = FusionDataset(
        data_dir,
        split='train',
        dataset_type=dataset_type,
        target_type='bce',
        out_channels_if_fusion=1,
        augment_factor=1,  # FIX: Không augment khi tính split indices
    )

    split_info = stratified_split_indices(full_dataset, val_ratio=val_ratio, fg_threshold=fg_threshold)

    # ==========================================================================
    # FIX: Kiểm tra DATA LEAKAGE - đảm bảo không có overlap
    # ==========================================================================
    train_set = set(split_info["train_indices"].tolist())
    val_set = set(split_info["val_indices"].tolist())
    overlap = train_set.intersection(val_set)
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping sample indices!")
    logger.info(f"Data leakage check PASSED: No overlap between train and val sets")

    # Lấy danh sách sample paths cho train và val
    train_samples = [full_dataset.samples[i] for i in split_info["train_indices"]]
    val_samples = [full_dataset.samples[i] for i in split_info["val_indices"]]

    # FIX: Tạo RIÊNG BIỆT train dataset (với augmentation) và val dataset (không augmentation)
    train_dataset_full = FusionDataset(
        data_dir,
        split='train',
        dataset_type=dataset_type,
        target_type='bce',
        out_channels_if_fusion=1,
        augment_factor=4,  # Augmentation cho train
    )
    train_dataset_full.samples = train_samples  # Chỉ chứa train samples

    val_dataset_full = FusionDataset(
        data_dir,
        split='val',
        dataset_type=dataset_type,
        target_type='bce',
        out_channels_if_fusion=1,
        augment_factor=1,  # Không augmentation cho val
    )
    val_dataset_full.samples = val_samples  # Chỉ chứa val samples

    logger.info(f"Original samples - Train: {len(train_samples)}, Val: {len(val_samples)}")

    return train_dataset_full, val_dataset_full, split_info


# --------------------------- TRAINING LOOP --------------------------- #

def train_resunet(
    data_dir: str = './data_ir',
    dataset_type: str = 'fusion',
    epochs: int = 100,
    batch_size: int = 8,
    lr: float = 0.001,
    output_dir: str = 'resunet_output',
    base_channels: int = 64,
    num_stages: int = 4,
    val_ratio: float = 0.2,
    fg_threshold: float = 0.05,
) -> float:
    dataset_type = dataset_type.lower()
    if dataset_type == 'ir':
        dataset_type = 'fusion'
    if dataset_type not in ('fusion', 'rgb'):
        raise ValueError("dataset_type must be 'fusion' or 'rgb'")

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info("Using device: %s", device)

    input_channels = 3 if dataset_type == 'rgb' else 1
    model = ResUNet(n_channels=input_channels, n_classes=1, base_channels=base_channels, num_stages=num_stages).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Model parameters: %.2fM (%d)", param_count / 1e6, param_count)

    train_dataset, val_dataset, split_info = build_datasets(data_dir, dataset_type, val_ratio, fg_threshold)
    logger.info("Stratified split -> Train=%d, Val=%d", len(train_dataset), len(val_dataset))
    logger.info("High FG samples - Train: %d, Val: %d", len(split_info['train_high']), len(split_info['val_high']))
    logger.info("Low FG samples  - Train: %d, Val: %d", len(split_info['train_low']), len(split_info['val_low']))

    log_dataset_overview(logger, "Train", train_dataset)
    log_dataset_overview(logger, "Val", val_dataset)

    num_workers = min(8, os.cpu_count() or 1)
    loader_kwargs = dict(num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **loader_kwargs)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history = {
        'train_loss': [],
        'val_loss': [],
        'train_iou': [],
        'val_iou': [],
        'train_dice': [],
        'val_dice': [],
    }
    best_iou = 0.0

    # FIX: Thêm early stopping
    patience = 15
    patience_counter = 0

    logger.info("Starting training for %d epochs...", epochs)
    logger.info("Early stopping patience: %d", patience)
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]', leave=False):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # FIX: Thêm gradient clipping để ổn định training
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            batch_metrics = calculate_metrics(outputs, labels)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]

        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        model.eval()
        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]', leave=False):
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                batch_metrics = calculate_metrics(outputs, labels)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]

        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['train_dice'].append(train_metrics['dice'])
        history['val_dice'].append(val_metrics['dice'])

        current_lr = optimizer.param_groups[0]['lr']
        iou_gap = train_metrics['iou'] - val_metrics['iou']
        logger.info(
            "Epoch %d: Train Loss=%.4f, Train IoU=%.4f, Train Dice=%.4f, "
            "Val Loss=%.4f, Val IoU=%.4f, Val Dice=%.4f, IoU Gap=%.4f, LR=%.6f",
            epoch + 1,
            train_loss,
            train_metrics['iou'],
            train_metrics['dice'],
            val_loss,
            val_metrics['iou'],
            val_metrics['dice'],
            iou_gap,
            current_lr,
        )

        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
            }, os.path.join(output_dir, f'best_resunet_{dataset_type}.pth'))
            logger.info("New best IoU: %.4f", best_iou)
            patience_counter = 0
        else:
            patience_counter += 1

        scheduler.step()

        # FIX: Early stopping thực sự hoạt động
        if patience_counter >= patience:
            logger.info("Early stopping triggered after %d epochs (no improvement for %d epochs)", epoch + 1, patience)
            break

        if (epoch + 1) % 10 == 0:
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'history': history,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1:03d}_{dataset_type}.pth'))

    history_path = os.path.join(output_dir, f'training_history_{dataset_type}.json')
    with open(history_path, 'w', encoding='utf-8') as f:
        json.dump(history, f, indent=2)
    logger.info("Training completed! Best IoU: %.4f | History saved to %s", best_iou, history_path)
    return best_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ResUNet for crack segmentation (fusion/RGB)')
    parser.add_argument('--data_dir', type=str, default='./data_ir', help='Data directory')
    parser.add_argument('--dataset_type', type=str, default='fusion', choices=['fusion', 'rgb', 'ir'], help='Dataset type')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--output_dir', type=str, default='resunet_output', help='Directory for checkpoints/logs')
    parser.add_argument('--base_channels', type=int, default=64, help='Number of channels in the first ResUNet block')
    parser.add_argument('--num_stages', type=int, choices=[3, 4], default=4, help='Encoder/decoder depth')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Validation split ratio')
    parser.add_argument('--fg_threshold', type=float, default=0.05, help='Foreground ratio threshold for stratification')

    args = parser.parse_args()

    train_resunet(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        output_dir=args.output_dir,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        val_ratio=args.val_ratio,
        fg_threshold=args.fg_threshold,
    )
