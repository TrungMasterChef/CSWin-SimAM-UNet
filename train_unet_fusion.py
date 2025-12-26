#!/usr/bin/env python3
"""
UNet Baseline: Lightweight UNet for Crack Segmentation
Maintains the advanced training pipeline from the CSWin variant while using a
pure convolutional encoder-decoder without transformer modules.

Key Features:
- Classic UNet encoder-decoder with multiscale skip connections
- Simplified training that mirrors the reference pipeline
- Compatible with both fusion and RGB crack datasets
- Designed for rapid experimentation and benchmarking
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
import numpy as np
import cv2
from PIL import Image
import glob
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
from torchvision.transforms import InterpolationMode, functional as TF
import math
import random
import gc
import time
from functools import partial
import torch.utils.checkpoint as checkpoint
import warnings
warnings.filterwarnings('ignore')

# Set up logging
LOG_FILE = Path(__file__).resolve().with_name("training_log.text")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_FILE, mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# GPU Performance Optimization
def optimize_gpu_performance():
    """Optimize GPU performance for maximum throughput"""
    if torch.cuda.is_available():
        # Enable TensorFloat-32 (TF32) for faster training on Ampere GPUs
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        # Enable cuDNN benchmark for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Set memory fraction to avoid fragmentation
        torch.cuda.empty_cache()
        gc.collect()

        logger.info("🚀 GPU Performance Optimizations Enabled:")
        logger.info("   ✅ TF32 enabled for faster training")
        logger.info("   ✅ cuDNN benchmark enabled")
        logger.info("   ✅ Memory optimizations applied")

# ======================== UNET ARCHITECTURE ========================
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv-[BN]-ReLU) lặp depth lần. Có thể tắt BN và thêm Dropout để ablation.
    """
    def __init__(self, in_channels, out_channels,
                 mid_channels=None,
                 depth: int = 2,
                 use_bn: bool = True,
                 dropout_p: float = 0.0):
        super().__init__()
        mid_channels = mid_channels or out_channels

        layers = []
        ch_in, ch_out = in_channels, mid_channels
        for i in range(depth):
            # Ở lần cuối dùng out_channels
            if i == depth - 1:
                ch_out = out_channels
            layers.append(nn.Conv2d(ch_in, ch_out, kernel_size=3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(ch_out))
            layers.append(nn.ReLU(inplace=True))
            if dropout_p > 0:
                layers.append(nn.Dropout2d(dropout_p))
            ch_in = ch_out
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool rồi DoubleConv (có thể kèm dropout/không BN)"""
    def __init__(self, in_channels, out_channels,
                 depth: int = 2, use_bn: bool = True, dropout_p: float = 0.0):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            DoubleConv(in_channels, out_channels, depth=depth, use_bn=use_bn, dropout_p=dropout_p),
        )
    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """
    Upsample + (tùy chọn) skip-connection rồi DoubleConv.
    up_mode: 'bilinear' | 'nearest' | 'bicubic' | 'transposed'
    use_skip=False để bỏ skip-connection (khi muốn giảm hiệu năng).
    """
    def __init__(self, in_channels, skip_channels, out_channels,
                 up_mode: str = "bilinear",
                 use_skip: bool = True,
                 depth: int = 2,
                 use_bn: bool = True,
                 dropout_p: float = 0.0):
        super().__init__()
        self.use_skip = use_skip

        if up_mode == "transposed":
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.reduce = nn.Identity()
        else:
            if up_mode in ("bilinear", "bicubic"):
                self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=False)
            else:  # 'nearest'…
                self.up = nn.Upsample(scale_factor=2, mode=up_mode)
            self.reduce = nn.Conv2d(in_channels, in_channels // 2, kernel_size=1)

        total_in = (in_channels // 2) + (skip_channels if use_skip else 0)
        self.conv = DoubleConv(total_in, out_channels, depth=depth, use_bn=use_bn, dropout_p=dropout_p)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = self.reduce(x1)
        # căn chỉnh kích thước nếu lệch
        if x2 is not None and self.use_skip:
            diff_y = x2.size(2) - x1.size(2)
            diff_x = x2.size(3) - x1.size(3)
            if diff_y != 0 or diff_x != 0:
                x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
            x = torch.cat([x2, x1], dim=1)
        else:
            x = x1
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=1,
                 base_channels=64,
                 num_stages: int = 4,
                 use_skip: bool = True,
                 use_bn: bool = True,
                 dropout_p: float = 0.0,
                 depth: int = 2,
                 up_mode: str = "bilinear"):
        super().__init__()

        assert num_stages in (3, 4), "num_stages chỉ hỗ trợ 3 hoặc 4"
        self.use_skip = use_skip

        # Kênh theo từng mức
        chs = [base_channels * (2 ** i) for i in range(num_stages + 1)]  # ví dụ 4 mức: [64,128,256,512,1024]

        self.inc = DoubleConv(input_channels, chs[0], depth=depth, use_bn=use_bn, dropout_p=dropout_p)
        # Encoder
        self.downs = nn.ModuleList([
            Down(chs[i], chs[i+1], depth=depth, use_bn=use_bn, dropout_p=dropout_p)
            for i in range(num_stages)
        ])

        # Decoder (đi ngược)
        self.ups = nn.ModuleList()
        for i in reversed(range(num_stages)):
            in_ch = chs[i+1]
            skip_ch = chs[i] if use_skip else 0
            out_ch = chs[i]
            self.ups.append(
                Up(in_ch, skip_ch, out_ch, up_mode=up_mode, use_skip=use_skip,
                   depth=depth, use_bn=use_bn, dropout_p=dropout_p)
            )

        self.out_conv = OutConv(chs[0], num_classes)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = [self.inc(x)]
        for down in self.downs:
            features.append(down(features[-1]))
        x = features[-1]
        # duyệt ups cùng chiều ngược, lấy skip nếu bật
        for i, up in enumerate(self.ups):
            skip = features[-(i+2)] if self.use_skip else None
            x = up(x, skip)
        return self.out_conv(x)

    
class FusionDataset(torch.utils.data.Dataset):
    """
    Dataset cho crack segmentation với hai chế độ:
      - dataset_type='rgb': ảnh RGB (3 kênh)
      - dataset_type='fusion': ảnh hồng ngoại/hợp nhất (grayscale, 1 kênh)
    Đảm bảo các phép biến đổi hình học áp dụng đồng bộ cho ảnh và mask.
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        img_size: int = 224,
        dataset_type: str = "fusion",    # 'rgb' hoặc 'fusion'
        augment_factor: int = 1,         # nhân dữ liệu theo chỉ số (lặp augmentation)
        normalize: bool = True,          # chuẩn hoá theo ImageNet cho RGB, (0.5,0.5) cho 1 kênh
        target_type: str = "ce",         # 'ce' -> long HxW; 'bce' -> float 1xHxW
        out_channels_if_fusion: int = 1  # 1 nếu model in_channels=1; 3 nếu muốn lặp kênh cho backbone pretrained
    ):
        """
        Args:
            data_dir: thư mục gốc của dataset; cần có các thư mục con:
              - '01-Visible images' hoặc '03-Fusion(50IRT) images'
              - '04-Ground truth'
            split: 'train' hoặc 'val'/'test'
            img_size: kích thước ảnh đầu vào (square)
            dataset_type: 'rgb' hoặc 'fusion'
            augment_factor: số lần lặp mỗi mẫu trong __len__ (dùng khi muốn oversampling)
            normalize: bật/tắt Normalize
            target_type: 'ce' (mask long HxW cho CrossEntropy) hoặc 'bce' (mask float 1xHxW cho BCEWithLogits)
            out_channels_if_fusion: số kênh xuất ra khi dataset_type='fusion' (1 hoặc 3)
        """
        super().__init__()
        assert dataset_type in ("rgb", "fusion")
        assert target_type in ("ce", "bce")
        assert out_channels_if_fusion in (1, 3)

        self.data_dir = data_dir
        self.split = split.lower()
        self.img_size = img_size
        self.dataset_type = dataset_type
        self.augment_factor = augment_factor if self.split == "train" else 1
        self.normalize = normalize
        self.target_type = target_type
        self.out_channels_if_fusion = out_channels_if_fusion

        # Chọn thư mục ảnh theo loại dataset
        if dataset_type == "rgb":
            self.image_dir = os.path.join(data_dir, "01-Visible images")
        else:
            self.image_dir = os.path.join(data_dir, "03-Fusion(50IRT) images")
        self.label_dir = os.path.join(data_dir, "04-Ground truth")

        # Tìm ảnh–mask hợp lệ (đệ quy)
        self.samples: List[Tuple[str, str]] = []
        for ext in ["**/*.png", "**/*.jpg", "**/*.jpeg"]:
            for img_path in sorted(glob.glob(os.path.join(self.image_dir, ext), recursive=True)):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                label_path_png = os.path.join(self.label_dir, img_name + ".png")
                label_path_jpg = os.path.join(self.label_dir, img_name + ".jpg")
                if os.path.exists(label_path_png):
                    self.samples.append((img_path, label_path_png))
                elif os.path.exists(label_path_jpg):
                    self.samples.append((img_path, label_path_jpg))
        self.samples.sort(key=lambda pair: pair[0])

        if len(self.samples) == 0:
            raise ValueError(
                f"No valid samples found in {self.image_dir} with labels in {self.label_dir}."
            )

        logger.info(f"Found {len(self.samples)} valid {dataset_type.upper()} samples for {self.split.upper()}.")

        # Chuẩn hoá
        self.normalize_tf = None
        if self.normalize:
            if self.dataset_type == "rgb":
                # ImageNet stats
                self.normalize_tf = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])
            else:
                # đơn kênh
                self.normalize_tf = transforms.Normalize(mean=[0.5], std=[0.5])

        # Color jitter (chỉ áp cho ảnh, không áp cho mask)
        if self.split == "train":
            # Mức độ tương đương với code gốc (có thể điều chỉnh)
            self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
        else:
            self.color_jitter = None

    def __len__(self):
        return len(self.samples) * self.augment_factor

    # --------- các helper cho paired augmentation ---------
    def _resize_pair(self, img: Image.Image, msk: Image.Image):
        img = TF.resize(img, (self.img_size, self.img_size), interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, (self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST)
        return img, msk

    def _maybe_hflip(self, img: Image.Image, msk: Image.Image, p: float = 0.6):
        if random.random() < p:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
        return img, msk

    def _maybe_vflip(self, img: Image.Image, msk: Image.Image, p: float = 0.6):
        if random.random() < p:
            img = TF.vflip(img)
            msk = TF.vflip(msk)
        return img, msk

    def _random_rotate(self, img: Image.Image, msk: Image.Image, low=-45, high=45):
        angle = random.uniform(low, high)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        msk = TF.rotate(msk, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        return img, msk

    def _random_affine(self, img: Image.Image, msk: Image.Image,
                       max_translate=0.15, scale_range=(0.75, 1.25), shear_range=(-8, 8)):
        tx = int(max_translate * self.img_size)
        ty = int(max_translate * self.img_size)
        translate = (random.randint(-tx, tx), random.randint(-ty, ty))
        scale = random.uniform(scale_range[0], scale_range[1])
        shear = random.uniform(shear_range[0], shear_range[1])
        # Áp cùng tham số cho ảnh và mask
        img = TF.affine(
            img, angle=0, translate=translate, scale=scale, shear=[shear, 0],
            interpolation=InterpolationMode.BILINEAR, fill=0
        )
        msk = TF.affine(
            msk, angle=0, translate=translate, scale=scale, shear=[shear, 0],
            interpolation=InterpolationMode.NEAREST, fill=0
        )
        return img, msk

    def _to_tensor_and_norm(self, img: Image.Image):
        t = transforms.ToTensor()(img)  # CxHxW, float32 in [0,1]
        if self.normalize_tf is not None:
            # đảm bảo số kênh đúng với normalize
            if self.dataset_type == "fusion" and t.shape[0] == 1 and isinstance(self.normalize_tf.mean, list):
                t = self.normalize_tf(t)
            elif self.dataset_type == "rgb" and t.shape[0] == 3:
                t = self.normalize_tf(t)
            else:
                # nếu mismatch, bỏ normalize để tránh lỗi
                pass
        return t

    def _prepare_label(self, msk: Image.Image):
        # PILToTensor → uint8 [0..255] → threshold > 0.5
        lb = transforms.PILToTensor()(msk).squeeze(0).float() / 255.0  # HxW in [0,1]
        if self.target_type == "ce":
            # CrossEntropyLoss: long HxW với {0,1}
            lb = (lb > 0.5).long()
        else:
            # BCEWithLogitsLoss: float 1xHxW
            lb = (lb > 0.5).float().unsqueeze(0)
        return lb

    def __getitem__(self, idx: int):
        original_idx = idx // self.augment_factor
        img_path, label_path = self.samples[original_idx]

        # Đọc ảnh/mask bằng PIL để tránh vấn đề BGR/RGB
        if self.dataset_type == "rgb":
            img = Image.open(img_path).convert("RGB")
        else:
            img = Image.open(img_path).convert("L")
        msk = Image.open(label_path).convert("L")

        # Resize trước cho đồng bộ
        img, msk = self._resize_pair(img, msk)

        # Aug train (đồng bộ hình học)
        if self.split == "train":
            img, msk = self._maybe_hflip(img, msk, p=0.6)
            img, msk = self._maybe_vflip(img, msk, p=0.6)
            img, msk = self._random_rotate(img, msk, low=-45, high=45)
            img, msk = self._random_affine(img, msk, max_translate=0.15, scale_range=(0.75, 1.25), shear_range=(-8, 8))

            # Color jitter chỉ cho ảnh
            if self.color_jitter is not None:
                img = self.color_jitter(img)

        # ToTensor + Normalize
        image = self._to_tensor_and_norm(img)
        label = self._prepare_label(msk)

        # Đảm bảo số kênh đầu vào phù hợp
        if self.dataset_type == "rgb":
            if image.shape[0] != 3:
                image = image.repeat(3, 1, 1)
        else:
            # fusion
            if self.out_channels_if_fusion == 3 and image.shape[0] == 1:
                image = image.repeat(3, 1, 1)  # lặp kênh để dùng backbone pretrained
            elif self.out_channels_if_fusion == 1 and image.shape[0] == 3:
                image = image[0:1, :, :]      # fallback: chỉ lấy 1 kênh

        return {
            "image": image,                              # Tensor: CxHxW
            "label": label,                              # Long HxW (CE) hoặc Float 1xHxW (BCE)
            "case_name": os.path.splitext(os.path.basename(img_path))[0]
        }

# ======================== ADVANCED LOSS FUNCTIONS ========================

class AdaptiveLoss(nn.Module):
    """ADAPTIVE loss that adjusts based on IoU level for breakthrough 0.6+ barrier"""
    def __init__(self, num_classes=1, label_smoothing=0.05):  # FIXED: Binary segmentation
        super(AdaptiveLoss, self).__init__()
        # Use BCEWithLogitsLoss for binary segmentation with high positive weight
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(20.0))  # High weight for cracks
        self.num_classes = num_classes
        self.alpha = 0.4  # Higher alpha for focal loss
        self.gamma = 3.0   # Higher gamma for harder examples
        self.current_iou = 0.0  # Track current IoU for adaptive weighting

        # Morphological kernels for connectivity
        self.register_buffer('connectivity_kernel', torch.ones(1, 1, 3, 3))
        self.register_buffer('dilation_kernel', torch.ones(1, 1, 5, 5))

    def lovasz_hinge_loss(self, pred, target):
        """Lovasz loss for better IoU optimization at high levels"""
        pred = F.softmax(pred, dim=1)[:, 1]  # Get foreground probabilities
        target = target.float()

        # Flatten
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)

        # Lovasz extension of IoU loss
        intersection = (pred_flat * target_flat).sum()
        union = pred_flat.sum() + target_flat.sum() - intersection
        iou = intersection / (union + 1e-6)

        # Lovasz hinge approximation
        errors = torch.abs(pred_flat - target_flat)
        errors_sorted, perm = torch.sort(errors, descending=True)
        gt_sorted = target_flat[perm]

        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(errors_sorted, grad)

        return loss

    def _lovasz_grad(self, gt_sorted):
        """Compute gradient for Lovasz loss"""
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1. - intersection / union
        if p > 1:
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def tversky_loss(self, pred, target, alpha=0.3, beta=0.7):
        """Tversky loss for handling class imbalance better"""
        smooth = 1e-6
        pred = F.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, self.num_classes).permute(0, 3, 1, 2).float()

        # Focus on foreground class
        pred_fg = pred[:, 1:2]
        target_fg = target_one_hot[:, 1:2]

        TP = (pred_fg * target_fg).sum(dim=(2, 3))
        FP = (pred_fg * (1 - target_fg)).sum(dim=(2, 3))
        FN = ((1 - pred_fg) * target_fg).sum(dim=(2, 3))

        tversky = (TP + smooth) / (TP + alpha * FP + beta * FN + smooth)
        return 1 - tversky.mean()

    def focal_loss(self, pred, target):
        """Enhanced focal loss with adaptive gamma"""
        ce_loss = F.cross_entropy(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)

        # Adaptive gamma based on current IoU
        adaptive_gamma = self.gamma + (1.0 if self.current_iou > 0.6 else 0.0)
        focal_loss = self.alpha * (1-pt)**adaptive_gamma * ce_loss
        return focal_loss.mean()

    def connectivity_loss(self, pred, target):
        """Connectivity loss for crack continuity - optimized for spreading cracks"""
        pred_prob = F.softmax(pred, dim=1)[:, 1:2]  # Foreground probability
        target_float = target.float().unsqueeze(1)

        # Ensure kernels have same dtype and device as input
        connectivity_kernel = self.connectivity_kernel.to(pred_prob.dtype).to(pred_prob.device)
        dilation_kernel = self.dilation_kernel.to(pred_prob.dtype).to(pred_prob.device)

        # Morphological operations for connectivity
        pred_dilated = F.conv2d(pred_prob, dilation_kernel, padding=2)
        target_dilated = F.conv2d(target_float.to(pred_prob.dtype), dilation_kernel, padding=2)

        # Connectivity preservation
        connectivity_diff = torch.abs(pred_dilated - target_dilated)
        connectivity_loss = connectivity_diff.mean()

        # Penalize disconnected components
        pred_binary = (pred_prob > 0.5).float()
        target_binary = target_float.to(pred_prob.dtype)

        # Edge detection for crack boundaries
        pred_edges = F.conv2d(pred_binary, connectivity_kernel, padding=1) - pred_binary
        target_edges = F.conv2d(target_binary, connectivity_kernel, padding=1) - target_binary

        edge_loss = F.mse_loss(pred_edges, target_edges)

        return connectivity_loss + 0.5 * edge_loss

    def update_iou(self, current_iou):
        """Update current IoU for adaptive loss weighting"""
        self.current_iou = current_iou

    def forward(self, pred_logits, target):
        """FIXED: Binary segmentation loss with NaN protection"""
        # Ensure target is in correct format for binary segmentation
        if target.dim() == 3:  # [B, H, W]
            target = target.unsqueeze(1).float()  # [B, 1, H, W]
        else:
            target = target.float()

        # Clamp predictions to prevent extreme values
        pred_logits = torch.clamp(pred_logits, min=-10, max=10)

        # Check for NaN/Inf in inputs
        if torch.isnan(pred_logits).any() or torch.isinf(pred_logits).any():
            logger.warning("NaN/Inf detected in predictions, replacing with zeros")
            pred_logits = torch.nan_to_num(pred_logits, nan=0.0, posinf=10.0, neginf=-10.0)

        if torch.isnan(target).any() or torch.isinf(target).any():
            logger.warning("NaN/Inf detected in targets, replacing with zeros")
            target = torch.nan_to_num(target, nan=0.0, posinf=1.0, neginf=0.0)

        # Use BCE loss for binary segmentation with numerical stability
        try:
            bce_loss = self.bce_loss(pred_logits, target)

            # Check if BCE loss is valid
            if torch.isnan(bce_loss) or torch.isinf(bce_loss):
                logger.warning("NaN/Inf in BCE loss, using fallback")
                bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')

        except Exception as e:
            logger.warning(f"BCE loss failed: {e}, using fallback")
            bce_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')

        # Add Dice loss for better overlap with numerical stability
        pred_probs = torch.sigmoid(pred_logits)
        pred_probs = torch.clamp(pred_probs, min=1e-7, max=1-1e-7)  # Prevent extreme values

        intersection = (pred_probs * target).sum()
        union = pred_probs.sum() + target.sum()
        dice_loss = 1 - (2 * intersection + 1e-6) / (union + 1e-6)

        # Check for NaN in dice loss
        if torch.isnan(dice_loss) or torch.isinf(dice_loss):
            logger.warning("NaN/Inf in Dice loss, setting to 1.0")
            dice_loss = torch.tensor(1.0, device=pred_logits.device)

        # Combine losses with numerical stability
        if self.current_iou < 0.3:
            # Early stage: Focus on BCE
            total_loss = 0.8 * bce_loss + 0.2 * dice_loss
        elif self.current_iou < 0.6:
            # Mid stage: Balance BCE and Dice
            total_loss = 0.6 * bce_loss + 0.4 * dice_loss
        else:
            # High IoU: Focus on Dice for fine-tuning
            total_loss = 0.4 * bce_loss + 0.6 * dice_loss

        # Final NaN check
        if torch.isnan(total_loss) or torch.isinf(total_loss):
            logger.warning("NaN/Inf in total loss, using fallback BCE")
            total_loss = F.binary_cross_entropy_with_logits(pred_logits, target, reduction='mean')

        return total_loss

def calculate_metrics(pred_logits, target, threshold=0.5):
    """FIXED: Binary segmentation metrics calculation"""
    # Apply sigmoid to get probabilities
    pred_probs = torch.sigmoid(pred_logits)

    # Apply threshold to get binary predictions
    pred_binary = (pred_probs > threshold).float()

    # Ensure target is in correct format
    if target.dim() == 3:  # [B, H, W]
        target = target.unsqueeze(1)  # [B, 1, H, W]

    target_binary = (target > 0.5).float()

    # Calculate IoU for each sample in batch
    batch_size = pred_binary.size(0)
    ious = []
    dices = []

    for i in range(batch_size):
        pred_i = pred_binary[i].flatten()
        target_i = target_binary[i].flatten()

        # IoU calculation
        intersection = (pred_i * target_i).sum()
        union = pred_i.sum() + target_i.sum() - intersection

        if union == 0:
            iou = 1.0 if intersection == 0 else 0.0
        else:
            iou = (intersection / (union + 1e-8)).item()

        # Dice calculation
        dice_denom = pred_i.sum() + target_i.sum()
        if dice_denom == 0:
            dice = 1.0 if intersection == 0 else 0.0
        else:
            dice = (2 * intersection / (dice_denom + 1e-8)).item()

        ious.append(iou)
        dices.append(dice)

    # Calculate accuracy
    correct = (pred_binary == target_binary).float()
    accuracy = correct.mean().item()

    return {
        'iou': np.mean(ious),
        'dice': np.mean(dices),
        'accuracy': accuracy
    }

# ======================== MODEL EMA ========================

class ModelEMA:
    """Model Exponential Moving Average for better generalization"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ======================== TRAINING PIPELINE ========================

def train_unet_fusion(
    data_dir='./data_ir',
    epochs=100,
    batch_size=None,
    lr=0.0001,
    target_iou=0.7,
    dataset_type='fusion',
    base_channels=32,
    num_stages=3,
    block_depth=2,
    use_skip=True,
):  # ULTRA-CONSERVATIVE LR
    """
    AGGRESSIVE training pipeline for VALIDATION IoU > 0.7 while allowing light-weight UNet variants.

    Args:
        base_channels: Number of channels in the first encoder block (controls model width).
        num_stages: Encoder/decoder depth (3 keeps the network compact, 4 matches classic UNet).
        block_depth: Number of conv layers inside each DoubleConv block.
        use_skip: Enable/disable skip connections to trade accuracy for parameter count.
    """

    if block_depth < 1:
        raise ValueError("block_depth must be >= 1")

    def _log_dataset_overview(name: str, dataset):
        size = len(dataset)
        logger.info("%s dataset size: %d samples", name, size)
        if size == 0:
            return
        try:
            sample = dataset[0]
        except Exception as exc:  # pragma: no cover - diagnostic only
            logger.warning("Unable to inspect %s dataset sample shape: %s", name, exc)
            return

        if isinstance(sample, dict):
            image = sample.get("image")
            label = sample.get("label")
        elif isinstance(sample, (list, tuple)):
            image = sample[0]
            label = sample[1] if len(sample) > 1 else None
        else:
            image, label = sample, None

        def _describe_tensor(t):
            if t is None:
                return "None", "None"
            if hasattr(t, "shape"):
                shape = tuple(t.shape)
            elif hasattr(t, "size"):
                size_attr = t.size
                shape = size_attr() if callable(size_attr) else size_attr
            else:
                shape = "Unknown"
            dtype = getattr(t, "dtype", type(t).__name__)
            return shape, dtype

        img_shape, img_dtype = _describe_tensor(image)
        lbl_shape, lbl_dtype = _describe_tensor(label)
        logger.info(
            "%s dataset sample shapes -> image=%s (%s), label=%s (%s)",
            name,
            img_shape,
            img_dtype,
            lbl_shape,
            lbl_dtype,
        )

    # DEBUG: Print received epochs
    print(f"DEBUG: Function received epochs = {epochs}")

    # Optimize GPU performance first
    optimize_gpu_performance()

    # Create checkpoint directory
    os.makedirs(f'checkpoints_unet_{dataset_type}', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"CUDA Memory: {gpu_memory:.1f} GB")

    # Use a default batch size when none is provided
    if batch_size is None:
        batch_size = 8
        logger.info("Batch size not provided. Using default value: 8")

    # Use gradient accumulation for larger effective batch size
    gradient_accumulation_steps = max(1, 8 // batch_size)
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"🔄 Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"📊 Effective batch size: {effective_batch_size}")

    # Create dataset based on dataset_type
    full_dataset = FusionDataset(data_dir, split='train', dataset_type=dataset_type)

    # Calculate foreground ratios for stratification
    fg_ratios = []
    for i in range(len(full_dataset)):
        sample = full_dataset[i]
        label = sample['label']
        fg_ratio = (label == 1).float().mean().item()
        fg_ratios.append(fg_ratio)

    # Create stratified indices
    import numpy as np
    fg_ratios = np.array(fg_ratios)
    # Split into high/low foreground groups
    high_fg_indices = np.where(fg_ratios > 0.05)[0]  # >5% foreground
    low_fg_indices = np.where(fg_ratios <= 0.05)[0]  # <=5% foreground

    # Stratified split
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(high_fg_indices)
    np.random.shuffle(low_fg_indices)

    train_high = high_fg_indices[:int(0.8 * len(high_fg_indices))]
    val_high = high_fg_indices[int(0.8 * len(high_fg_indices)):]
    train_low = low_fg_indices[:int(0.8 * len(low_fg_indices))]
    val_low = low_fg_indices[int(0.8 * len(low_fg_indices)):]

    train_indices = np.concatenate([train_high, train_low])
    val_indices = np.concatenate([val_high, val_low])

    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_base_dataset = FusionDataset(data_dir, split='val', dataset_type=dataset_type)
    val_base_dataset.samples = list(full_dataset.samples)
    val_dataset = torch.utils.data.Subset(val_base_dataset, val_indices)

    logger.info(f"Stratified split: Train={len(train_dataset)}, Val={len(val_dataset)}")
    logger.info(f"High FG samples - Train: {len(train_high)}, Val: {len(val_high)}")
    logger.info(f"Low FG samples - Train: {len(train_low)}, Val: {len(val_low)}")

    _log_dataset_overview("Train", train_dataset)
    _log_dataset_overview("Val", val_dataset)

    # Create data loaders with maximum performance optimization
    num_workers = min(8, os.cpu_count())  # Use more workers for better performance
    logger.info(f"🚀 Using {num_workers} workers for data loading")

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,  # Increased prefetch for better GPU utilization
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
        drop_last=False
    )

    # Create model with appropriate input channels - FIXED: Binary segmentation
    input_channels = 3 if dataset_type == 'rgb' else 1
    model = UNet(
        num_classes=1,
        input_channels=input_channels,
        base_channels=base_channels,
        num_stages=num_stages,
        depth=block_depth,
        use_skip=use_skip,
    ).to(device)

    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model architecture -> base_channels=%d, stages=%d, block_depth=%d, use_skip=%s",
        base_channels,
        num_stages,
        block_depth,
        use_skip,
    )
    logger.info(f"Total trainable parameters: {param_count/1e6:.2f}M ({param_count:,})")

    # FIXED: Binary loss function for breakthrough 0.6+ IoU barrier
    criterion = AdaptiveLoss(num_classes=1, label_smoothing=0.05)

    # FIXED: Stronger regularization to prevent overfitting
    initial_weight_decay = 0.05  # Increased weight decay
    optimizer = optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=initial_weight_decay,
        eps=1e-8,
        betas=(0.9, 0.999)
    )

    # FIXED: Aggressive early stopping to prevent overfitting
    best_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dices, val_dices = [], []
    patience = 15  # FIXED: Much shorter patience to prevent overfitting
    patience_counter = 0

    logger.info("=" * 80)
    logger.info("UNet training started")
    logger.info("=" * 80)
    logger.info(f"Target validation IoU: {target_iou}")
    logger.info(f"Total epochs: {epochs}")
    logger.info(f"Batch size: {batch_size} (effective: {effective_batch_size})")
    logger.info(f"Learning rate: {lr}")
    logger.info("Loss function: Adaptive focal + dice + IoU blend")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info("=" * 80)

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}

        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")):
            images = batch['image'].to(device, non_blocking=True)
            labels = batch['label'].to(device, non_blocking=True)

            # Disable autocast completely for stability
            # with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
            outputs = model(images)

            # Check outputs for NaN/Inf before loss calculation
            if torch.isnan(outputs).any() or torch.isinf(outputs).any():
                logger.warning("🚨 NaN/Inf in model outputs - EMERGENCY SKIP")
                continue

            loss = criterion(outputs, labels)

            # Check loss for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.warning("🚨 NaN/Inf in loss - EMERGENCY SKIP")
                continue

            # Scale loss for gradient accumulation
            loss = loss / gradient_accumulation_steps

            loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping to prevent NaN
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Check for NaN gradients
                has_nan_grad = False
                for param in model.parameters():
                    if param.grad is not None and (torch.isnan(param.grad).any() or torch.isinf(param.grad).any()):
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    logger.warning("🚨 NaN gradients detected - EMERGENCY RESET")
                    optimizer.zero_grad()
                    reset_model_if_unstable(model)
                    # Reset optimizer state
                    optimizer.state = {}
                    # Reduce learning rate drastically
                    for param_group in optimizer.param_groups:
                        param_group['lr'] *= 0.1
                        logger.warning(f"🔥 Emergency LR reduction: {param_group['lr']:.8f}")
                else:
                    optimizer.step()
                    optimizer.zero_grad()

                    # Check model stability after optimizer step
                    if not check_model_stability(model):
                        logger.warning("🚨 Model unstable after step - EMERGENCY RESET")
                        reset_model_if_unstable(model)
                        optimizer.state = {}
                        for param_group in optimizer.param_groups:
                            param_group['lr'] *= 0.5

            train_loss += loss.item() * gradient_accumulation_steps

            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(outputs, labels)
                for key in train_metrics:
                    train_metrics[key] += metrics[key]

        # Validation with ORIGINAL model (NO EMA)
        model.eval()
        # ema.apply_shadow()

        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'dice': 0.0, 'accuracy': 0.0}

        with torch.no_grad():
            for batch in val_loader:
                images = batch['image'].to(device, non_blocking=True)
                labels = batch['label'].to(device, non_blocking=True)

                with autocast(device_type='cuda' if torch.cuda.is_available() else 'cpu'):
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                val_loss += loss.item()

                metrics = calculate_metrics(outputs, labels)
                for key in val_metrics:
                    val_metrics[key] += metrics[key]

        # ema.restore()

        # Calculate averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        for key in train_metrics:
            train_metrics[key] /= len(train_loader)
            val_metrics[key] /= len(val_loader)

        current_lr = optimizer.param_groups[0]['lr']

        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_ious.append(train_metrics['iou'])
        val_ious.append(val_metrics['iou'])
        train_dices.append(train_metrics['dice'])
        val_dices.append(val_metrics['dice'])

        # Update adaptive loss with current IoU
        criterion.update_iou(val_metrics['iou'])


        # Calculate train/val gap for overfitting detection
        iou_gap = train_metrics['iou'] - val_metrics['iou']

        logger.info(
            f"Epoch {epoch+1}: "
            f"Train Loss={avg_train_loss:.4f}, Train IoU={train_metrics['iou']:.4f}, Train Dice={train_metrics['dice']:.4f}, "
            f"Val Loss={avg_val_loss:.4f}, Val IoU={val_metrics['iou']:.4f}, Val Dice={val_metrics['dice']:.4f}, "
            f"IoU Gap={iou_gap:.4f}, LR={current_lr:.6f}"
        )

        # Check for model collapse (all zeros)
        if train_metrics['iou'] == 0.0 and val_metrics['iou'] == 0.0:
            logger.error("🚨 MODEL COLLAPSE DETECTED - All IoU = 0.0")
            logger.error("🔄 EMERGENCY RESTART with new initialization")

            # Reinitialize model completely
            model = UNet(
                num_classes=1,
                input_channels=input_channels,
                base_channels=base_channels,
                num_stages=num_stages,
                depth=block_depth,
                use_skip=use_skip,
            ).to(device)

            # Reinitialize optimizer with much lower LR
            emergency_lr = current_lr * 0.01
            optimizer = optim.AdamW(
                model.parameters(),
                lr=emergency_lr,
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999)
            )


            logger.warning(f"🚨 EMERGENCY RESTART: New LR = {emergency_lr:.8f}")
            continue

        # Save best model and early stopping logic
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'best_dice': val_metrics['dice']
            }, 'best_unet_fusion.pth')
            logger.info(f"🏆 New best IoU: {best_iou:.4f} | Val Dice: {val_metrics['dice']:.4f}")
            patience_counter = 0
        else:
            patience_counter += 1

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_iou': best_iou,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_ious': train_ious,
                'val_ious': val_ious,
                'train_dices': train_dices,
                'val_dices': val_dices
            }, f'checkpoints_unet_{dataset_type}/checkpoint_epoch_{epoch+1:03d}_{dataset_type}.pth')


    # Generate results
    logger.info("\n" + "="*60)
    logger.info("🎯 GENERATING COMPREHENSIVE RESULTS")
    logger.info("="*60)

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_ious, val_ious, train_dices, val_dices)

    # Generate predictions
    generate_predictions(model, val_loader, device, dataset_type)

    return best_iou

def plot_training_curves(train_losses, val_losses, train_ious, val_ious, train_dices, val_dices):
    """Plot comprehensive training curves"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    epochs = range(1, len(train_losses) + 1)

    # Loss curves
    ax1.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    ax1.plot(epochs, val_losses, label='Val Loss', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Loss Curves')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # IoU and Dice curves
    ax2.plot(epochs, train_ious, label='Train IoU', linewidth=2)
    ax2.plot(epochs, val_ious, label='Val IoU', linewidth=2)
    ax2.plot(epochs, train_dices, label='Train Dice', linewidth=2, linestyle='--')
    ax2.plot(epochs, val_dices, label='Val Dice', linewidth=2, linestyle='--')
    ax2.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Target IoU (0.9)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Score')
    ax2.set_title('IoU / Dice Progress')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1)

    # Loss difference
    loss_diff = [val - train for val, train in zip(val_losses, train_losses)]
    ax3.plot(epochs, loss_diff, color='red', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Val Loss - Train Loss')
    ax3.set_title('Overfitting Indicator')
    ax3.grid(True, alpha=0.3)

    # IoU improvement
    iou_improvement = [0] + [val_ious[i] - val_ious[i-1] for i in range(1, len(val_ious))]
    dice_improvement = [0] + [val_dices[i] - val_dices[i-1] for i in range(1, len(val_dices))]
    ax4.plot(epochs, iou_improvement, color='green', linewidth=2, label='IoU Improvement')
    ax4.plot(epochs, dice_improvement, color='blue', linewidth=2, linestyle='--', label='Dice Improvement')
    ax4.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Metric Improvement')
    ax4.set_title('Validation Improvement Rate')
    ax4.grid(True, alpha=0.3)
    ax4.legend()

    plt.suptitle('UNet Training Analysis - Fusion Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('unet_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("📈 Training curves saved as 'unet_training_curves.png'")

def generate_predictions(model, val_loader, device, dataset_type='fusion'):
    """Generate predictions with ORIGINAL model (not EMA)"""
    model.eval()
    # DON'T use EMA for inference - use original model

    # Create checkpoint directory
    os.makedirs(f'checkpoints_unet_{dataset_type}', exist_ok=True)
    os.makedirs('predictions_fusion_unet', exist_ok=True)

    all_ious = []
    all_dices = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Generating predictions")):
            if batch_idx >= 10:  # Limit to first 10 batches for demo
                break

            images = batch['image'].to(device)
            labels = batch['label'].to(device)
            case_names = batch['case_name']

            # Use original model, not EMA
            outputs = model(images)
            # FIXED: Binary segmentation - use sigmoid + threshold instead of argmax
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            for i in range(images.size(0)):
                # Calculate IoU for this sample
                pred_i = predictions[i:i+1]
                label_i = labels[i:i+1]
                metrics = calculate_metrics(outputs[i:i+1], label_i)
                iou = metrics['iou']
                dice = metrics['dice']
                all_ious.append(iou)
                all_dices.append(dice)

                # Convert to numpy for visualization
                img_fusion = images[i, 0].cpu().numpy()  # First channel (Fusion)
                label_np = labels[i].cpu().numpy()
                pred_np = predictions[i, 0].cpu().numpy()  # FIXED: Remove extra dimension

                # Create visualization with original filename
                case_name = case_names[i] if isinstance(case_names, list) else case_names[i].item()
                fig, axes = plt.subplots(1, 4, figsize=(16, 4))

                # Use grayscale for fusion images to preserve original appearance
                axes[0].imshow(img_fusion, cmap='gray', vmin=0, vmax=1)
                axes[0].set_title('Fusion Image')
                axes[0].axis('off')

                axes[1].imshow(label_np, cmap='Blues')
                axes[1].set_title('Ground Truth')
                axes[1].axis('off')

                axes[2].imshow(pred_np.squeeze(), cmap='Reds')  # FIXED: Remove all extra dimensions
                axes[2].set_title('Prediction')
                axes[2].axis('off')

                # Overlay
                axes[3].imshow(img_fusion, cmap='gray', alpha=0.7)
                axes[3].imshow(pred_np.squeeze(), cmap='Reds', alpha=0.5)  # FIXED: Remove all extra dimensions
                axes[3].set_title('Overlay')
                axes[3].axis('off')

                color = 'green' if iou > 0.1 else 'orange' if iou > 0.05 else 'red'
                plt.suptitle(
                    f'{case_name}_predict - IoU: {iou:.4f} | Dice: {dice:.4f}',
                    fontsize=14,
                    fontweight='bold',
                    color=color
                )

                plt.tight_layout()
                plt.savefig(f'predictions_fusion_unet/{case_name}_predict.png',
                           dpi=150, bbox_inches='tight')
                plt.close()

    # Summary statistics
    logger.info(f"\n📊 Prediction Summary:")
    logger.info(f"Total samples: {len(all_ious)}")
    logger.info(f"Mean IoU: {np.mean(all_ious):.4f}")
    logger.info(f"Std IoU: {np.std(all_ious):.4f}")
    logger.info(f"Min IoU: {np.min(all_ious):.4f}")
    logger.info(f"Max IoU: {np.max(all_ious):.4f}")
    logger.info(f"Mean Dice: {np.mean(all_dices):.4f}")
    logger.info(f"Std Dice: {np.std(all_dices):.4f}")
    logger.info(f"Min Dice: {np.min(all_dices):.4f}")
    logger.info(f"Max Dice: {np.max(all_dices):.4f}")
    logger.info(f"Samples with IoU > 0.1: {sum(1 for iou in all_ious if iou > 0.1)}/{len(all_ious)}")
    logger.info(f"Samples with IoU > 0.05: {sum(1 for iou in all_ious if iou > 0.05)}/{len(all_ious)}")

# ======================== MAIN EXECUTION ========================



def main():
    """Main function with argument parsing"""
    import argparse

    parser = argparse.ArgumentParser(description='UNet Training')
    parser.add_argument('--dataset_type', type=str, choices=['fusion', 'rgb'], default='fusion',
                       help='Dataset type (fusion or rgb)')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./data_ir',
                       help='Data directory')
    parser.add_argument('--base_channels', type=int, default=32,
                        help='Base number of channels in the first UNet block (controls width/params)')
    parser.add_argument('--num_stages', type=int, choices=[3, 4], default=3,
                        help='Encoder/decoder depth (3=lighter, 4=classic UNet)')
    parser.add_argument('--block_depth', type=int, choices=[1, 2], default=2,
                        help='Number of convolutions per UNet block')
    parser.add_argument('--disable_skip', action='store_true',
                        help='Disable skip connections to minimize parameters further')

    args = parser.parse_args()

    # Train with parsed arguments
    train_unet_fusion(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        target_iou=0.7,   # Target VALIDATION IoU 0.7 during training
        dataset_type=args.dataset_type,
        base_channels=args.base_channels,
        num_stages=args.num_stages,
        block_depth=args.block_depth,
        use_skip=not args.disable_skip,
    )

def check_model_stability(model):
    """Check model for numerical stability"""
    has_nan = False
    has_inf = False

    for name, param in model.named_parameters():
        if param.data is not None:
            if torch.isnan(param.data).any():
                logger.warning(f"NaN detected in parameter: {name}")
                has_nan = True
            if torch.isinf(param.data).any():
                logger.warning(f"Inf detected in parameter: {name}")
                has_inf = True

        if param.grad is not None:
            if torch.isnan(param.grad).any():
                logger.warning(f"NaN detected in gradient: {name}")
                has_nan = True
            if torch.isinf(param.grad).any():
                logger.warning(f"Inf detected in gradient: {name}")
                has_inf = True

    return not (has_nan or has_inf)

def reset_model_if_unstable(model):
    """Reset model parameters if they become unstable - AGGRESSIVE RESET"""
    logger.warning("🚨 AGGRESSIVE MODEL RESET - Reinitializing ALL parameters")

    # Complete model reinitialization
    for name, param in model.named_parameters():
        if param.data is not None:
            if 'weight' in name:
                if len(param.shape) >= 2:
                    # Much smaller initialization
                    nn.init.xavier_uniform_(param.data, gain=0.1)
                else:
                    nn.init.normal_(param.data, 0, 0.001)
            elif 'bias' in name:
                nn.init.constant_(param.data, 0)

            # Clamp all parameters to safe range
            param.data = torch.clamp(param.data, min=-1.0, max=1.0)

if __name__ == "__main__":
    main()



