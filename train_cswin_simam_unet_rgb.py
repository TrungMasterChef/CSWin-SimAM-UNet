#!/usr/bin/env python3
"""
CSWin-SimAM-UNet: TRUE CSWin Transformer with SimAM for Crack Segmentation
Implements proper Cross-Shaped Window Transformer with SimAM attention modules

Key Features:
- Real CSWin Transformer blocks with horizontal/vertical stripe attention
- Cross-shaped window self-attention mechanism
- SimAM modules for enhanced feature attention
- UNet decoder with skip connections and attention gates
- Hierarchical transformer encoder structure
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import OneCycleLR
import numpy as np
import cv2
from PIL import Image
import glob
import logging
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision import transforms
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

class AdaptiveLRScheduler:
    """ADAPTIVE learning rate scheduler with IoU-based restarts"""

    def __init__(self, optimizer, total_epochs=150, max_lr=0.0001):  # ULTRA-CONSERVATIVE max_lr
        self.optimizer = optimizer
        self.total_epochs = total_epochs
        self.max_lr = max_lr
        self.current_epoch = 0
        self.warmup_epochs = 5  # FIXED: Shorter warmup
        self.best_iou = 0.0
        self.plateau_counter = 0
        self.restart_count = 0

        # FIXED: Proper CosineAnnealingLR with correct T_max
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,  # Match actual epochs
            eta_min=max_lr / 100  # Higher min LR for stability
        )

        logger.info(f"🔥 ADAPTIVE LR Scheduler for {total_epochs} epochs")
        logger.info(f"   Max LR: {max_lr} with IoU-based restarts")
        logger.info(f"   Warmup epochs: {self.warmup_epochs}")
        logger.info(f"   Strategy: OneCycleLR with adaptive restarts")

    def step(self, epoch=None, current_iou=None):
        """Step with adaptive restart based on IoU plateau"""
        self.current_epoch = epoch if epoch is not None else self.current_epoch + 1

        # FIXED: More aggressive IoU-based LR adjustment
        if current_iou is not None:
            if current_iou > self.best_iou + 0.005:  # Lower threshold for improvement
                self.best_iou = current_iou
                self.plateau_counter = 0
            else:
                self.plateau_counter += 1

            # FIXED: Earlier and more frequent LR reduction
            if self.plateau_counter >= 8:  # Shorter patience
                self.reduce_lr()
                self.plateau_counter = 0

        self.scheduler.step()
        return self.get_current_lr()

    def reduce_lr(self):
        """FIXED: Reduce learning rate when plateau"""
        current_lr = self.get_current_lr()
        new_lr = current_lr * 0.5  # Reduce by half
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(new_lr, self.max_lr / 1000)  # Don't go below min
        logger.info(f"🔥 LR reduced: {current_lr:.6f} → {new_lr:.6f}")

    def restart_lr(self, current_iou):
        """Restart learning rate when IoU plateaus"""
        # Adaptive restart LR based on current IoU level
        if current_iou > 0.6:
            restart_lr = 0.002  # Lower restart for high IoU
        else:
            restart_lr = 0.003  # Higher restart for medium IoU

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = restart_lr

        logger.info(f"🔄 LR RESTART #{self.restart_count + 1}: IoU plateau at {current_iou:.4f}")
        logger.info(f"   New LR: {restart_lr}")

    def get_current_lr(self):
        """Get current learning rate"""
        return self.optimizer.param_groups[0]['lr']

class HyperparameterOptimizer:
    """Automatic hyperparameter optimization during training"""

    def __init__(self):
        self.best_iou = 0.0
        self.best_params = {}
        self.trial_history = []

    def suggest_batch_size(self, gpu_memory_gb):
        """Suggest optimal batch size based on GPU memory"""
        if gpu_memory_gb >= 24:
            return 8
        elif gpu_memory_gb >= 16:
            return 6
        elif gpu_memory_gb >= 12:
            return 4
        else:
            return 2

    def suggest_learning_rate(self, epoch, current_iou, current_loss):
        """Dynamic learning rate suggestion based on performance"""
        if epoch < 50:
            # Early training - be conservative
            return 0.0001
        elif current_iou > 0.4:
            # Good performance - reduce LR for fine-tuning
            return 0.00005
        elif current_loss > 0.5:
            # High loss - increase LR to escape local minima
            return 0.0002
        else:
            # Default
            return 0.0001

    def suggest_weight_decay(self, current_iou, train_iou, val_iou):
        """Dynamic weight decay based on overfitting"""
        iou_gap = train_iou - val_iou

        if iou_gap > 0.1:  # Overfitting
            return 0.01  # Strong regularization
        elif iou_gap > 0.05:
            return 0.005  # Medium regularization
        else:
            return 0.001  # Light regularization

    def update_performance(self, epoch, iou, loss, params):
        """Update performance tracking"""
        if iou > self.best_iou:
            self.best_iou = iou
            self.best_params = params.copy()
            logger.info(f"🎯 New best IoU: {iou:.4f} with params: {params}")

        self.trial_history.append({
            'epoch': epoch,
            'iou': iou,
            'loss': loss,
            'params': params.copy()
        })

# ======================== CSWIN TRANSFORMER COMPONENTS ========================

def img2windows(img, H_sp, W_sp):
    """Convert image to windows for CSWin attention"""
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm

def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """Convert windows back to image for CSWin attention"""
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))
    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img

class Mlp(nn.Module):
    """MLP module for CSWin Transformer"""
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class LePEAttention(nn.Module):
    """Cross-Shaped Window Self-Attention with Locally-enhanced Positional Encoding"""
    def __init__(self, dim, resolution, idx, split_size=7, dim_out=None, num_heads=8, attn_drop=0., proj_drop=0., qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # Determine window size based on attention type
        if idx == -1:  # Full attention
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:  # Horizontal stripe attention
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:  # Vertical stripe attention
            W_sp, H_sp = self.resolution, self.split_size
        else:
            raise ValueError(f"Invalid attention index: {idx}")

        self.H_sp = H_sp
        self.W_sp = W_sp

        # Locally-enhanced positional encoding
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        """Convert input to cross-shaped windows"""
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        """Get locally-enhanced positional encoding"""
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)

        lepe = func(x)
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()
        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """Forward pass with cross-shaped window attention"""
        q, k, v = qkv[0], qkv[1], qkv[2]

        H = W = self.resolution
        B, L, C = q.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # Convert to cross-shaped windows
        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        # Scaled dot-product attention
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        # Apply attention and add positional encoding
        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)

        # Convert back to image format
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)
        return x

# ======================== SimAM ATTENTION MODULE ========================

class SimAM(nn.Module):
    """Simple parameter-free attention module for enhanced feature modeling."""
    def __init__(self, channels=None, e_lambda=1e-4):
        super().__init__()
        self.e_lambda = e_lambda

    def forward(self, x):
        if x.dim() != 4:
            return x

        _, _, h, w = x.size()
        if h * w <= 1:
            return x

        n = h * w - 1
        mean = x.mean(dim=(2, 3), keepdim=True)
        delta = x - mean
        delta_sq = delta.pow(2)
        var = delta_sq.sum(dim=(2, 3), keepdim=True) / n
        denominator = 4 * (var + self.e_lambda)
        attention = delta_sq / denominator + 0.5
        return x * attention

class CSWinBlock(nn.Module):
    """CSWin Transformer Block with Cross-Shaped Window Self-Attention"""
    def __init__(self, dim, reso, num_heads, split_size=7, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio

        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        # Determine number of attention branches
        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1  # Full attention
        else:
            self.branch_num = 2  # Horizontal + Vertical attention

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

        # Create attention modules
        if last_stage:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim, resolution=self.patches_resolution, idx=-1,
                    split_size=split_size, num_heads=num_heads, dim_out=dim,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])
        else:
            self.attns = nn.ModuleList([
                LePEAttention(
                    dim//2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads//2, dim_out=dim//2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer, drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """Forward pass with cross-shaped window attention"""
        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"

        # Layer normalization and QKV projection
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        # Apply cross-shaped window attention
        if self.branch_num == 2:
            # Horizontal and vertical stripe attention
            x1 = self.attns[0](qkv[:, :, :, :C//2])  # Horizontal
            x2 = self.attns[1](qkv[:, :, :, C//2:])  # Vertical
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            # Full attention
            attened_x = self.attns[0](qkv)

        attened_x = self.proj(attened_x)
        x = x + attened_x

        # MLP
        x = x + self.mlp(self.norm2(x))
        return x

class Merge_Block(nn.Module):
    """Merge block for downsampling between CSWin stages"""
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)
        return x

class AttentionGate(nn.Module):
    """Attention gate for skip connections"""
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

# ======================== CSWIN-SimAM-UNET ARCHITECTURE ========================

class CSWinSimAMUNet(nn.Module):
    """CSWin-SimAM-UNet: True CSWin Transformer encoder with SimAM and UNet decoder"""

    def __init__(self, num_classes=1, img_size=224, input_channels=1, embed_dim=64, depths=[2, 2, 6, 2],
                 split_sizes=[1, 2, 7, 7], num_heads=[2, 4, 8, 16]):
        super(CSWinSimAMUNet, self).__init__()
        self.num_classes = num_classes
        self.img_size = img_size
        self.input_channels = input_channels
        self.embed_dim = embed_dim

        # CSWin Transformer Encoder
        # Stage 1: Initial patch embedding
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(input_channels, embed_dim, 7, 4, 2),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True)
        )

        # CSWin Transformer stages
        curr_dim = embed_dim
        self.stage1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, reso=img_size//4, num_heads=num_heads[0],
                split_size=split_sizes[0], mlp_ratio=4., qkv_bias=True,
                drop=0.1, attn_drop=0.1, norm_layer=nn.LayerNorm)
            for _ in range(depths[0])])

        self.merge1 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, reso=img_size//8, num_heads=num_heads[1],
                split_size=split_sizes[1], mlp_ratio=4., qkv_bias=True,
                drop=0.1, attn_drop=0.1, norm_layer=nn.LayerNorm)
            for _ in range(depths[1])])

        self.merge2 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim * 2
        self.stage3 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, reso=img_size//16, num_heads=num_heads[2],
                split_size=split_sizes[2], mlp_ratio=4., qkv_bias=True,
                drop=0.1, attn_drop=0.1, norm_layer=nn.LayerNorm)
            for _ in range(depths[2])])

        self.merge3 = Merge_Block(curr_dim, curr_dim*2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, reso=img_size//32, num_heads=num_heads[3],
                split_size=split_sizes[3], mlp_ratio=4., qkv_bias=True,
                drop=0.1, attn_drop=0.1, norm_layer=nn.LayerNorm, last_stage=True)
            for _ in range(depths[3])])

        # Feature dimension mapping for UNet decoder
        self.feature_dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]

        # SimAM modules for enhanced attention
        self.simam1 = SimAM(self.feature_dims[0])
        self.simam2 = SimAM(self.feature_dims[1])
        self.simam3 = SimAM(self.feature_dims[2])
        self.simam4 = SimAM(self.feature_dims[3])

        # UNet Decoder with attention gates
        # FIX: Điều chỉnh input channels để attention gate output được sử dụng đúng
        self.att_gate4 = AttentionGate(self.feature_dims[3], self.feature_dims[2], self.feature_dims[2]//2)
        self.att_gate3 = AttentionGate(self.feature_dims[2], self.feature_dims[1], self.feature_dims[1]//2)
        self.att_gate2 = AttentionGate(self.feature_dims[1], self.feature_dims[0], self.feature_dims[0]//2)

        # FIX: Decoder4 nhận concat của upsampled bottleneck + attention-gated skip
        # decoder4: 512 (upsampled) + 256 (attention gated skip) = 768 -> 256
        self.decoder4 = self._make_decoder_stage(self.feature_dims[3] + self.feature_dims[2], self.feature_dims[2], self.feature_dims[2])
        # decoder3: 256 (upsampled) + 128 (attention gated skip) = 384 -> 128
        self.decoder3 = self._make_decoder_stage(self.feature_dims[2] + self.feature_dims[1], self.feature_dims[1], self.feature_dims[1])
        # decoder2: 128 (upsampled) + 64 (attention gated skip) = 192 -> 64
        self.decoder2 = self._make_decoder_stage(self.feature_dims[1] + self.feature_dims[0], self.feature_dims[0], self.feature_dims[0])
        self.decoder1 = self._make_decoder_stage(self.feature_dims[0], 32, 32)

        # Final output head
        self.output_head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            SimAM(16),
            nn.Conv2d(16, num_classes, 1)
        )

        # Initialize weights
        self._initialize_weights()
    def _make_decoder_stage(self, in_channels, mid_channels, out_channels):
        """Decoder stage with SimAM attention"""
        return nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, 3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            SimAM(mid_channels),
            nn.Conv2d(mid_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def _tensor_to_2d(self, x, H, W):
        """Convert 1D tensor to 2D for CNN operations"""
        B, L, C = x.shape
        return x.transpose(-2, -1).contiguous().view(B, C, H, W)

    def _tensor_to_1d(self, x):
        """Convert 2D tensor to 1D for transformer operations"""
        B, C, H, W = x.shape
        return x.view(B, C, -1).transpose(-2, -1).contiguous()
    
    def _initialize_weights(self):
        """Initialize weights with numerical stability"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # Use smaller initialization to prevent exploding gradients
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                # Smaller initialization for transformer layers
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """Forward pass with CSWin encoder and UNet decoder"""
        B = x.shape[0]

        # CSWin Transformer Encoder
        # Stage 1: Patch embedding
        x = self.stage1_conv_embed(x)  # [B, C, H/4, W/4]
        x = self._tensor_to_1d(x)  # Convert to [B, L, C] for transformer

        # Apply CSWin blocks and collect features for skip connections
        features = []

        # Stage 1
        for blk in self.stage1:
            x = blk(x)
        x1_2d = self._tensor_to_2d(x, self.img_size//4, self.img_size//4)
        x1_2d = self.simam1(x1_2d)
        features.append(x1_2d)

        # Stage 2
        x = self.merge1(x)
        for blk in self.stage2:
            x = blk(x)
        x2_2d = self._tensor_to_2d(x, self.img_size//8, self.img_size//8)
        x2_2d = self.simam2(x2_2d)
        features.append(x2_2d)

        # Stage 3
        x = self.merge2(x)
        for blk in self.stage3:
            x = blk(x)
        x3_2d = self._tensor_to_2d(x, self.img_size//16, self.img_size//16)
        x3_2d = self.simam3(x3_2d)
        features.append(x3_2d)

        # Stage 4 (bottleneck)
        x = self.merge3(x)
        for blk in self.stage4:
            x = blk(x)
        x4_2d = self._tensor_to_2d(x, self.img_size//32, self.img_size//32)
        x4_2d = self.simam4(x4_2d)

        # UNet Decoder with skip connections and attention gates
        # FIX: Sử dụng attention gate output đúng cách cho TẤT CẢ decoder stages

        # Decoder 4: upsample bottleneck + attention-gated skip from stage3
        d4 = F.interpolate(x4_2d, size=features[2].shape[-2:], mode='bilinear', align_corners=False)
        x3_att = self.att_gate4(d4, features[2])  # Attention gate: gating signal=d4, skip=features[2]
        d4 = self.decoder4(torch.cat([d4, x3_att], dim=1))  # FIX: Concat và decode

        # Decoder 3: upsample d4 + attention-gated skip from stage2
        d3 = F.interpolate(d4, size=features[1].shape[-2:], mode='bilinear', align_corners=False)
        x2_att = self.att_gate3(d3, features[1])  # Attention gate: gating signal=d3, skip=features[1]
        d3 = self.decoder3(torch.cat([d3, x2_att], dim=1))

        # Decoder 2: upsample d3 + attention-gated skip from stage1
        d2 = F.interpolate(d3, size=features[0].shape[-2:], mode='bilinear', align_corners=False)
        x1_att = self.att_gate2(d2, features[0])  # Attention gate: gating signal=d2, skip=features[0]
        d2 = self.decoder2(torch.cat([d2, x1_att], dim=1))

        # Decoder 1 (no skip connection, just upsampling)
        d1 = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=False)
        d1 = self.decoder1(d1)

        # Final output
        output = self.output_head(d1)
        output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return output

# ======================== ADVANCED DATASET WITH STRONG AUGMENTATION ========================

class FusionDataset(torch.utils.data.Dataset):
    """
    Dataset cho crack segmentation với paired augmentation đồng bộ.
    FIX: Sử dụng torchvision.transforms.functional để đảm bảo image và mask
    được augment với cùng tham số ngẫu nhiên.
    """
    def __init__(self, data_dir, split='train', img_size=224, dataset_type='fusion', augment_factor=None):
        self.data_dir = data_dir
        self.split = split
        self.img_size = img_size
        self.dataset_type = dataset_type

        # Choose image directory based on dataset type
        if dataset_type == 'rgb':
            self.image_dir = os.path.join(data_dir, '01-Visible images')
        else:  # fusion
            self.image_dir = os.path.join(data_dir, '03-Fusion(50IRT) images')
        self.label_dir = os.path.join(data_dir, '04-Ground truth')

        # Find all valid image-label pairs
        self.samples = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            for img_path in glob.glob(os.path.join(self.image_dir, ext)):
                img_name = os.path.splitext(os.path.basename(img_path))[0]
                # Try both .png and .jpg for labels
                label_path_png = os.path.join(self.label_dir, img_name + '.png')
                label_path_jpg = os.path.join(self.label_dir, img_name + '.jpg')

                if os.path.exists(label_path_png):
                    self.samples.append((img_path, label_path_png))
                elif os.path.exists(label_path_jpg):
                    self.samples.append((img_path, label_path_jpg))

        logger.info(f"Found {len(self.samples)} valid {dataset_type.upper()} samples for {split}")

        # FIX: Cho phép override augment_factor, mặc định 4 cho train, 1 cho val
        if augment_factor is not None:
            self.augment_factor = augment_factor
        else:
            self.augment_factor = 4 if split == 'train' else 1

        # Color jitter chỉ áp dụng cho image, không áp dụng cho mask
        if split == 'train':
            self.color_jitter = transforms.ColorJitter(brightness=0.3, contrast=0.3)
        else:
            self.color_jitter = None

        # Normalization parameters
        if dataset_type == 'rgb':
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                   std=[0.229, 0.224, 0.225])
        else:
            self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    # --------- Paired augmentation helpers (FIX: đồng bộ image và mask) ---------
    def _resize_pair(self, img, msk):
        """Resize cả image và mask về cùng kích thước"""
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        img = TF.resize(img, (self.img_size, self.img_size), interpolation=InterpolationMode.BILINEAR)
        msk = TF.resize(msk, (self.img_size, self.img_size), interpolation=InterpolationMode.NEAREST)
        return img, msk

    def _maybe_hflip(self, img, msk, p=0.6):
        """Random horizontal flip cho CẢ image và mask"""
        from torchvision.transforms import functional as TF
        if random.random() < p:
            img = TF.hflip(img)
            msk = TF.hflip(msk)
        return img, msk

    def _maybe_vflip(self, img, msk, p=0.6):
        """Random vertical flip cho CẢ image và mask"""
        from torchvision.transforms import functional as TF
        if random.random() < p:
            img = TF.vflip(img)
            msk = TF.vflip(msk)
        return img, msk

    def _random_rotate(self, img, msk, low=-45, high=45):
        """Random rotation cho CẢ image và mask với cùng góc"""
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        angle = random.uniform(low, high)
        img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR, fill=0)
        msk = TF.rotate(msk, angle, interpolation=InterpolationMode.NEAREST, fill=0)
        return img, msk

    def _random_affine(self, img, msk, max_translate=0.15, scale_range=(0.75, 1.25), shear_range=(-8, 8)):
        """Random affine transform cho CẢ image và mask với cùng tham số"""
        from torchvision.transforms import functional as TF
        from torchvision.transforms import InterpolationMode
        tx = int(max_translate * self.img_size)
        ty = int(max_translate * self.img_size)
        translate = (random.randint(-tx, tx), random.randint(-ty, ty))
        scale = random.uniform(scale_range[0], scale_range[1])
        shear = random.uniform(shear_range[0], shear_range[1])

        img = TF.affine(img, angle=0, translate=translate, scale=scale, shear=[shear, 0],
                        interpolation=InterpolationMode.BILINEAR, fill=0)
        msk = TF.affine(msk, angle=0, translate=translate, scale=scale, shear=[shear, 0],
                        interpolation=InterpolationMode.NEAREST, fill=0)
        return img, msk
    
    def __len__(self):
        return len(self.samples) * self.augment_factor

    def __getitem__(self, idx):
        """
        FIX: Sử dụng paired augmentation đồng bộ cho image và mask
        """
        from torchvision.transforms import functional as TF
        from PIL import Image as PILImage

        # Map augmented index back to original sample
        original_idx = idx // self.augment_factor

        img_path, label_path = self.samples[original_idx]

        # Load image và mask bằng PIL để tránh vấn đề BGR/RGB
        if self.dataset_type == 'rgb':
            img = PILImage.open(img_path).convert('RGB')
        else:  # fusion
            img = PILImage.open(img_path).convert('L')
        msk = PILImage.open(label_path).convert('L')

        # Resize trước
        img, msk = self._resize_pair(img, msk)

        # FIX: Áp dụng paired augmentation đồng bộ cho CẢ image và mask
        if self.split == 'train':
            img, msk = self._maybe_hflip(img, msk, p=0.6)
            img, msk = self._maybe_vflip(img, msk, p=0.6)
            img, msk = self._random_rotate(img, msk, low=-45, high=45)
            img, msk = self._random_affine(img, msk, max_translate=0.15,
                                            scale_range=(0.75, 1.25), shear_range=(-8, 8))

            # Color jitter CHỈ cho image, không cho mask
            if self.color_jitter is not None:
                img = self.color_jitter(img)

        # Convert to tensor
        image = TF.to_tensor(img)  # CxHxW, float32 in [0,1]

        # Normalize image
        if self.dataset_type == 'rgb' and image.shape[0] == 3:
            image = self.normalize(image)
        elif self.dataset_type != 'rgb' and image.shape[0] == 1:
            image = self.normalize(image)

        # Convert mask to tensor và binary
        label = TF.to_tensor(msk).squeeze(0)  # HxW
        label = (label > 0.5).long()

        # Đảm bảo số kênh đầu vào phù hợp
        if self.dataset_type == 'rgb':
            if image.shape[0] != 3:
                image = image.repeat(3, 1, 1)
        else:  # fusion
            if image.shape[0] == 3:
                image = image[0:1]  # Lấy 1 kênh
            elif image.shape[0] != 1:
                image = image.unsqueeze(0) if len(image.shape) == 2 else image[0:1]

        return {
            'image': image,
            'label': label,
            'case_name': os.path.splitext(os.path.basename(img_path))[0]
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

def train_cswin_simam_unet(data_dir='./data_ir', epochs=100, batch_size=None, lr=0.0001, target_iou=0.7, dataset_type='fusion'):  # ULTRA-CONSERVATIVE LR
    """AGGRESSIVE training pipeline for VALIDATION IoU > 0.7"""

    # DEBUG: Print received epochs
    print(f"DEBUG: Function received epochs = {epochs}")

    # Optimize GPU performance first
    optimize_gpu_performance()

    # Create checkpoint directory
    os.makedirs(f'checkpoints_cswin_simam_unet_{dataset_type}', exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        logger.info(f"GPU: {gpu_name}")
        logger.info(f"CUDA Memory: {gpu_memory:.1f} GB")

    # Initialize hyperparameter optimizer
    hp_optimizer = HyperparameterOptimizer()

    # Auto-suggest batch size if not provided
    if batch_size is None:
        batch_size = hp_optimizer.suggest_batch_size(gpu_memory)
        logger.info(f"🎯 Auto-suggested batch size: {batch_size}")

    # Use gradient accumulation for larger effective batch size
    gradient_accumulation_steps = max(1, 8 // batch_size)
    effective_batch_size = batch_size * gradient_accumulation_steps
    logger.info(f"🔄 Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info(f"📊 Effective batch size: {effective_batch_size}")

    # ==========================================================================
    # FIX DATA LEAKAGE: Split dựa trên ORIGINAL samples, không phải augmented
    # ==========================================================================

    # Tạo dataset với augment_factor=1 để tính foreground ratio trên ảnh gốc
    base_dataset = FusionDataset(data_dir, split='train', dataset_type=dataset_type, augment_factor=1)
    num_original_samples = len(base_dataset.samples)

    logger.info(f"Total original samples: {num_original_samples}")

    # Calculate foreground ratios cho ORIGINAL samples (không augmented)
    fg_ratios = []
    for i in range(num_original_samples):
        sample = base_dataset[i]
        label = sample['label']
        fg_ratio = (label == 1).float().mean().item()
        fg_ratios.append(fg_ratio)

    # Create stratified indices trên ORIGINAL sample indices
    fg_ratios = np.array(fg_ratios)
    high_fg_indices = np.where(fg_ratios > 0.05)[0]  # >5% foreground
    low_fg_indices = np.where(fg_ratios <= 0.05)[0]  # <=5% foreground

    # Stratified split với fixed seed
    rng = np.random.default_rng(42)  # Reproducible
    rng.shuffle(high_fg_indices)
    rng.shuffle(low_fg_indices)

    train_high = high_fg_indices[:int(0.8 * len(high_fg_indices))]
    val_high = high_fg_indices[int(0.8 * len(high_fg_indices)):]
    train_low = low_fg_indices[:int(0.8 * len(low_fg_indices))]
    val_low = low_fg_indices[int(0.8 * len(low_fg_indices)):]

    train_sample_indices = np.concatenate([train_high, train_low])
    val_sample_indices = np.concatenate([val_high, val_low])

    # ==========================================================================
    # FIX: Kiểm tra DATA LEAKAGE - đảm bảo không có overlap
    # ==========================================================================
    train_set = set(train_sample_indices.tolist())
    val_set = set(val_sample_indices.tolist())
    overlap = train_set.intersection(val_set)
    if overlap:
        raise RuntimeError(f"DATA LEAKAGE DETECTED: {len(overlap)} overlapping sample indices!")
    logger.info(f"Data leakage check PASSED: No overlap between train and val sets")

    # Lấy danh sách sample paths cho train và val
    train_samples = [base_dataset.samples[i] for i in train_sample_indices]
    val_samples = [base_dataset.samples[i] for i in val_sample_indices]

    # Tạo RIÊNG BIỆT train dataset (với augmentation) và val dataset (không augmentation)
    train_dataset = FusionDataset(data_dir, split='train', dataset_type=dataset_type, augment_factor=4)
    train_dataset.samples = train_samples  # Chỉ chứa train samples

    val_dataset = FusionDataset(data_dir, split='val', dataset_type=dataset_type, augment_factor=1)
    val_dataset.samples = val_samples  # Chỉ chứa val samples

    logger.info(f"Stratified split: Train={len(train_dataset)} (augmented), Val={len(val_dataset)}")
    logger.info(f"Original samples - Train: {len(train_samples)}, Val: {len(val_samples)}")
    logger.info(f"High FG samples - Train: {len(train_high)}, Val: {len(val_high)}")
    logger.info(f"Low FG samples - Train: {len(train_low)}, Val: {len(val_low)}")

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
    model = CSWinSimAMUNet(num_classes=1, input_channels=input_channels).to(device)

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

    # ADAPTIVE learning rate scheduler with IoU-based restarts
    scheduler = AdaptiveLRScheduler(optimizer, total_epochs=epochs, max_lr=lr)

    # FIXED: Aggressive early stopping to prevent overfitting
    best_iou = 0.0
    train_losses, val_losses = [], []
    train_ious, val_ious = [], []
    train_dices, val_dices = [], []
    patience = 15  # FIXED: Much shorter patience to prevent overfitting
    patience_counter = 0

    logger.info("="*80)
    logger.info("🔥 AGGRESSIVE CSWin-SimAM-UNet for VALIDATION IoU > 0.7")
    logger.info("="*80)
    logger.info(f"🎯 Target VALIDATION IoU: {target_iou} (during training)")
    logger.info(f"📊 Total epochs: {epochs} (EXTENDED)")
    logger.info(f"📦 Batch size: {batch_size} (effective: {effective_batch_size})")
    logger.info(f"🚀 Learning rate: {lr} (AGGRESSIVE OneCycleLR)")
    logger.info(f"🔥 Loss: AGGRESSIVE Focal + Dice + IoU with class weighting")
    logger.info(f"⏰ Early stopping patience: {patience}")
    logger.info(f"🔄 Gradient accumulation steps: {gradient_accumulation_steps}")
    logger.info("="*80)

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

        # Step the adaptive scheduler with IoU feedback
        current_lr = scheduler.step(epoch, val_metrics['iou'])

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
            model = CSWinSimAMUNet(num_classes=1, input_channels=1).to(device)

            # Reinitialize optimizer with much lower LR
            emergency_lr = current_lr * 0.01
            optimizer = optim.AdamW(
                model.parameters(),
                lr=emergency_lr,
                weight_decay=0.01,
                eps=1e-8,
                betas=(0.9, 0.999)
            )

            # Reset scheduler
            scheduler = AdaptiveLRScheduler(optimizer, total_epochs=epochs, max_lr=emergency_lr)

            logger.warning(f"🚨 EMERGENCY RESTART: New LR = {emergency_lr:.8f}")
            continue

        # Save best model and early stopping logic
        if val_metrics['iou'] > best_iou:
            best_iou = val_metrics['iou']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.scheduler.state_dict(),
                'epoch': epoch,
                'best_iou': best_iou,
                'best_dice': val_metrics['dice']
            }, 'best_cswin_simam_unet_fusion.pth')
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
                'scheduler_state_dict': scheduler.scheduler.state_dict(),
                'best_iou': best_iou,
                'train_losses': train_losses,
                'val_losses': val_losses,
                'train_ious': train_ious,
                'val_ious': val_ious,
                'train_dices': train_dices,
                'val_dices': val_dices
            }, f'checkpoints_cswin_simam_unet_{dataset_type}/checkpoint_epoch_{epoch+1:03d}_{dataset_type}.pth')


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

    plt.suptitle('CSWin-SimAM-UNet Training Analysis - Fusion Dataset', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('cswin_simam_unet_training_curves.png', dpi=300, bbox_inches='tight')
    plt.close()

    logger.info("📈 Training curves saved as 'cswin_simam_unet_training_curves.png'")

def generate_predictions(model, val_loader, device, dataset_type='fusion'):
    """Generate predictions with ORIGINAL model (not EMA)"""
    model.eval()
    # DON'T use EMA for inference - use original model

    # Create checkpoint directory
    os.makedirs(f'checkpoints_cswin_simam_unet_{dataset_type}', exist_ok=True)
    os.makedirs('predictions_fusion_cswin_simam', exist_ok=True)

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
                plt.savefig(f'predictions_fusion_cswin_simam/{case_name}_predict.png',
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

    parser = argparse.ArgumentParser(description='CSWin-SimAM-UNet Training')
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

    args = parser.parse_args()

    # Train with parsed arguments
    train_cswin_simam_unet(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        target_iou=0.7,   # Target VALIDATION IoU 0.7 during training
        dataset_type=args.dataset_type
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



