#!/usr/bin/env python3

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import argparse
import logging
from tqdm import tqdm
import json
from datetime import datetime


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
                 split_sizes=[1, 2, 7, 7], num_heads=[2, 4, 8, 16], n_classes=None, n_channels=None):
        super(CSWinSimAMUNet, self).__init__()
        if n_classes is not None:
            num_classes = n_classes
        if n_channels is not None:
            input_channels = n_channels
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
        self.att_gate4 = AttentionGate(self.feature_dims[3], self.feature_dims[2], self.feature_dims[2]//2)
        self.att_gate3 = AttentionGate(self.feature_dims[2], self.feature_dims[1], self.feature_dims[1]//2)
        self.att_gate2 = AttentionGate(self.feature_dims[1], self.feature_dims[0], self.feature_dims[0]//2)

        self.decoder4 = self._make_decoder_stage(self.feature_dims[3], self.feature_dims[2], self.feature_dims[2])
        self.decoder3 = self._make_decoder_stage(self.feature_dims[2] + self.feature_dims[2]//2, self.feature_dims[1], self.feature_dims[1])
        self.decoder2 = self._make_decoder_stage(self.feature_dims[1] + self.feature_dims[1]//2, self.feature_dims[0], self.feature_dims[0])
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
        # Decoder 4
        d4 = F.interpolate(x4_2d, size=features[2].shape[-2:], mode='bilinear', align_corners=False)
        x3_att = self.att_gate4(d4, features[2])
        d4 = self.decoder4(d4)

        # Decoder 3
        d3 = F.interpolate(d4, size=features[1].shape[-2:], mode='bilinear', align_corners=False)
        x2_att = self.att_gate3(d3, features[1])
        d3 = self.decoder3(torch.cat([d3, x2_att], dim=1))

        # Decoder 2
        d2 = F.interpolate(d3, size=features[0].shape[-2:], mode='bilinear', align_corners=False)
        x1_att = self.att_gate2(d2, features[0])
        d2 = self.decoder2(torch.cat([d2, x1_att], dim=1))

        # Decoder 1 (no skip connection, just upsampling)
        d1 = F.interpolate(d2, scale_factor=4, mode='bilinear', align_corners=False)
        d1 = self.decoder1(d1)

        # Final output
        output = self.output_head(d1)
        output = F.interpolate(output, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return output



def _collect_samples(data_dir, dataset_type):
    if dataset_type != 'fusion':
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Only 'fusion' is supported.")

    img_dir = os.path.join(data_dir, '03-Fusion(50IRT) images')
    label_dir = os.path.join(data_dir, '04-Ground truth')

    samples = []
    for img_name in sorted(os.listdir(img_dir)):
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(img_dir, img_name)
            label_path = _find_label_path(label_dir, img_name)
            if label_path is not None:
                samples.append((img_path, label_path))
    if not samples:
        raise ValueError(f"No valid image/label pairs found in: {img_dir}")
    return samples


def _find_label_path(label_dir, img_name):
    base_name = os.path.splitext(img_name)[0]
    for ext in ('.png', '.jpg', '.jpeg'):
        candidate = os.path.join(label_dir, base_name + ext)
        if os.path.exists(candidate):
            return candidate
    return None


def _split_samples(samples, split_ratio=0.8, seed=42):
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(samples))
    split_idx = int(split_ratio * len(samples))
    train_samples = [samples[i] for i in indices[:split_idx]]
    val_samples = [samples[i] for i in indices[split_idx:]]
    return train_samples, val_samples


# Dataset class (same as U-Net)
class CrackDataset(Dataset):
    def __init__(self, data_dir, dataset_type='fusion', split='train', samples=None, split_ratio=0.8, seed=42):
        self.data_dir = data_dir
        self.dataset_type = dataset_type
        self.split = split

        if samples is not None:
            self.samples = samples
            return

        self.samples = _collect_samples(data_dir, dataset_type)

        if split in ('train', 'val'):
            train_samples, val_samples = _split_samples(self.samples, split_ratio=split_ratio, seed=seed)
            self.samples = train_samples if split == 'train' else val_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label_path = self.samples[idx]

        # Load image
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Failed to read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = np.transpose(image, (2, 0, 1))  # HWC to CHW

        # Load label
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
        if label is None:
            raise FileNotFoundError(f"Failed to read label: {label_path}")
        label = cv2.resize(label, (224, 224), interpolation=cv2.INTER_NEAREST)
        label = (label > 127).astype(np.float32)

        return torch.from_numpy(image), torch.from_numpy(label)


def calculate_metrics(pred_logits, target, threshold=0.5):
    """Calculate IoU, Dice, Precision, Recall"""
    pred_probs = torch.sigmoid(pred_logits)
    pred_binary = (pred_probs > threshold).float()

    # Flatten tensors
    pred_flat = pred_binary.view(pred_binary.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    # Calculate metrics
    intersection = (pred_flat * target_flat).sum(dim=1)
    union = pred_flat.sum(dim=1) + target_flat.sum(dim=1) - intersection

    iou = intersection / (union + 1e-8)
    dice = 2 * intersection / (pred_flat.sum(dim=1) + target_flat.sum(dim=1) + 1e-8)

    tp = intersection
    fp = pred_flat.sum(dim=1) - intersection
    fn = target_flat.sum(dim=1) - intersection

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)

    return {
        'iou': iou.mean().item(),
        'dice': dice.mean().item(),
        'precision': precision.mean().item(),
        'recall': recall.mean().item()
    }


def dice_loss_with_logits(logits, target, eps=1e-6):
    probs = torch.sigmoid(logits)
    probs = probs.view(probs.size(0), -1)
    target = target.view(target.size(0), -1)
    intersection = (probs * target).sum(dim=1)
    denominator = probs.sum(dim=1) + target.sum(dim=1)
    dice = (2 * intersection + eps) / (denominator + eps)
    return 1 - dice.mean()


def calculate_iou_sweep(pred_logits, target, thresholds):
    probs = torch.sigmoid(pred_logits)
    probs = probs.view(probs.size(0), -1)
    target_flat = target.view(target.size(0), -1)

    thresholds = torch.tensor(thresholds, device=probs.device, dtype=probs.dtype).view(-1, 1, 1)
    pred_binary = (probs.unsqueeze(0) > thresholds).float()
    target_expanded = target_flat.unsqueeze(0)

    intersection = (pred_binary * target_expanded).sum(dim=2)
    union = pred_binary.sum(dim=2) + target_expanded.sum(dim=2) - intersection
    iou = intersection / (union + 1e-8)
    return iou.mean(dim=1)


def train_CSWinSimAMUNet(data_dir, dataset_type='fusion', epochs=100, batch_size=8, lr=0.001, output_dir='CSWinSimAMUNet_output', dice_weight=1.0):
    """Train ResU-Net model"""

    # Setup logging
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(output_dir, 'training.log')),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Model
    if dataset_type != 'fusion':
        raise ValueError(f"Unsupported dataset_type: {dataset_type}. Only 'fusion' is supported.")
    model = CSWinSimAMUNet(n_channels=3, n_classes=1).to(device)

    # Dataset and DataLoader
    all_samples = _collect_samples(data_dir, dataset_type)
    train_samples, val_samples = _split_samples(all_samples, split_ratio=0.8, seed=42)
    train_dataset = CrackDataset(data_dir, dataset_type, 'train', samples=train_samples)
    val_dataset = CrackDataset(data_dir, dataset_type, 'val', samples=val_samples)

    train_set = set(train_samples)
    val_set = set(val_samples)
    overlap = train_set.intersection(val_set)
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} overlapping samples in train/val split.")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training history
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 'val_iou_best': [], 'val_iou_best_thresh': []}
    best_iou = 0.0
    iou_thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    logger.info(f"Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}

        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs} [Train]'):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            logits = outputs.squeeze(1)
            loss = criterion(logits, labels) + dice_weight * dice_loss_with_logits(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

            # Calculate metrics
            batch_metrics = calculate_metrics(logits, labels)
            for key in train_metrics:
                train_metrics[key] += batch_metrics[key]

        train_loss /= len(train_loader)
        for key in train_metrics:
            train_metrics[key] /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        val_metrics = {'iou': 0.0, 'dice': 0.0, 'precision': 0.0, 'recall': 0.0}
        val_iou_sweep = torch.zeros(len(iou_thresholds), device=device)

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch + 1}/{epochs} [Val]'):
                images, labels = images.to(device), labels.to(device)

                outputs = model(images)
                logits = outputs.squeeze(1)
                loss = criterion(logits, labels) + dice_weight * dice_loss_with_logits(logits, labels)
                val_loss += loss.item()

                # Calculate metrics
                batch_metrics = calculate_metrics(logits, labels)
                for key in val_metrics:
                    val_metrics[key] += batch_metrics[key]
                val_iou_sweep += calculate_iou_sweep(logits, labels, iou_thresholds)

        val_loss /= len(val_loader)
        for key in val_metrics:
            val_metrics[key] /= len(val_loader)
        val_iou_sweep = val_iou_sweep / len(val_loader)
        best_idx = int(torch.argmax(val_iou_sweep).item())
        best_iou_epoch = float(val_iou_sweep[best_idx].item())
        best_thresh_epoch = float(iou_thresholds[best_idx])

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_metrics['iou'])
        history['val_iou'].append(val_metrics['iou'])
        history['val_iou_best'].append(best_iou_epoch)
        history['val_iou_best_thresh'].append(best_thresh_epoch)

        # Log progress
        logger.info(f"Epoch {epoch + 1}: Train Loss={train_loss:.4f}, Train IoU={train_metrics['iou']:.4f}, "
                    f"Val Loss={val_loss:.4f}, Val IoU@0.5={val_metrics['iou']:.4f}, "
                    f"Val Best IoU={best_iou_epoch:.4f} (thr={best_thresh_epoch:.2f})")

        # Save best model
        if best_iou_epoch > best_iou:
            best_iou = best_iou_epoch
            torch.save(model.state_dict(), os.path.join(output_dir, f'best_CSWinSimAMUNet_{dataset_type}.pth'))
            logger.info(f"New best IoU: {best_iou:.4f}")

        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(),
                       os.path.join(output_dir, f'checkpoint_epoch_{epoch + 1:03d}_{dataset_type}.pth'))

        scheduler.step()

    # Save final results
    with open(os.path.join(output_dir, f'training_history_{dataset_type}.json'), 'w') as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training completed! Best IoU: {best_iou:.4f}")
    return best_iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CSWinSimAMUNet for crack segmentation')
    parser.add_argument('--data_dir', type=str, default='./data_ir', help='Data directory')
    parser.add_argument('--dataset_type', type=str, default='fusion', choices=['fusion'], help='Dataset type (fusion only)')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--dice_weight', type=float, default=1.0, help='Dice loss weight')
    parser.add_argument('--output_dir', type=str, default='resunet_output', help='Output directory')

    args = parser.parse_args()

    train_CSWinSimAMUNet(
        data_dir=args.data_dir,
        dataset_type=args.dataset_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        output_dir=args.output_dir,
        dice_weight=args.dice_weight
    )
