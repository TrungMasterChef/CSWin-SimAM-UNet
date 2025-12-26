# Crack Segmentation Training Suite

Unified research framework for training and evaluating three complementary architectures on the IR/RGB crack datasets:

- **UNet** (lightweight CNN encoder/decoder)
- **ResUNet** (residual encoder with UNet decoder)
- **CSWin-SimAM-UNet** (hybrid CSWin Transformer + SimAM attention + UNet decoder)

All trainers share the same stratified data pipeline, logging format, and metric reporting so experiments remain comparable.

## Environment

- Python 3.9+
- PyTorch 2.x, torchvision, numpy, Pillow, OpenCV, matplotlib, tqdm (install manually or via `pip install -r requirements.txt`).
- CUDA-capable GPU strongly recommended; scripts automatically fall back to CPU.

## Dataset Layout

Each `data_dir` must mirror the original dataset folders:

```
data_ir/
├── 01-Visible images/            # RGB images (3 channels)
├── 03-Fusion(50IRT) images/      # Fusion/IR images (1 channel)
└── 04-Ground truth/              # Binary crack masks (.png/.jpg)
```

`FusionDataset` handles augmentation, normalization, and a foreground-aware stratified split with sample-shape logging for both train/validation subsets.

## Training UNet (Lightweight CNN)

```
python train_unet_fusion.py \
  --dataset_type fusion \            # or rgb
  --data_dir ./data_ir \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 1e-4 \
  --base_channels 32 \               # controls width
  --num_stages 3 \                   # 3=compact, 4=classic UNet depth
  --block_depth 2 \                  # convs per DoubleConv block (1 or 2)
  --disable_skip                     # optional flag to drop skip connections
```

Highlights:

- Aggressive GPU optimizations, NaN/Inf safety checks, gradient accumulation.
+- Binary loss blend (BCE + Dice/IoU) with adaptive weighting.
- Stratified split logging + dataset shape inspection before training.
- Logs each epoch as `Epoch N: Train Loss=… Train IoU=… Train Dice=… Val Loss=… Val IoU=… Val Dice=… IoU Gap=… LR=…`.
- Checkpoints: `best_unet_fusion.pth` and periodic `checkpoints_unet_<dataset_type>/…`.

## Training ResUNet (Residual CNN)

```
python train_resUNet_fusion.py \
  --dataset_type fusion \            # accepts fusion, rgb, ir (mapped to fusion)
  --data_dir ./data_ir \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --base_channels 64 \
  --num_stages 4 \                   # 3 or 4 residual stages
  --val_ratio 0.2 \
  --fg_threshold 0.05
```

Highlights:

- Reuses `FusionDataset` stratified pipeline + detailed dataset introspection.
- Residual encoder/decoder with configurable width/depth (`base_channels`, `num_stages`).
- Tracks `train/val` IoU & Dice each epoch and stores histories in `resunet_output/training_history_<dataset>.json`.
- Best checkpoint at `resunet_output/best_resunet_<dataset>.pth`, snapshots every 10 epochs.

## Training CSWin-SimAM-UNet (Hybrid Transformer)

```
python train_cswin_simam_unet_rgb.py \
  --dataset_type fusion \            # fusion (IR) or rgb
  --data_dir ./data_ir \
  --epochs 100 \
  --batch_size 8 \
  --learning_rate 1e-4
```

Architecture highlights:

- **CSWin Transformer encoder** with genuine cross-shaped window self-attention (horizontal/vertical stripe heads) in every stage.
- **SimAM refinement** blocks applied to transformer outputs and skip tensors for lightweight attention without extra parameters.
- **UNet-style decoder** with attention gates and multi-scale skips for precise crack delineation.
- **Hierarchical pyramid** of patch embeddings + depth-wise feed-forward layers mirroring the UNet decoder resolutions.

Training pipeline extras:

- Shares the same `FusionDataset` stratified sampling/augmentation as the CNN trainers.
- Gradient accumulation, extensive stability guards (NaN detection, emergency re-init), and automatic foreground ratio tracking.
- Custom `AdaptiveLRScheduler` (cosine annealing + IoU-based restarts) plus on-line hyperparameter suggestions (batch size, LR, weight decay).
- Best checkpoint saved as `best_cswin_simam_unet_fusion.pth` (per dataset type) with rolling snapshots in `checkpoints_cswin_<dataset_type>/`.
- Built-in visualization utility saves qualitative predictions to `predictions_fusion_cswin_simam/` with IoU/Dice overlays.

> **Tip:** The CSWin encoder is memory hungry. Reduce `--batch_size`, image size, or window count inside the script if training on <16 GB GPUs.

## Inference / Prediction

`load_prediction.py` runs batched inference over `03-Fusion(50IRT) images` and writes `{image}_input.png`, `{image}_mask.png`, `{image}_overlay.png` to `prediction_all/`.

```
python load_prediction.py
```

Behavior:

- Defaults to `Result/ResUNet/best_resunet_fusion.pth` but you can switch `MODEL_PATH` and `MODEL_ARCH` (`"resunet"`/`"unet"`) to load other checkpoints.
- Reconstructs model hyperparameters (channels, stages, skip usage) directly from the state dict, so lightweight variants load without shape mismatches.
- Uses the same preprocessing as training (Resize→ToTensor) to keep predictions consistent.

## Logging & Monitoring

- All training scripts stream to stdout **and** append to `training_log.text` (rewritten each run). Tail this file during long trainings for progress updates.
- Metric histories (`*.json`) can be plotted with pandas/matplotlib for experiment tracking.

## Tips

1. **Match parameter counts:** Adjust `base_channels`, `num_stages`, and (for UNet) `block_depth`/`--disable_skip` to keep experiments fair across architectures.
2. **Balance datasets:** The stratified splitter groups samples by crack coverage; tweak `fg_threshold` if your dataset has different foreground ratios.
3. **Custom inference:** Point `IMAGE_DIR` (and transforms) in `load_prediction.py` to any folder of images to batch-process new data.
4. **Reproducibility:** Random seeds are set inside the dataset split utilities; add `torch.manual_seed(...)` if you need deterministic augmentation as well.
5. **Resource constraints:** For CSWin models, lower the image size or enable gradient checkpointing (already supported) when running on memory-limited GPUs.

Happy experimenting!
