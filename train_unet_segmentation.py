import os
import numpy as np
import cv2
from glob import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")


# ==================== Data Augmentation ====================
class AugmentationTransform:
    """
    Data augmentation for image segmentation.
    Applies same transforms to both image and mask to maintain alignment.
    """
    def __init__(self, flip_prob=0.5, rotate_prob=0.25, crop_scale=(0.75, 1.0)):
        """
        Args:
            flip_prob: Probability of applying horizontal/vertical flip
            rotate_prob: Probability of applying rotation per angle
            crop_scale: Tuple (min, max) for random crop scale
        """
        self.flip_prob = flip_prob
        self.rotate_prob = rotate_prob
        self.crop_scale = crop_scale
    
    def __call__(self, image, mask):
        """
        Apply augmentation to image and mask.
        
        Args:
            image: numpy array (H, W, C) - RGB image
            mask: numpy array (H, W) - grayscale mask
            
        Returns:
            augmented_image, augmented_mask
        """
        # Random horizontal flip
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 1)  # 1 = horizontal
            mask = cv2.flip(mask, 1)
        
        # Random vertical flip
        if np.random.random() < self.flip_prob:
            image = cv2.flip(image, 0)  # 0 = vertical
            mask = cv2.flip(mask, 0)
        
        # Random rotation (0°, 90°, 180°, 270°)
        if np.random.random() < self.rotate_prob:
            angle = np.random.choice([0, 90, 180, 270])
            if angle == 90:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_CLOCKWISE)
            elif angle == 180:
                image = cv2.rotate(image, cv2.ROTATE_180)
                mask = cv2.rotate(mask, cv2.ROTATE_180)
            elif angle == 270:
                image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                mask = cv2.rotate(mask, cv2.ROTATE_90_COUNTERCLOCKWISE)
        
        # Random crop and resize
        h, w = image.shape[:2]
        crop_scale = np.random.uniform(self.crop_scale[0], self.crop_scale[1])
        new_h, new_w = int(h * crop_scale), int(w * crop_scale)
        
        # Random crop position
        top = np.random.randint(0, h - new_h + 1) if h > new_h else 0
        left = np.random.randint(0, w - new_w + 1) if w > new_w else 0
        
        # Crop
        image = image[top:top+new_h, left:left+new_w]
        mask = mask[top:top+new_h, left:left+new_w]
        
        # Resize back to original size
        image = cv2.resize(image, (w, h))
        mask = cv2.resize(mask, (w, h))
        
        return image, mask


# ==================== Data Loader ====================
class SegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_size=(448, 448), augment=False):
        """
        Dataset cho image segmentation
        
        Args:
            image_dir: Đường dẫn thư mục chứa ảnh
            mask_dir: Đường dẫn thư mục chứa mask
            image_size: Kích thước ảnh đầu ra (height, width)
            augment: Có sử dụng data augmentation không (chỉ cho training)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_size = image_size
        self.augment = augment
        
        # Khởi tạo augmentation transform nếu cần
        if self.augment:
            self.transform = AugmentationTransform(
                flip_prob=0.5,
                rotate_prob=0.25,
                crop_scale=(0.75, 1.0)
            )
            print("Data augmentation enabled: Flip, Rotate, Random Crop")
        else:
            self.transform = None
        
        # Lấy danh sách tất cả file ảnh
        self.image_paths = sorted(glob(os.path.join(image_dir, "*.jpg")))
        
        # Kiểm tra xem có ảnh không
        if len(self.image_paths) == 0:
            raise ValueError(f"Không tìm thấy ảnh trong thư mục: {image_dir}")
        
        print(f"Tìm thấy {len(self.image_paths)} ảnh trong dataset")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        # Đọc ảnh - Sử dụng np.fromfile để hỗ trợ Unicode path
        img_path = self.image_paths[idx]
        
        # Đọc file bằng numpy để xử lý Unicode path
        image_array = np.fromfile(img_path, dtype=np.uint8)
        image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError(f"Không thể đọc ảnh: {img_path}")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Lấy tên file để tìm mask tương ứng
        img_name = os.path.basename(img_path)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Đọc mask
        if os.path.exists(mask_path):
            mask_array = np.fromfile(mask_path, dtype=np.uint8)
            mask = cv2.imdecode(mask_array, cv2.IMREAD_GRAYSCALE)
            
            if mask is None:
                print(f"Cảnh báo: Không thể đọc mask cho {img_name}, tạo mask rỗng")
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        else:
            print(f"Cảnh báo: Không tìm thấy mask cho {img_name}")
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Resize về kích thước mong muốn
        image = cv2.resize(image, self.image_size)
        mask = cv2.resize(mask, self.image_size)
        
        # Apply augmentation TRƯỚC khi normalize (if enabled)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        
        # Chuẩn hóa ảnh về [0, 1]
        image = image.astype(np.float32) / 255.0
        mask = mask.astype(np.float32) / 255.0
        
        # Chuyển sang tensor: (H, W, C) -> (C, H, W)
        image = torch.from_numpy(image).permute(2, 0, 1)
        mask = torch.from_numpy(mask).unsqueeze(0)  # Thêm channel dimension
        
        return image, mask


# ==================== UNet Model ====================

class DoubleConv(nn.Module):
    """(Conv2D -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Concatenate
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return self.sigmoid(logits)


# ==================== Metrics ====================
def dice_coefficient(pred, target, smooth=1e-6):
    """Tính Dice Coefficient"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    dice = (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
    return dice.item()


def iou_score(pred, target, smooth=1e-6):
    """Tính IoU (Intersection over Union)"""
    pred = pred.view(-1)
    target = target.view(-1)
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + smooth) / (union + smooth)
    return iou.item()


# ==================== Evaluation Function ====================
def evaluate_model(model, data_loader, criterion, device):
    """
    Đánh giá model trên tập dữ liệu
    """
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    
    with torch.no_grad():
        for images, masks in data_loader:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Threshold predictions
            preds = (outputs > 0.5).float()
            
            # Tính metrics
            batch_dice = dice_coefficient(preds, masks)
            batch_iou = iou_score(preds, masks)
            
            # Cập nhật metrics
            total_loss += loss.item()
            total_dice += batch_dice
            total_iou += batch_iou
    
    # Tính trung bình
    avg_loss = total_loss / len(data_loader)
    avg_dice = total_dice / len(data_loader)
    avg_iou = total_iou / len(data_loader)
    
    return avg_loss, avg_dice, avg_iou


# ==================== Training Function ====================
def train_model(model, train_loader, test_loader, criterion, optimizer, scheduler, device, num_epochs=100):
    """
    Training loop với metrics tracking cho cả train và test, kèm Learning Rate Scheduler
    """
    history = {
        'train_loss': [],
        'train_dice': [],
        'train_iou': [],
        'test_loss': [],
        'test_dice': [],
        'test_iou': [],
        'learning_rates': []
    }
    
    for epoch in range(num_epochs):
        # ========== TRAINING ==========
        model.train()
        epoch_loss = 0.0
        epoch_dice = 0.0
        epoch_iou = 0.0
        
        # Progress bar
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [TRAIN]')
        
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Tính metrics
            with torch.no_grad():
                # Threshold predictions
                preds = (outputs > 0.5).float()
                
                batch_dice = dice_coefficient(preds, masks)
                batch_iou = iou_score(preds, masks)
            
            # Cập nhật metrics
            epoch_loss += loss.item()
            epoch_dice += batch_dice
            epoch_iou += batch_iou
            
            # Cập nhật progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Dice': f'{batch_dice:.4f}',
                'IoU': f'{batch_iou:.4f}'
            })
        
        # Tính trung bình metrics cho train
        train_loss = epoch_loss / len(train_loader)
        train_dice = epoch_dice / len(train_loader)
        train_iou = epoch_iou / len(train_loader)
        
        # ========== TESTING ==========
        test_loss, test_dice, test_iou = evaluate_model(model, test_loader, criterion, device)
        
        # ========== LEARNING RATE SCHEDULER ==========
        if scheduler is not None:
            scheduler.step(test_loss)
            current_lr = optimizer.param_groups[0]['lr']
        else:
            current_lr = optimizer.param_groups[0]['lr']
        
        # Lưu vào history
        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['train_iou'].append(train_iou)
        history['test_loss'].append(test_loss)
        history['test_dice'].append(test_dice)
        history['test_iou'].append(test_iou)
        history['learning_rates'].append(current_lr)
        
        # In kết quả epoch
        print(f'\n{"="*70}')
        print(f'Epoch {epoch+1}/{num_epochs}:')
        print(f'  [TRAIN] Loss: {train_loss:.4f} | Dice: {train_dice:.4f} | IoU: {train_iou:.4f}')
        print(f'  [TEST]  Loss: {test_loss:.4f} | Dice: {test_dice:.4f} | IoU: {test_iou:.4f}')
        print(f'  [LR]    Learning Rate: {current_lr:.6f}')
        print(f'{"="*70}\n')
    
    return history


# ==================== Main Training Script ====================
def main():
    # Đường dẫn dữ liệu
    IMAGE_DIR = r"D:\Thực tập 2025\Code_model_5\image_volker"
    MASK_DIR = r"D:\Thực tập 2025\Code_model_5\mask_volker"
    
    # Hyperparameters
    IMAGE_SIZE = (448, 448)
    BATCH_SIZE = 4  # UNet lighter than CSWinUNet
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.001  # Higher LR for CNN
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
    
    # Regularization parameters
    WEIGHT_DECAY = 1e-4
    LR_SCHEDULER_FACTOR = 0.5
    LR_SCHEDULER_PATIENCE = 5
    LR_SCHEDULER_MIN_LR = 1e-7
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Sử dụng device: {device}')
    
    # Tạo datasets - Separate for train (with augment) and test (no augment)
    print("\nĐang load dữ liệu...")
    
    # Dataset with augmentation for training
    train_dataset_full = SegmentationDataset(
        IMAGE_DIR, MASK_DIR, 
        image_size=IMAGE_SIZE, 
        augment=True
    )
    
    # Dataset without augmentation for testing
    test_dataset_full = SegmentationDataset(
        IMAGE_DIR, MASK_DIR, 
        image_size=IMAGE_SIZE, 
        augment=False
    )
    
    # Chia train/test split
    dataset_size = len(train_dataset_full)
    indices = list(range(dataset_size))
    train_indices, test_indices = train_test_split(
        indices, 
        test_size=TEST_SPLIT, 
        random_state=RANDOM_SEED
    )
    
    print(f"Tổng số mẫu: {dataset_size}")
    print(f"Tập train: {len(train_indices)} mẫu (with augmentation)")
    print(f"Tập test: {len(test_indices)} mẫu (no augmentation)")
    
    # Tạo subset cho train và test
    train_dataset = torch.utils.data.Subset(train_dataset_full, train_indices)
    test_dataset = torch.utils.data.Subset(test_dataset_full, test_indices)
    
    # Tạo dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Khởi tạo UNet model
    print("\nKhởi tạo UNet model...")
    model = UNet(n_channels=3, n_classes=1).to(device)
    
    # Loss function và optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        model.parameters(), 
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning Rate Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=LR_SCHEDULER_FACTOR,
        patience=LR_SCHEDULER_PATIENCE,
        min_lr=LR_SCHEDULER_MIN_LR,
        verbose=True
    )
    
    print("="*70)
    print("TRAINING CONFIGURATION - UNet")
    print("="*70)
    print(f"Tổng số epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"\n--- Data Augmentation (Training Only) ---")
    print(f"Horizontal/Vertical Flip: 50% probability")
    print(f"Random Rotation: 25% (90°, 180°, 270°)")
    print(f"Random Crop: 75-100% scale")
    print(f"\n--- Model Architecture ---")
    print(f"Model: UNet (Standard CNN-based)")
    print(f"Encoder: [64, 128, 256, 512, 1024]")
    print(f"Decoder: [512, 256, 128, 64]")
    print(f"\n--- Regularization ---")
    print(f"Weight decay (L2): {WEIGHT_DECAY}")
    print(f"Batch Normalization: Yes")
    print(f"Initial learning rate: {LEARNING_RATE}")
    print(f"LR scheduler factor: {LR_SCHEDULER_FACTOR}")
    print(f"LR scheduler patience: {LR_SCHEDULER_PATIENCE} epochs")
    print("="*70 + "\n")
    
    history = train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS
    )
    
    # Lưu model
    print("\nLưu model...")
    torch.save(model.state_dict(), 'unet_segmentation_model.pth')
    print("Model đã được lưu tại: unet_segmentation_model.pth")
    
    # Vẽ biểu đồ metrics
    print("\nVẽ biểu đồ metrics...")
    plot_metrics(history)
    print("Biểu đồ đã được lưu!")
    
    # Xuất metrics ra file CSV
    save_metrics_to_csv(history)
    print("Metrics đã được lưu ra file CSV!")


def plot_metrics(history):
    """Vẽ biểu đồ metrics cho cả train và test, kèm Learning Rate"""
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', linewidth=2, label='Train')
    axes[0].plot(epochs, history['test_loss'], 'r-', linewidth=2, label='Test')
    axes[0].set_title('Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Dice Coefficient
    axes[1].plot(epochs, history['train_dice'], 'b-', linewidth=2, label='Train')
    axes[1].plot(epochs, history['test_dice'], 'r-', linewidth=2, label='Test')
    axes[1].set_title('Dice Coefficient', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Dice')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # IoU
    axes[2].plot(epochs, history['train_iou'], 'b-', linewidth=2, label='Train')
    axes[2].plot(epochs, history['test_iou'], 'r-', linewidth=2, label='Test')
    axes[2].set_title('IoU Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('IoU')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # Learning Rate
    axes[3].plot(epochs, history['learning_rates'], 'g-', linewidth=2)
    axes[3].set_title('Learning Rate', fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('Learning Rate')
    axes[3].set_yscale('log')
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('unet_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_to_csv(history):
    """Lưu metrics ra file CSV cho cả train và test, kèm Learning Rate"""
    import csv
    
    with open('unet_training_metrics.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Epoch', 'Train_Loss', 'Train_Dice', 'Train_IoU', 
                        'Test_Loss', 'Test_Dice', 'Test_IoU', 'Learning_Rate'])
        
        for i in range(len(history['train_loss'])):
            writer.writerow([
                i + 1,
                f"{history['train_loss'][i]:.6f}",
                f"{history['train_dice'][i]:.6f}",
                f"{history['train_iou'][i]:.6f}",
                f"{history['test_loss'][i]:.6f}",
                f"{history['test_dice'][i]:.6f}",
                f"{history['test_iou'][i]:.6f}",
                f"{history['learning_rates'][i]:.8f}"
            ])


if __name__ == "__main__":
    main()
