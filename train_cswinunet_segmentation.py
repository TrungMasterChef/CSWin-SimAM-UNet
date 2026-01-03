import os
import numpy as np
import cv2
from glob import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, trunc_normal_
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
    def __init__(self, image_dir, mask_dir, image_size=(224, 224), augment=False):
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


# ==================== CSWinUNet Model ====================

class Mlp(nn.Module):
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


def img2windows(img, H_sp, W_sp):
    """
    img: B C H W
    """
    B, C, H, W = img.shape
    img_reshape = img.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
    img_perm = img_reshape.permute(0, 2, 4, 3, 5, 1).contiguous().reshape(-1, H_sp * W_sp, C)
    return img_perm


def windows2img(img_splits_hw, H_sp, W_sp, H, W):
    """
    img_splits_hw: B' H W C
    """
    B = int(img_splits_hw.shape[0] / (H * W / H_sp / W_sp))

    img = img_splits_hw.view(B, H // H_sp, W // W_sp, H_sp, W_sp, -1)
    img = img.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return img


class LePEAttention(nn.Module):
    def __init__(self, dim, resolution, idx, split_size, dim_out=None, num_heads=9, attn_drop=0., proj_drop=0.,
                 qk_scale=None):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out or dim
        self.resolution = resolution
        self.split_size = split_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if idx == -1:
            H_sp, W_sp = self.resolution, self.resolution
        elif idx == 0:
            H_sp, W_sp = self.resolution, self.split_size
        elif idx == 1:
            W_sp, H_sp = self.resolution, self.split_size
        else:
            print("ERROR MODE", idx)
            exit(0)
        self.H_sp = H_sp
        self.W_sp = W_sp
        stride = 1
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)

        self.attn_drop = nn.Dropout(attn_drop)

    def im2cswin(self, x):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = img2windows(x, self.H_sp, self.W_sp)
        x = x.reshape(-1, self.H_sp * self.W_sp, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3).contiguous()
        return x

    def get_lepe(self, x, func):
        B, N, C = x.shape
        H = W = int(np.sqrt(N))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        H_sp, W_sp = self.H_sp, self.W_sp
        x = x.view(B, C, H // H_sp, H_sp, W // W_sp, W_sp)
        x = x.permute(0, 2, 4, 1, 3, 5).contiguous().reshape(-1, C, H_sp, W_sp)  ### B', C, H', W'

        lepe = func(x)  ### B', C, H', W'
        lepe = lepe.reshape(-1, self.num_heads, C // self.num_heads, H_sp * W_sp).permute(0, 1, 3, 2).contiguous()

        x = x.reshape(-1, self.num_heads, C // self.num_heads, self.H_sp * self.W_sp).permute(0, 1, 3, 2).contiguous()
        return x, lepe

    def forward(self, qkv):
        """
        x: B L C
        """
        q, k, v = qkv[0], qkv[1], qkv[2]

        ### Img2Window
        H = W = self.resolution
        B, L, C = q.shape

        assert L == H * W, "flatten img_tokens has wrong size"

        q = self.im2cswin(q)
        k = self.im2cswin(k)
        v, lepe = self.get_lepe(v, self.get_v)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))  # B head N C @ B head C N --> B head N N
        attn = nn.functional.softmax(attn, dim=-1, dtype=attn.dtype)
        attn = self.attn_drop(attn)

        x = (attn @ v) + lepe
        x = x.transpose(1, 2).reshape(-1, self.H_sp * self.W_sp, C)  # B head N N @ B head N C

        ### Window2Img
        x = windows2img(x, self.H_sp, self.W_sp, H, W).view(B, -1, C)  # B H' W' C

        return x


class CSWinBlock(nn.Module):

    def __init__(self, dim, reso, num_heads,
                 split_size, mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 last_stage=False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.patches_resolution = reso
        self.split_size = split_size
        self.mlp_ratio = mlp_ratio
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.norm1 = norm_layer(dim)

        if self.patches_resolution == split_size:
            last_stage = True
        if last_stage:
            self.branch_num = 1
        else:
            self.branch_num = 2
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)

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
                    dim // 2, resolution=self.patches_resolution, idx=i,
                    split_size=split_size, num_heads=num_heads // 2, dim_out=dim // 2,
                    qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
                for i in range(self.branch_num)])

        mlp_hidden_dim = int(dim * mlp_ratio)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, out_features=dim, act_layer=act_layer,
                       drop=drop)
        self.norm2 = norm_layer(dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """

        H = W = self.patches_resolution
        B, L, C = x.shape
        assert L == H * W, "flatten img_tokens has wrong size"
        img = self.norm1(x)
        qkv = self.qkv(img).reshape(B, -1, 3, C).permute(2, 0, 1, 3)

        if self.branch_num == 2:
            x1 = self.attns[0](qkv[:, :, :, :C // 2])
            x2 = self.attns[1](qkv[:, :, :, C // 2:])
            attened_x = torch.cat([x1, x2], dim=2)
        else:
            attened_x = self.attns[0](qkv)
        attened_x = self.proj(attened_x)
        x = x + self.drop_path(attened_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class Merge_Block(nn.Module):
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


class CARAFE(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x


class CARAFE4(nn.Module):
    def __init__(self, dim, dim_out, kernel_size=3, up_factor=4):
        super().__init__()
        self.kernel_size = kernel_size
        self.up_factor = up_factor
        self.down = nn.Conv2d(dim, dim // 4, 1)
        self.encoder = nn.Conv2d(dim // 4, self.up_factor ** 2 * self.kernel_size ** 2,
                                 self.kernel_size, 1, self.kernel_size // 2)
        self.out = nn.Conv2d(dim, dim_out, 1)

    def forward(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        # N,C,H,W -> N,C,delta*H,delta*W
        # kernel prediction module
        kernel_tensor = self.down(x)  # (N, Cm, H, W)
        kernel_tensor = self.encoder(kernel_tensor)  # (N, S^2 * Kup^2, H, W)
        kernel_tensor = F.pixel_shuffle(kernel_tensor,
                                        self.up_factor)  # (N, S^2 * Kup^2, H, W)->(N, Kup^2, S*H, S*W)
        kernel_tensor = F.softmax(kernel_tensor, dim=1)  # (N, Kup^2, S*H, S*W)
        kernel_tensor = kernel_tensor.unfold(2, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W*S, S)
        kernel_tensor = kernel_tensor.unfold(3, self.up_factor, step=self.up_factor)  # (N, Kup^2, H, W, S, S)
        kernel_tensor = kernel_tensor.reshape(B, self.kernel_size ** 2, H, W,
                                              self.up_factor ** 2)  # (N, Kup^2, H, W, S^2)
        kernel_tensor = kernel_tensor.permute(0, 2, 3, 1, 4)  # (N, H, W, Kup^2, S^2)

        # content-aware reassembly module
        # tensor.unfold: dim, size, step
        w = F.pad(x, pad=(self.kernel_size // 2, self.kernel_size // 2,
                          self.kernel_size // 2, self.kernel_size // 2),
                  mode='constant', value=0)  # (N, C, H+Kup//2+Kup//2, W+Kup//2+Kup//2)
        w = w.unfold(2, self.kernel_size, step=1)  # (N, C, H, W+Kup//2+Kup//2, Kup)
        w = w.unfold(3, self.kernel_size, step=1)  # (N, C, H, W, Kup, Kup)
        w = w.reshape(B, C, H, W, -1)  # (N, C, H, W, Kup^2)
        w = w.permute(0, 2, 3, 1, 4)  # (N, H, W, C, Kup^2)

        x = torch.matmul(w, kernel_tensor)  # (N, H, W, C, S^2)
        x = x.reshape(B, H, W, -1)
        x = x.permute(0, 3, 1, 2)
        x = F.pixel_shuffle(x, self.up_factor)
        x = self.out(x)
        B, C = x.shape[:2]
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        return x


class CSWinTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1, embed_dim=64, depth=[1, 2, 9, 1],
                 split_size=[1, 2, 7, 7],
                 num_heads=[2, 4, 8, 16], mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, use_chk=False):
        super().__init__()
        self.use_chk = use_chk
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        heads = num_heads

        # encoder
        self.stage1_conv_embed = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, 7, 4, 2),
            Rearrange('b c h w -> b (h w) c', h=img_size // 4, w=img_size // 4),
            nn.LayerNorm(embed_dim)
        )

        curr_dim = embed_dim

        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, np.sum(depth))]  # stochastic depth decay rule

        self.stage1 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
                for i in range(depth[0])])
        self.merge1 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
        self.merge2 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        temp_stage3 = []
        temp_stage3.extend(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])])

        self.stage3 = nn.ModuleList(temp_stage3)
        self.merge3 = Merge_Block(curr_dim, curr_dim * 2)
        curr_dim = curr_dim * 2
        self.stage4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.norm = norm_layer(curr_dim)

        # decoder
        self.stage_up4 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[3], reso=img_size // 32, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[-1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:-1]) + i], norm_layer=norm_layer, last_stage=True)
                for i in range(depth[-1])])

        self.upsample4 = CARAFE(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear4 = nn.Linear(512, 256)
        self.stage_up3 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[2], reso=img_size // 16, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[2],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:2]) + i], norm_layer=norm_layer)
                for i in range(depth[2])]
        )

        self.upsample3 = CARAFE(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear3 = nn.Linear(256, 128)
        self.stage_up2 = nn.ModuleList(
            [CSWinBlock(
                dim=curr_dim, num_heads=heads[1], reso=img_size // 8, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[1],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[np.sum(depth[:1]) + i], norm_layer=norm_layer)
                for i in range(depth[1])])
        self.upsample2 = CARAFE(curr_dim, curr_dim // 2)
        curr_dim = curr_dim // 2

        self.concat_linear2 = nn.Linear(128, 64)
        self.stage_up1 = nn.ModuleList([
            CSWinBlock(
                dim=curr_dim, num_heads=heads[0], reso=img_size // 4, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale, split_size=split_size[0],
                drop=drop_rate, attn_drop=attn_drop_rate,
                drop_path=dpr[i], norm_layer=norm_layer)
            for i in range(depth[0])])

        self.upsample1 = CARAFE4(curr_dim, 64)
        self.norm_up = norm_layer(embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    # Encoder and Bottleneck
    def forward_features(self, x):
        x = self.stage1_conv_embed(x)

        x = self.pos_drop(x)

        for blk in self.stage1:
            x = blk(x)
        self.x1 = x
        x = self.merge1(x)

        for blk in self.stage2:
            x = blk(x)
        self.x2 = x
        x = self.merge2(x)

        for blk in self.stage3:
            x = blk(x)
        self.x3 = x
        x = self.merge3(x)

        for blk in self.stage4:
            x = blk(x)

        x = self.norm(x)

        return x

    # Decoder and Skip connection
    def forward_up_features(self, x):
        for blk in self.stage_up4:
            x = blk(x)
        x = self.upsample4(x)
        x = torch.cat([self.x3, x], -1)
        x = self.concat_linear4(x)
        for blk in self.stage_up3:
            x = blk(x)
        x = self.upsample3(x)
        x = torch.cat([self.x2, x], -1)
        x = self.concat_linear3(x)
        for blk in self.stage_up2:
            x = blk(x)
        x = self.upsample2(x)
        x = torch.cat([self.x1, x], -1)
        x = self.concat_linear2(x)
        for blk in self.stage_up1:
            x = blk(x)
        x = self.norm_up(x)  # B L C
        return x

    def up_x4(self, x):
        B, new_HW, C = x.shape
        H = W = int(np.sqrt(new_HW))
        x = self.upsample1(x)
        x = x.view(B, 4 * H, 4 * W, -1)
        x = x.permute(0, 3, 1, 2)  # B,C,H,W
        x = self.output(x)

        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_up_features(x)
        x = self.up_x4(x)
        return torch.sigmoid(x)


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
        # Giảm learning rate khi test loss không cải thiện
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
    IMAGE_SIZE = (448, 448)  # Kích thước ảnh đầu vào 448x448
    BATCH_SIZE = 2  # Giảm batch size vì 448x448 cần nhiều memory hơn
    NUM_EPOCHS = 100
    LEARNING_RATE = 0.0001  # Learning rate thấp hơn cho transformer
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
        augment=True  # Enable augmentation for training
    )
    
    # Dataset without augmentation for testing
    test_dataset_full = SegmentationDataset(
        IMAGE_DIR, MASK_DIR, 
        image_size=IMAGE_SIZE, 
        augment=False  # No augmentation for testing
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
        num_workers=4,  # Giảm num_workers để tiết kiệm bộ nhớ
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Khởi tạo CSWinTransformer model
    print("\nKhởi tạo CSWinTransformer model...")
    model = CSWinTransformer(
        img_size=IMAGE_SIZE[0],  # 448x448
        in_chans=3,
        num_classes=1,
        embed_dim=64,
        depth=[1, 2, 9, 1],
        split_size=[1, 2, 7, 7],
        num_heads=[2, 4, 8, 16],
        mlp_ratio=4.,
        drop_rate=0.3,
        attn_drop_rate=0.3,
        drop_path_rate=0.3
    ).to(device)
    
    # Loss function và optimizer
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(
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
    print("TRAINING CONFIGURATION - CSWinUNet")
    print("="*70)
    print(f"Tổng số epochs: {NUM_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"\n--- Data Augmentation (Training Only) ---")
    print(f"Horizontal/Vertical Flip: 50% probability")
    print(f"Random Rotation: 25% (90°, 180°, 270°)")
    print(f"Random Crop: 75-100% scale")
    print(f"\n--- Model Architecture ---")
    print(f"Model: CSWinTransformer (Cross-Shaped Window Transformer)")
    print(f"Embed dim: 64")
    print(f"Depth: [1, 2, 9, 1]")
    print(f"Num heads: [2, 4, 8, 16]")
    print(f"Split size: [1, 2, 7, 7]")
    print(f"\n--- Regularization ---")
    print(f"Weight decay (L2): {WEIGHT_DECAY}")
    print(f"Drop rate: 0.3")
    print(f"Attention drop rate: 0.3")
    print(f"Drop path rate: 0.3")
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
    torch.save(model.state_dict(), 'cswinunet_segmentation_model.pth')
    print("Model đã được lưu tại: cswinunet_segmentation_model.pth")
    
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
    plt.savefig('cswinunet_training_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()


def save_metrics_to_csv(history):
    """Lưu metrics ra file CSV cho cả train và test, kèm Learning Rate"""
    import csv
    
    with open('cswinunet_training_metrics.csv', 'w', newline='', encoding='utf-8') as f:
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
