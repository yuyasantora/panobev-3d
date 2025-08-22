# train4.py - ViT+VGGTハイブリッドモデル版
import os
import yaml
import shutil
import random
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import ViTModel
import SimpleITK as sitk
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR  # 変更
from tqdm import tqdm
import argparse

# ... set_seed関数とDatasetクラスは train3.py と同じ ...
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===================================================================
# Dataset (dense lung BEV targets)
# ===================================================================
class PanoBEVDataset3(Dataset):
    def __init__(self, dataset_dir: str, resize_shape: tuple, augmentation_config: dict, bev_cfg: dict):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        self.depth_dir  = os.path.join(dataset_dir, 'depths')
        self.resize_shape = resize_shape
        self.use_augmentation = augmentation_config.get('use_augmentation', False)

        self.bev_mode = bev_cfg.get('bev_target_mode', 'binary')  # 肺野は密なのでbinaryでもOK
        self.bev_gauss_sigma = float(bev_cfg.get('bev_gaussian_sigma', 2.0))
        self.normalize_input = bool(bev_cfg.get('normalize_input', True))

        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])
        self._cache = {}
        self.max_cache_size = 100  # 適切なサイズに調整

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
            
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        base_filename = image_filename.replace('.npy', '')
        patient_id = base_filename.split('_')[0]
        bev_filename = f"{patient_id}_bev.npy"
        bev_path = os.path.join(self.target_dir, bev_filename)
        depth_filename = f"{base_filename}_depth.npy"
        depth_path = os.path.join(self.depth_dir, depth_filename)

        image = np.load(image_path)
        bev_target_np = np.load(bev_path)      # 2D lung BEV
        depth_target = np.load(depth_path)     # 2D

        # to SITK
        image_sitk = sitk.GetImageFromArray(image)
        bev_sitk   = sitk.GetImageFromArray(bev_target_np)
        depth_sitk = sitk.GetImageFromArray(depth_target)

        def resize_image(sitk_image, interpolator):
            ref_image = sitk.Image(self.resize_shape, sitk_image.GetPixelIDValue())
            old_size = sitk_image.GetSize()
            old_spacing = sitk_image.GetSpacing()
            new_spacing = [old_sp * (old_sz / new_sz) for old_sp, old_sz, new_sz in zip(old_spacing, old_size, self.resize_shape)]
            ref_image.SetSpacing(new_spacing)
            ref_image.SetOrigin(sitk_image.GetOrigin())
            ref_image.SetDirection(sitk_image.GetDirection())
            return sitk.Resample(sitk_image, ref_image, sitk.Transform(), interpolator, sitk_image.GetPixelIDValue())

        resized_image = sitk.GetArrayFromImage(resize_image(image_sitk, sitk.sitkLinear))
        resized_bev_bin = sitk.GetArrayFromImage(resize_image(bev_sitk, sitk.sitkNearestNeighbor))
        resized_depth = sitk.GetArrayFromImage(resize_image(depth_sitk, sitk.sitkLinear))

        # Binary BEV (lung mask)
        bev_binary = (resized_bev_bin > 0).astype(np.float32)

        # Continuous BEV target (for dense targets, simpler processing)
        if self.bev_mode == 'binary':
            bev_cont = bev_binary
        elif self.bev_mode == 'gaussian':
            # Light gaussian for smoothing
            bev_img = sitk.GetImageFromArray(bev_binary.astype(np.float32))
            gauss = sitk.DiscreteGaussian(bev_img, variance=float(self.bev_gauss_sigma**2))
            bev_cont = sitk.GetArrayFromImage(gauss).astype(np.float32)
            m = bev_cont.max()
            if m > 1e-6:
                bev_cont = bev_cont / m
        else:
            bev_cont = bev_binary  # fallback

        # Depth normalization [0,1]
        if np.max(resized_depth) > 1e-6:
            resized_depth = resized_depth / np.max(resized_depth)

        # Input normalization [-1,1]
        img = resized_image.astype(np.float32)
        m, M = img.min(), img.max()
        if M > m:
            img = (img - m) / (M - m)
        else:
            img = np.zeros_like(img, dtype=np.float32)
        if self.normalize_input:
            img = (img - 0.5) / 0.5

        image_tensor = torch.from_numpy(img).float().unsqueeze(0)         # [1,H,W]
        bev_cont_tensor = torch.from_numpy(bev_cont).float().unsqueeze(0) # [1,H,W]
        bev_bin_tensor  = torch.from_numpy(bev_binary).float().unsqueeze(0)
        depth_tensor = torch.from_numpy(resized_depth).float().unsqueeze(0)

        result = image_tensor, bev_cont_tensor, depth_tensor, bev_bin_tensor
        if len(self._cache) < self.max_cache_size:
            self._cache[idx] = result
            
        return result


# ===================================================================
# VGGT Backbone
# ===================================================================
class VGGTBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # プーリングを分離
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.token_mixer = nn.Conv2d(out_channels, out_channels, 1)
        self.channel_mixer = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * 2, 1),
            nn.GELU(),
            nn.Conv2d(out_channels * 2, out_channels, 1)
        )
        
    def forward(self, x):
        # 空間特徴抽出
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        
        # トークンミキシング
        identity = x
        x = self.token_mixer(x)
        x = x + identity
        
        # チャネルミキシング
        identity = x
        x = self.channel_mixer(x)
        x = x + identity
        
        # プーリング
        x = self.pool(x)
        
        return x

# 1. VGGTバックボーンの構造を明確に
def create_vggt_backbone():
    return nn.ModuleList([
        VGGTBlock(3, 64),      # 224 -> 112, channels: 3 -> 64
        VGGTBlock(64, 128),    # 112 -> 56,  channels: 64 -> 128
        VGGTBlock(128, 256),   # 56 -> 28,   channels: 128 -> 256
        VGGTBlock(256, 512)    # 28 -> 14,   channels: 256 -> 512
    ])

# ===================================================================
# Hybrid Model (ViT + VGGT)
# ===================================================================
class HybridViTPanoBEV(nn.Module):
    def __init__(self, vit_model_name, output_size=224):
        super().__init__()
        # ViTパス
        self.vit = ViTModel.from_pretrained(vit_model_name, output_hidden_states=True)
        self.vit_hidden_size = self.vit.config.hidden_size
        
        # VGGTパス
        self.vggt = create_vggt_backbone()
        self.vggt_pools = nn.ModuleList([
            nn.MaxPool2d(2) for _ in range(4)
        ])
        
        # 特徴変換（ViT）
        self.vit_convs = nn.ModuleList([
            nn.Conv2d(self.vit_hidden_size, 256, 1),  # 細部
            nn.Conv2d(self.vit_hidden_size, 256, 1),  # 中間
            nn.Conv2d(self.vit_hidden_size, 256, 1)   # 大域
        ])
        
        # 特徴変換（VGGT）- チャネル数を修正
        self.vggt_convs = nn.ModuleList([
            nn.Conv2d(64, 256, 1),    # 第1層 (112x112, 64ch -> 256ch)
            nn.Conv2d(128, 256, 1),   # 第2層 (56x56, 128ch -> 256ch)
            nn.Conv2d(256, 256, 1)    # 第3層 (28x28, 256ch -> 256ch)
        ])
        
        # 強化された特徴融合
        self.fusion = nn.Sequential(
            nn.Conv2d(256*6, 512, 3, padding=1),  # ViT(3) + VGGT(3) = 6
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1)
        )
        
        # デコーダー（train3.pyと同じ）
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        
        # 出力ヘッド
        self.bev_head = nn.Conv2d(32, 1, 1)
        # 深度ヘッドを修正
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, dilation=2, padding=2),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, dilation=4, padding=4),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()
        )

        # 出力層を分離
        self.depth_output = nn.Sequential(
            nn.Conv2d(32, 1, 1),
            nn.Softplus()
        )

        # 形状一貫性モジュールのチャネル数を修正
        self.shape_consistency = nn.Sequential(
            nn.Conv2d(33, 64, 3, padding=1),  # 32 + 1 = 33チャネル
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 1, 1)
        )
        
        self.output_size = output_size
        
    def forward(self, x):
        B = x.shape[0]
        
        # ViTパス
        if x.shape[1] == 1:
            x_vit = x.repeat(1, 3, 1, 1)
        else:
            x_vit = x
            
        vit_outputs = self.vit(x_vit)
        vit_features = []
        
        # ViT特徴の抽出（3スケール）
        for i, layer_idx in enumerate([4, 8, 12]):
            h = vit_outputs.hidden_states[layer_idx]
            h = h[:, 1:].transpose(1, 2).reshape(B, self.vit_hidden_size, 14, 14)
            h = self.vit_convs[i](h)
            if i > 0:
                h = F.interpolate(h, size=(14, 14), mode='bilinear', align_corners=False)
            vit_features.append(h)
            
        # VGGTパス
        vggt_features = []
        x_vggt = x_vit  # 入力共有
        
        # 特徴抽出（最初の3層のみ使用）
        for i, block in enumerate(self.vggt):
            if i >= 3:  # 最後の層は使用しない
                break
            x_vggt = block(x_vggt)
            vggt_features.append(x_vggt)
            
        # 特徴変換（3スケール）- サイズを明示的に制御
        vggt_processed = []
        for i, feat in enumerate(vggt_features):
            # 現在のサイズを取得
            _, _, H, W = feat.shape
            
            # 特徴変換
            h = self.vggt_convs[i](feat)
            
            # 全ての特徴を14x14にリサイズ
            if H != 14 or W != 14:
                h = F.interpolate(h, size=(14, 14), mode='bilinear', align_corners=False)
            
            vggt_processed.append(h)
            
        
        # 特徴融合（ViT + VGGT）
        fused = self.fusion(torch.cat(vit_features + vggt_processed, dim=1))
        
        # デコード
        decoded = self.decoder(fused)
        
        # BEVとDepthの処理を修正
        bev = self.bev_head(decoded)
        depth_features = self.depth_head(decoded)
        
        # 特徴の結合を修正
        combined = torch.cat([depth_features, bev], dim=1)  # [B, 33, H, W]
        shape_guidance = self.shape_consistency(combined)
        
        # 最終的な深度出力
        depth = self.depth_output(depth_features + shape_guidance)
        
        # サイズ調整
        bev = F.interpolate(bev, size=(self.output_size, self.output_size),
                          mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=(self.output_size, self.output_size),
                            mode='bilinear', align_corners=False)
        
        return torch.sigmoid(bev), depth  # depthはすでにSoftplusを通過

# 新しい損失関数
def depth_shape_loss(bev_pred, depth_pred, bev_target, depth_target):
    """3D形状を考慮した深度損失"""
    # 基本的なMSE損失
    mse_loss = F.mse_loss(depth_pred, depth_target)
    
    # 勾配の連続性
    depth_dx = depth_pred[:, :, :, 1:] - depth_pred[:, :, :, :-1]
    depth_dy = depth_pred[:, :, 1:, :] - depth_pred[:, :, :-1, :]
    target_dx = depth_target[:, :, :, 1:] - depth_target[:, :, :, :-1]
    target_dy = depth_target[:, :, 1:, :] - depth_target[:, :, :-1, :]
    
    gradient_loss = F.l1_loss(depth_dx, target_dx) + F.l1_loss(depth_dy, target_dy)
    
    # BEVマスク領域での深度の一貫性
    bev_mask = (bev_pred > 0.5).float()
    consistency_loss = F.l1_loss(depth_pred * bev_mask, depth_target * bev_mask)
    
    return mse_loss + 0.1 * gradient_loss + 0.5 * consistency_loss

# ... Loss関数は train3.py と同じ（ただしgrad_lossのバグ修正） ...
def grad_loss(pred, tgt):
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_t = tgt[..., :, 1:] - tgt[..., :, :-1]
    dy_t = tgt[..., 1:, :] - tgt[..., :-1, :]  # 修正

    dx = torch.abs(dx_p - dx_t).mean()
    dy = torch.abs(dy_p - dy_t).mean()
    return (dx + dy) * 0.5

def soft_dice_loss(pred, tgt_bin, eps=1e-6):
    p = pred.clamp(0,1)
    t = (tgt_bin > 0.5).float()
    inter = (p * t).sum()
    union = p.sum() + t.sum()
    return 1.0 - (2*inter + eps) / (union + eps)

def calculate_rmse(predicted, target):
    return torch.sqrt(F.mse_loss(predicted, target)).item()

def thresh_dice(pred, tgt_bin, th=0.3, eps=1e-6):
    pm = (pred > th).float()
    tm = (tgt_bin > 0.5).float()
    inter = (pm * tm).sum()
    union = pm.sum() + tm.sum()
    return ((2*inter + eps) / (union + eps)).item()

def physics_consistency_loss(bev_pred, depth_pred, eps=1e-6):
    """物理的妥当性を強制する損失（安全版）"""
    try:
        # 1. 体積整合性（より安全に）
        bev_mask = (bev_pred > 0.3).float()
        bev_area = bev_mask.sum(dim=(2,3)) + eps  # ゼロ除算回避
        
        # depth_predの値も制限
        depth_clamped = torch.clamp(depth_pred, 0.0, 1.0)
        depth_avg = (depth_clamped * bev_mask).sum(dim=(2,3)) / bev_area
        
        estimated_volume = bev_area * depth_avg
        target_volume = torch.tensor(0.5, device=bev_pred.device)
        
        # NaN/Inf チェック
        if torch.isnan(estimated_volume).any() or torch.isinf(estimated_volume).any():
            return torch.tensor(0.0, device=bev_pred.device)
        
        volume_loss = F.mse_loss(estimated_volume.mean(), target_volume)
        
        # 2. 形状整合性（より単純に）
        high_bev_mask = (bev_pred > 0.5).float()
        if high_bev_mask.sum() > 0:
            depth_consistency = F.mse_loss(depth_clamped * high_bev_mask, 
                                          high_bev_mask * 0.5)
        else:
            depth_consistency = torch.tensor(0.0, device=bev_pred.device)
        
        # 重みを小さく
        total_physics_loss = 0.01 * volume_loss + 0.01 * depth_consistency
        
        # 最終的なNaNチェック
        if torch.isnan(total_physics_loss) or torch.isinf(total_physics_loss):
            return torch.tensor(0.0, device=bev_pred.device)
        
        return total_physics_loss
        
    except Exception as e:
        print(f"Physics loss error: {e}")
        return torch.tensor(0.0, device=bev_pred.device)

def depth_consistency_loss(depth_pred, depth_tgt, bev_mask):
    """深度の連続性を保証する損失"""
    # 勾配の連続性
    dx = depth_pred[:, :, :, 1:] - depth_pred[:, :, :, :-1]
    dy = depth_pred[:, :, 1:, :] - depth_pred[:, :, :-1, :]
    
    grad_loss = torch.mean(torch.abs(dx)) + torch.mean(torch.abs(dy))
    
    # マスク領域での深度損失
    masked_loss = F.mse_loss(depth_pred * bev_mask, depth_tgt * bev_mask)
    
    # 深度の単調性を促進
    monotonicity_loss = torch.relu(-dx).mean() + torch.relu(-dy).mean()
    
    return masked_loss + 0.1 * grad_loss + 0.05 * monotonicity_loss

def get_adaptive_depth_weight(epoch):
    if epoch < 50:
        return 0.05   # 軽めから開始
    elif epoch < 100:
        return 0.1    # 徐々に増加
    else:
        return 0.15   # 最終的に適度な重み

# Warmup付き
def get_warmup_factor(epoch, warmup_epochs=5):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    return 1.0

def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)

def main(config_path, resume_from=None):
    # ... 設定読み込みなど（train3.pyと同じ） ...
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    set_seed(config.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # experiment dir
    if resume_from:
        exp_dir = resume_from
        exp_name = os.path.basename(exp_dir)
        print(f"--- Resuming Experiment: {exp_name} ---")
    else:
        exp_name = f"{datetime.now().strftime('%y%m%d_%H%M')}_lungBEV_lr{config['learning_rate']}"
        exp_dir = os.path.join(config['output_dir'], exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(exp_dir, 'config.yaml'))
        print(f"--- New Experiment: {exp_name} ---")

    print(f"Configuration loaded from {config_path}")
    print(f"Results will be saved to {exp_dir}")
    print(f"Using device: {device}")

    # Data
    bev_cfg = {
        'bev_target_mode': config.get('bev_target_mode', 'binary'),
        'bev_gaussian_sigma': config.get('bev_gaussian_sigma', 2.0),
        'normalize_input': True
    }
    train_dataset = PanoBEVDataset3(
        dataset_dir=os.path.join(config['data_dir'], 'train'),
        resize_shape=config['resize_shape'],
        augmentation_config=config,
        bev_cfg=bev_cfg
    )
    val_dataset = PanoBEVDataset3(
        dataset_dir=os.path.join(config['data_dir'], 'val'),
        resize_shape=config['resize_shape'],
        augmentation_config={'use_augmentation': False},
        bev_cfg=bev_cfg
    )
    
    # No weighted sampling needed for dense targets
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                          shuffle=True, num_workers=config['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                          shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    
    # モデル
    model = HybridViTPanoBEV(
        vit_model_name=config['vit_model_name'],
        output_size=config['resize_shape'][0]
    ).to(device)
    if resume_from:
        model_path = os.path.join(resume_from, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Resumed model weights from: {model_path}")
    
    criterion_bev_reg = nn.MSELoss()
    criterion_depth   = nn.MSELoss()
    bev_grad_weight   = float(config.get('bev_grad_weight', 0.3))
    dice_loss_weight  = float(config.get('dice_loss_weight', 0.2))   # 軽めに
    
    # オプティマイザ（3つのパラメータグループ）
    optimizer = AdamW([
        {'params': model.vit.parameters(), 'lr': config['learning_rate']},
        {'params': model.vggt.parameters(), 'lr': float(config.get('vggt_learning_rate', 2e-4))},
        {'params': list(model.fusion.parameters()) + 
                  list(model.decoder.parameters()) + 
                  list(model.bev_head.parameters()) + 
                  list(model.depth_head.parameters()), 
         'lr': float(config.get('head_learning_rate', 1e-3))}
    ])
    
    # Cosineスケジューラ
    scheduler = CosineAnnealingLR(
        optimizer, 
        T_max=config['num_epochs'],
        eta_min=float(config.get('min_learning_rate', 1e-7))
    )
    patience, bad = 75, 0  # 密ターゲットなので少し長めに
    best_val = float('inf')  # 複合指標は大きい方が良いため
    scaler = torch.cuda.amp.GradScaler()
    
    # 保存用の変数を追加
    best_val = float('inf')
    best_dice = 0.0
    best_epoch = 0

    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        # 動的深度重み計算
        adaptive_depth_weight = get_adaptive_depth_weight(epoch)
        
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            images, bev_cont, depth_tgt, bev_bin = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bev_pred, depth_pred = model(images)
                
                # 損失計算（深度重みを動的調整）
                loss_bev_reg = criterion_bev_reg(bev_pred, bev_cont)
                loss_bev_grad = grad_loss(bev_pred, bev_cont)
                loss_dice_aux = soft_dice_loss(bev_pred, bev_bin)
                loss_depth = criterion_depth(depth_pred, depth_tgt)
                
                warmup_factor = get_warmup_factor(epoch)
                total_loss = loss_bev_reg + \
                           warmup_factor * (bev_grad_weight * loss_bev_grad + \
                                          dice_loss_weight * loss_dice_aux) + \
                           adaptive_depth_weight * depth_shape_loss(bev_pred, depth_pred, bev_cont, depth_tgt)
                # + physics_consistency_loss(bev_pred, depth_pred)  # 一旦コメントアウト

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)  # 1.0→0.5に強化
            scaler.step(optimizer)
            scaler.update()
            train_loss += total_loss.item()

        # Val
        model.eval()
        val_loss = val_bev_reg = val_bev_grad = val_depth = val_dice = 0.0
        dice_sum_01 = dice_sum_02 = dice_sum_03 = 0.0
        val_count = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, bev_cont, depth_tgt, bev_bin = [b.to(device) for b in batch]
                bev_pred, depth_pred = model(images)

                l_reg = criterion_bev_reg(bev_pred, bev_cont).item()
                l_grad= grad_loss(bev_pred, bev_cont).item()
                l_dice= soft_dice_loss(bev_pred, bev_bin).item()
                l_dep = criterion_depth(depth_pred, depth_tgt).item()

                val_bev_reg += l_reg
                val_bev_grad+= l_grad
                val_dice    += l_dice
                val_depth   += l_dep
                val_loss    += l_reg + bev_grad_weight * l_grad + \
                              dice_loss_weight * l_dice + adaptive_depth_weight * l_dep

                # Multiple threshold Dice
                dice_sum_01 += thresh_dice(bev_pred, bev_bin, th=0.1)
                dice_sum_02 += thresh_dice(bev_pred, bev_bin, th=0.2)
                dice_sum_03 += thresh_dice(bev_pred, bev_bin, th=0.3)
                val_count += 1

        avg_train = train_loss / max(1, len(train_loader))
        avg_val   = val_loss   / max(1, len(val_loader))
        avg_bev_r = val_bev_reg/ max(1, len(val_loader))
        avg_bev_g = val_bev_grad/ max(1, len(val_loader))
        avg_depth = val_depth/ max(1, len(val_loader))
        avg_dice  = val_dice/ max(1, len(val_loader))
        avg_dice_01 = dice_sum_01 / max(1, val_count)
        avg_dice_02 = dice_sum_02 / max(1, val_count)
        avg_dice_03 = dice_sum_03 / max(1, val_count)

        print(f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, "
              f"Val BEV Reg: {avg_bev_r:.4f}, Val BEV Grad: {avg_bev_g:.4f}, "
              f"Val Depth: {avg_depth:.4f}, Val Dice: {avg_dice:.4f}, "
              f"Dice@0.1: {avg_dice_01:.4f}, Dice@0.2: {avg_dice_02:.4f}, Dice@0.3: {avg_dice_03:.4f}")

        # モデルの保存（複合指標を使用）
        composite_metric = avg_dice_03 - 0.1 * avg_val  # Diceスコアを重視
        if composite_metric > best_dice - 1e-4:  # 改善があれば保存
            best_dice = composite_metric
            best_epoch = epoch
            
            # チェックポイントの保存
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_dice': best_dice,
                'best_val_loss': avg_val,
                'config': config,
                'metrics': {
                    'dice_03': avg_dice_03,
                    'val_loss': avg_val,
                    'bev_reg': avg_bev_r,
                    'bev_grad': avg_bev_g,
                    'depth': avg_depth,
                    'composite': composite_metric
                }
            }
            
            # ベストモデルの保存
            torch.save(checkpoint, os.path.join(exp_dir, "best_model.pth"))
            print(f"✨ New best model saved! (Epoch {epoch+1}, Dice@0.3: {avg_dice_03:.4f}, Val Loss: {avg_val:.4f})")
            
            # 最新のモデルも保存（リカバリ用）
            torch.save(checkpoint, os.path.join(exp_dir, "latest_model.pth"))

        # 定期的なチェックポイント（10エポックごと）
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(exp_dir, f"checkpoint_epoch_{epoch+1}.pth")
            torch.save(checkpoint, checkpoint_path)
            print(f"📦 Checkpoint saved at epoch {epoch+1}")

        # Early stopping
        if epoch - best_epoch > patience:
            print(f"🛑 Early stopping triggered! No improvement for {patience} epochs")
            break

        # スケジューラのステップ
        scheduler.step()

    # 学習終了時の情報保存
    final_info = {
        'best_epoch': best_epoch,
        'best_dice': best_dice,
        'training_time': str(datetime.now() - start_time),
        'final_metrics': {
            'dice_03': avg_dice_03,
            'val_loss': avg_val,
            'composite': composite_metric
        }
    }
    
    # 学習情報をJSONで保存
    import json
    with open(os.path.join(exp_dir, 'training_info.json'), 'w') as f:
        json.dump(final_info, f, indent=4)

    print(f"\n🎉 Training completed!")
    print(f"Best model was saved at epoch {best_epoch+1}")
    print(f"Best Dice@0.3: {best_dice:.4f}")

if __name__ == '__main__':
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="PanoBEV-3D Training Script (Hybrid ViT+VGGT)")
    parser.add_argument('--config', type=str, default='config_lung.yaml', help="設定ファイルのパス")
    parser.add_argument('--resume_from', type=str, default=None, help="再開する実験ディレクトリパス")
    args = parser.parse_args()
    main(args.config, args.resume_from)