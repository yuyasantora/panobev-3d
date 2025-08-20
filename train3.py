# train3.py
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
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import argparse

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

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
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

        return image_tensor, bev_cont_tensor, depth_tensor, bev_bin_tensor

# ===================================================================
# Model (same as train2)
# ===================================================================
class ViTPanoBEV3_MultiScale(nn.Module):
    def __init__(self, vit_model_name, output_size=224):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name, output_hidden_states=True)
        self.hidden_size = self.vit.config.hidden_size
        self.output_size = output_size

        # 各スケールの特徴変換
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(self.hidden_size, 256, 1),  # 細部 (layer 4)
            nn.Conv2d(self.hidden_size, 256, 1),  # 中間 (layer 8)
            nn.Conv2d(self.hidden_size, 256, 1)   # 大域 (layer 12)
        ])

        # 強化された特徴融合
        self.fusion = nn.Sequential(
            nn.Conv2d(256*3, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 128, 1)
        )

        # デコーダー
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # 改良された出力ヘッド
        self.bev_head = nn.Conv2d(32, 1, 1)
        self.depth_head = nn.Sequential(
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1)
        )

    def forward(self, x):
        # 入力処理
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # ViT特徴抽出
        outputs = self.vit(x)
        hidden_states = outputs.hidden_states

        # 各スケールの特徴を抽出・処理
        features = []
        for i, layer_idx in enumerate([4, 8, 12]):
            # hidden_statesから特徴マップを復元
            h = hidden_states[layer_idx]
            B = h.shape[0]
            h = h[:, 1:].transpose(1, 2).reshape(B, self.hidden_size, 14, 14)
            
            # スケール特徴の変換
            h = self.scale_convs[i](h)
            
            # サイズ調整
            if i > 0:  # 中間・大域特徴を細部特徴のサイズに合わせる
                h = F.interpolate(h, size=(14, 14), mode='bilinear', align_corners=False)
            
            features.append(h)

        # 特徴融合
        fused = self.fusion(torch.cat(features, dim=1))
        
        # デコード
        decoded = self.decoder(fused)

        # 最終出力
        bev = self.bev_head(decoded)
        depth = self.depth_head(decoded)

        # サイズ調整とシグモイド
        bev = F.interpolate(bev, size=(self.output_size, self.output_size), 
                          mode='bilinear', align_corners=False)
        depth = F.interpolate(depth, size=(self.output_size, self.output_size), 
                            mode='bilinear', align_corners=False)

        return torch.sigmoid(bev), torch.sigmoid(depth)

# ===================================================================
# Loss/metrics (optimized for dense targets)
# ===================================================================
def grad_loss(pred, tgt):
    # pred,tgt: [B,1,H,W], in [0,1]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_t = tgt[..., :, 1:] - tgt[..., :, :-1]
    dy_t = tgt[..., 1:, :] - pred[..., :-1, :]

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

def get_adaptive_depth_weight(epoch):
    """エポックに応じて深度損失重みを調整（より保守的に）"""
    if epoch < 30:    # 30エポックまでBEV専念
        return 0.0
    elif epoch < 80:  # 段階的導入をゆっくり
        return 0.1
    else:
        return 0.2    # 最大でも0.2に抑制

# ===================================================================
# Main
# ===================================================================
def main(config_path, resume_from=None):
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

    # Model
    model = ViTPanoBEV3_MultiScale(
        vit_model_name=config['vit_model_name'],
        output_size=config['resize_shape'][0]  # 384を渡す
    ).to(device)
    if resume_from:
        model_path = os.path.join(resume_from, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Resumed model weights from: {model_path}")

    # Loss (simplified for dense targets)
    criterion_bev_reg = nn.MSELoss()
    criterion_depth   = nn.MSELoss()
    bev_grad_weight   = float(config.get('bev_grad_weight', 0.3))
    dice_loss_weight  = float(config.get('dice_loss_weight', 0.2))   # 軽めに

    # Optimizer (encoder low LR, heads high LR)
    head_params = list(model.scale_convs.parameters()) + \
                  list(model.fusion.parameters()) + \
                  list(model.decoder.parameters()) + \
                  list(model.bev_head.parameters()) + \
                  list(model.depth_head.parameters())
    optimizer = AdamW([
        {'params': model.vit.parameters(), 'lr': config['learning_rate']},
        {'params': head_params, 'lr': float(config.get('head_learning_rate', 1e-3))},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8, verbose=True)

    # Train
    patience, bad = 75, 0  # 密ターゲットなので少し長めに
    best_val = float('inf')  # 複合指標は大きい方が良いため
    scaler = torch.cuda.amp.GradScaler()
    
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
                
                total_loss = loss_bev_reg + bev_grad_weight * loss_bev_grad + \
                           dice_loss_weight * loss_dice_aux + adaptive_depth_weight * loss_depth
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
        avg_bev_g = val_bev_grad/max(1, len(val_loader))
        avg_dice  = val_dice   / max(1, len(val_loader))
        avg_dep   = val_depth  / max(1, len(val_loader))
        
        avg_dice_01 = dice_sum_01 / max(1, val_count)
        avg_dice_02 = dice_sum_02 / max(1, val_count)
        avg_dice_03 = dice_sum_03 / max(1, val_count)

        print(f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}")
        print(f"Val BEV(reg): {avg_bev_r:.4f}, BEV(grad): {avg_bev_g:.4f}, Dice: {avg_dice:.4f}, Depth: {avg_dep:.4f}")
        print(f"Avg Dice@0.1: {avg_dice_01:.4f}, @0.2: {avg_dice_02:.4f}, @0.3: {avg_dice_03:.4f}")
        print(f"Adaptive Depth Weight: {adaptive_depth_weight:.3f}")  # 追加

        # 深度統計も追加
        with torch.no_grad():
            depth_mean = depth_pred.mean().item()
            depth_std = depth_pred.std().item()
            print(f"Depth Stats: mean={depth_mean:.4f}, std={depth_std:.4f}")

        scheduler.step(avg_val)

        # より賢い早期終了（371行目付近を変更）
        # Val Lossでなく、複合指標で判断
        composite_metric = avg_dice_03 - 0.05 * avg_val  # Dice重視、Loss軽視
        print(f"  Bad epochs: {bad}/{patience}, Best val: {best_val:.4f}, Current: {avg_val:.4f}")
        if avg_val < best_val - 1e-4:  # Val Lossは小さい方が良い
            best_val = avg_val; bad = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            print(f"  -> Best model saved with Val Loss: {best_val:.4f}")
        else:
            bad += 1
            # if bad >= patience: 
            #     print("Early stop."); break

if __name__ == '__main__':
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="PanoBEV-3D Training Script (Lung BEV)")
    parser.add_argument('--config', type=str, default='config_lung.yaml', help="設定ファイルのパス")
    parser.add_argument('--resume_from', type=str, default=None, help="再開する実験ディレクトリパス")
    args = parser.parse_args()
    main(args.config, args.resume_from)