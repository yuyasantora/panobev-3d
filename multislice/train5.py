# train5.py - Multi-slice BEV approach
import os
import sys
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
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import argparse

# --- 親ディレクトリをパスに追加 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- 初期設定 ---
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===================================================================
# Dataset for Multi-slice BEV
# ===================================================================
class PanoBEVMultiSliceDataset(Dataset):
    def __init__(self, dataset_dir: str, resize_shape: tuple):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        self.resize_shape = resize_shape
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # DRR画像（入力）の読み込み
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        image = np.load(image_path)
        
        # マルチスライスBEV（教師データ）の読み込み
        patient_id = image_filename.split('_')[0]
        target_filename = f"{patient_id}_multislice_bev.npy"
        target_path = os.path.join(self.target_dir, target_filename)
        multislice_bev = np.load(target_path) # Shape: [16, Z, X]

        # --- リサイズ処理 ---
        # DRR
        image_sitk = sitk.GetImageFromArray(image.astype(np.float32))
        ref_img = sitk.Image(self.resize_shape, image_sitk.GetPixelIDValue())
        resized_image_sitk = sitk.Resample(image_sitk, ref_img, sitk.Transform(), sitk.sitkLinear)
        resized_image = sitk.GetArrayFromImage(resized_image_sitk)

        # Multi-slice BEV Target
        # [16, Z, X] -> [16, H, W]
        resized_slices = []
        for i in range(multislice_bev.shape[0]):
            slice_sitk = sitk.GetImageFromArray(multislice_bev[i].astype(np.float32))
            ref_slice = sitk.Image(self.resize_shape, slice_sitk.GetPixelIDValue())
            resized_slice_sitk = sitk.Resample(slice_sitk, ref_slice, sitk.Transform(), sitk.sitkNearestNeighbor)
            resized_slices.append(sitk.GetArrayFromImage(resized_slice_sitk))
        resized_multislice = np.stack(resized_slices, axis=0)

        # --- 正規化 ---
        # 入力DRR: [-1, 1]
        img_norm = resized_image.astype(np.float32)
        m, M = img_norm.min(), img_norm.max()
        if M > m: img_norm = (img_norm - m) / (M - m)
        img_norm = (img_norm - 0.5) / 0.5
        
        # --- Tensorへ変換 ---
        image_tensor = torch.from_numpy(img_norm).float().unsqueeze(0) # [1, H, W]
        target_tensor = torch.from_numpy(resized_multislice).float()  # [16, H, W]

        return image_tensor, target_tensor

# ===================================================================
# Model for Multi-slice BEV
# ===================================================================
class ViTMultiSliceBEV(nn.Module):
    def __init__(self, vit_model_name, num_bev_slices=16, output_size=224):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name, output_hidden_states=True)
        self.hidden_size = self.vit.config.hidden_size
        self.output_size = output_size

        # (train3.pyと同様のマルチスケール融合)
        self.scale_convs = nn.ModuleList([
            nn.Conv2d(self.hidden_size, 256, 1) for _ in range(3)
        ])
        self.fusion = nn.Sequential(
            nn.Conv2d(256 * 3, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.Conv2d(256, 128, 1)
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU()
        )
        # --- 出力ヘッドの変更 ---
        # 1チャンネルではなく、スライス数(16)チャンネルを出力
        self.bev_head = nn.Conv2d(32, num_bev_slices, 1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        outputs = self.vit(x)
        hidden_states = outputs.hidden_states
        B = x.shape[0]

        features = []
        for i, layer_idx in enumerate([4, 8, 12]):
            h = hidden_states[layer_idx][:, 1:].transpose(1, 2).reshape(B, self.hidden_size, 14, 14)
            h = self.scale_convs[i](h)
            h = F.interpolate(h, size=(14, 14), mode='bilinear', align_corners=False)
            features.append(h)

        fused = self.fusion(torch.cat(features, dim=1))
        decoded = self.decoder(fused)
        
        # マルチスライスBEVを予測
        bev_slices = self.bev_head(decoded)

        bev_slices = F.interpolate(bev_slices, size=(self.output_size, self.output_size), mode='bilinear', align_corners=False)
        
        return torch.sigmoid(bev_slices) # Shape: [B, 16, H, W]

# ===================================================================
# Loss & Metrics
# ===================================================================
class DiceBCELoss(nn.Module):
    """DiceとBCEを組み合わせた損失"""
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs: [B, 16, H, W], targets: [B, 16, H, W]
        inputs = inputs.reshape(-1)
        targets = targets.reshape(-1)
        
        # Dice
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        # BCE
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        
        return BCE + dice_loss

def calculate_dice_metric(pred, target, threshold=0.5):
    """マルチスライス全体のDiceスコアを計算"""
    pred_bin = (pred > threshold).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    dice = (2. * intersection + 1e-6) / (union + 1e-6)
    return dice.item()

# ===================================================================
# Main Training Loop
# ===================================================================
def main(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    set_seed(config.get('seed', 42))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(config['output_dir'], f"{datetime.now().strftime('%y%m%d_%H%M')}_MultiSlice")
    os.makedirs(exp_dir, exist_ok=True)
    shutil.copy(config_path, os.path.join(exp_dir, 'config.yaml'))

    print(f"--- Experiment: {os.path.basename(exp_dir)} ---")
    print(f"Using device: {device}")

    train_dataset = PanoBEVMultiSliceDataset(
        dataset_dir=os.path.join(config['data_dir'], 'train'),
        resize_shape=config['resize_shape']
    )
    val_dataset = PanoBEVMultiSliceDataset(
        dataset_dir=os.path.join(config['data_dir'], 'val'),
        resize_shape=config['resize_shape']
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])
    
    model = ViTMultiSliceBEV(
        vit_model_name=config['vit_model_name'],
        num_bev_slices=config['num_bev_slices'],
        output_size=config['resize_shape'][0]
    ).to(device)

    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['num_epochs'], eta_min=1e-7)
    criterion = DiceBCELoss()
    scaler = torch.cuda.amp.GradScaler()
    
    best_dice = 0.0
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            images, targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                preds = model(images)
                loss = criterion(preds, targets)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        total_dice = 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, targets = [b.to(device) for b in batch]
                preds = model(images)
                loss = criterion(preds, targets)
                val_loss += loss.item()
                total_dice += calculate_dice_metric(preds, targets)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = total_dice / len(val_loader)
        
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val Dice: {avg_dice:.4f}")

        if avg_dice > best_dice:
            best_dice = avg_dice
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            print(f"✨ New best model saved! Dice: {best_dice:.4f}")
        
        scheduler.step()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config_multislice.yaml', help="設定ファイル")
    args = parser.parse_args()
    main(args.config) 