import os
import yaml
import shutil
import random
from datetime import datetime
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from transformers import ViTModel
import SimpleITK as sitk
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import torchvision.transforms as T
import argparse

def set_seed(seed):
    """乱数シードを固定して再現性を確保する"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 決定的なアルゴリズムを使用する
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===================================================================
# Dataset Class
# ===================================================================
class PanoBEVDataset(Dataset):
    def __init__(self, dataset_dir: str, resize_shape: tuple, augmentation_config: dict):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        self.depth_dir = os.path.join(dataset_dir, 'depths')
        self.resize_shape = resize_shape
        self.use_augmentation = augmentation_config['use_augmentation']
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

        if self.use_augmentation:
            self.augmentation_transforms = T.Compose([
                T.RandomRotation(augmentation_config['rotation_degrees']),
                T.ColorJitter(
                    brightness=augmentation_config['color_jitter_brightness'], 
                    contrast=augmentation_config['color_jitter_contrast']
                ),
            ])

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
        bev_target = np.load(bev_path)
        depth_target = np.load(depth_path)

        image_sitk = sitk.GetImageFromArray(image)
        bev_sitk = sitk.GetImageFromArray(bev_target)
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
        resized_bev = sitk.GetArrayFromImage(resize_image(bev_sitk, sitk.sitkNearestNeighbor))
        resized_depth = sitk.GetArrayFromImage(resize_image(depth_sitk, sitk.sitkLinear))
        
        epsilon = 1e-6
        if np.max(resized_depth) > epsilon:
            resized_depth = resized_depth / np.max(resized_depth)
        
        image_tensor = torch.from_numpy(resized_image).float().unsqueeze(0)
        bev_tensor = torch.from_numpy(resized_bev).float().unsqueeze(0)
        depth_tensor = torch.from_numpy(resized_depth).float().unsqueeze(0)

        if self.use_augmentation:
            image_tensor = self.augmentation_transforms(image_tensor)
       
        return image_tensor, bev_tensor, depth_tensor

# ===================================================================
# Model Class (TransUNet inspired - Simplified)
# ===================================================================
class ViTPanoBEV(nn.Module):
    def __init__(self, vit_model_name):
        super().__init__()

        # --- 1. ViT Encoder ---
        self.vit = ViTModel.from_pretrained(vit_model_name, output_hidden_states=True)
        self.hidden_size = self.vit.config.hidden_size  # 768

        # --- 2. U-Net Decoder Blocks ---
        class DecoderBlock(nn.Module):
            def __init__(self, in_channels, skip_channels, out_channels):
                super().__init__()
                # まず入力画像を2倍にアップサンプリング
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
                # アップサンプリングされた特徴量と、エンコーダからのスキップ接続を結合
                # 結合後のチャンネル数は (out_channels + skip_channels) になる
                self.conv = nn.Sequential(
                    nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

            def forward(self, x, skip_x):
                x = self.up(x)
                if skip_x.shape[1] != x.shape[2:]:
                    skip_x = nn.functional.interpolate(skip_x, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_x], dim=1)
                return self.conv(x)

        # ボトルネック層: ViTの最終出力を受け取る
        self.bottleneck = nn.Conv2d(self.hidden_size, 512, kernel_size=1)

        # 各デコーダーブロックを定義
        # in_channels, skip_channels, out_channels
        self.decoder4 = DecoderBlock(512, self.hidden_size, 256)
        self.decoder3 = DecoderBlock(256, self.hidden_size, 128)
        self.decoder2 = DecoderBlock(128, self.hidden_size, 64)
        
        # 最初のスキップ接続はViTの入力埋め込み層から取得する
        # この層のチャンネル数は hidden_size ではなく embedding_size (通常同じだが念のため)
        self.decoder1 = DecoderBlock(64, self.vit.config.hidden_size, 32)
        
        # --- 3. Final Convolution and Output Heads ---
        self.final_conv = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.bev_head = nn.Conv2d(16, 1, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        
        # --- 1. ViT Encoder Forward ---
        vit_outputs = self.vit(x)
        hidden_states = vit_outputs.hidden_states
        
        # --- 2. 特徴マップの整形 & スキップ接続の準備 ---
        batch_size = x.shape[0]
        patch_size = self.vit.config.patch_size
        feature_map_size = x.shape[2] // patch_size

        # スキップ接続に使用する中間層の特徴マップを整形
        # hidden_states[0] はパッチ埋め込み層
        skip1_features = hidden_states[1][:, 1:].permute(0, 2, 1).reshape(batch_size, self.hidden_size, feature_map_size, feature_map_size)
        skip2_features = hidden_states[4][:, 1:].permute(0, 2, 1).reshape(batch_size, self.hidden_size, feature_map_size, feature_map_size)
        skip3_features = hidden_states[8][:, 1:].permute(0, 2, 1).reshape(batch_size, self.hidden_size, feature_map_size, feature_map_size)
        
        # ボトルネック用の特徴マップ (ViTの最終層)
        bottleneck_features = hidden_states[-1][:, 1:].permute(0, 2, 1).reshape(batch_size, self.hidden_size, feature_map_size, feature_map_size)

        # --- 3. U-Net Decoder Forward ---
        x = self.bottleneck(bottleneck_features)
        x = self.decoder4(x, skip3_features)
        x = self.decoder3(x, skip2_features)
        x = self.decoder2(x, skip1_features)
        # decoder1は入力埋め込みと結合したいが、サイズが合わない可能性があるため、
        # ここではよりシンプルな最終アップサンプリングに切り替える
        
        x = self.final_conv(x)
        
        # --- 4. Output Heads ---
        bev_map = self.bev_head(x)
        depth_map = self.depth_head(x)
        
        # 元の画像サイズ (224x224) にリサイズ
        bev_map = nn.functional.interpolate(bev_map, size=(224, 224), mode='bilinear', align_corners=False)
        depth_map = nn.functional.interpolate(depth_map, size=(224, 224), mode='bilinear', align_corners=False)
        depth_map = torch.sigmoid(depth_map)
        
        return bev_map, depth_map

# ===================================================================
# Loss Functions
# ===================================================================
class DiceLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, predicted, target):
        predicted = torch.sigmoid(predicted)
        intersection = (predicted * target).sum(dim=(2,3))
        union = predicted.sum(dim=(2,3)) + target.sum(dim=(2,3))
        dice_score = (2. * intersection + self.epsilon) / (union + self.epsilon)
        return 1. - dice_score.mean()

# ===================================================================
# Metrics
# ===================================================================
def calculate_dice_coefficient(predicted, target, epsilon=1e-6):
    predicted_mask = (torch.sigmoid(predicted) > 0.5).float()
    target_mask = (target > 0.5).float()
    intersection = (predicted_mask * target_mask).sum()
    union = predicted_mask.sum() + target_mask.sum()
    return ((2. * intersection + epsilon) / (union + epsilon)).item()

def calculate_rmse(predicted, target):
    return torch.sqrt(nn.functional.mse_loss(predicted, target)).item()

# ===================================================================
# Main Training Function
# ===================================================================
def main(config_path, resume_from=None):
    """
    設定ファイルに基づいて学習と検証を行うメイン関数
    """
    # --- 0. 設定ファイルの読み込み & シードの固定 ---
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    set_seed(config.get('seed', 42))

    # --- 1. Experiment Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # ★ 学習再開時と新規学習時で実験ディレクトリを分ける
    if resume_from:
        exp_dir = resume_from
        exp_name = os.path.basename(exp_dir)
        print(f"--- Resuming Experiment: {exp_name} ---")
    else:
        exp_name = f"{datetime.now().strftime('%y%m%d_%H%M')}_pos{config['pos_weight']}_lr{config['learning_rate']}"
        exp_dir = os.path.join(config['output_dir'], exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(exp_dir, 'config.yaml'))
        print(f"--- New Experiment: {exp_name} ---")

    print(f"Configuration loaded from {config_path}")
    print(f"Results will be saved to {exp_dir}")
    print(f"Using device: {device}")

    # --- 2. DataLoaders (変更なし) ---
    train_dataset = PanoBEVDataset(
        dataset_dir=os.path.join(config['data_dir'], 'train'), 
        resize_shape=config['resize_shape'], 
        augmentation_config=config
    )
    val_dataset = PanoBEVDataset(
        dataset_dir=os.path.join(config['data_dir'], 'val'), 
        resize_shape=config['resize_shape'], 
        augmentation_config={'use_augmentation': False} # Validation set should not be augmented
    )
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=config['num_workers'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=config['num_workers'])

    # --- 3. Model, Loss, Optimizer ---
    model = ViTPanoBEV(vit_model_name=config['vit_model_name']).to(device)
    
    # ★★★ 学習再開ロジック ★★★
    if resume_from:
        model_path = os.path.join(resume_from, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Resumed model weights from: {model_path}")
        else:
            print(f"Warning: --resume_from was specified, but best_model.pth not found in {resume_from}. Starting from scratch.")

    pos_weight_tensor = torch.tensor([config['pos_weight']]).to(device)
    criterion_bev_bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_bev_dice = DiceLoss().to(device)
    criterion_depth = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'])
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    # --- 4. Training Loop ---
    best_val_loss = float('inf')
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch + 1}/{config['num_epochs']}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            images, bev_targets, depth_targets = [b.to(device) for b in batch]
            optimizer.zero_grad()
            predicted_bev, predicted_depth = model(images)
            
            loss_bce = criterion_bev_bce(predicted_bev, bev_targets)
            loss_dice = criterion_bev_dice(predicted_bev, bev_targets)
            loss_bev = loss_bce + loss_dice

            loss_depth = criterion_depth(predicted_depth, depth_targets)
            total_loss = loss_bev + config['depth_loss_weight'] * loss_depth
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += total_loss.item()
        
        # Validation phase
        model.eval()
        val_loss, val_bev_loss, val_depth_loss, val_dice, val_rmse = 0.0, 0.0, 0.0, 0.0, 0.0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, bev_targets, depth_targets = [b.to(device) for b in batch]
                predicted_bev, predicted_depth = model(images)

                bce_loss_item = criterion_bev_bce(predicted_bev, bev_targets).item()
                dice_loss_item = criterion_bev_dice(predicted_bev, bev_targets).item()
                bev_loss_item = bce_loss_item + dice_loss_item
                depth_loss_item = criterion_depth(predicted_depth, depth_targets).item()

                val_bev_loss += bev_loss_item
                val_depth_loss += depth_loss_item
                val_loss += bev_loss_item + config['depth_loss_weight'] * depth_loss_item

                val_dice += calculate_dice_coefficient(predicted_bev, bev_targets)
                val_rmse += calculate_rmse(predicted_depth, depth_targets)

        # Logging
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_val_bev_loss = val_bev_loss / len(val_loader)
        avg_val_depth_loss = val_depth_loss / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_rmse = val_rmse / len(val_loader)
        print(f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, "
              f"Val BEV Loss: {avg_val_bev_loss:.4f}, Val Depth Loss: {avg_val_depth_loss:.4f}, "
              f"Val Dice: {avg_val_dice:.4f}, Val RMSE: {avg_val_rmse:.4f}")

        # Learning rate scheduler step
        scheduler.step(avg_val_loss)

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
            print(f"  -> Best model saved with Val Loss: {best_val_loss:.4f}")

# ===================================================================
# Script Entry Point
# ===================================================================
if __name__ == '__main__':
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    
    # ★★★ argparse を使ってコマンドライン引数を設定 ★★★
    parser = argparse.ArgumentParser(description="PanoBEV-3D Training Script")
    parser.add_argument(
        '--config', 
        type=str, 
        default='config.yaml', 
        help="設定ファイルのパス (default: config.yaml)"
    )
    parser.add_argument(
        '--resume_from', 
        type=str, 
        default=None, 
        help="学習を再開する実験ディレクトリのパス (例: experiments/240523...)"
    )
    args = parser.parse_args()
        
    main(args.config, args.resume_from)
