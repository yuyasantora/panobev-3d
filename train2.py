# train2.py
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
from torch.utils.data import WeightedRandomSampler

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ===================================================================
# Dataset (continuous BEV target on-the-fly)
# ===================================================================
class PanoBEVDataset2(Dataset):
    def __init__(self, dataset_dir: str, resize_shape: tuple, augmentation_config: dict, bev_cfg: dict):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        self.depth_dir  = os.path.join(dataset_dir, 'depths')
        self.resize_shape = resize_shape
        self.use_augmentation = augmentation_config.get('use_augmentation', False)

        self.bev_mode = bev_cfg.get('bev_target_mode', 'gaussian')  # 'gaussian' | 'distance' | 'binary'
        self.bev_gauss_sigma = float(bev_cfg.get('bev_gaussian_sigma', 1.5))
        self.bev_distance_tau = float(bev_cfg.get('bev_distance_tau', 3.0))
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
        bev_target_np = np.load(bev_path)      # 2D
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

        # Binary BEV (for logging Dice, etc)
        bev_binary = (resized_bev_bin > 0).astype(np.float32)

        # Continuous BEV target
        if self.bev_mode == 'binary':
            bev_cont = bev_binary
        elif self.bev_mode == 'gaussian':
            # Gaussian blur on binary mask
            bev_img = sitk.GetImageFromArray(bev_binary.astype(np.float32))
            gauss = sitk.DiscreteGaussian(bev_img, variance=float(self.bev_gauss_sigma**2))
            bev_cont = sitk.GetArrayFromImage(gauss).astype(np.float32)
            m = bev_cont.max()
            if m > 1e-6:
                bev_cont = bev_cont / m
        elif self.bev_mode == 'distance':
            # Signed distance -> soft heatmap exp(-|d|/tau)
            bin_img = sitk.GetImageFromArray(bev_binary.astype(np.uint8))
            dist = sitk.SignedMaurerDistanceMap(bin_img, insideIsPositive=True, squaredDistance=False, useImageSpacing=False)
            d = np.abs(sitk.GetArrayFromImage(dist)).astype(np.float32)
            bev_cont = np.exp(-(d / float(self.bev_distance_tau)))
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
# Model (apply sigmoid to both heads for [0,1] range)
# ===================================================================
class ViTPanoBEV2(nn.Module):
    def __init__(self, vit_model_name):
        super().__init__()
        self.vit = ViTModel.from_pretrained(vit_model_name, output_hidden_states=True)
        self.hidden_size = self.vit.config.hidden_size

        class DecoderBlock(nn.Module):
            def __init__(self, in_channels, skip_channels, out_channels):
                super().__init__()
                self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
                self.conv = nn.Sequential(
                    nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )
            def forward(self, x, skip_x):
                x = self.up(x)
                if skip_x.shape[2:] != x.shape[2:]:
                    skip_x = F.interpolate(skip_x, size=x.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_x], dim=1)
                return self.conv(x)

        self.bottleneck = nn.Conv2d(self.hidden_size, 512, kernel_size=1)
        self.decoder4 = DecoderBlock(512, self.hidden_size, 256)
        self.decoder3 = DecoderBlock(256, self.hidden_size, 128)
        self.decoder2 = DecoderBlock(128, self.hidden_size, 64)
        self.final_conv = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)
        self.bev_head = nn.Conv2d(16, 1, kernel_size=1)
        self.depth_head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        vit_outputs = self.vit(x)
        hidden_states = vit_outputs.hidden_states

        bsz = x.shape[0]
        patch = self.vit.config.patch_size
        fmap = x.shape[2] // patch

        skip1 = hidden_states[1][:, 1:].permute(0,2,1).reshape(bsz, self.hidden_size, fmap, fmap)
        skip2 = hidden_states[4][:, 1:].permute(0,2,1).reshape(bsz, self.hidden_size, fmap, fmap)
        skip3 = hidden_states[8][:, 1:].permute(0,2,1).reshape(bsz, self.hidden_size, fmap, fmap)
        bott  = hidden_states[-1][:, 1:].permute(0,2,1).reshape(bsz, self.hidden_size, fmap, fmap)

        x = self.bottleneck(bott)
        x = self.decoder4(x, skip3)
        x = self.decoder3(x, skip2)
        x = self.decoder2(x, skip1)
        x = self.final_conv(x)

        bev = self.bev_head(x)
        depth = self.depth_head(x)

        bev  = F.interpolate(bev,  size=(224,224), mode='bilinear', align_corners=False)
        depth= F.interpolate(depth,size=(224,224), mode='bilinear', align_corners=False)

        bev = torch.sigmoid(bev)     # regression target in [0,1]
        depth = torch.sigmoid(depth) # [0,1]
        return bev, depth

# ===================================================================
# Loss/metrics
# ===================================================================
def grad_loss(pred, tgt):
    # pred,tgt: [B,1,H,W], in [0,1]
    dx_p = pred[..., :, 1:] - pred[..., :, :-1]
    dy_p = pred[..., 1:, :] - pred[..., :-1, :]
    dx_t = tgt[..., :, 1:] - tgt[..., :, :-1]
    dy_t = tgt[..., 1:, :] - tgt[..., :-1, :]

    # pad back to [B,1,H,W]
    dx = F.pad(torch.abs(dx_p - dx_t), (0,1,0,0), mode='replicate')
    dy = F.pad(torch.abs(dy_p - dy_t), (0,0,0,1), mode='replicate')
    return (dx.mean() + dy.mean()) * 0.5

def calculate_rmse(predicted, target):
    return torch.sqrt(F.mse_loss(predicted, target)).item()

def thresh_dice(pred, tgt_bin, th=0.3, eps=1e-6):
    pm = (pred > th).float()
    tm = (tgt_bin > 0.5).float()
    inter = (pm * tm).sum()
    union = pm.sum() + tm.sum()
    return ((2*inter + eps) / (union + eps)).item()

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
        exp_name = f"{datetime.now().strftime('%y%m%d_%H%M')}_contBEV_lr{config['learning_rate']}"
        exp_dir = os.path.join(config['output_dir'], exp_name)
        os.makedirs(exp_dir, exist_ok=True)
        shutil.copy(config_path, os.path.join(exp_dir, 'config.yaml'))
        print(f"--- New Experiment: {exp_name} ---")

    print(f"Configuration loaded from {config_path}")
    print(f"Results will be saved to {exp_dir}")
    print(f"Using device: {device}")

    # Data
    bev_cfg = {
        'bev_target_mode': config.get('bev_target_mode', 'gaussian'),
        'bev_gaussian_sigma': config.get('bev_gaussian_sigma', 1.5),
        'bev_distance_tau': config.get('bev_distance_tau', 3.0),
        'normalize_input': True
    }
    train_dataset = PanoBEVDataset2(
        dataset_dir=os.path.join(config['data_dir'], 'train'),
        resize_shape=config['resize_shape'],
        augmentation_config=config,
        bev_cfg=bev_cfg
    )
    val_dataset = PanoBEVDataset2(
        dataset_dir=os.path.join(config['data_dir'], 'val'),
        resize_shape=config['resize_shape'],
        augmentation_config={'use_augmentation': False},
        bev_cfg=bev_cfg
    )
    train_target_dir = os.path.join(config['data_dir'], 'train', 'targets')
    weights = []
    for f in train_dataset.image_files:
        pid = f.replace('.npy','').split('_')[0]
        bev = np.load(os.path.join(train_target_dir, f"{pid}_bev.npy"))
        weights.append(3.0 if (bev > 0).any() else 1.0)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                          sampler=sampler, shuffle=False, num_workers=config['num_workers'], pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                          shuffle=False, num_workers=config['num_workers'], pin_memory=True)

    # Model
    model = ViTPanoBEV2(vit_model_name=config['vit_model_name']).to(device)
    if resume_from:
        model_path = os.path.join(resume_from, "best_model.pth")
        if os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            print(f"Resumed model weights from: {model_path}")

    # Loss
    criterion_bev_reg = nn.MSELoss()
    criterion_depth   = nn.MSELoss()
    bev_grad_weight   = float(config.get('bev_grad_weight', 0.5))
    depth_loss_weight = float(config.get('depth_loss_weight', 0.0))  # まずBEVに集中

    # Optimizer (encoder low LR, heads high LR)
    head_params = list(model.bottleneck.parameters()) + \
                  list(model.decoder4.parameters()) + \
                  list(model.decoder3.parameters()) + \
                  list(model.decoder2.parameters()) + \
                  list(model.final_conv.parameters()) + \
                  list(model.bev_head.parameters()) + \
                  list(model.depth_head.parameters())
    optimizer = AdamW([
        {'params': model.vit.parameters(), 'lr': config['learning_rate']},
        {'params': head_params, 'lr': float(config.get('head_learning_rate', 1e-3))},
    ])
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    # Train
    patience, bad = 8, 0
    best_val = float('inf')
    scaler = torch.cuda.amp.GradScaler()
    for epoch in range(config['num_epochs']):
        print(f"\nEpoch {epoch+1}/{config['num_epochs']}")
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc="Training"):
            images, bev_cont, depth_tgt, bev_bin = [b.to(device) for b in batch]
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():
                bev_pred, depth_pred = model(images)
                loss_bev_reg = criterion_bev_reg(bev_pred, bev_cont)
                loss_bev_grad= grad_loss(bev_pred, bev_cont)
                loss_bev = loss_bev_reg + bev_grad_weight * loss_bev_grad
                loss_depth = criterion_depth(depth_pred, depth_tgt)
                total_loss = loss_bev + depth_loss_weight * loss_depth
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
            train_loss += total_loss.item()

        # Val
        model.eval()
        val_loss = val_bev_reg = val_bev_grad = val_depth = 0.0
        dice_nz_sum = 0.0; nz_batches = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                images, bev_cont, depth_tgt, bev_bin = [b.to(device) for b in batch]
                bev_pred, depth_pred = model(images)

                l_reg = criterion_bev_reg(bev_pred, bev_cont).item()
                l_grad= grad_loss(bev_pred, bev_cont).item()
                l_bev = l_reg + bev_grad_weight * l_grad
                l_dep = criterion_depth(depth_pred, depth_tgt).item()

                val_bev_reg += l_reg
                val_bev_grad+= l_grad
                val_depth   += l_dep
                val_loss    += l_bev + depth_loss_weight * l_dep

                # Dice vs binary mask (参考)
                nz = (bev_bin.sum(dim=(2,3)) > 0)
                if nz.any():
                    d = thresh_dice(bev_pred[nz], bev_bin[nz], th=0.3)
                    dice_nz_sum += d
                    nz_batches += 1

        avg_train = train_loss / max(1, len(train_loader))
        avg_val   = val_loss   / max(1, len(val_loader))
        avg_bev_r = val_bev_reg/ max(1, len(val_loader))
        avg_bev_g = val_bev_grad/max(1, len(val_loader))
        avg_dep   = val_depth  / max(1, len(val_loader))
        avg_dice_nz = (dice_nz_sum / nz_batches) if nz_batches>0 else 0.0

        print(f"Train Loss: {avg_train:.4f}, Val Loss: {avg_val:.4f}, "
              f"Val BEV(reg): {avg_bev_r:.4f}, Val BEV(grad): {avg_bev_g:.4f}, "
              f"Val Depth: {avg_dep:.4f}, Avg DiceNZ@0.3: {avg_dice_nz:.4f} (nz_batches={nz_batches})")

        # depth_loss_weightが0でない運用も想定し、BEVのみでLRを下げる
        scheduler.step(avg_bev_r + bev_grad_weight * avg_bev_g)

        if avg_val < best_val - 1e-4:
            best_val = avg_val; bad = 0
            torch.save(model.state_dict(), os.path.join(exp_dir, "best_model.pth"))
        else:
            bad += 1
            if bad >= patience: 
                print("Early stop."); break

if __name__ == '__main__':
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    parser = argparse.ArgumentParser(description="PanoBEV-3D Training Script (continuous BEV)")
    parser.add_argument('--config', type=str, default='config.yaml', help="設定ファイルのパス")
    parser.add_argument('--resume_from', type=str, default=None, help="再開する実験ディレクトリパス")
    args = parser.parse_args()
    main(args.config, args.resume_from)
