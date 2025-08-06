import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import ViTModel
import SimpleITK as sitk # ★ SimpleITKをインポート
# ★★★ オプティマイザをインポート ★★★
from torch.optim import AdamW
from tqdm import tqdm # tqdmをインポート
import torchvision.transforms as T # ★ torchvision.transformsをインポート

# --- 設定項目 ---
DATASET_DIR = 'dataset'
BATCH_SIZE = 4 # バッチサイズ（一度に処理するデータ数）
NUM_WORKERS = 2 # データ読み込みの並列プロセス数 (Windowsでは0を推奨する場合あり)
RESIZE_SHAPE = (224, 224) # ★ 学習に使う画像の固定サイズを定義

VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k' # Hugging Faceの事前学習済みViTモデル名
# ★★★ 学習に関する設定を追加 ★★★
LEARNING_RATE = 1e-4 # 学習率
# ★★★ 学習エポック数を追加 ★★★
NUM_EPOCHS = 1000 # データセット全体を何周学習させるか

class PanoBEVDataset(Dataset):
    """
    DRR画像、BEVターゲット、深度マップを読み込み、
    データ増強を適用し、固定サイズにリサイズするカスタムデータセットクラス
    """
    def __init__(self, dataset_dir: str, resize_shape: tuple, is_train: bool = True):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        self.depth_dir = os.path.join(dataset_dir, 'depths')
        self.resize_shape = resize_shape
        self.is_train = is_train # ★ 学習用か検証用かをフラグで管理
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

        # ★ 学習データにのみデータ増強を適用する
        if self.is_train:
            self.augmentation_transforms = T.Compose([
                T.RandomRotation(10), # ±10度の範囲でランダムに回転
                T.ColorJitter(brightness=0.2, contrast=0.2), # 明るさとコントラストをランダムに変化
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- 1. パスの取得 ---
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        
        base_filename = image_filename.replace('.npy', '')
        patient_id = base_filename.split('_')[0]
        
        bev_filename = f"{patient_id}_bev.npy"
        bev_path = os.path.join(self.target_dir, bev_filename)
        
        
        depth_filename = f"{base_filename}_depth.npy" 
        depth_path = os.path.join(self.depth_dir, depth_filename)

        # --- 2. NumPy配列の読み込み ---
        image = np.load(image_path)
        bev_target = np.load(bev_path)
        depth_target = np.load(depth_path)

        # --- 3. SimpleITK Imageに変換 ---
        image_sitk = sitk.GetImageFromArray(image)
        bev_sitk = sitk.GetImageFromArray(bev_target)
        depth_sitk = sitk.GetImageFromArray(depth_target)

        # --- 4. 共通のリサイズ関数を定義 ---
        def resize_image(sitk_image, interpolator):
            ref_image = sitk.Image(self.resize_shape, sitk_image.GetPixelIDValue())
            old_size = sitk_image.GetSize()
            old_spacing = sitk_image.GetSpacing()
            new_spacing = [old_sp * (old_sz / new_sz) for old_sp, old_sz, new_sz in zip(old_spacing, old_size, self.resize_shape)]
            ref_image.SetSpacing(new_spacing)
            ref_image.SetOrigin(sitk_image.GetOrigin())
            ref_image.SetDirection(sitk_image.GetDirection())
            return sitk.Resample(sitk_image, ref_image, sitk.Transform(), interpolator, sitk_image.GetPixelIDValue())

        # ★★★ 修正点2: 適切な補間方法を選択 ★★★
        resized_image_sitk = resize_image(image_sitk, sitk.sitkLinear)
        resized_bev_sitk = resize_image(bev_sitk, sitk.sitkNearestNeighbor)
        resized_depth_sitk = resize_image(depth_sitk, sitk.sitkLinear)

        # --- 5. NumPy配列に戻す ---
        resized_image = sitk.GetArrayFromImage(resized_image_sitk)
        resized_bev = sitk.GetArrayFromImage(resized_bev_sitk)
        resized_depth = sitk.GetArrayFromImage(resized_depth_sitk)

        # ★★★ 新しい修正点: 深度マップの正規化 ★★★
        # 深度マップの最大値で割ることで、値を0-1の範囲に収める
        # ゼロ除算を避けるために、微小な値(epsilon)を加える
        epsilon = 1e-6
        if np.max(resized_depth) > epsilon:
            resized_depth = resized_depth / np.max(resized_depth)

        # ★★★ 修正点3: テンソルに変換し、float型に指定 ★★★
        image_tensor = torch.from_numpy(resized_image).float().unsqueeze(0)
        bev_tensor = torch.from_numpy(resized_bev).float().unsqueeze(0)
        depth_tensor = torch.from_numpy(resized_depth).float().unsqueeze(0)
       
        # ★★★ 新しい修正点: 学習データの場合のみデータ増強を適用 ★★★
        if self.is_train:
            image_tensor = self.augmentation_transforms(image_tensor)
       
        return image_tensor, bev_tensor, depth_tensor
    
class ViTPanoBEV(nn.Module):
    """
    Vision TransformerエンコーダーとCNNデコーダーを組み合わせた
    組み合わせたマルチタスクモデル
    """
    def __init__(self, vit_model_name):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(vit_model_name)
        hidden_size = self.vit.config.hidden_size
        self.output_shape = self.vit.config.image_size

         # --- デコーダーの共通部分を関数として定義 ---
        def create_decoder():
            return nn.Sequential(
                nn.Linear(hidden_size, 512),
                nn.ReLU(),
                nn.Linear(512, 14 * 14 * 128),
                nn.ReLU(),
                nn.Unflatten(1, (128, 14, 14)),
                nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # -> 28x28
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # -> 56x56
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # -> 112x112
                nn.ReLU(),
                nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # -> 224x224
                nn.ReLU(),
                nn.Conv2d(16, 1, kernel_size=3, padding=1)
                # 活性化関数(Sigmoid)は損失関数側で考慮するため、ここでは外す
            )
        
        # --- 深度マップとBEVマップ用の二つのデコーダーを作成
        self.bev_decoder = create_decoder()
        self.depth_decoder = create_decoder()

    def forward(self, x):
        # --- ★★★ forwardパスをシンプルに修正 ★★★ ---
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. ViTエンコーダーで特徴を抽出
        outputs = self.vit(x)
        encoder_output = outputs.last_hidden_state[:, 0, :] # (バッチ, 768)

        # 2. デコーダーで一気に画像を生成
        bev_map = self.bev_decoder(encoder_output)
        depth_map = self.depth_decoder(encoder_output)
        
        return bev_map, depth_map

def calculate_dice_coefficient(predicted, target, epsilon=1e-6):
    """
    Dice係数を計算する。
    予測とターゲットは(バッチサイズ, 1, 高さ, 幅)のテンソルを想定。
    """
    # 予測を確率に変換し、0.5を閾値としてバイナリマスクを作成
    predicted_mask = (torch.sigmoid(predicted) > 0.5).float()
    
    # ターゲットも0/1のマスクであることを確認
    target_mask = (target > 0.5).float()
    
    intersection = (predicted_mask * target_mask).sum()
    union = predicted_mask.sum() + target_mask.sum()
    
    dice = (2. * intersection + epsilon) / (union + epsilon)
    return dice.item()

def calculate_rmse(predicted, target):
    """
    RMSE (Root Mean Square Error) を計算する。
    """
    mse_loss = nn.MSELoss()
    return torch.sqrt(mse_loss(predicted, target)).item()


def main():
    """
    学習と検証プロセスのメイン関数
    """
    # --- 1. 準備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ★ データローダーを学習用と検証用に分けて作成 (is_trainフラグを追加)
    print("データセットを読み込んでいます...")
    train_dataset = PanoBEVDataset(dataset_dir=os.path.join(DATASET_DIR, 'train'), resize_shape=RESIZE_SHAPE, is_train=True)
    val_dataset = PanoBEVDataset(dataset_dir=os.path.join(DATASET_DIR, 'val'), resize_shape=RESIZE_SHAPE, is_train=False) # 検証データには増強しない

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    print(f"学習データ数: {len(train_dataset)}, 検証データ数: {len(val_dataset)}")

    # モデル
    model = ViTPanoBEV(vit_model_name=VIT_MODEL_NAME).to(device)

    # 損失関数とオプティマイザ
    pos_weight = torch.tensor([5.0]).to(device)
    criterion_bev = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    criterion_depth = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    # ★ 最良モデルを保存するための変数を初期化
    best_val_loss = float('inf')

    print("\n--- 学習開始 ---")
    
    # --- 2. 学習ループ ---
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 20)

        # --- 学習フェーズ ---
        model.train()
        running_train_loss = 0.0
        
        for images, bev_targets, depth_targets in tqdm(train_loader, desc="Training"):
            images, bev_targets, depth_targets = images.to(device), bev_targets.to(device), depth_targets.to(device)
            
            optimizer.zero_grad()
            
            predicted_bev, predicted_depth = model(images)
            loss_bev = criterion_bev(predicted_bev, bev_targets)
            loss_depth = criterion_depth(predicted_depth, depth_targets)
            total_loss = loss_bev + loss_depth
            
            total_loss.backward()
            optimizer.step()
            
            running_train_loss += total_loss.item() * images.size(0)
            
        epoch_train_loss = running_train_loss / len(train_dataset)

        # --- 検証フェーズ ---
        model.eval()
        running_val_loss = 0.0
        # ★ 評価指標を記録するためのリストを初期化
        val_dice_scores = []
        val_rmse_scores = []
        
        with torch.no_grad():
            for images, bev_targets, depth_targets in tqdm(val_loader, desc="Validation"):
                images, bev_targets, depth_targets = images.to(device), bev_targets.to(device), depth_targets.to(device)

                predicted_bev, predicted_depth = model(images)
                loss_bev = criterion_bev(predicted_bev, bev_targets)
                loss_depth = criterion_depth(predicted_depth, depth_targets)
                total_loss = loss_bev + loss_depth
                
                running_val_loss += total_loss.item() * images.size(0)

                # ★ 評価指標を計算してリストに追加
                dice_score = calculate_dice_coefficient(predicted_bev, bev_targets)
                rmse_score = calculate_rmse(predicted_depth, depth_targets)
                val_dice_scores.append(dice_score)
                val_rmse_scores.append(rmse_score)

        epoch_val_loss = running_val_loss / len(val_dataset)
        # ★ エポックごとの平均評価指標を計算
        avg_val_dice = np.mean(val_dice_scores)
        avg_val_rmse = np.mean(val_rmse_scores)
        
        # ★ 評価指標も一緒に表示
        print(f"Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f}")
        print(f"  -> Val BEV Dice: {avg_val_dice:.4f}, Val Depth RMSE: {avg_val_rmse:.4f}")

        # ★ 検証ロスが改善した場合、モデルを保存
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            torch.save(model.state_dict(), "panobev_model_best.pth")
            print(f"  -> Best model saved with validation loss: {best_val_loss:.4f}")

    print("\n--- 学習完了 ---")
    print(f"最も良かった検証ロス: {best_val_loss:.4f}")
    print("最も性能の良いモデルが 'panobev_model_best.pth' として保存されました。")


if __name__ == '__main__':
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    main()
