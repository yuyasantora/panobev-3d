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

# --- 設定項目 ---
DATASET_DIR = 'dataset'
BATCH_SIZE = 4 # バッチサイズ（一度に処理するデータ数）
NUM_WORKERS = 2 # データ読み込みの並列プロセス数 (Windowsでは0を推奨する場合あり)
RESIZE_SHAPE = (224, 224) # ★ 学習に使う画像の固定サイズを定義

VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k' # Hugging Faceの事前学習済みViTモデル名
# ★★★ 学習に関する設定を追加 ★★★
LEARNING_RATE = 1e-4 # 学習率
# ★★★ 学習エポック数を追加 ★★★
NUM_EPOCHS = 20 # データセット全体を何周学習させるか

class PanoBEVDataset(Dataset):
    """
    DRR画像、BEVターゲット、深度マップを読み込み、
    固定サイズにリサイズするカスタムデータセットクラス
    """
    def __init__(self, dataset_dir: str, resize_shape: tuple):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        self.depth_dir = os.path.join(dataset_dir, 'depths')
        self.resize_shape = resize_shape
        
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

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
        
        # ★★★ 修正点1: 正しい深度マップのファイル名を構築 ★★★
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


def main():
    """
    学習プロセスのメイン関数
    """
    # --- 1. 準備 ---
    # データローダー
    print("データセットを読み込んでいます...")
    # ★ PanoBEVDataset を使用するように変更
    dataset = PanoBEVDataset(dataset_dir=DATASET_DIR, resize_shape=RESIZE_SHAPE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"データセットの総数: {len(dataset)}")

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # ★ ViTForPanoBEV モデルを使用するように変更
    model = ViTPanoBEV(vit_model_name=VIT_MODEL_NAME).to(device)

    # ★ 損失関数を2つ定義
    criterion_bev = nn.BCEWithLogitsLoss() # BEV用 (Sigmoid不要)
    criterion_depth = nn.MSELoss()         # 深度用
    
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n--- 学習開始 ---")
    
    # --- 2. 学習ループ ---
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        model.train()
        
        running_loss = 0.0
        
        # ★ データローダーの出力を3つに更新
        for i, (images, bev_targets, depth_targets) in enumerate(tqdm(dataloader, desc="Training")):
            # データをデバイスに送る
            images = images.to(device)
            bev_targets = bev_targets.to(device)
            depth_targets = depth_targets.to(device)
            
            # --- 順伝播 ---
            # ★ モデルから2つの出力を受け取る
            predicted_bev, predicted_depth = model(images)
            
            # ★ 2つの損失を計算
            loss_bev = criterion_bev(predicted_bev, bev_targets)
            loss_depth = criterion_depth(predicted_depth, depth_targets)
            
            # ★ 損失を合計する (正規化後は重みを一旦1.0に戻す)
            total_loss = loss_bev + loss_depth
            
            # --- 逆伝播と最適化 ---
            optimizer.zero_grad()
            total_loss.backward() # 合計損失で逆伝播
            optimizer.step()
            
            running_loss += total_loss.item()
            
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}")

        # ★ モデルの保存 (10エポックごと)
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"panobev_model_epoch_{epoch + 1}.pth")

    print("\n--- 学習完了 ---")

    # ★ 学習済みモデルの最終版を保存
    model_save_path = "panobev_model_final.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"学習済みモデルを '{model_save_path}' に保存しました。")


if __name__ == '__main__':
    # Windowsで 'An attempt has been made to start a new process before...' エラーを防ぐ
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    main()
