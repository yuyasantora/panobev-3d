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
NUM_EPOCHS = 200 # データセット全体を何周学習させるか

class BEVDataset(Dataset):
    """
    DRR画像とBEVターゲットを読み込み、固定サイズにリサイズするカスタムデータセットクラス
    """
    def __init__(self, dataset_dir: str, resize_shape: tuple):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        self.resize_shape = resize_shape # ★ 固定サイズを保存
        
        # 全ての画像ファイルのリストを取得
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

    def __len__(self):
        # データセットの総数を返す (画像の枚数)
        return len(self.image_files)

    def __getitem__(self, idx):
        # パスの取得とNumpy配列の読み込み
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        patient_id = image_filename.split('_')[0]
        target_filename = f"{patient_id}_bev.npy"
        target_path = os.path.join(self.target_dir, target_filename)

        image = np.load(image_path)
        target = np.load(target_path)

        # NumPy配列をSimpleITK Imageに変換
        image_sitk = sitk.GetImageFromArray(image, isVector=False)
        target_sitk = sitk.GetImageFromArray(target, isVector=False)

        # --- リサイズ処理 ---
        # 変換の基となる、何もしない変換（Identity Transform）を定義
        identity_transform = sitk.Transform()

        # --- 画像(DRR)のリサイズ ---
        # 目標サイズの参照グリッドを作成
        ref_image = sitk.Image(self.resize_shape, image_sitk.GetPixelIDValue())
        # 物理的な縦横比を維持するための新しいスペーシングを計算
        old_size = image_sitk.GetSize()
        old_spacing = image_sitk.GetSpacing()
        new_spacing = [old_sp * (old_sz / new_sz) for old_sp, old_sz, new_sz in zip(old_spacing, old_size, self.resize_shape)]
        ref_image.SetSpacing(new_spacing)
        ref_image.SetOrigin(image_sitk.GetOrigin())
        ref_image.SetDirection(image_sitk.GetDirection())
        # 高レベルなResample関数を使用
        resized_image_sitk = sitk.Resample(image_sitk, ref_image, identity_transform, sitk.sitkLinear, image_sitk.GetPixelIDValue())

        # --- ターゲット(BEV)のリサイズ ---
        # 目標サイズの参照グリッドを作成
        ref_target = sitk.Image(self.resize_shape, target_sitk.GetPixelIDValue())
        # BEVも同様に、縦横比を維持するスペーシングを計算
        old_size_tgt = target_sitk.GetSize()
        old_spacing_tgt = target_sitk.GetSpacing()
        new_spacing_tgt = [old_sp * (old_sz / new_sz) for old_sp, old_sz, new_sz in zip(old_spacing_tgt, old_size_tgt, self.resize_shape)]
        ref_target.SetSpacing(new_spacing_tgt)
        ref_target.SetOrigin(target_sitk.GetOrigin())
        ref_target.SetDirection(target_sitk.GetDirection())
        # 高レベルなResample関数を使用
        resized_target_sitk = sitk.Resample(target_sitk, ref_target, identity_transform, sitk.sitkNearestNeighbor, 0)
        
        # SimpleITK Imageを再びNumPy配列に戻す
        resized_image = sitk.GetArrayFromImage(resized_image_sitk)
        resized_target = sitk.GetArrayFromImage(resized_target_sitk)

        # PyTorchのテンソルに変換
        image_tensor = torch.from_numpy(resized_image).float().unsqueeze(0) 
        target_tensor = torch.from_numpy(resized_target).float()

        return image_tensor, target_tensor
    
class ViTForBEVGeneration(nn.Module):
    """
    Vision TransformerエンコーダーとCNNデコーダーを組み合わせた
    BEVマップ生成モデル
    """
    def __init__(self, vit_model_name, output_shape):
        super().__init__()
        
        self.vit = ViTModel.from_pretrained(vit_model_name)
        hidden_size = self.vit.config.hidden_size
        self.output_shape = output_shape

        # --- ★★★ デコーダーの定義を一つに統合 ★★★ ---
        # ViTの出力（潜在表現）を受け取り、目標のBEVマップサイズまでアップサンプリングする
        self.decoder = nn.Sequential(
            # 1. MLPで特徴量を拡張
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 14 * 14 * 128),
            nn.ReLU(),
            
            # 2. ベクトルを画像形式に変形
            nn.Unflatten(1, (128, 14, 14)),
            
            # 3. 畳み込み層でアップサンプリング
            nn.ConvTranspose2d(128, 128, kernel_size=4, stride=2, padding=1), # 14x14 -> 28x28
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 28x28 -> 56x56
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 56x56 -> 112x112
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 112x112 -> 224x224
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # --- ★★★ forwardパスをシンプルに修正 ★★★ ---
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. ViTエンコーダーで特徴を抽出
        outputs = self.vit(x)
        encoder_output = outputs.last_hidden_state[:, 0, :] # (バッチ, 768)

        # 2. デコーダーで一気に画像を生成
        bev_map = self.decoder(encoder_output)
        
        return bev_map


def main():
    """
    学習プロセスのメイン関数
    """
    # --- 1. 準備 ---
    # データローダー
    print("データセットを読み込んでいます...")
    dataset = BEVDataset(dataset_dir=DATASET_DIR, resize_shape=RESIZE_SHAPE)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    print(f"データセットの総数: {len(dataset)}")

    # デバイス
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # モデル
    model = ViTForBEVGeneration(vit_model_name=VIT_MODEL_NAME, output_shape=RESIZE_SHAPE).to(device)

    # 損失関数とオプティマイザ
    criterion = nn.MSELoss()
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("\n--- 学習開始 ---")
    
    # --- 2. 学習ループ ---
    for epoch in range(NUM_EPOCHS):
        # 現在のエポック数を表示
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        
        model.train() # モデルを訓練モードに設定
        
        running_loss = 0.0 # エポック内の損失を記録するための変数
        
        # データローダーからバッチ単位でデータを取り出す
        # tqdmを使って進捗バーを表示
        for i, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
            # データをデバイスに送る
            images = images.to(device)
            targets = targets.to(device)
            
            # --- 順伝播 (Forward pass) ---
            predicted_bev = model(images)
            loss = criterion(predicted_bev.squeeze(1), targets)
            
            # --- 逆伝播 (Backward pass) と最適化 ---
            # 1. 勾配をリセット
            optimizer.zero_grad()
            # 2. 損失を基に勾配を計算
            loss.backward()
            # 3. 計算した勾配を基にモデルの重みを更新
            optimizer.step()
            
            # 損失を記録
            running_loss += loss.item()
            
        # エポックごとの平均損失を計算して表示
        epoch_loss = running_loss / len(dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {epoch_loss:.4f}")
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"bev_generation_model_{epoch + 1}.pth")

    print("\n--- 学習完了 ---")

    # (オプション) 学習済みモデルの重みを保存
    model_save_path = "bev_generation_last.pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"学習済みモデルを '{model_save_path}' に保存しました。")


if __name__ == '__main__':
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    main()
