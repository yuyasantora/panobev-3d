import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from transformers import ViTModel
import SimpleITK as sitk # ★ SimpleITKをインポート

# --- 設定項目 ---
DATASET_DIR = 'dataset'
BATCH_SIZE = 4 # バッチサイズ（一度に処理するデータ数）
NUM_WORKERS = 2 # データ読み込みの並列プロセス数 (Windowsでは0を推奨する場合あり)
RESIZE_SHAPE = (224, 224) # ★ 学習に使う画像の固定サイズを定義

VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k' # Hugging Faceの事前学習済みViTモデル名

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
    
class BEVModel(nn.Module):
    """
    Vision TransformerエンコーダとCNNエンコーダを組み合わせた
    BEVマップ生成モデル
    """
    def __init__(self, vit_model_name, output_shape):
        super().__init__()

        # Vision Transformerエンコーダのインスタンス
        self.vit = ViTModel.from_pretrained(vit_model_name)

        # ViTの出力特徴量の次元数を取得
        hidden_size = self.vit.config.hidden_size

        #2. CNN デコーダ
        # ViTの出力を受け取り、目標のBEVマップサイズまでアップサンプリング
        self.decoder = nn.Sequential(
             nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, 24 * 24 * 64), # 24x24の画像に展開
            nn.ReLU(),
            # ここで (バッチ, 24*24*64) -> (バッチ, 64, 24, 24) に変形
            
            # --- ここから畳み込み層 ---
            # nn.Unflatten(1, (64, 24, 24)), # reshapeの代わり
            nn.ConvTranspose2d(64, 128, kernel_size=4, stride=2, padding=1), # 24x24 -> 48x48
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1), # 48x48 -> 96x96
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 96x96 -> 192x192
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1),  # 192x192 -> 384x384
            nn.ReLU(),
            nn.Conv2d(16, 1, kernel_size=3, padding=1), # 最終的な出力チャンネルを1に
            nn.Sigmoid() # 出力を0-1の範囲に収める
        )
        self.output_shape = output_shape
        self.unflatten = nn.Unflatten(1, (64, 24, 24))

    def forward(self, x):
        # 入力画像のチャネルは１なので、チャネル次元を追加
        if x.shape[1] == 1:
            x = x.repeat(1,3,1,1)

        # 1. ViTエンコーダ
        outputs = self.vit(x)
        encoder_output = outputs.last_hidden_state[:, 0, :] 
        
        # デコーダーで画像を生成
        decoded = self.decoder[0:5](encoder_output)
        decoded = self.unflatten(decoded) # (バッチ, 64, 24, 24)

        # 畳み込みそうでアップサンプリング
        bev_map = self.decoder[5:](decoded)

        return bev_map


def main():
    """
    学習プロセスのメイン関数
    """
    # --- データローダーの準備 ---
    print("データセットを読み込んでいます...")
    # ★ データセット作成時にリサイズ後の形状を渡す
    dataset = BEVDataset(dataset_dir=DATASET_DIR, resize_shape=RESIZE_SHAPE)
    
    # データローダーを作成
    # shuffle=Trueにすることで、エポックごとにデータの順序がシャッフルされる
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    # --- モデルの定義 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = BEVModel(vit_model_name=VIT_MODEL_NAME, output_shape=RESIZE_SHAPE)
    model.to(device)

    
      # --- モデルの動作確認 ---
    print("\nモデルに最初のバッチを入力して動作確認...")
    images, _ = next(iter(dataloader))
    images = images.to(device)
    
    # モデルの推論を実行
    with torch.no_grad(): # 勾配計算を無効にして、純粋な推論のみ行う
        predicted_bev = model(images)
    
    print(f"  入力画像の形状: {images.shape}")
    print(f"  モデルが出力したBEVマップの形状: {predicted_bev.shape}")

    # (学習ループは次のステップで実装)
    print("\nモデルの定義と動作確認が完了しました。次は損失関数とオプティマイザの定義です。")


if __name__ == '__main__':
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    main()
