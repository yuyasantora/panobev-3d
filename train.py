import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- 設定項目 ---
DATASET_DIR = 'dataset'
BATCH_SIZE = 4 # バッチサイズ（一度に処理するデータ数）
NUM_WORKERS = 2 # データ読み込みの並列プロセス数 (Windowsでは0を推奨する場合あり)

class BEVDataset(Dataset):
    """
    DRR画像とBEVターゲットを読み込むためのカスタムデータセットクラス
    """
    def __init__(self, dataset_dir: str):
        self.image_dir = os.path.join(dataset_dir, 'images')
        self.target_dir = os.path.join(dataset_dir, 'targets')
        
        # 全ての画像ファイルのリストを取得
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.npy')])

    def __len__(self):
        # データセットの総数を返す (画像の枚数)
        return len(self.image_files)

    def __getitem__(self, idx):
        # --- 1. 画像ファイルのパスを取得 ---
        image_filename = self.image_files[idx]
        image_path = os.path.join(self.image_dir, image_filename)
        
        # --- 2. ファイル名から患者IDを抽出し、対応するターゲットのパスを作成 ---
        patient_id = image_filename.split('_')[0]
        target_filename = f"{patient_id}_bev.npy"
        target_path = os.path.join(self.target_dir, target_filename)

        # --- 3. NumPy配列としてファイルを読み込む ---
        image = np.load(image_path)
        target = np.load(target_path)
        
        # --- 4. PyTorchのテンソルに変換 ---
        # 画像にはチャンネルの次元を追加 (H, W) -> (C, H, W)
        # PyTorchの多くのモデルはチャンネル次元を期待するため
        image_tensor = torch.from_numpy(image).float().unsqueeze(0) 
        target_tensor = torch.from_numpy(target).float()

        return image_tensor, target_tensor


def main():
    """
    学習プロセスのメイン関数
    """
    # --- データローダーの準備 ---
    print("データセットを読み込んでいます...")
    dataset = BEVDataset(dataset_dir=DATASET_DIR)
    
    # データローダーを作成
    # shuffle=Trueにすることで、エポックごとにデータの順序がシャッフルされる
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    
    print(f"データセットの総数: {len(dataset)}")
    print(f"1バッチあたりのデータ数: {BATCH_SIZE}")
    print(f"総バッチ数: {len(dataloader)}")

    # --- データローダーの動作確認 ---
    print("\nデータローダーから最初のバッチを取得して確認...")
    
    # イテレーターから最初のバッチを取得
    images, targets = next(iter(dataloader))
    
    print(f"  取得した画像のバッチの形状: {images.shape}")
    print(f"  取得したターゲットのバッチの形状: {targets.shape}")
    
    # 最初の1枚を可視化
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(images[0].squeeze().numpy(), cmap='gray')
    axes[0].set_title('Sample Image from DataLoader')
    axes[1].imshow(targets[0].numpy(), cmap='hot')
    axes[1].set_title('Sample Target from DataLoader')
    plt.show()

    # (ここに今後、モデルの定義と学習ループが追加される)
    print("\nデータローダーの準備が完了しました。次はモデルの定義です。")


if __name__ == '__main__':
    # Windows環境でmultiprocessingを使う際の定型句
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()
    main()
