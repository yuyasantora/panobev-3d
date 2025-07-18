import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# train.pyからモデル定義とデータローダー定義をインポート
from train import ViTForBEVGeneration, BEVDataset
from torch.utils.data import DataLoader

# --- 設定項目 ---
# 学習済みモデルの重みファイルへのパス
MODEL_PATH = "bev_generation_model_91.pth" 
# データセットのディレクトリ
DATASET_DIR = 'dataset'
# モデル定義 (train.pyと一致させる)
RESIZE_SHAPE = (224, 224)
VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'

def inference():
    """
    推論を実行し、結果を可視化するメイン関数
    """
    # --- 1. 準備 ---
    # デバイスの指定
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- 2. モデルのロード ---
    print(f"学習済みモデルを '{MODEL_PATH}' からロードしています...")
    # まず、モデルの「器」を定義
    model = ViTForBEVGeneration(vit_model_name=VIT_MODEL_NAME, output_shape=RESIZE_SHAPE)
    # 次に、学習済みの重みをロード
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    # モデルをデバイスに送る
    model.to(device)
    # ★★★ モデルを評価モードに設定 ★★★
    model.eval() 
    print("モデルのロードが完了しました。")

    # --- 3. データローダーの準備 ---
    # 推論時にはデータをシャッフルする必要はない
    dataset = BEVDataset(dataset_dir=DATASET_DIR, resize_shape=RESIZE_SHAPE)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True) # バッチサイズ1で1枚ずつ処理

    # --- 4. 推論と可視化 ---
    print("\n推論を開始します... (ウィンドウを閉じると次の画像が表示されます)")
    
    # torch.no_grad()ブロック内で推論を行うことで、不要な勾配計算を無効化し、メモリを節約
    with torch.no_grad():
        for i, (image, target) in enumerate(dataloader):
            image = image.to(device)
            target = target.to(device)

            # モデルによる予測
            predicted_bev = model(image)

            # --- 5. 結果をCPUに戻し、NumPy配列に変換して可視化 ---
            image_np = image.squeeze().cpu().numpy()
            target_np = target.squeeze().cpu().numpy()
            predicted_np = predicted_bev.squeeze().cpu().numpy()

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title('Input DRR')
            axes[0].axis('off')

            axes[1].imshow(target_np, cmap='hot')
            axes[1].set_title('Ground Truth BEV')
            axes[1].axis('off')

            axes[2].imshow(predicted_np, cmap='hot')
            axes[2].set_title('Predicted BEV')
            axes[2].axis('off')
            
            plt.suptitle(f"Inference Result #{i+1}")
            plt.show()

            # (例として5枚表示したら終了)
            if i >= 4:
                break

if __name__ == '__main__':
    inference()
