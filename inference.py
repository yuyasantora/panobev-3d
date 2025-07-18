import os
import numpy as np
import torch
import matplotlib.pyplot as plt

# train.pyから新しいモデルとデータセット定義をインポート
from train import ViTPanoBEV, PanoBEVDataset 
from torch.utils.data import DataLoader

# --- 設定項目 ---
# 学習済みモデルの重みファイルへのパス
MODEL_PATH = "panobev_model_final.pth" 
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    # --- 2. モデルのロード ---
    print(f"学習済みモデルを '{MODEL_PATH}' からロードしています...")
    # ★ 新しいモデルの器を定義
    model = ViTPanoBEV(vit_model_name=VIT_MODEL_NAME)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval() # モデルを評価モードに設定
    print("モデルのロードが完了しました。")

    # --- 3. データローダーの準備 ---
    # ★ 新しいデータセットを使用
    dataset = PanoBEVDataset(dataset_dir=DATASET_DIR, resize_shape=RESIZE_SHAPE)
    # 推論なのでシャッフルはFalseでOK
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False) 

    # --- 4. 推論と可視化 ---
    print("\n推論を開始します... (ウィンドウを閉じると次の画像が表示されます)")
    
    with torch.no_grad():
        # ★ データローダーから3つの値を受け取る
        for i, (image, bev_target, depth_target) in enumerate(dataloader):
            image = image.to(device)
            bev_target = bev_target.to(device)
            depth_target = depth_target.to(device)

            # ★ モデルから2つの予測結果を受け取る
            predicted_bev_logits, predicted_depth = model(image)
            
            # BEVの予測を確率に変換 (Sigmoidを適用)
            predicted_bev = torch.sigmoid(predicted_bev_logits)

            # --- 5. 結果をCPUに戻し、NumPy配列に変換して可視化 ---
            image_np = image.squeeze().cpu().numpy()
            bev_target_np = bev_target.squeeze().cpu().numpy()
            depth_target_np = depth_target.squeeze().cpu().numpy()
            predicted_bev_np = predicted_bev.squeeze().cpu().numpy()
            predicted_depth_np = predicted_depth.squeeze().cpu().numpy()

            # ★ 5つのプロットを作成
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            
            axes[0].imshow(image_np, cmap='gray')
            axes[0].set_title('Input DRR')
            axes[0].axis('off')

            axes[1].imshow(predicted_bev_np, cmap='hot', vmin=0, vmax=1)
            axes[1].set_title('Predicted BEV')
            axes[1].axis('off')

            axes[2].imshow(bev_target_np, cmap='hot', vmin=0, vmax=1)
            axes[2].set_title('Ground Truth BEV')
            axes[2].axis('off')

            axes[3].imshow(predicted_depth_np, cmap='viridis')
            axes[3].set_title('Predicted Depth')
            axes[3].axis('off')
            
            axes[4].imshow(depth_target_np, cmap='viridis')
            axes[4].set_title('Ground Truth Depth')
            axes[4].axis('off')
            
            plt.suptitle(f"Inference Result #{i+1}")
            plt.tight_layout()
            plt.show()

            # 5枚表示したら終了
            if i >= 4:
                break

if __name__ == '__main__':
    inference()
