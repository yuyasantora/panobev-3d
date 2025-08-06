import os
import torch
import numpy as np
import pyvista as pv
import SimpleITK as sitk
from tqdm import tqdm

# train.pyからモデル定義とデータセット定義をインポート
from train import ViTPanoBEV, PanoBEVDataset 
from torch.utils.data import DataLoader

# --- 設定項目 ---
MODEL_PATH = "panobev_model_best.pth" 
DATASET_DIR = 'dataset/test'
RESIZE_SHAPE = (224, 224)
VIT_MODEL_NAME = 'google/vit-base-patch16-224-in21k'
# ★★★ 修正点: 閾値を0.5から0.1に下げて、弱い反応も拾えるようにする ★★★
PROBABILITY_THRESHOLD = 0.1 

def reconstruct_3d_from_bev(output_filename="3d_reconstruction.png"):
    """
    モデルのBEV予測結果を3D空間に逆射影し、可視化する
    """
    # --- 1. モデルとデータローダーの準備 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用デバイス: {device}")

    if not os.path.exists(MODEL_PATH):
        print(f"エラー: モデルファイル '{MODEL_PATH}' が見つかりません。")
        return
        
    model = ViTPanoBEV(vit_model_name=VIT_MODEL_NAME).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    dataset = PanoBEVDataset(dataset_dir=DATASET_DIR, resize_shape=RESIZE_SHAPE)
    # テストセットから最初のデータだけ使う
    if len(dataset) == 0:
        print(f"エラー: テストデータが '{DATASET_DIR}' に見つかりません。")
        return
    
    image, _, _ = dataset[0] # 最初のテストデータを取得
    image = image.unsqueeze(0).to(device) # バッチ次元を追加してデバイスへ

    # --- 2. 推論を実行し、予測BEVマップを取得 ---
    print("推論を実行してBEVマップを生成...")
    with torch.no_grad():
        predicted_bev_logits, _ = model(image)
        predicted_bev = torch.sigmoid(predicted_bev_logits)
    
    bev_np = predicted_bev.squeeze().cpu().numpy()
    
    # 閾値以下の値を0にする
    bev_np[bev_np < PROBABILITY_THRESHOLD] = 0.0

    # --- 3. BEVマップを3D空間に逆射影 ---
    print("BEVマップを3D空間に逆射影...")
    # 元のDRRのサイズに合わせて奥行き方向の次元を追加
    # ここでは仮に奥行きも同じサイズとする
    depth_dim = RESIZE_SHAPE[0] 
    
    # (高さ, 幅) -> (高さ, 奥行き, 幅) に拡張
    # np.newaxisで新しい軸を追加し、np.repeatで値をコピー
    bev_3d_np = np.repeat(bev_np[:, np.newaxis, :], depth_dim, axis=1)

    # --- 4. PyVistaで3Dボリュームとして可視化 ---
    print(f"3D再構成をレンダリングし、'{output_filename}' に保存します...")
    grid = pv.ImageData()
    grid.dimensions = np.array(bev_3d_np.shape)[::-1]
    grid.spacing = (1, 1, 1)  # 等方的なボクセルと仮定
    grid.point_data["probability"] = bev_3d_np.flatten(order="F")

    plotter = pv.Plotter(off_screen=True)
    
    # add_volumeのopacity引数で、確率に応じた不透明度を設定
    # 確率0は透明、1に近づくにつれて不透明になるように設定
    opacity_map = [0, 0.2, 0.4, 0.6, 0.8] 
    plotter.add_volume(
        grid, 
        cmap="hot", 
        clim=[0.0, 1.0], # 確率は0-1の範囲
        opacity=opacity_map,
        shade=True
    )
    
    plotter.camera_position = 'iso'
    plotter.screenshot(output_filename)
    plotter.close()

    print(f"\n保存完了！ '{output_filename}' を確認してください。")

if __name__ == '__main__':
    reconstruct_3d_from_bev() 