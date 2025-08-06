import os
import yaml
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# train.pyからモデルとデータセットのクラス定義をインポート
from train import ViTPanoBEV, PanoBEVDataset

def inference(exp_dir: str):
    """
    指定された実験ディレクトリの結果を用いて、推論と可視化を行う。
    """
    # --- 1. パスの検証と設定ファイルの読み込み ---
    if not os.path.isdir(exp_dir):
        print(f"エラー: 実験ディレクトリが見つかりません: {exp_dir}")
        return

    config_path = os.path.join(exp_dir, 'config.yaml')
    model_path = os.path.join(exp_dir, 'best_model.pth')

    if not os.path.exists(config_path) or not os.path.exists(model_path):
        print(f"エラー: {exp_dir} 内に 'config.yaml' または 'best_model.pth' が見つかりません。")
        return

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"--- '{exp_dir}' の設定で推論を開始 ---")

    # --- 2. 準備 (デバイス、モデル) ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ViTPanoBEV(vit_model_name=config['vit_model_name'])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"モデル '{model_path}' をロードしました。")

    # --- 3. データローダーの準備 ---
    test_data_dir = os.path.join(config['data_dir'], 'test')
    if not os.path.isdir(test_data_dir):
        print(f"エラー: テストデータディレクトリが見つかりません: {test_data_dir}")
        return

    dataset = PanoBEVDataset(
        dataset_dir=test_data_dir, 
        resize_shape=config['resize_shape'],
        augmentation_config={'use_augmentation': False} # 推論時に増強はしない
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"テストデータ {len(dataset)} 件を '{test_data_dir}' から読み込みました。")

    # --- 4. 推論と可視化 ---
    with torch.no_grad():
        for i, (image, bev_target, depth_target) in enumerate(dataloader):
            image, bev_target, depth_target = image.to(device), bev_target.to(device), depth_target.to(device)

            predicted_bev_logits, predicted_depth = model(image)
            predicted_bev = torch.sigmoid(predicted_bev_logits)

            # 結果をCPUに戻してNumPy配列に変換
            image_np = image.squeeze().cpu().numpy()
            bev_target_np = bev_target.squeeze().cpu().numpy()
            depth_target_np = depth_target.squeeze().cpu().numpy()
            predicted_bev_np = predicted_bev.squeeze().cpu().numpy()
            predicted_depth_np = predicted_depth.squeeze().cpu().numpy()

            # 可視化
            fig, axes = plt.subplots(1, 5, figsize=(25, 5))
            
            axes[0].imshow(image_np, cmap='gray'); axes[0].set_title('Input DRR'); axes[0].axis('off')
            axes[1].imshow(predicted_bev_np, cmap='hot', vmin=0, vmax=1); axes[1].set_title('Predicted BEV'); axes[1].axis('off')
            axes[2].imshow(bev_target_np, cmap='hot', vmin=0, vmax=1); axes[2].set_title('Ground Truth BEV'); axes[2].axis('off')
            
            im_pred = axes[3].imshow(predicted_depth_np, cmap='viridis'); axes[3].set_title('Predicted Depth'); axes[3].axis('off')
            fig.colorbar(im_pred, ax=axes[3])
            
            im_gt = axes[4].imshow(depth_target_np, cmap='viridis'); axes[4].set_title('Ground Truth Depth'); axes[4].axis('off')
            fig.colorbar(im_gt, ax=axes[4])
            
            plt.suptitle(f"Test Result #{i+1} from {os.path.basename(exp_dir)}")
            plt.tight_layout()
            plt.show()

            if i >= 4: # 最初の5件を表示して終了
                break

if __name__ == '__main__':
    # コマンドライン引数をパース
    parser = argparse.ArgumentParser(description="PanoBEV-3D Inference Script")
    parser.add_argument(
        "experiment_directory", 
        type=str, 
        help="推論対象の実験結果が保存されているディレクトリのパス (例: experiments/240523_1030_...)"
    )
    args = parser.parse_args()
    
    inference(args.experiment_directory)
