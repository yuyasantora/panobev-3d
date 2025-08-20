# reconstruct_3d_eval.py
import os
import numpy as np
import torch
import yaml
import SimpleITK as sitk
from train3 import ViTPanoBEV3, PanoBEVDataset3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import cdist

def load_model(model_path, config_path, device):
    """学習済みモデルをロード"""
    with open(config_path, 'r', encoding='utf-8') as f:  # encoding='utf-8' を追加
        config = yaml.safe_load(f)
    
    model = ViTPanoBEV3(config['vit_model_name']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, config

def bev_depth_to_3d_points(bev_pred, depth_pred, threshold=0.5, spacing=(1.0, 1.0)):
    """
    予測BEV+Depthから3D点群を生成
    
    Args:
        bev_pred: [H, W] BEV予測（sigmoid後）
        depth_pred: [H, W] Depth予測（正規化済み）
        threshold: BEV二値化閾値
        spacing: (x_spacing, z_spacing) 物理空間での画素間隔
    
    Returns:
        points_3d: [N, 3] (X, Y, Z)座標の点群
    """
    H, W = bev_pred.shape
    
    # BEVで肺野領域をマスク
    mask = bev_pred > threshold
    
    # マスク内の座標を取得
    z_indices, x_indices = np.where(mask)  # numpy行列は(row, col) = (Z, X)
    
    if len(x_indices) == 0:
        return np.empty((0, 3))
    
    # 画像座標→物理座標
    x_coords = x_indices * spacing[0]  # X方向
    z_coords = z_indices * spacing[1]  # Z方向（奥行き）
    
    # Depthから高さ（Y座標）を取得
    y_coords = depth_pred[z_indices, x_indices]  # 対応するDepth値
    
    # Y座標を物理単位にスケール（例：0-1 → 0-300mm）
    y_coords = y_coords * 300.0  # 適宜調整
    
    points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    return points_3d

def ct_to_3d_points(ct_image, hu_threshold=-500):
    """
    GT CTから肺野の3D点群を生成
    
    Args:
        ct_image: SimpleITK Image（等方性リサンプル済み）
        hu_threshold: 肺野抽出のHU閾値
    
    Returns:
        points_3d: [N, 3] 物理座標の点群
    """
    ct_array = sitk.GetArrayFromImage(ct_image)  # [Z, Y, X]
    
    # 肺野マスク作成
    lung_mask = (ct_array < hu_threshold) & (ct_array > -1000)
    
    # 連結成分分析で大きな領域のみ保持（簡易版）
    # より厳密にはscipy.ndimage.label等を使用
    
    # マスク内の座標取得
    z_indices, y_indices, x_indices = np.where(lung_mask)
    
    # 画像座標→物理座標変換
    spacing = ct_image.GetSpacing()  # (x, y, z)
    origin = ct_image.GetOrigin()
    
    x_coords = x_indices * spacing[0] + origin[0]
    y_coords = y_indices * spacing[1] + origin[1]
    z_coords = z_indices * spacing[2] + origin[2]
    
    points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    return points_3d

def calculate_chamfer_distance(points1, points2, max_points=5000):
    """
    2つの点群間のChamfer距離を計算
    
    Args:
        points1, points2: [N, 3], [M, 3] 点群
        max_points: 計算高速化のための点数制限
    
    Returns:
        chamfer_dist: Chamfer距離
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # 点数制限（ランダムサンプリング）
    if len(points1) > max_points:
        indices = np.random.choice(len(points1), max_points, replace=False)
        points1 = points1[indices]
    if len(points2) > max_points:
        indices = np.random.choice(len(points2), max_points, replace=False)
        points2 = points2[indices]
    
    # 最近傍距離計算
    dist1 = cdist(points1, points2)  # [N, M]
    dist2 = cdist(points2, points1)  # [M, N]
    
    # Chamfer距離
    chamfer = np.mean(np.min(dist1, axis=1)) + np.mean(np.min(dist2, axis=1))
    return chamfer

def calculate_iou_3d(points1, points2, voxel_size=2.0):
    """
    ボクセル化による3D IoU計算
    
    Args:
        points1, points2: [N, 3], [M, 3] 点群
        voxel_size: ボクセルサイズ（mm）
    
    Returns:
        iou: 3D IoU
    """
    if len(points1) == 0 and len(points2) == 0:
        return 1.0
    if len(points1) == 0 or len(points2) == 0:
        return 0.0
    
    # 両点群の範囲を取得
    all_points = np.vstack([points1, points2])
    min_coords = np.min(all_points, axis=0)
    max_coords = np.max(all_points, axis=0)
    
    # ボクセルグリッド作成
    grid_size = ((max_coords - min_coords) / voxel_size).astype(int) + 1
    
    def points_to_voxel(points):
        voxel_coords = ((points - min_coords) / voxel_size).astype(int)
        # 範囲外クリップ
        voxel_coords = np.clip(voxel_coords, 0, grid_size - 1)
        
        voxel_grid = np.zeros(grid_size, dtype=bool)
        voxel_grid[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = True
        return voxel_grid
    
    voxel1 = points_to_voxel(points1)
    voxel2 = points_to_voxel(points2)
    
    intersection = np.sum(voxel1 & voxel2)
    union = np.sum(voxel1 | voxel2)
    
    if union == 0:
        return 1.0
    return intersection / union

def visualize_3d_comparison(pred_points, gt_points, save_path=None):
    """3D点群の比較可視化"""
    fig = plt.figure(figsize=(15, 5))
    
    # 予測点群
    ax1 = fig.add_subplot(131, projection='3d')
    if len(pred_points) > 0:
        ax1.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', s=0.1, alpha=0.6, label='Predicted')
    ax1.set_title('Predicted 3D Points')
    ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
    
    # GT点群
    ax2 = fig.add_subplot(132, projection='3d')
    if len(gt_points) > 0:
        ax2.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                   c='blue', s=0.1, alpha=0.6, label='Ground Truth')
    ax2.set_title('Ground Truth 3D Points')
    ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
    
    # 重ね合わせ
    ax3 = fig.add_subplot(133, projection='3d')
    if len(pred_points) > 0:
        ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', s=0.1, alpha=0.6, label='Predicted')
    if len(gt_points) > 0:
        ax3.scatter(gt_points[:, 0], gt_points[:, 1], gt_points[:, 2], 
                   c='blue', s=0.1, alpha=0.6, label='Ground Truth')
    ax3.set_title('Overlay')
    ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
    ax3.legend()
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

def evaluate_3d_reconstruction(model_path, config_path, data_dir, output_dir):
    """3D再構成の包括評価"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルロード
    model, config = load_model(model_path, config_path, device)
    
    # データセット準備
    bev_cfg = {
        'bev_target_mode': config.get('bev_target_mode', 'binary'),
        'bev_gaussian_sigma': config.get('bev_gaussian_sigma', 2.0),
        'normalize_input': True
    }
    
    val_dataset = PanoBEVDataset3(
        dataset_dir=os.path.join(data_dir, 'val'),
        resize_shape=config['resize_shape'],
        augmentation_config={'use_augmentation': False},
        bev_cfg=bev_cfg
    )
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(min(10, len(val_dataset))):  # 最初の10サンプル評価
        # データ取得
        image, bev_cont, depth_tgt, bev_bin = val_dataset[i]
        patient_id = val_dataset.image_files[i].split('_')[0]
        
        # 推論
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            bev_pred, depth_pred = model(image_batch)
            
            bev_pred = bev_pred.squeeze().cpu().numpy()    # [H, W]
            depth_pred = depth_pred.squeeze().cpu().numpy() # [H, W]
        
        # 3D点群生成
        pred_points = bev_depth_to_3d_points(bev_pred, depth_pred, threshold=0.3)
        
        # GT CT読み込み（実際のパスに要調整）
        # この部分は元のCTデータへのパスが必要
        print(f"Patient {patient_id}: Predicted points = {len(pred_points)}")
        
        # 可視化保存
        if len(pred_points) > 0:
            fig = plt.figure(figsize=(10, 8))
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                      c='red', s=0.5, alpha=0.6)
            ax.set_title(f'Patient {patient_id} - Predicted 3D Lung')
            ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
            plt.savefig(os.path.join(output_dir, f'{patient_id}_3d.png'), 
                       dpi=150, bbox_inches='tight')
            plt.close()
        
        results.append({
            'patient_id': patient_id,
            'pred_points': len(pred_points),
            'bev_dice': np.mean((bev_pred > 0.3) == (bev_bin.squeeze().numpy() > 0.5))
        })
    
    # 結果出力
    print("\n=== 3D Reconstruction Evaluation ===")
    for result in results:
        print(f"Patient {result['patient_id']}: "
              f"3D points = {result['pred_points']}, "
              f"BEV Dice = {result['bev_dice']:.4f}")

if __name__ == '__main__':
    model_path = 'experiments_lung/250820_0957_lungBEV_lr5e-06/best_model.pth'  # 要調整
    config_path = 'config_lung.yaml'
    data_dir = 'dataset_lung'
    output_dir = 'evaluation_3d'
    
    evaluate_3d_reconstruction(model_path, config_path, data_dir, output_dir)
    
    


