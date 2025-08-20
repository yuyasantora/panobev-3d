# visual_comparison_3d.py
import os
import numpy as np
import torch
import yaml
import SimpleITK as sitk
from train3 import ViTPanoBEV3, PanoBEVDataset3
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, binary_closing, binary_opening
import drr_generator as drr_gen

def load_original_ct(patient_id, lidc_root='data/manifest-1752629384107/LIDC-IDRI/'):
    """元のCTデータを読み込み"""
    patient_dir = os.path.join(lidc_root, patient_id)
    
    # CTスキャンを探す
    dicom_series = {}
    for root, _, files in os.walk(patient_dir):
        if any(f.lower().endswith('.dcm') for f in files):
            reader = sitk.ImageSeriesReader()
            try:
                dicom_names = reader.GetGDCMSeriesFileNames(root)
                if dicom_names:
                    dicom_series[root] = len(dicom_names)
            except RuntimeError:
                continue
    
    if not dicom_series:
        return None
    
    # 最もスライス数の多いシリーズを使用
    best_series_path = max(dicom_series, key=dicom_series.get)
    
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(best_series_path)
    reader.SetFileNames(dicom_names)
    original_image = reader.Execute()
    
    # 等方性リサンプリング
    resampled_image = drr_gen.resample_image_to_isotropic(original_image)
    return resampled_image

def extract_gt_lung_3d(ct_image, hu_threshold=-500):
    """GTのCTから肺野3D点群を抽出（改良版）"""
    ct_array = sitk.GetArrayFromImage(ct_image)  # [Z, Y, X]
    
    # 肺野マスク作成
    lung_mask = (ct_array < hu_threshold) & (ct_array > -1000)
    
    # モルフォロジー処理で小さなノイズ除去
    lung_mask = binary_closing(lung_mask, structure=np.ones((3,3,3)))
    lung_mask = binary_opening(lung_mask, structure=np.ones((2,2,2)))
    
    # 連結成分分析で最大2つの領域（左右肺）のみ保持
    labeled_array, num_features = label(lung_mask)
    if num_features > 0:
        # 各領域のサイズを計算
        region_sizes = []
        for i in range(1, num_features + 1):
            size = np.sum(labeled_array == i)
            region_sizes.append((size, i))
        
        # サイズでソートして上位2つを保持
        region_sizes.sort(reverse=True)
        keep_regions = [region_sizes[i][1] for i in range(min(2, len(region_sizes)))]
        
        # 小さな領域を除去
        min_size = 10000  # 最小領域サイズ
        final_mask = np.zeros_like(lung_mask)
        for region_id in keep_regions:
            if region_sizes[region_id-1][0] > min_size:
                final_mask |= (labeled_array == region_id)
        
        lung_mask = final_mask
    
    # マスク内の座標取得
    z_indices, y_indices, x_indices = np.where(lung_mask)
    
    if len(z_indices) == 0:
        return np.empty((0, 3))
    
    # 画像座標→物理座標変換
    spacing = ct_image.GetSpacing()  # (x, y, z)
    origin = ct_image.GetOrigin()
    
    x_coords = x_indices * spacing[0] + origin[0]
    y_coords = y_indices * spacing[1] + origin[1]
    z_coords = z_indices * spacing[2] + origin[2]
    
    points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    return points_3d

def bev_depth_to_3d_points_improved(bev_pred, depth_pred, ct_image, threshold=0.3):
    """改良版：CTの物理座標系に合わせた3D点群生成"""
    H, W = bev_pred.shape
    
    # BEVで肺野領域をマスク
    mask = bev_pred > threshold
    z_indices, x_indices = np.where(mask)
    
    if len(x_indices) == 0:
        return np.empty((0, 3))
    
    # CTの物理パラメータを取得
    spacing = ct_image.GetSpacing()
    origin = ct_image.GetOrigin()
    size = ct_image.GetSize()
    
    # BEV座標→CT物理座標への変換
    # BEVは(224, 224)でCTのXZ平面に対応
    ct_x_range = size[0] * spacing[0]
    ct_z_range = size[2] * spacing[2]
    
    x_coords = (x_indices / W) * ct_x_range + origin[0]
    z_coords = (z_indices / H) * ct_z_range + origin[2]
    
    # Depthから高さ（Y座標）を取得
    y_depths = depth_pred[z_indices, x_indices]
    # Depth正規化を物理座標に変換
    ct_y_range = size[1] * spacing[1]
    y_coords = y_depths * ct_y_range + origin[1]
    
    points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    return points_3d

def visualize_gt_vs_prediction(patient_id, model_path, config_path, data_dir, lidc_root):
    """GT vs 予測の視覚比較"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # モデルロード
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    model = ViTPanoBEV3(config['vit_model_name']).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
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
    
    # 該当患者のデータを探す
    target_idx = None
    for i, filename in enumerate(val_dataset.image_files):
        if patient_id in filename:
            target_idx = i
            break
    
    if target_idx is None:
        print(f"Patient {patient_id} not found in validation set")
        return
    
    # 予測実行
    image, bev_cont, depth_tgt, bev_bin = val_dataset[target_idx]
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        bev_pred, depth_pred = model(image_batch)
        
        bev_pred = bev_pred.squeeze().cpu().numpy()
        depth_pred = depth_pred.squeeze().cpu().numpy()
    
    # 元のCT読み込み
    print(f"Loading original CT for {patient_id}...")
    ct_image = load_original_ct(patient_id, lidc_root)
    if ct_image is None:
        print(f"Could not load CT for {patient_id}")
        return
    
    # GT肺野3D抽出
    print("Extracting GT lung from CT...")
    gt_points = extract_gt_lung_3d(ct_image)
    
    # 予測3D生成
    print("Generating predicted 3D...")
    pred_points = bev_depth_to_3d_points_improved(bev_pred, depth_pred, ct_image)
    
    print(f"GT points: {len(gt_points)}, Predicted points: {len(pred_points)}")
    
    # 可視化
    fig = plt.figure(figsize=(20, 6))
    
    # GT点群
    ax1 = fig.add_subplot(131, projection='3d')
    if len(gt_points) > 0:
        # サンプリング（表示用）
        if len(gt_points) > 50000:
            indices = np.random.choice(len(gt_points), 50000, replace=False)
            gt_sample = gt_points[indices]
        else:
            gt_sample = gt_points
        
        ax1.scatter(gt_sample[:, 0], gt_sample[:, 1], gt_sample[:, 2], 
                   c='blue', s=0.1, alpha=0.6, label='Ground Truth')
    ax1.set_title(f'{patient_id} - Ground Truth Lung')
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    
    # 予測点群
    ax2 = fig.add_subplot(132, projection='3d')
    if len(pred_points) > 0:
        ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', s=0.5, alpha=0.8, label='Predicted')
    ax2.set_title(f'{patient_id} - Predicted Lung')
    ax2.set_xlabel('X (mm)'); ax2.set_ylabel('Y (mm)'); ax2.set_zlabel('Z (mm)')
    
    # 重ね合わせ
    ax3 = fig.add_subplot(133, projection='3d')
    if len(gt_points) > 0:
        if len(gt_points) > 30000:
            indices = np.random.choice(len(gt_points), 30000, replace=False)
            gt_sample = gt_points[indices]
        else:
            gt_sample = gt_points
        ax3.scatter(gt_sample[:, 0], gt_sample[:, 1], gt_sample[:, 2], 
                   c='blue', s=0.1, alpha=0.3, label='Ground Truth')
    
    if len(pred_points) > 0:
        ax3.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', s=0.5, alpha=0.7, label='Predicted')
    
    ax3.set_title(f'{patient_id} - GT vs Predicted')
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)'); ax3.set_zlabel('Z (mm)')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig(f'{patient_id}_gt_vs_pred.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return gt_points, pred_points

if __name__ == '__main__':
    patient_id = 'LIDC-IDRI-0005'
    model_path = 'experiments_lung/250820_0957_lungBEV_lr5e-06/best_model.pth'
    config_path = 'config_lung.yaml'
    data_dir = 'dataset_lung'
    lidc_root = 'data/manifest-1752629384107/LIDC-IDRI/'  # 調整してください
    
    gt_points, pred_points = visualize_gt_vs_prediction(
        patient_id, model_path, config_path, data_dir, lidc_root
    )