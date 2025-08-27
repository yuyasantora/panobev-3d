# inference.py (後処理機能付き)
import os
import sys
import yaml
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import zoom
# ★★★ ノイズ除去のためのDBSCANをインポート ★★★
from sklearn.cluster import DBSCAN

# --- パス設定 ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
plan_b_dir = os.path.join(parent_dir, '3d_e2e')
if os.path.isdir(plan_b_dir):
    sys.path.append(plan_b_dir)

from multislice.train5 import ViTMultiSliceBEV
import drr_generator as drr_gen
from build_dataset import extract_lung_mask

# --- 設定 ---
PATIENT_ID = 'LIDC-IDRI-0077'
MODEL_PATH = 'experiments_multislice/250827_1159_MultiSlice/best_model.pth'
CONFIG_PATH = 'config_multislice.yaml'
LIDC_IDRI_ROOT = 'C:/Users/ohhara/PanoBEV-3D/data/manifest-1752629384107/LIDC-IDRI'
RECONSTRUCTION_THRESHOLD = 0.65
MAX_POINTS_TO_PLOT = 50000

# ★★★ ノイズ除去を行う後処理関数 ★★★
def post_process_points(points, eps=5.0, min_samples=100):
    """
    DBSCANを使用して点群からノイズを除去し、最大のクラスター（肺）のみを返す
    """
    if len(points) == 0:
        return points
    
    # DBSCANでクラスタリング
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points)
    labels = clustering.labels_
    
    # -1はノイズとして分類された点
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)
    
    # ノイズ以外の最大のクラスターを見つける
    if len(counts) > 0:
        largest_cluster_label = unique_labels[np.argmax(counts)]
        return points[labels == largest_cluster_label]
    else:
        # ノイズしか見つからなかった場合は空を返す
        return np.array([])

def reconstruct_3d_from_multislice(multislice_bev, original_ct_sitk, threshold=0.5):
    # (この関数は変更なし)
    num_slices, H, W = multislice_bev.shape
    original_ct_array = sitk.GetArrayFromImage(original_ct_sitk)
    original_ct_shape = original_ct_array.shape
    binary_slices = (multislice_bev > threshold)
    reconstructed_mask = np.zeros(original_ct_shape, dtype=np.uint8)
    y_indices = np.where(np.any(original_ct_array, axis=(0, 2)))[0]
    if len(y_indices) == 0: y_min_idx, y_max_idx = 0, original_ct_shape[1]
    else: y_min_idx, y_max_idx = y_indices.min(), y_indices.max()
    y_bins_idx = np.linspace(y_min_idx, y_max_idx, num_slices + 1, dtype=int)
    for i in range(num_slices):
        slice_2d = binary_slices[i, :, :]
        if np.sum(slice_2d) == 0: continue
        target_shape = (original_ct_shape[0], original_ct_shape[2])
        zoom_factors = (target_shape[0] / slice_2d.shape[0], target_shape[1] / slice_2d.shape[1])
        resampled_slice = zoom(slice_2d, zoom_factors, order=0, mode='nearest')
        y_start_idx, y_end_idx = y_bins_idx[i], y_bins_idx[i+1]
        if y_start_idx < y_end_idx:
            reconstructed_mask[:, y_start_idx:y_end_idx, :] = np.expand_dims(resampled_slice, axis=1)
    return reconstructed_mask

def main():
    # ... (モデルロードと予測の部分は変更なし) ...
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f: config = yaml.safe_load(f)
    model = ViTMultiSliceBEV(vit_model_name=config['vit_model_name'], num_bev_slices=config['num_bev_slices'], output_size=config['resize_shape'][0]).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device)); model.eval()
    original_ct_sitk = drr_gen.find_best_ct_series(os.path.join(LIDC_IDRI_ROOT, PATIENT_ID))
    resampled_ct_sitk = drr_gen.resample_image_to_isotropic(original_ct_sitk)
    gt_lung_mask_3d = extract_lung_mask(sitk.GetArrayFromImage(resampled_ct_sitk))
    drr_images = drr_gen.create_drr_from_isotropic_ct(resampled_ct_sitk, views=['AP'])
    drr_array = sitk.GetArrayFromImage(drr_images['AP'])
    drr_sitk = sitk.GetImageFromArray(drr_array.astype(np.float32))
    ref_img = sitk.Image(config['resize_shape'], drr_sitk.GetPixelIDValue())
    resized_drr_sitk = sitk.Resample(drr_sitk, ref_img, sitk.Transform(), sitk.sitkLinear)
    resized_drr = sitk.GetArrayFromImage(resized_drr_sitk)
    m, M = resized_drr.min(), resized_drr.max();
    if M > m: resized_drr = (resized_drr - m) / (M - m)
    img_norm = (resized_drr - 0.5) / 0.5
    input_tensor = torch.from_numpy(img_norm).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        predicted_slices = torch.sigmoid(logits).squeeze().cpu().numpy()
    pred_lung_mask_3d = reconstruct_3d_from_multislice(predicted_slices, resampled_ct_sitk, threshold=RECONSTRUCTION_THRESHOLD)

    # --- 3D可視化 ---
    fig = plt.figure(figsize=(18, 9))
    
    # Ground Truth
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    z, y, x = np.where(gt_lung_mask_3d > 0)
    if len(x) > MAX_POINTS_TO_PLOT:
        indices = np.random.choice(len(x), MAX_POINTS_TO_PLOT, replace=False)
        x, y, z = x[indices], y[indices], z[indices]
    ax1.scatter(x, y, z, c='blue', s=1, alpha=0.2); ax1.set_title(f'Ground Truth ({len(x)} points)'); ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z'); ax1.view_init(elev=0, azim=-90)

    # Prediction
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    z_pred, y_pred, x_pred = np.where(pred_lung_mask_3d > 0)
    
    # ★★★ 予測された点群に後処理を適用 ★★★
    if len(x_pred) > 0:
        points_pred = np.vstack([x_pred, y_pred, z_pred]).T
        points_pred_cleaned = post_process_points(points_pred)
        
        if len(points_pred_cleaned) > 0:
            x_pred, y_pred, z_pred = points_pred_cleaned[:, 0], points_pred_cleaned[:, 1], points_pred_cleaned[:, 2]
        else: # ノイズ除去後に点が残らなかった場合
            x_pred, y_pred, z_pred = np.array([]), np.array([]), np.array([])

    if len(x_pred) > MAX_POINTS_TO_PLOT:
        indices = np.random.choice(len(x_pred), MAX_POINTS_TO_PLOT, replace=False)
        x_pred, y_pred, z_pred = x_pred[indices], y_pred[indices], z_pred[indices]
        
    if len(x_pred) > 0: ax2.scatter(x_pred, y_pred, z_pred, c='red', s=1, alpha=0.2)
    ax2.set_title(f'Prediction (Cleaned, {len(x_pred)} points)'); ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z'); ax2.view_init(elev=0, azim=-90)
    if len(x) > 0: ax2.set_xlim(ax1.get_xlim()); ax2.set_ylim(ax1.get_ylim()); ax2.set_zlim(ax1.get_zlim())
    plt.tight_layout(); plt.show()

if __name__ == '__main__':
    main()
