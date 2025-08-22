# visual_comparison_3d_v4.py
import os
import numpy as np
import torch
import yaml
import SimpleITK as sitk
from train4 import PanoBEVDataset3, HybridViTPanoBEV  # train4の新しいモデルを使用
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import label, binary_closing, binary_opening
import drr_generator as drr_gen
# improved_lung_extraction.py
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, binary_closing, binary_opening, binary_fill_holes
from skimage.morphology import remove_small_objects, convex_hull_image
from skimage.segmentation import clear_border

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

def extract_lungs_improved(ct_image, hu_threshold=-500):
    """改良版肺野抽出（より解剖学的に正確）"""
    ct_array = sitk.GetArrayFromImage(ct_image)  # [Z, Y, X]
    
    print(f"CT shape: {ct_array.shape}, HU range: [{ct_array.min():.1f}, {ct_array.max():.1f}]")
    
    # 1. 肺野の粗抽出（HU値ベース）
    lung_mask = (ct_array < hu_threshold) & (ct_array > -1000)
    print(f"Initial lung voxels: {np.sum(lung_mask)}")
    
    # 2. 体外領域（空気）を除去
    # 画像境界に接している大きな空気領域を除去
    lung_mask_clean = clear_border(lung_mask)
    print(f"After border cleaning: {np.sum(lung_mask_clean)}")
    
    # 3. 小さなノイズ除去
    lung_mask_clean = remove_small_objects(lung_mask_clean, min_size=1000)
    print(f"After small object removal: {np.sum(lung_mask_clean)}")
    
    # 4. モルフォロジー処理
    # 閉じる（小さな穴を埋める）
    lung_mask_clean = binary_closing(lung_mask_clean, structure=np.ones((3,3,3)))
    
    # 穴埋め（スライスごと）
    for z in range(lung_mask_clean.shape[0]):
        lung_mask_clean[z] = binary_fill_holes(lung_mask_clean[z])
    
    # 開く（小さな突起を除去）
    lung_mask_clean = binary_opening(lung_mask_clean, structure=np.ones((2,2,2)))
    
    print(f"After morphology: {np.sum(lung_mask_clean)}")
    
    # 5. 連結成分分析で左右肺を特定
    labeled_array, num_features = label(lung_mask_clean)
    print(f"Connected components found: {num_features}")
    
    if num_features == 0:
        return np.empty((0, 3))
    
    # 各成分のサイズと位置を分析
    components_info = []
    for i in range(1, num_features + 1):
        component_mask = (labeled_array == i)
        size = np.sum(component_mask)
        
        # 成分の重心
        z_coords, y_coords, x_coords = np.where(component_mask)
        centroid = [np.mean(x_coords), np.mean(y_coords), np.mean(z_coords)]
        
        components_info.append({
            'id': i,
            'size': size,
            'centroid': centroid,
            'mask': component_mask
        })
    
    # サイズでソート
    components_info.sort(key=lambda x: x['size'], reverse=True)
    
    # 肺らしい成分を選択（通常は上位2つが左右肺）
    min_lung_size = 50000  # より現実的なサイズ閾値
    lung_components = []
    
    for comp in components_info:
        if comp['size'] > min_lung_size and len(lung_components) < 2:
            lung_components.append(comp)
            print(f"Lung component {comp['id']}: size={comp['size']}, centroid={comp['centroid']}")
    
    if len(lung_components) == 0:
        print("No lung-like components found!")
        return np.empty((0, 3))
    
    # 最終マスク作成
    final_mask = np.zeros_like(lung_mask_clean)
    for comp in lung_components:
        final_mask |= comp['mask']
    
    print(f"Final lung voxels: {np.sum(final_mask)}")
    
    # 6. 点群生成（間引きして軽量化）
    z_indices, y_indices, x_indices = np.where(final_mask)
    
    # 間引き（表面重視）
    # エッジ検出で境界付近の点を優先的に保持
    edge_mask = np.zeros_like(final_mask)
    
    # 各スライスで境界を検出
    for z in range(final_mask.shape[0]):
        if np.any(final_mask[z]):
            from scipy.ndimage import binary_erosion
            eroded = binary_erosion(final_mask[z])
            edge_mask[z] = final_mask[z] & ~eroded
    
    # 境界点を優先的に選択
    edge_z, edge_y, edge_x = np.where(edge_mask)
    interior_z, interior_y, interior_x = np.where(final_mask & ~edge_mask)
    
    # 境界点を全て含め、内部点を間引き
    if len(interior_z) > 20000:  # 内部点を制限
        indices = np.random.choice(len(interior_z), 20000, replace=False)
        interior_z = interior_z[indices]
        interior_y = interior_y[indices]  
        interior_x = interior_x[indices]
    
    # 結合
    final_z = np.concatenate([edge_z, interior_z])
    final_y = np.concatenate([edge_y, interior_y])
    final_x = np.concatenate([edge_x, interior_x])
    
    print(f"Final points (edge + interior): {len(final_z)}")
    
    # 物理座標変換
    spacing = ct_image.GetSpacing()
    origin = ct_image.GetOrigin()
    
    x_coords = final_x * spacing[0] + origin[0]
    y_coords = final_y * spacing[1] + origin[1]
    z_coords = final_z * spacing[2] + origin[2]
    
    points_3d = np.stack([x_coords, y_coords, z_coords], axis=1)
    return points_3d

def visualize_extraction_process(ct_image, save_prefix="lung_extraction"):
    """肺野抽出プロセスの可視化"""
    ct_array = sitk.GetArrayFromImage(ct_image)
    
    # 中央スライスを選択
    mid_slice = ct_array.shape[0] // 2
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 原画像
    axes[0,0].imshow(ct_array[mid_slice], cmap='gray', vmin=-1000, vmax=500)
    axes[0,0].set_title('Original CT')
    
    # HU閾値適用
    lung_mask = (ct_array < -500) & (ct_array > -1000)
    axes[0,1].imshow(lung_mask[mid_slice], cmap='gray')
    axes[0,1].set_title('HU Threshold (-500)')
    
    # ボーダークリア
    lung_clean = clear_border(lung_mask)
    axes[0,2].imshow(lung_clean[mid_slice], cmap='gray')
    axes[0,2].set_title('Border Cleared')
    
    # モルフォロジー処理後
    lung_morph = binary_closing(lung_clean, structure=np.ones((3,3,3)))
    axes[1,0].imshow(lung_morph[mid_slice], cmap='gray')
    axes[1,0].set_title('After Morphology')
    
    # 連結成分
    labeled, _ = label(lung_morph)
    axes[1,1].imshow(labeled[mid_slice], cmap='nipy_spectral')
    axes[1,1].set_title('Connected Components')
    
    # 最終結果
    final_points = extract_lungs_improved(ct_image)
    if len(final_points) > 0:
        # 該当スライスの点を表示
        slice_tolerance = 5
        slice_points = final_points[
            np.abs(final_points[:, 2] - (mid_slice * ct_image.GetSpacing()[2] + ct_image.GetOrigin()[2])) < slice_tolerance
        ]
        
        axes[1,2].imshow(ct_array[mid_slice], cmap='gray', vmin=-1000, vmax=500)
        if len(slice_points) > 0:
            axes[1,2].scatter(slice_points[:, 0], slice_points[:, 1], c='red', s=1, alpha=0.7)
        axes[1,2].set_title(f'Final Result ({len(final_points)} points)')
    
    plt.tight_layout()
    plt.savefig(f'{save_prefix}_process.png', dpi=150, bbox_inches='tight')
    plt.show()

# visual_comparison_3d.py のextract_gt_lung_3d関数を置き換え
def extract_gt_lung_3d(ct_image, hu_threshold=-500):
    """改良版GT肺野抽出"""
    return extract_lungs_improved(ct_image, hu_threshold)

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

def post_process_predicted_points(pred_points, min_cluster_size=1000):
    """予測点群の後処理（GTと同等の品質向上）"""
    if len(pred_points) == 0:
        return pred_points
    
    from sklearn.cluster import DBSCAN
    
    # 1. クラスタリングで離散点を除去
    clustering = DBSCAN(eps=5.0, min_samples=50)
    labels = clustering.fit_predict(pred_points)
    
    # 2. 最大クラスタのみ保持（主要な肺野構造）
    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
    if len(unique_labels) == 0:
        return pred_points
    
    # 大きなクラスタを選択
    large_clusters = unique_labels[counts > min_cluster_size]
    if len(large_clusters) == 0:
        return pred_points
    
    mask = np.isin(labels, large_clusters)
    return pred_points[mask]


def visualize_gt_vs_prediction(patient_id, model_path, config_path, data_dir, lidc_root):
    """GT vs 予測の視覚比較（HybridViTPanoBEV用）"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # モデルロード
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # HybridViTPanoBEVモデルの初期化
    model = HybridViTPanoBEV(
        vit_model_name=config['vit_model_name'],
        output_size=config['resize_shape'][0]
    ).to(device)
    
    # チェックポイントのロード
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Best metrics: {checkpoint.get('metrics', 'unknown')}")
    else:
        model.load_state_dict(checkpoint)
        print("Loaded legacy checkpoint format")
    
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
    print(f"Running inference for patient {patient_id}")
    image, bev_cont, depth_tgt, bev_bin = val_dataset[target_idx]
    with torch.no_grad():
        image_batch = image.unsqueeze(0).to(device)
        with torch.cuda.amp.autocast():  # AMP対応
            bev_pred, depth_pred = model(image_batch)
        
        bev_pred = bev_pred.squeeze().cpu().numpy()
        depth_pred = depth_pred.squeeze().cpu().numpy()
    
    # 予測結果の基本統計量を表示
    print("\nPrediction statistics:")
    print(f"BEV range: [{bev_pred.min():.3f}, {bev_pred.max():.3f}]")
    print(f"Depth range: [{depth_pred.min():.3f}, {depth_pred.max():.3f}]")
    
    # 元のCT読み込み
    print(f"\nLoading original CT for {patient_id}...")
    ct_image = load_original_ct(patient_id, lidc_root)
    if ct_image is None:
        print(f"Could not load CT for {patient_id}")
        return
    
    # GT肺野3D抽出
    print("Extracting GT lung from CT...")
    gt_points = extract_gt_lung_3d(ct_image)
    
    # 予測3D生成
    print("Generating predicted 3D points...")
    pred_points = bev_depth_to_3d_points_improved(bev_pred, depth_pred, ct_image)
    print("Post-processing predicted points...")
    pred_points = post_process_predicted_points(pred_points)
    
    print(f"\nPoint cloud statistics:")
    print(f"GT points: {len(gt_points)}")
    print(f"Predicted points: {len(pred_points)}")
    
    # 結果の保存ディレクトリ作成
    save_dir = os.path.join(os.path.dirname(model_path), 'visualizations')
    os.makedirs(save_dir, exist_ok=True)
    
    # 可視化（3Dプロット）
    fig = plt.figure(figsize=(20, 6))
    
    # GT点群
    ax1 = fig.add_subplot(131, projection='3d')
    if len(gt_points) > 0:
        if len(gt_points) > 50000:
            indices = np.random.choice(len(gt_points), 50000, replace=False)
            gt_sample = gt_points[indices]
        else:
            gt_sample = gt_points
        
        ax1.scatter(gt_sample[:, 0], gt_sample[:, 1], gt_sample[:, 2], 
                   c='blue', s=0.1, alpha=0.6, label='Ground Truth')
    ax1.set_title(f'{patient_id}\nGround Truth Lung')
    ax1.set_xlabel('X (mm)'); ax1.set_ylabel('Y (mm)'); ax1.set_zlabel('Z (mm)')
    
    # 予測点群
    ax2 = fig.add_subplot(132, projection='3d')
    if len(pred_points) > 0:
        ax2.scatter(pred_points[:, 0], pred_points[:, 1], pred_points[:, 2], 
                   c='red', s=0.5, alpha=0.8, label='Predicted')
    ax2.set_title(f'{patient_id}\nPredicted Lung')
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
    
    ax3.set_title(f'{patient_id}\nGT vs Predicted')
    ax3.set_xlabel('X (mm)'); ax3.set_ylabel('Y (mm)'); ax3.set_zlabel('Z (mm)')
    ax3.legend()
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{patient_id}_gt_vs_pred_v4.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved visualization to: {save_path}")
    plt.show()
    
    # BEVとDepthの可視化も追加
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    
    axes[0,0].imshow(bev_cont.squeeze(), cmap='gray')
    axes[0,0].set_title('GT BEV')
    
    axes[0,1].imshow(bev_pred, cmap='gray')
    axes[0,1].set_title('Predicted BEV')
    
    axes[1,0].imshow(depth_tgt.squeeze(), cmap='viridis')
    axes[1,0].set_title('GT Depth')
    
    axes[1,1].imshow(depth_pred, cmap='viridis')
    axes[1,1].set_title('Predicted Depth')
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{patient_id}_bev_depth_v4.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved BEV/Depth visualization to: {save_path}")
    plt.show()
    
    return gt_points, pred_points

if __name__ == '__main__':
    patient_id = 'LIDC-IDRI-0005'
    # train4.pyの新しいモデルのパスを指定
    model_path = 'experiments_lung/250822_1443_lungBEV_lr5e-06/best_model.pth'
    config_path = 'config_lung.yaml'
    data_dir = 'dataset_lung'
    lidc_root = 'data/manifest-1752629384107/LIDC-IDRI/'
    
    gt_points, pred_points = visualize_gt_vs_prediction(
        patient_id, model_path, config_path, data_dir, lidc_root
    )