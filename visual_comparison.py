# improved_lung_extraction.py
import numpy as np
import SimpleITK as sitk
from scipy.ndimage import label, binary_closing, binary_opening, binary_fill_holes
from skimage.morphology import remove_small_objects, convex_hull_image
from skimage.segmentation import clear_border
import matplotlib.pyplot as plt

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