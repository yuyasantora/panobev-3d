# build_dataset_multislice.py
import os
import SimpleITK as sitk
import numpy as np
import drr_generator as drr_gen
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

# --- 設定 ---
LIDC_IDRI_ROOT = 'data/manifest-1752629384107/LIDC-IDRI/'
OUTPUT_DIR = 'dataset_lung_multislice'  # 新しいデータセット用のディレクトリ
NUM_SLICES = 16  # Y軸（高さ）の分割数
HU_THRESHOLD = -500  # 肺野を抽出するためのHU値の閾値

# --- 関数 ---

def extract_lung_mask(ct_array, hu_threshold=-500):
    """3D CT配列から肺野のバイナリマスクを抽出する"""
    from scipy.ndimage import label, binary_fill_holes
    from skimage.morphology import remove_small_objects, binary_closing
    from skimage.segmentation import clear_border

    # 1. 閾値処理で肺の候補領域を抽出
    binary_mask = (ct_array < hu_threshold) & (ct_array > -1000)

    # 2. 境界に接している空気領域（体外）を除去
    cleared_mask = clear_border(binary_mask)

    # 3. モルフォロジー処理でマスクを整形
    closed_mask = binary_closing(cleared_mask, structure=np.ones((5,5,5)))

    # 4. 穴埋め
    filled_mask = np.zeros_like(closed_mask)
    for i in range(filled_mask.shape[0]):
        filled_mask[i, :, :] = binary_fill_holes(closed_mask[i, :, :])

    # 5. 小さな連結成分を除去
    # min_sizeは画像の解像度に応じて調整が必要
    min_size = int(np.prod(filled_mask.shape) * 0.001) 
    cleaned_mask = remove_small_objects(filled_mask, min_size=min_size)

    # 6. 連結成分分析で最大の2つの領域（左右の肺）を特定
    labeled_mask, num_labels = label(cleaned_mask)
    if num_labels > 1:
        label_sizes = np.bincount(labeled_mask.ravel())[1:]
        top_two_labels = np.argsort(label_sizes)[-2:] + 1
        final_mask = np.isin(labeled_mask, top_two_labels)
    else:
        final_mask = cleaned_mask

    return final_mask.astype(np.uint8)

def create_multislice_bev(lung_mask_3d, num_slices=16):
    """
    3D肺マスクをY軸（高さ）方向に分割し、マルチスライスのBEVを生成する。
    入力: lung_mask_3d (Z, Y, X)
    出力: multislice_bev (num_slices, Z, X)
    """
    if lung_mask_3d.sum() == 0:
        return np.zeros((num_slices, lung_mask_3d.shape[0], lung_mask_3d.shape[2]), dtype=np.uint8)

    # 肺が存在するY軸の範囲を特定
    y_indices = np.where(np.any(lung_mask_3d, axis=(0, 2)))[0]
    if len(y_indices) == 0:
        return np.zeros((num_slices, lung_mask_3d.shape[0], lung_mask_3d.shape[2]), dtype=np.uint8)
        
    y_min, y_max = y_indices.min(), y_indices.max()

    # Y軸をnum_slices個の区間に分割
    y_bins = np.linspace(y_min, y_max + 1, num_slices + 1, dtype=int)

    bev_slices = []
    for i in range(num_slices):
        # i番目の区間に対応するYスライスを取得
        y_start, y_end = y_bins[i], y_bins[i+1]
        y_mask_slice = lung_mask_3d[:, y_start:y_end, :]

        # その区間のBEVを生成（Z-X平面へのプロジェクション）
        bev_slice = np.max(y_mask_slice, axis=1)
        bev_slices.append(bev_slice)

    # 全てのスライスをスタック
    multislice_bev = np.stack(bev_slices, axis=0)
    return multislice_bev.astype(np.uint8)


def process_patient(patient_id):
    """
    一人の患者データに対して前処理を実行する
    1. CT読み込みとリサンプリング
    2. DRR生成
    3. 肺マスク抽出
    4. マルチスライスBEV生成
    """
    try:
        # 1. CT読み込みと等方性リサンプリング
        original_image = drr_gen.find_best_ct_series(os.path.join(LIDC_IDRI_ROOT, patient_id))
        if original_image is None:
            return f"No valid CT series found for {patient_id}"
        resampled_image = drr_gen.resample_image_to_isotropic(original_image)
        ct_array = sitk.GetArrayFromImage(resampled_image)

        # 2. 肺マスク抽出
        lung_mask_3d = extract_lung_mask(ct_array, HU_THRESHOLD)
        if lung_mask_3d.sum() < 1000: # 抽出がうまくいかなかった場合はスキップ
            return f"Lung mask extraction failed for {patient_id}"

        # 3. DRR生成（3方向から）
        drr_images = drr_gen.create_drr_from_isotropic_ct(resampled_image, views=['AP', 'LAT', 'OBL'])

        # 4. マルチスライスBEV生成
        multislice_bev_target = create_multislice_bev(lung_mask_3d, NUM_SLICES)

        # 5. 保存
        saved_files = []
        for view, drr_img_sitk in drr_images.items():
            drr_array = sitk.GetArrayFromImage(drr_img_sitk)

            # DRR画像（入力）
            image_filename = f"{patient_id}_{view}.npy"
            np.save(os.path.join(OUTPUT_DIR, 'images', image_filename), drr_array)

            # マルチスライスBEV（教師データ）はDRRごとに同じものを保存
            target_filename = f"{patient_id}_multislice_bev.npy"
            np.save(os.path.join(OUTPUT_DIR, 'targets', target_filename), multislice_bev_target)
            
            saved_files.append(image_filename)

        return f"Successfully processed {patient_id}, saved {len(saved_files)} views."

    except Exception as e:
        return f"Error processing {patient_id}: {e}"

def main():
    # 出力ディレクトリ作成
    os.makedirs(os.path.join(OUTPUT_DIR, 'images'), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_DIR, 'targets'), exist_ok=True)
    
    # フィルタリングされていない患者IDリストを取得
    patient_ids = [pid for pid in os.listdir(LIDC_IDRI_ROOT) if pid.startswith('LIDC-IDRI-')]
    print(f"Found {len(patient_ids)} patients.")

    # マルチプロセスでデータセットを生成
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1 or 1) as executor:
        futures = {executor.submit(process_patient, pid): pid for pid in patient_ids}
        
        with tqdm(total=len(patient_ids), desc="Processing Patients") as pbar:
            for future in as_completed(futures):
                result = future.result()
                pbar.set_postfix_str(result, refresh=True)
                pbar.update(1)

    print("\nDataset generation finished.")
    print(f"Data saved to: {OUTPUT_DIR}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore', '.*GetGDCMSeriesFileNames.*')
    main() 