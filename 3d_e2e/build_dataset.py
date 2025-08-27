import os
import SimpleITK as sitk
import numpy as np
import sys
import yaml
# モジュールのパスを追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import drr_generator as drr_gen
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
from sklearn.model_selection import train_test_split


# --- 設定 --- 
LIDC_IDRI_ROOT = 'C:/Users/ohhara/PanoBEV-3D/data/manifest-1752629384107/LIDC-IDRI'
OUTPUT_DIR = 'dataset_lung_3d_e2e'

# --- 関数 --- 

def extract_lung_mask(ct_array, hu_threshold=-500):
    from scipy.ndimage import label, binary_fill_holes
    from skimage.morphology import remove_small_objects, binary_closing
    from skimage.segmentation import clear_border

    binary_mask = (ct_array < hu_threshold) & (ct_array > -1000)
    cleared_mask = clear_border(binary_mask)
    closed_mask = binary_closing(cleared_mask, footprint=np.ones((5,5,5)))
    filled_mask = np.zeros_like(closed_mask)
    for i in range(filled_mask.shape[0]):
        filled_mask[i, :, :] = binary_fill_holes(closed_mask[i, :, :])
    min_size = int(np.prod(filled_mask.shape) * 0.001) 
    cleaned_mask = remove_small_objects(filled_mask, min_size=min_size)
    labeled_mask, num_labels = label(cleaned_mask)
    if num_labels > 1:
        label_sizes = np.bincount(labeled_mask.ravel())[1:]
        top_two_labels = np.argsort(label_sizes)[-2:] + 1
        final_mask = np.isin(labeled_mask, top_two_labels)
    else:
        final_mask = cleaned_mask
    return final_mask.astype(np.uint8)

def create_3d_voxel_grid(lung_mask_3d, output_size=(96, 96, 96)):
    """
    3D肺マスクを指定された解像度のボクセルグリッドにリサンプリングする。
    """
    mask_sitk = sitk.GetImageFromArray(lung_mask_3d.astype(np.float32))

    # 元の画像の情報を取得
    mask_sitk.SetSpacing([sz * sp / nsz for sz, sp, nsz in zip(lung_mask_3d.shape[::-1], [1,1,1], output_size)])
    mask_sitk.SetOrigin([0,0,0])

    # ターゲットとなる参照画像を作成
    ref_image = sitk.Image(output_size, sitk.sitkFloat32)
    ref_image.SetSpacing([1,1,1])
    ref_image.SetOrigin([0,0,0])

    # リサンプリング
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(ref_image)
    resampler.SetInterpolator(sitk.sitkNearestNeighbor) # バイナリマスクなので最近傍法

    resampled_mask = resampler.Execute(mask_sitk)

    voxel_grid = sitk.GetArrayFromImage(resampled_mask)
    return (voxel_grid > 0.5).astype(np.uint8)

def process_patient(patient_id, config, output_dir):
    """一人の患者データに対して3Dボクセルの前処理を実行し、指定ディレクトリに保存する"""
    try:
        # 保存先ディレクトリを構築
        images_dir = os.path.join(output_dir, 'images')
        targets_dir = os.path.join(output_dir, 'targets')

        original_image = drr_gen.find_best_ct_series(os.path.join(config['LIDC_IDRI_ROOT'], patient_id))
        if original_image is None: return f"No CT series for {patient_id}"
        
        resampled_image = drr_gen.resample_image_to_isotropic(original_image)
        ct_array = sitk.GetArrayFromImage(resampled_image)

        lung_mask_3d = extract_lung_mask(ct_array)
        if lung_mask_3d.sum() < 1000: return f"Lung mask failed for {patient_id}"

        # 3Dボクセルグリッドを生成
        voxel_target = create_3d_voxel_grid(lung_mask_3d, output_size=tuple(config['VOXEL_SIZE']))
        
        drr_images = drr_gen.create_drr_from_isotropic_ct(resampled_image, views=['AP', 'LAT', 'OBL'])

        # ターゲットファイル名は患者IDごとに一意
        target_filename = f"{patient_id}_voxel.npy"
        target_path = os.path.join(targets_dir, target_filename)
        # ターゲットは一度だけ保存
        if not os.path.exists(target_path):
            np.save(target_path, voxel_target)

        for view, drr_img_sitk in drr_images.items():
            drr_array = sitk.GetArrayFromImage(drr_img_sitk)
            
            # DRR画像（入力）
            image_filename = f"{patient_id}_{view}.npy"
            np.save(os.path.join(images_dir, image_filename), drr_array)
            
        return f"Success: {patient_id}"
    except Exception as e:
        return f"Error on {patient_id}: {e}"

def run_processing(patient_ids, config, output_dir, description):
    """指定された患者IDリストのデータセット生成を実行するヘルパー関数"""
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1 or 1) as executor:
        futures = {executor.submit(process_patient, pid, config, output_dir): pid for pid in patient_ids}
        with tqdm(total=len(patient_ids), desc=description) as pbar:
            for future in as_completed(futures):
                pbar.set_postfix_str(future.result(), refresh=True)
                pbar.update(1)

def main():
    config_path = os.path.join(current_dir, 'config_3d.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # 1. 出力ディレクトリ構造を作成
    base_output_dir = config['OUTPUT_DIR']
    train_dir = os.path.join(base_output_dir, 'train')
    val_dir = os.path.join(base_output_dir, 'val')

    os.makedirs(os.path.join(train_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(train_dir, 'targets'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(val_dir, 'targets'), exist_ok=True)
    
    # 2. 患者IDをリストアップし、訓練用と検証用に分割
    all_patient_ids = [pid for pid in os.listdir(config['LIDC_IDRI_ROOT']) if pid.startswith('LIDC-IDRI-')]
    print(f"Found {len(all_patient_ids)} total patients.")
    
    # configファイルから設定を読み込む（なければデフォルト値を使用）
    val_split_size = config.get('VAL_SPLIT_SIZE', 0.2)
    random_state = config.get('RANDOM_STATE', 42)

    train_ids, val_ids = train_test_split(
        all_patient_ids, 
        test_size=val_split_size, 
        random_state=random_state
    )
    print(f"Splitting into {len(train_ids)} training and {len(val_ids)} validation patients.")

    # 3. 訓練データセットを生成
    print("\n--- Generating Training Set ---")
    run_processing(train_ids, config, train_dir, "Processing Train Set")

    # 4. 検証データセットを生成
    print("\n--- Generating Validation Set ---")
    run_processing(val_ids, config, val_dir, "Processing Validation Set")

    print(f"\n3D Voxel dataset generation finished. Saved to: {config['OUTPUT_DIR']}")

if __name__ == '__main__':
    warnings.filterwarnings('ignore', '.*GetGDCMSeriesFileNames.*')
    main()


