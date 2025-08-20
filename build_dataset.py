import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
import random # ★ randomライブラリをインポート

import drr_generator as drr_gen

# --- 設定項目 ---
ROOT_DATA_DIR = 'data/manifest-1752629384107/LIDC-IDRI/'
OUTPUT_DIR = 'dataset_lung'
ROTATION_ANGLES_X = [-30, 0, 30]
# ★ データ分割の比率と乱数シード
SPLIT_RATIOS = {'train': 0.8, 'val': 0.1, 'test': 0.1}
RANDOM_SEED = 42

def find_scan_and_xml_robust(patient_dir: str):
    """
    指定された患者ディレクトリ内を再帰的に探索し、最もスライス数の多いCTスキャンシリーズと、
    アノテーションXMLファイルを探し出す、より堅牢なバージョンの関数。
    """
    dicom_series = {}
    xml_path = None
    for root, _, files in os.walk(patient_dir):
        if not xml_path and any(f.lower().endswith('.xml') for f in files):
            xml_path = os.path.join(root, [f for f in files if f.lower().endswith('.xml')][0])
        if any(f.lower().endswith('.dcm') for f in files):
            reader = sitk.ImageSeriesReader()
            try:
                dicom_names = reader.GetGDCMSeriesFileNames(root)
                if dicom_names:
                    dicom_series[root] = len(dicom_names)
            except RuntimeError:
                continue
    if not dicom_series: return None, xml_path
    best_series_path = max(dicom_series, key=dicom_series.get)
    return best_series_path, xml_path

def process_patient(patient_dir, output_base_dir):
    """
    一人の患者データを処理し、指定された出力ディレクトリに保存する関数（肺野版）
    """
    patient_id = os.path.basename(patient_dir)
    image_dir = os.path.join(output_base_dir, 'images')
    target_dir = os.path.join(output_base_dir, 'targets')
    depth_dir = os.path.join(output_base_dir, 'depths')

    dicom_path, xml_path = find_scan_and_xml_robust(patient_dir)
    if not dicom_path:
        print(f"  [!] {patient_id}: CTが見つからずスキップ。")
        return

    try:
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
        reader.SetFileNames(dicom_names)
        original_image = reader.Execute()
        resampled_image = drr_gen.resample_image_to_isotropic(original_image)
        
        # 肺野マスク作成
        ct_array = sitk.GetArrayFromImage(resampled_image)
        lung_mask_3d = extract_lung_mask(ct_array)
        bev_target = create_lung_bev(lung_mask_3d)

        # 肺野BEV保存
        target_filename = f"{patient_id}_bev.npy"
        np.save(os.path.join(target_dir, target_filename), bev_target)

        # DRRと深度は従来通り
        for angle in ROTATION_ANGLES_X:
            angle_rad = np.deg2rad(angle)
            drr = drr_gen.generate_drr(resampled_image, rotation_x_rad=angle_rad)
            drr_array = np.squeeze(sitk.GetArrayFromImage(drr))
            base_filename = f"{patient_id}_rotX{angle:+04d}"
            np.save(os.path.join(image_dir, f"{base_filename}.npy"), drr_array)

            depth_map = drr_gen.generate_depth_map(resampled_image, rotation_x_rad=angle_rad)
            depth_map_array = sitk.GetArrayFromImage(depth_map)
            np.save(os.path.join(depth_dir, f"{base_filename}_depth.npy"), depth_map_array)
    except Exception as e:
        print(f"  [!!!] {patient_id}: 処理中にエラー発生: {e}")

def build_dataset():
    """
    データセット構築のメイン関数。全患者をtrain/val/testに分割して保存する。
    """
    # ★ 1. 出力ディレクトリを作成
    for split in SPLIT_RATIOS.keys():
        for folder in ['images', 'targets', 'depths']:
            os.makedirs(os.path.join(OUTPUT_DIR, split, folder), exist_ok=True)

    # ★ 2. 患者リストを取得し、シャッフル
    all_patient_dirs = [os.path.join(ROOT_DATA_DIR, d) for d in os.listdir(ROOT_DATA_DIR) if os.path.isdir(os.path.join(ROOT_DATA_DIR, d))]
    random.seed(RANDOM_SEED)
    random.shuffle(all_patient_dirs)

    # ★ 3. 患者リストを分割 (より堅牢な方法に修正)
    num_patients = len(all_patient_dirs)
    
    n_train = int(num_patients * SPLIT_RATIOS['train'])
    n_val = int(num_patients * SPLIT_RATIOS['val'])

    # データが少なくn_valが0になった場合でも、残りの患者がいれば1人割り当てる
    if n_val == 0 and n_train < num_patients:
        n_val = 1
    
    # 割り当ての結果、合計が全体数を超えないようにtrainを調整
    if n_train + n_val >= num_patients:
        n_train = num_patients - n_val
    
    train_patients = all_patient_dirs[:n_train]
    val_patients = all_patient_dirs[n_train:n_train + n_val]
    test_patients = all_patient_dirs[n_train + n_val:]

    patient_splits = {
        'train': train_patients,
        'val': val_patients,
        'test': test_patients
    }
    
    print("データセット構築を開始します...")
    print(f"合計: {num_patients}人 -> Train: {len(train_patients)}人, Val: {len(val_patients)}人, Test: {len(test_patients)}人")

    # ★ 4. 各セットごとに患者を処理
    for split, patient_list in patient_splits.items():
        print(f"\n--- {split.upper()} セットの処理を開始 ---")
        output_base_dir = os.path.join(OUTPUT_DIR, split)
        for patient_dir in tqdm(patient_list, desc=f"Processing {split} set"):
            process_patient(patient_dir, output_base_dir)

    print("\nデータセットの構築と分割が完了しました！")

# build_dataset.py に追加する関数
def extract_lung_mask(ct_array, hu_threshold=-500):
    """CTから肺野領域を抽出"""
    # HU値で粗い肺野抽出 + モルフォロジー処理
    lung_mask = (ct_array < hu_threshold) & (ct_array > -1000)
    # 連結成分分析で大きな領域のみ保持
    # 穴埋め処理等
    return lung_mask.astype(np.uint8)

def create_lung_bev(lung_mask_3d):
    """3D肺野マスク → BEV投影"""
    # Z軸（上下）方向に最大値投影
    bev = np.max(lung_mask_3d, axis=0)  # (Y,X) → (X,Z)に転置
    return bev.T

if __name__ == '__main__':
    build_dataset()





