import os
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

import drr_generator as drr_gen

# --- 設定項目 ---
ROOT_DATA_DIR = 'data/manifest-1752629384107/LIDC-IDRI/'
OUTPUT_DIR = 'dataset'
ROTATION_ANGLES_X = [-30, 0, 30]

def find_scan_and_xml_robust(patient_dir: str):
    """
    指定された患者ディレクトリ内を再帰的に探索し、最もスライス数の多いCTスキャンシリーズと、
    アノテーションXMLファイルを探し出す、より堅牢なバージョンの関数。
    """
    dicom_series = {}  # key: series_path, value: slice_count
    xml_path = None

    # os.walkで患者フォルダ内の全サブディレクトリを探索
    for root, _, files in os.walk(patient_dir):
        # 最初に見つかったXMLファイルを取得
        if not xml_path:
            for file in files:
                if file.lower().endswith('.xml'):
                    xml_path = os.path.join(root, file)
                    # ここでループを抜けないのは、DICOMシリーズが別のサブフォルダにあるかもしれないため

        # 現在のディレクトリに.dcmファイルが存在するかチェック
        if any(f.lower().endswith('.dcm') for f in files):
            reader = sitk.ImageSeriesReader()
            try:
                # この'root'フォルダが有効なDICOMシリーズか確認
                dicom_names = reader.GetGDCMSeriesFileNames(root)
                if dicom_names:
                    # 有効なシリーズであれば、パスとスライス数を辞書に保存
                    dicom_series[root] = len(dicom_names)
            except RuntimeError:
                # 有効なシリーズではない場合（例: RTSTRUCTファイルなど）、無視して次に進む
                continue

    if not dicom_series:
        return None, xml_path

    # 見つかった全シリーズの中から、スライス数が最大のものを選択
    best_series_path = max(dicom_series, key=dicom_series.get)
    
    return best_series_path, xml_path


def build_dataset():
    """
    データセット構築のメイン関数
    """
    image_dir = os.path.join(OUTPUT_DIR, 'images')
    target_dir = os.path.join(OUTPUT_DIR, 'targets')
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(target_dir, exist_ok=True)

    patient_dirs = [os.path.join(ROOT_DATA_DIR, d) for d in os.listdir(ROOT_DATA_DIR) if os.path.isdir(os.path.join(ROOT_DATA_DIR, d))]
    
    print(f"{len(patient_dirs)} 人の患者データが見つかりました。データセット構築を開始します...")

    for patient_dir in tqdm(patient_dirs, desc="患者データ処理中"):
        patient_id = os.path.basename(patient_dir)
        print(f"\n--- 患者ID: {patient_id} の処理を開始 ---")

        # ★ 堅牢な探索関数を呼び出すように変更
        dicom_path, xml_path = find_scan_and_xml_robust(patient_dir)

        if not dicom_path or not xml_path:
            print(f"  [!] CTスキャンまたはXMLが見つからなかったため、スキップします。")
            continue
        
        print(f"  CTシリーズ: {os.path.basename(dicom_path)}")
        print(f"  XMLファイル: {os.path.basename(xml_path)}")
        
        try:
            reader = sitk.ImageSeriesReader()
            dicom_names = reader.GetGDCMSeriesFileNames(dicom_path)
            reader.SetFileNames(dicom_names)
            original_image = reader.Execute()
            resampled_image = drr_gen.resample_image_to_isotropic(original_image)
            nodule_mask_3d = drr_gen.create_nodule_mask(xml_path, original_image, resampled_image)
            bev_target = drr_gen.create_bev_target(nodule_mask_3d)

            # --- ★★★ ここが修正点 ★★★ ---
            # 3. BEVターゲットを保存
            # np.squeeze() を追加して、2D配列に変換してから保存する
            bev_array = np.squeeze(sitk.GetArrayFromImage(bev_target))
            target_filename = f"{patient_id}_bev.npy"
            np.save(os.path.join(target_dir, target_filename), bev_array)
            print(f"  BEVターゲットを保存しました: {target_filename}")

            # 4. 各角度でDRRを生成して保存
            for angle in ROTATION_ANGLES_X:
                drr = drr_gen.generate_drr(resampled_image, rotation_x_rad=np.deg2rad(angle))
                drr_array = np.squeeze(sitk.GetArrayFromImage(drr))
                image_filename = f"{patient_id}_rotX{angle:+04d}.npy"
                np.save(os.path.join(image_dir, image_filename), drr_array)
                print(f"  DRRを保存しました: {image_filename}")

        except Exception as e:
            print(f"  [!!!] 処理中に予期せぬエラーが発生しました: {e}")
            continue

    print("\nデータセットの構築が完了しました！")
    print(f"  入力画像は '{image_dir}' に保存されています。")
    print(f"  教師ターゲットは '{target_dir}' に保存されています。")


if __name__ == '__main__':
    build_dataset()





