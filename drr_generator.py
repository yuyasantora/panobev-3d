# drr_generator.py (Multi-slice BEVタスク向け 最終版)
import os
import SimpleITK as sitk
import numpy as np

def find_best_ct_series(patient_dir):
    """
    os.walkを使って手動でDICOMシリーズを検索する、最も確実なバージョン。
    サブディレクトリを探索し、最も.dcmファイルの数が多いものをCTシリーズとして特定する。
    """
    best_series_root = ""
    max_slices = 0
    reader = sitk.ImageSeriesReader()

    # os.walkで全てのサブディレクトリを網羅的に探索
    for root, _, files in os.walk(patient_dir):
        # .dcmファイルを含むディレクトリのみを対象
        if any(f.lower().endswith('.dcm') for f in files):
            dcm_file_count = len([f for f in files if f.lower().endswith('.dcm')])
            
            # これまでに見つけたシリーズよりもスライス数が多いかチェック
            if dcm_file_count > max_slices:
                # このディレクトリが本当に3D画像として読めるか最終確認
                try:
                    dicom_names = reader.GetGDCMSeriesFileNames(root)
                    if not dicom_names: continue

                    test_reader = sitk.ImageFileReader()
                    test_reader.SetFileName(dicom_names[0])
                    img = test_reader.Execute()
                    
                    # 3D画像であり、かつセグメンテーション（通常スライス数が少ない）ではないことを確認
                    if img.GetDimension() >= 3 and dcm_file_count > 20: 
                        max_slices = dcm_file_count
                        best_series_root = root
                except Exception:
                    # 有効なシリーズとして読めない場合はスキップ
                    continue
    
    if not best_series_root:
        return None # 有効なシリーズが一つも見つからなかった
        
    # 見つかった最良のシリーズを読み込んで返す
    dicom_names = reader.GetGDCMSeriesFileNames(best_series_root)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    
    return image

def resample_image_to_isotropic(image, interpolator=sitk.sitkLinear):
    """
    SimpleITK画像を等方性（全軸で同じスペーシング）にリサンプリングする。
    """
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    min_spacing = min(original_spacing)
    
    new_spacing = [min_spacing, min_spacing, min_spacing]
    new_size = [int(round(osz * ospc / min_spacing)) for osz, ospc in zip(original_size, original_spacing)]
    
    resample = sitk.ResampleImageFilter()
    resample.SetOutputSpacing(new_spacing)
    resample.SetSize(new_size)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())
    resample.SetTransform(sitk.Transform())
    
    # CTの空気領域のHU値（-1000）をデフォルト値として設定するのが最も安全
    resample.SetDefaultPixelValue(-1000)

    resample.SetInterpolator(interpolator)
    
    return resample.Execute(image)

def create_drr_from_isotropic_ct(resampled_image, views=['AP']):
    """
    NumpyベースのレイトレーシングでDRRを生成する。RTKや特殊機能は不要。
    """
    ct_array = sitk.GetArrayFromImage(resampled_image) # (Z, Y, X)
    spacing = resampled_image.GetSpacing() # ITK/SITKは (X, Y, Z)順
    
    # HU値を線減衰係数に変換（簡易的なモデル）
    mu = (ct_array.astype(np.float32) + 1000) * 0.02 
    mu[mu < 0] = 0

    drr_images = {}

    for view in views:
        if view == 'AP': # Y軸に沿って投影 (Anterior-Posterior)
            drr_array = np.sum(mu, axis=1) # (Z, Y, X) -> Y軸で積分 -> (Z, X)
        elif view == 'LAT': # X軸に沿って投影 (Lateral)
            drr_array = np.sum(mu, axis=2) # (Z, Y, X) -> X軸で積分 -> (Z, Y)
        elif view == 'OBL': # 45度回転して投影（簡易版）
            from scipy.ndimage import rotate
            # Y軸を中心にZ-X平面を45度回転
            rotated_mu = rotate(mu, 45, axes=(0, 2), reshape=False, mode='nearest')
            drr_array = np.sum(rotated_mu, axis=2) # X軸で積分
        else:
            continue

        # 指数変換してX線写真のような見た目にする
        # spacing[1]はY軸（視線方向）のピクセルの厚み
        drr_array = np.exp(-drr_array * spacing[1]) 
        
        # [0, 255]の範囲に正規化
        min_val, max_val = drr_array.min(), drr_array.max()
        if max_val > min_val:
            drr_array = 255 * (drr_array - min_val) / (max_val - min_val)
        
        drr_image_sitk = sitk.GetImageFromArray(drr_array.astype(np.uint8))
        drr_images[view] = drr_image_sitk
            
    return drr_images












