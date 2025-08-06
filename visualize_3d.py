import os
import SimpleITK as sitk
import pyvista as pv
from tqdm import tqdm
import numpy as np

def visualize_dicom_volume(dicom_dir: str, output_filename="3d_model_screenshot.png"):
    """
    指定されたDICOMディレクトリのCTスキャンを読み込み、
    3Dモデルのスクリーンショットをファイルに保存する。
    """
    if not os.path.isdir(dicom_dir):
        print(f"エラー: 指定されたディレクトリが存在しません: {dicom_dir}")
        return

    print(f"--- DICOMシリーズの読み込み開始: {dicom_dir} ---")
    
    reader = sitk.ImageSeriesReader()
    try:
        # まずは指定されたディレクトリで直接シリーズ読み込みを試みる
        dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
        if not dicom_names:
            # 失敗した場合、サブディレクトリを探索する
            series_ids = sitk.ImageSeriesReader.GetGDCMSeriesIDs(dicom_dir)
            if not series_ids:
                print(f"  [!!] ディレクトリ '{dicom_dir}' からDICOMファイルを読み込めませんでした。")
                return
            series_file_names = {s_id: sitk.ImageSeriesReader.GetGDCMSeriesFileNames(dicom_dir, s_id) for s_id in series_ids}
            best_series_id = max(series_file_names, key=lambda s: len(series_file_names[s]))
            dicom_names = series_file_names[best_series_id]
        
        reader.SetFileNames(dicom_names)
        image = reader.Execute()
    except RuntimeError as e:
        print(f"  [!!] DICOMシリーズの読み込み中にエラーが発生しました: {e}")
        return
    
    print(f"読み込み完了。画像サイズ: {image.GetSize()}, スペーシング: {['{:.2f}'.format(s) for s in image.GetSpacing()]}")

    print("\n--- PyVistaでの3Dレンダリング準備 ---")
    volume_np = sitk.GetArrayFromImage(image)
    grid = pv.ImageData()
    grid.dimensions = np.array(volume_np.shape)[::-1]
    grid.origin = image.GetOrigin()
    grid.spacing = image.GetSpacing()
    grid.point_data["values"] = volume_np.flatten(order="F")

    print(f"オフスクリーンでレンダリングし、'{output_filename}' に保存します...")
    
    plotter = pv.Plotter(off_screen=True)
    
    # ★★★ 最終修正: 不透明度を数値リストで直接指定 ★★★
    # このリストは、climで指定した範囲(-900~1200)にマッピングされる
    # 最初の20%は不透明度0(透明)にし、そこから徐々に不透明にする
    opacity_map = [0.0, 0.0, 0.2, 0.4, 0.8]

    plotter.add_volume(
        grid, 
        cmap="bone", 
        shade=True, 
        clim=[-900, 1200],
        opacity=opacity_map # ★ ここで数値リストを指定
    )
    
    plotter.camera_position = 'iso'
    plotter.screenshot(output_filename)
    plotter.close()
    
    print(f"\n保存完了！ プロジェクトフォルダ内の '{output_filename}' を確認してください。")


if __name__ == '__main__':
    # 可視化したい患者データのパスを指定
    # 例: LIDC-IDRIの患者1人分のデータが含まれるディレクトリ
    # 注意: このパスはご自身の環境に合わせて変更してください
    TARGET_DICOM_DIR = r'data/manifest-1752629384107/LIDC-IDRI/LIDC-IDRI-0003/01-01-2000-NA-NA-94866/3000611.000000-NA-03264'
    
    visualize_dicom_volume(TARGET_DICOM_DIR) 