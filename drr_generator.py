import SimpleITK as sitk
import os
#import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET

def resample_image_to_isotropic(image: sitk.Image, new_spacing: list = [1.0, 1.0, 1.0]) -> sitk.Image:
    """
    3D画像をアイソトロピック（等方性）にリサンプリングする
    """
    print("アイソトロピック・リサンプリングを開始...")
    
    # 元の画像の情報を取得
    original_spacing = image.GetSpacing()
    original_size = image.GetSize()
    
    # 新しい画像の物理的なサイズを計算
    original_physical_size = [sz * sp for sz, sp in zip(original_size, original_spacing)]
    
    # 新しい画像のピクセルサイズを計算 (物理サイズ / 新しいスペーシング)
    new_size = [int(round(phys_sz / spc)) for phys_sz, spc in zip(original_physical_size, new_spacing)]
    
    print(f"  新しいサイズ: {new_size}")
    print(f"  新しいスペーシング: {new_spacing}")

    # リサンプリングを実行するフィルターを準備
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing(new_spacing)
    resampler.SetSize(new_size)
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetTransform(sitk.Transform()) # 移動や回転はしない
    resampler.SetDefaultPixelValue(image.GetPixelIDValue())
    resampler.SetInterpolator(sitk.sitkLinear) # 線形補間
    
    return resampler.Execute(image)

def generate_drr(image: sitk.Image,
                 rotation_x_rad: float = 0,
                 rotation_y_rad: float = 0,
                 rotation_z_rad: float = 0,
                 default_pixel_value: int = -1000,  # ★ デフォルト値を受け取れるように引数を追加
                 interpolator: int = sitk.sitkLinear # ★ 補間方法を受け取れるように引数を追加
                 ) -> sitk.Image:
    """
    指定された角度で3D画像を回転させ、DRRを生成する
    """
    print(f"\nDRR生成中... X回転: {np.rad2deg(rotation_x_rad):.1f}°, Y回転: {np.rad2deg(rotation_y_rad):.1f}°")

    image_center_phys = image.TransformContinuousIndexToPhysicalPoint([idx/2.0 for idx in image.GetSize()])
    
    transform = sitk.Euler3DTransform()
    transform.SetCenter(image_center_phys)
    transform.SetRotation(rotation_x_rad, rotation_y_rad, rotation_z_rad)
    
    rotated_image = sitk.Resample(image,
                                  image,
                                  transform,
                                  interpolator,          # ★ 引数で受け取った補間方法を使用
                                  float(default_pixel_value), # ★ 引数で受け取ったデフォルト値を使用
                                  image.GetPixelID())
    
    projector = sitk.SumProjectionImageFilter()
    projector.SetProjectionDimension(1)
    drr_image = projector.Execute(rotated_image)
    
    return drr_image

def create_nodule_mask(xml_path: str, original_image: sitk.Image, resampled_image: sitk.Image) -> sitk.Image:
    """
    LIDCのXMLファイルから結節情報を読み込み、3Dマスク画像を生成する
    """
    print(f"\nXMLファイルから結節マスクを生成: {os.path.basename(xml_path)}")
    
    # 1. リサンプリング後の画像と同じサイズの空の画像を作成
    nodule_mask = sitk.Image(resampled_image.GetSize(), sitk.sitkUInt8)
    nodule_mask.CopyInformation(resampled_image)

    # 2. XMLファイルを解析
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"エラー: XMLファイルの解析に失敗しました: {xml_path}")
        print(e)
        # エラーが発生した場合、空のマスクを返す
        return sitk.Image(resampled_image.GetSize(), sitk.sitkUInt8)

    root = tree.getroot()
    
    # LIDCのXMLの名前空間を定義 (★ここが修正点です)
    # {キー: 値} の形式の「辞書」であることを確認してください
    ns = {'ns': 'http://www.nih.gov'}
    
    nodule_count = 0
    # 全てのROI（関心領域）をループ
    for roi in root.findall('.//ns:roi', ns):
        # 結節のz座標を取得
        z_pos = float(roi.find("ns:imageZposition",ns).text)
        
        # 結節の輪郭座標の平均を計算して中心点を求める
        x_coords = [float(edge.find("ns:xCoord", ns).text) for edge in roi.findall(".//ns:edgeMap", ns)]
        y_coords = [float(edge.find("ns:yCoord", ns).text) for edge in roi.findall(".//ns:edgeMap", ns)]
        x_center_pixel = np.mean(x_coords)
        y_center_pixel = np.mean(y_coords)

        # 3. 元画像のピクセル画像から物理画像への変換
        # まず、z物理座標から元画像のzスライスインデックスを求める
        z_index = int(round(z_pos - original_image.GetOrigin()[2] / original_image.GetSpacing()[2]))

        # 元画像のピクセルインデックスから3D物理座標を取得
        # TransformIndexToPhysicalPointは整数のタプルを期待するため、キャストする
        point_in_original_image = (int(x_center_pixel), int(y_center_pixel), z_index)
        physical_point = original_image.TransformIndexToPhysicalPoint(point_in_original_image)
        
        # 4. 3D物理座標をリサンプリング後の画像のインデックスに変換
        resampled_index = resampled_image.TransformPhysicalPointToIndex(physical_point)

        # 境界チェック
        if all(0 <= idx < size for idx, size in zip(resampled_index, resampled_image.GetSize())):
            nodule_mask[resampled_index] = 1 # マスク画像の対応するピクセルを1にする
            nodule_count += 1

    print(f"  {nodule_count}個の結節をマスクに設定しました。")
    return nodule_mask



# --- データセット作成のための関数 ---
def create_bev_target(nodule_mask_3d: sitk.Image) -> sitk.Image:
    """
    3D結節マスクからBEVターゲットを生成する。
    Z軸方向に積分することで実現する。
    """
    print("BEVターゲットの生成中...")

    projector = sitk.SumProjectionImageFilter()
    projector.SetProjectionDimension(0)
    bev_image = projector.Execute(nodule_mask_3d)

    print(f"BEVターゲットの生成完了")
    return bev_image


def generate_depth_map(image: sitk.Image,
                       rotation_x_rad: float = 0,
                       rotation_y_rad: float = 0,
                       rotation_z_rad: float = 0,
                       default_pixel_value: int = -1000,
                       interpolator: int = sitk.sitkLinear
                       ) -> sitk.Image:
    """
    指定された角度で3D画像を回転させ、DRRと同じ視点からの深度マップを生成する。
    深度は、視線方向（Y軸）で最初に物質（非背景ピクセル）が現れる位置（Y座標インデックス）として定義する。
    """
    print(f"\n深度マップ生成中... X回転: {np.rad2deg(rotation_x_rad):.1f}°, Y回転: {np.rad2deg(rotation_y_rad):.1f}°")

    # 1. generate_drr と同じ回転を適用し、視点を揃える
    image_center_phys = image.TransformContinuousIndexToPhysicalPoint([idx/2.0 for idx in image.GetSize()])
    transform = sitk.Euler3DTransform()
    transform.SetCenter(image_center_phys)
    transform.SetRotation(rotation_x_rad, rotation_y_rad, rotation_z_rad)
    
    rotated_image = sitk.Resample(image,
                                  image,
                                  transform,
                                  interpolator,
                                  float(default_pixel_value),
                                  image.GetPixelID())

    # 2. NumPy配列に変換 (SimpleITKの(x,y,z)からNumPyの(z,y,x)へ)
    volume_np = sitk.GetArrayFromImage(rotated_image)

    # 3. 物質が存在する領域のマスクを作成（背景ピクセルより大きい値を持つかで判断）
    tissue_mask = volume_np > default_pixel_value
    
    # 4. Y軸（視線方向, axis=1）で最初の物質のインデックスを検索
    # np.argmaxは、ブール配列に対して適用すると、最初のTrueのインデックス（=距離）を返す
    # 物質が全くない視線（すべてFalse）の場合、argmaxは0を返してしまうため、
    # 物質が存在するピクセル（ray_has_tissue）についてのみ計算する
    ray_has_tissue = np.any(tissue_mask, axis=1)
    depth_map_np = np.zeros_like(ray_has_tissue, dtype=np.float32)
    depth_map_np[ray_has_tissue] = np.argmax(tissue_mask, axis=1)[ray_has_tissue]

    # 5. NumPy配列をSimpleITKイメージに変換
    depth_map_image = sitk.GetImageFromArray(depth_map_np)

    # 6. DRRと整合性が取れるようにメタデータ（Spacing, Origin, Direction）を設定
    # Y軸で射影するため、回転後の3D画像のX軸とZ軸の情報を引き継ぐ
    rotated_spacing = rotated_image.GetSpacing()
    rotated_origin = rotated_image.GetOrigin()
    rotated_direction = rotated_image.GetDirection()
    
    depth_map_image.SetSpacing([rotated_spacing[0], rotated_spacing[2]])
    depth_map_image.SetOrigin([rotated_origin[0], rotated_origin[2]])
    
    # 3x3の方向行列からY軸成分を除いた2x2行列を作成して設定
    new_direction = [rotated_direction[0], rotated_direction[2], 
                     rotated_direction[6], rotated_direction[8]]
    depth_map_image.SetDirection(new_direction)
    
    print("深度マップの生成完了")
    return depth_map_image


# def main():
#     # --- DICOMデータの読み込み ---
#     dicom_dir = r'data\manifest-1752629384107\LIDC-IDRI\LIDC-IDRI-0001\01-01-2000-NA-NA-30178\3000566.000000-NA-03192'
#     xml_path = os.path.join(dicom_dir, '069.xml')
    
#     if not os.path.isdir(dicom_dir):
#         print(f"エラー: ディレクトリが存在しません: {dicom_dir}")
#         return

#     reader = sitk.ImageSeriesReader()
#     dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)
#     reader.SetFileNames(dicom_names)
#     original_image = reader.Execute()
#     resampled_image = resample_image_to_isotropic(original_image)

#     # --- 結節マスクの作成 ---
#     nodule_mask_3d = create_nodule_mask(xml_path, original_image, resampled_image)

#     # DRRとマスクDRRの生成
#     # CT画像のDRRを生成（デフォルト値は-1000, 補間は線形）
#     drr_frontal = generate_drr(resampled_image, 0, 0, 0)
#     drr_rotated_x = generate_drr(resampled_image, rotation_x_rad=np.deg2rad(30))
    
#     # マスク画像のDRRを生成（デフォルト値は0, 補間は最近傍法）
#     mask_frontal = generate_drr(nodule_mask_3d, 0, 0, 0, 
#                                 default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
#     mask_rotated_x = generate_drr(nodule_mask_3d, rotation_x_rad=np.deg2rad(30),
#                                   default_pixel_value=0, interpolator=sitk.sitkNearestNeighbor)
    
#     # --- 生成したDRRを並べて表示 ---
#     fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
#     axes_flat = axes.flat

#     axes_flat[0].imshow(np.rot90(np.squeeze(sitk.GetArrayFromImage(drr_frontal))), cmap='gray')
#     axes_flat[0].imshow(np.rot90(np.squeeze(sitk.GetArrayFromImage(mask_frontal))) > 0, cmap='Reds', alpha=0.5)
#     axes_flat[0].set_title('Frontal View with Nodule Mask')
#     axes_flat[0].axis('off')

#     axes_flat[1].imshow(np.rot90(np.squeeze(sitk.GetArrayFromImage(drr_rotated_x))), cmap='gray')
#     axes_flat[1].imshow(np.rot90(np.squeeze(sitk.GetArrayFromImage(mask_rotated_x))) > 0, cmap='Reds', alpha=0.5)
#     axes_flat[1].set_title('Rotated View with Nodule Mask')
#     axes_flat[1].axis('off')
    
   


# if __name__ == '__main__':
#     main()












