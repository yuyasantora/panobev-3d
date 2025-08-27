import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random
import glob

def visualize_sample(dataset_dir):
    """
    データセットからランダムに"成功した"サンプルを1つ選び、DRR画像と3Dボクセルを可視化する。
    """
    data_subset_to_verify = 'train' 
    images_dir = os.path.join(dataset_dir, data_subset_to_verify, 'images')
    targets_dir = os.path.join(dataset_dir, data_subset_to_verify, 'targets')

    if not os.path.exists(targets_dir) or not os.listdir(targets_dir):
        print(f"エラー: ターゲットディレクトリが見つからないか、空です: {targets_dir}")
        print("ヒント: build_dataset.py を実行して、データセットを生成してください。")
        return

    # ★★★★★ 修正点: ターゲットからランダムに選択 ★★★★★
    # 1. ランダムなターゲットファイルを選択 (これは成功が保証されている)
    random_target_name = random.choice(os.listdir(targets_dir))
    target_path = os.path.join(targets_dir, random_target_name)

    # 2. 対応する画像ファイルを見つける
    # 例: "LIDC-IDRI-0001_voxel.npy" -> "LIDC-IDRI-0001"
    patient_id = random_target_name.replace('_voxel.npy', '')
    
    # この患者IDに一致するDRR画像を探す (AP, LAT, OBLのいずれか)
    # globを使ってワイルドカード検索
    possible_images = glob.glob(os.path.join(images_dir, f"{patient_id}_*.npy"))
    
    if not possible_images:
        print(f"エラー: 対応する画像ファイルが見つかりません: {patient_id}_*.npy")
        return
    
    # 見つかった画像の中からランダムに1つ選ぶ
    image_path = random.choice(possible_images)
    image_name = os.path.basename(image_path)


    # 3. データを読み込む
    print(f"読み込み中:\n  - 画像: {image_name}\n  - ターゲット: {random_target_name}")
    drr_image = np.load(image_path)
    voxel_grid = np.load(target_path)

    # 4. 可視化
    fig = plt.figure(figsize=(12, 6))
    
    # 左側: DRR画像
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(drr_image, cmap='gray')
    ax1.set_title(f"Input DRR Image\n{image_name}")
    ax1.axis('off')

    # 右側: 3Dボクセルデータ
    ax2 = fig.add_subplot(1, 2, 2, projection='3d')
    if np.any(voxel_grid):
        z, y, x = np.where(voxel_grid > 0)
        ax2.scatter(x, y, z, c='red', s=1)
    
    ax2.set_title(f"Target 3D Voxel Grid\nShape: {voxel_grid.shape}")
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_zlabel('Z')
    # 軸の範囲を揃える
    max_dim = max(voxel_grid.shape)
    ax2.set_xlim(0, max_dim)
    ax2.set_ylim(0, max_dim)
    ax2.set_zlim(0, max_dim)
    ax2.set_aspect('auto') # アスペクト比を自動調整

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # configファイルからデータセットのパスを読み込む
    try:
        with open('config_3d.yaml', 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        dataset_dir = config.get('OUTPUT_DIR')
        if not dataset_dir:
            print("エラー: config_3d.yamlに'OUTPUT_DIR'が設定されていません。")
        else:
            visualize_sample(dataset_dir)

    except FileNotFoundError:
        print("エラー: 3d_e2e/config_3d.yaml が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")








