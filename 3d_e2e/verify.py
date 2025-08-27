import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

def visualize_sample(dataset_dir):
    """
    データセットからランダムにサンプルを1つ選び、DRR画像と3Dボクセルを可視化する。
    """
    # ★★★★★ 修正点 1: 'train' サブディレクトリを見るようにパスを修正 ★★★★★
    data_subset_to_verify = 'train' 
    images_dir = os.path.join(dataset_dir, data_subset_to_verify, 'images')
    targets_dir = os.path.join(dataset_dir, data_subset_to_verify, 'targets')

    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"エラー: 画像ディレクトリが見つからないか、空です: {images_dir}")
        print("ヒント: build_dataset.py を実行して、データセットが正しく生成されているか確認してください。")
        return

    # 1. ランダムな画像ファイルを選択
    random_image_name = random.choice(os.listdir(images_dir))
    image_path = os.path.join(images_dir, random_image_name)

    # ★★★★★ 修正点 2: 患者IDの抽出をより堅牢な方法に修正 ★★★★★
    # 例: "LIDC-IDRI-0001_AP.npy" -> "LIDC-IDRI-0001"
    #     ファイル名の最後の'_'より前をすべて取得します。
    patient_id = random_image_name.rsplit('_', 1)[0]
    target_name = f"{patient_id}_voxel.npy"
    target_path = os.path.join(targets_dir, target_name)

    if not os.path.exists(target_path):
        print(f"エラー: 対応するターゲットファイルが見つかりません: {target_path}")
        return

    # 3. データを読み込む
    print(f"読み込み中:\n  - 画像: {random_image_name}\n  - ターゲット: {target_name}")
    drr_image = np.load(image_path)
    voxel_grid = np.load(target_path)

    # 4. 可視化
    fig = plt.figure(figsize=(12, 6))
    
    # 左側: DRR画像
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(drr_image, cmap='gray')
    ax1.set_title(f"Input DRR Image\n{random_image_name}")
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








