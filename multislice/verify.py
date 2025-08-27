# verify_dataset.py
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# 親ディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# --- 設定 ---
DATASET_DIR = 'dataset_lung_multislice'

def visualize_random_sample():
    """
    データセットからランダムなサンプルを1つ選び、DRRとマルチスライスBEVを可視化する。
    """
    images_dir = os.path.join(DATASET_DIR, 'images')
    targets_dir = os.path.join(DATASET_DIR, 'targets')

    if not os.path.exists(images_dir) or not os.listdir(images_dir):
        print(f"エラー: '{images_dir}' が空か、存在しません。")
        print("まず build_dataset.py を実行してデータセットを生成してください。")
        return

    # --- 1. ランダムなDRR画像を読み込み ---
    random_image_file = np.random.choice(os.listdir(images_dir))
    image_path = os.path.join(images_dir, random_image_file)
    drr_image = np.load(image_path)

    print(f"可視化するサンプル:")
    print(f"  - DRR画像: {random_image_file}")
    print(f"  - DRR形状: {drr_image.shape}")

    # --- 2. 対応するマルチスライスBEVを読み込み ---
    patient_id = random_image_file.split('_')[0]
    target_filename = f"{patient_id}_multislice_bev.npy"
    target_path = os.path.join(targets_dir, target_filename)
    
    if not os.path.exists(target_path):
        print(f"エラー: 対応するターゲットファイルが見つかりません: {target_path}")
        return
        
    multislice_bev = np.load(target_path)
    print(f"  - マルチスライスBEV形状: {multislice_bev.shape}")

    # --- 3. 可視化 ---
    fig = plt.figure(figsize=(18, 8))
    
    # 左側にDRR画像を表示
    ax_drr = fig.add_subplot(1, 2, 1)
    ax_drr.imshow(drr_image, cmap='gray')
    ax_drr.set_title(f'Input DRR\n({random_image_file})', fontsize=16)
    ax_drr.axis('off')
    
    # 右側に16枚のBEVスライスをグリッドで表示
    num_slices = multislice_bev.shape[0]
    grid_size = int(np.ceil(np.sqrt(num_slices))) # 4x4 グリッド

    gs = fig.add_gridspec(grid_size, grid_size, left=0.55, right=0.95, wspace=0.1, hspace=0.1)
    fig.suptitle('Ground Truth: Multi-slice BEV (16 slices from bottom to top)', fontsize=20, y=0.98)

    for i in range(num_slices):
        ax = fig.add_subplot(gs[i // grid_size, i % grid_size])
        ax.imshow(multislice_bev[i, :, :], cmap='hot')
        ax.text(0.5, 0.95, f'Slice {i}', color='white', ha='center', va='top', transform=ax.transAxes, fontsize=8)
        ax.axis('off')

    plt.show()

if __name__ == '__main__':
    visualize_random_sample()