import numpy as np
import matplotlib.pyplot as plt
import os

# --- 設定 ---
DATASET_DIR = 'dataset'

def verify():
    image_dir = os.path.join(DATASET_DIR, 'images')
    target_dir = os.path.join(DATASET_DIR, 'targets')

    # --- フォルダが存在し、中身が空でないことを確認 ---
    if not os.path.exists(image_dir) or not os.listdir(image_dir):
        print(f"エラー: '{image_dir}' が存在しないか、空です。")
        return
    if not os.path.exists(target_dir) or not os.listdir(target_dir):
        print(f"エラー: '{target_dir}' が存在しないか、空です。")
        return

    # --- 最初の画像ファイルとターゲットファイルを取得 ---
    first_image_file = sorted(os.listdir(image_dir))[0]
    patient_id = first_image_file.split('_')[0]
    first_target_file = f"{patient_id}_bev.npy"
    
    image_path = os.path.join(image_dir, first_image_file)
    target_path = os.path.join(target_dir, first_target_file)

    if not os.path.exists(target_path):
        print(f"エラー: 画像 '{first_image_file}' に対応するターゲット '{first_target_file}' が見つかりません。")
        return
        
    print(f"読み込み中:")
    print(f"  画像: {image_path}")
    print(f"  ターゲット: {target_path}")

    # --- NPYファイルを読み込む ---
    drr_image = np.load(image_path)
    bev_target = np.load(target_path)

    # --- BEVマスクの統計情報を出力 ---
    print(f"\n=== BEVマスクの統計情報 ===")
    print(f"Shape: {bev_target.shape}")
    print(f"Data type: {bev_target.dtype}")
    print(f"Min value: {np.min(bev_target)}")
    print(f"Max value: {np.max(bev_target)}")
    print(f"Mean value: {np.mean(bev_target)}")
    print(f"Unique values: {np.unique(bev_target)}")
    print(f"Non-zero pixels: {np.count_nonzero(bev_target)} / {bev_target.size}")
    print(f"Positive pixels percentage: {100 * np.count_nonzero(bev_target) / bev_target.size:.4f}%")

    # --- 可視化 ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].imshow(drr_image, cmap='gray')
    axes[0].set_title('Sample DRR Image')
    axes[0].axis('off')

    axes[1].imshow(bev_target, cmap='hot')
    axes[1].set_title('Corresponding BEV Target')
    axes[1].axis('off')
    
    plt.suptitle('Dataset Verification')
    plt.show()


if __name__ == '__main__':
    verify() 