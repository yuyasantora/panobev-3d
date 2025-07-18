import SimpleITK as sitk
import os

# --- 確認してください ---
# 1. 実際のDICOMシリーズ（.dcmファイル群）が格納されているフォルダパスを指定します
#    Windowsの場合は、パスの先頭に r を付けるとバックスラッシュの問題を避けられます
#    例: dicom_dir = r'C:\Users\ohhara\PanoBEV\data\LIDC-IDRI-0001\1.3.6.1.4.1.14519.5.2.1.6279.6001.298806137288633453246975630178\1.3.6.1.4.1.14519.5.2.1.6279.6001.179049373636438705059720643130'
dicom_dir = 'data/manifest-1752629384107/LIDC-IDRI/LIDC-IDRI-0001/01-01-2000-NA-NA-30178/3000566.000000-NA-03192' 

# 2. 指定したパスが存在するかどうかを確認
if not os.path.isdir(dicom_dir):
    print(f"エラー: 指定されたディレクトリが存在しません: {dicom_dir}")
    exit() # 存在しない場合はここで終了

print(f"指定されたディレクトリ: {dicom_dir}")


reader = sitk.ImageSeriesReader()

# DICOMシリーズのファイル名リストを取得
dicom_names = reader.GetGDCMSeriesFileNames(dicom_dir)

# 3. SimpleITKが見つけたファイルリストが空でないか確認
if not dicom_names:
    print("エラー: SimpleITKがDICOMシリーズをフォルダ内で見つけられませんでした。")
    print("ヒント: 'dicom_dir'のパスが、.dcmファイルが直接含まれているフォルダを指しているか確認してください。")
    exit() # ファイルが見つからない場合はここで終了

print(f"見つかったDICOMファイル数: {len(dicom_names)}")

# 4. 読み込み処理を実行
try:
    reader.SetFileNames(dicom_names)
    image_3d = reader.Execute()

    print("\n3D画像の読み込みに成功しました！")
    print("Image size:", image_3d.GetSize())
    print("Image spacing:", image_3d.GetSpacing())

except Exception as e:
    print(f"\nエラー: 3D画像の読み込み中に例外が発生しました: {e}")










