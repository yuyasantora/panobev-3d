## PanoBEV-3D: DRRからの3D情報復元プロジェクト

本プロジェクトは、単一のDRR（2D疑似X線）から、鳥瞰図（BEV）と深度マップを同時に予測し、3次元的なシーン理解を目指す深層学習プロジェクトです。

### 目的
- **BEVセグメンテーション**: 空間内の重要オブジェクト（本プロジェクトでは肺結節）の(X, Z)位置を特定
- **深度予測**: 各ピクセルの視点からの距離(Y)を推定

---

## リポジトリ構成（主要ファイル）
- `build_dataset.py`: LIDC-IDRI から DRR / BEV / Depth を生成し、`dataset/{train,val,test}` に保存
- `train.py`: 学習・検証サイクル（最良モデル `best_model.pth` を保存）
- `inference.py`: 学習済みモデルで推論・可視化
- `reconstruct_3d_from_bev.py`: 予測BEVを3Dへ逆射影
- `visualize_3d.py`: CTデータの3D可視化
- `verify_dataset.py`: データの統計・可視化

---

## セットアップ
1) Python環境（推奨: 3.10 付近）を用意  
2) 依存関係をインストール
```bash
pip install -r requirements.txt
```

---

## データセット作成
- LIDC-IDRI のルートを `build_dataset.py` の `ROOT_DATA_DIR` に設定し実行
```bash
python build_dataset.py
```
- 出力は `dataset/{train,val,test}/{images,targets,depths}`

---

## 学習（Training）
- 設定: `config.yaml` を編集
  - 例（BEV学習に集中する初期推奨値）
    - `num_epochs: 80 ~ 100`
    - `learning_rate: 1e-5`
    - `depth_loss_weight: 0.0`（立ち上げ時はBEVに全集中）
    - `use_augmentation: false`（まず安定化優先）
    - `pos_weight: 200`（現在の損失はFocalを使用するため未使用。実験名表示にのみ反映）

- 実行
```bash
python train.py --config config.yaml
```

- 出力
  - `experiments/<timestamp>_pos{...}_lr{...}/best_model.pth`
  - ログには以下の評価が出力されます:
    - `soft-Dice`: ソフト（連続値）Dice
    - `Dice@[0.05,0.1,0.2,0.3,0.5]`: 閾値ごとのDice
    - `DiceNZ@[...]`: 非空ターゲットのみでのDice（主KPI）
    - `val p.mean / p.max`: 予測活性の平均/最大値

---

## 推論（Inference）
- 学習済み `best_model.pth` を指定して実行
```bash
python inference.py --config config.yaml --resume_from experiments/<your_exp_dir>
```

---

## 実装のポイント（最新）
- 入力正規化: DRR を [0,1] → [-1,1] に正規化（ViTの前提に合わせる）
- 最適化: パラメタグループで学習率分離
  - `model.vit`: 低LR（例: 1e-5）
  - デコーダ/ヘッド: 高LR（例: 1e-3）
- 損失（BEV）: 極端な不均衡に対応
  - `FocalWithLogitsLoss(alpha≈0.99, gamma≈1.5–2.0)`
  - `TverskyLoss(alpha=0.7, beta=0.3)`
  - 教師BEVを `max_pool2d(k=5)` で膨張した版も併用（初期は膨張寄与を強めに）
- 評価: 空マスク対策
  - `DiceNZ`（非空のみ）を主KPIに
  - `Dice@...=1.0 かつ nz_ratio=0` は「教師も予測も空」の見かけの1.0なので無視
- 運用指針
  - 立ち上げ時は `depth_loss_weight: 0.0`（BEVに集中）
  - `DiceNZ` が安定して上がったら `depth_loss_weight: 0.1` に戻す
  - 膨張寄与を段階的に下げ、最終的に元教師のみへ

---

## 進捗の目安
- Val BEV Loss: ≈ 1.0 付近で安定
- `DiceNZ@0.2/0.3`: 0 → 0.01 → 0.02… と漸増
- `p.mean`: 立ち上がりに伴って徐々に上昇（極端な暴発はNG）

---

## これまでの成果と知見（履歴）
- v0.1: ベースライン構築（フチ反応などの課題を確認）
- v0.2: 深度予測の導入（損失スケール不一致の課題）
- v0.3: 深度正規化でロス安定化、`pos_weight` 未設定だと検出ゼロに
- v0.4: `pos_weight` 導入で偽陰性低減（強すぎると偽陽性増加）
- v0.5: データ増強導入でアーチファクト過反応を緩和
- v0.6: 入力正規化（[-1,1]）と最適化の分離でViTを安定化
- v0.7: Focal + Tversky と膨張教師で極端なスパースに対処
- v0.8: soft-Dice / Diceスイープ / DiceNZ を導入（空マスク対策）
- v0.9: 陽性率（train≈0.0089%、val≈0.0028%）を確認し、Focal中心に移行
- v1.0: 運用指針確立（BEV集中→段階復帰、KPIはDiceNZ）

---



