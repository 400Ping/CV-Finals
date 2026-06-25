# 元智大學低光源垃圾分類輔助系統：飲料紙杯物件偵測

本專案以元智大學校園中的飲料紙杯為偵測目標，使用自行拍攝的校園影像建立資料集。資料集包含光線充足與低光源等不同亮度條件下的飲料紙杯照片，並以 OpenCV 的 HOG 特徵搭配線性 SVM 訓練一個飲料紙杯物件偵測器。系統可對單張圖片、影片或即時攝影機畫面進行偵測，適合作為低光源垃圾分類輔助系統的前端偵測模組。

## 專案結構

```text
.
├── Data/                  # 原始 HEIC 校園照片
├── dataset/
│   ├── images/            # 訓練用 JPG 圖片
│   ├── labels/            # 訓練用標註檔，每張圖一個 .txt
│   └── test/
│       ├── images/        # 獨立測試用 JPG 圖片
│       └── labels/        # 獨立測試用標註檔
├── models/                # 訓練後的模型
├── outputs/               # 偵測輸出圖片或影片
├── scripts/
│   ├── convert_heic_macos.py # 將 HEIC 轉成 JPG
│   ├── annotate.py        # 互動式標註工具
│   ├── train_hog_svm.py   # 訓練 HOG + SVM 偵測器
│   ├── detect.py          # 圖片、影片、攝影機偵測
│   └── evaluate.py        # Precision / Recall / F1 評估
└── requirements.txt
```

## 環境安裝

建議使用 Python 3.10 以上版本。

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 資料集與標註

`Data/` 內保留原始 HEIC 訓練照片，`DataTest/` 內保留額外拍攝的 HEIC 測試照片；`dataset/images/` 內為轉換後的訓練 JPG，`dataset/test/images/` 內為獨立測試 JPG。這些照片由元智大學校園實際拍攝而來，包含日間光線充足、陰影處、室內或光線較不充足等不同亮度條件。若需要重新轉換，可在 macOS 執行：

```bash
python scripts/convert_heic_macos.py --input Data --output dataset/images
python scripts/convert_heic_macos.py --input DataAdd --output dataset/images
python scripts/convert_heic_macos.py --input DataTest --output dataset/test/images
```

標註格式採 YOLO bounding box：

```text
class_id center_x center_y width height
```

其中 `class_id = 0` 代表飲料紙杯，座標皆為 0 到 1 的正規化比例。

執行標註工具：

```bash
python scripts/annotate.py --images dataset/images --labels dataset/labels
```

測試集也需要標註，才能計算 Precision、Recall 與 F1。測試集標註請使用另一個資料夾，避免混入訓練集：

```bash
python scripts/annotate.py --images dataset/test/images --labels dataset/test/labels
```

操作方式：

- 滑鼠拖曳：框選飲料紙杯
- `s`：儲存目前圖片標註
- `n`：儲存並切到下一張
- `u`：復原上一個框
- `q`：儲存並離開

## 訓練

完成標註後執行：

```bash
python scripts/train_hog_svm.py \
  --images dataset/images \
  --labels dataset/labels \
  --model models/cup_hog_svm.yml
```

訓練流程會從標註框擷取飲料紙杯正樣本，並從不重疊區域隨機擷取負樣本，再訓練線性 SVM 分類器。由於訓練資料本身包含不同亮度條件，模型會直接從真實拍攝資料中學習光線充足與低光源場景下的飲料紙杯外觀。

本專案針對飲料紙杯外型使用 `128x128` HOG 視窗，較符合紙杯接近方形的比例。訓練腳本也會執行 hard negative mining：先訓練初始模型，再掃描訓練圖片，把被誤判為紙杯的背景區域加入負樣本後重新訓練，以降低背景誤判。

## 偵測

偵測單張圖片：

```bash
python scripts/detect.py \
  --model models/cup_hog_svm.yml \
  --source dataset/images/IMG_9018.jpg \
  --output outputs/result.jpg
```

使用筆電或 USB 攝影機即時偵測：

```bash
python scripts/detect.py --model models/cup_hog_svm.yml --source 0
```

視窗開啟後按 `q` 結束。若要錄製展示影片，可使用：

```bash
python scripts/detect.py \
  --model models/cup_hog_svm.yml \
  --source 0 \
  --output outputs/demo.mp4
```

## 評估

使用訓練集標註資料計算 Precision、Recall 與 F1：

```bash
python scripts/evaluate.py \
  --images dataset/images \
  --labels dataset/labels \
  --model models/cup_hog_svm.yml
```

使用獨立測試集評估模型：

```bash
python scripts/evaluate.py \
  --images dataset/test/images \
  --labels dataset/test/labels \
  --model models/cup_hog_svm.yml
```

若誤判數量偏高，可提高 `--score-threshold`。threshold 越高，模型會保留較少偵測框，通常 Precision 會上升，但 Recall 可能下降：

```bash
python scripts/evaluate.py \
  --images dataset/test/images \
  --labels dataset/test/labels \
  --model models/cup_hog_svm.yml \
  --score-threshold 0.5
```

報告中建議以獨立測試集結果作為主要測試結果，訓練集結果只作為模型是否成功學到訓練資料的參考。

輸出範例：

```text
Labeled images: 45
TP: 38  FP: 12  FN: 7
Precision: 0.760
Recall: 0.844
F1: 0.800
```
