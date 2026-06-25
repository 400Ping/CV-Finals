# 元智大學低光源垃圾分類輔助系統：寶特瓶物件偵測

本專案以元智大學校園中的寶特瓶為偵測目標，使用自行拍攝的校園影像建立資料集。資料集包含光線充足與低光源等不同亮度條件下的寶特瓶照片，並以 OpenCV 的 HOG 特徵搭配線性 SVM 訓練一個寶特瓶物件偵測器。系統可對單張圖片、影片或即時攝影機畫面進行偵測，適合作為低光源垃圾分類輔助系統的前端偵測模組。

## 專案結構

```text
.
├── Data/                  # 原始 HEIC 校園照片
├── dataset/
│   ├── images/            # 已轉換成 OpenCV 可讀的 JPG 圖片
│   └── labels/            # YOLO 格式標註檔，每張圖一個 .txt
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

`Data/` 內保留原始 HEIC 照片，`dataset/images/` 內為轉換後的 JPG 版本。這批照片由元智大學校園實際拍攝而來，包含日間光線充足、陰影處、室內或光線較不充足等不同亮度條件。若需要重新轉換，可在 macOS 執行：

```bash
python scripts/convert_heic_macos.py --input Data --output dataset/images
```

標註格式採 YOLO bounding box：

```text
class_id center_x center_y width height
```

其中 `class_id = 0` 代表寶特瓶，座標皆為 0 到 1 的正規化比例。

執行標註工具：

```bash
python scripts/annotate.py --images dataset/images --labels dataset/labels
```

操作方式：

- 滑鼠拖曳：框選寶特瓶
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
  --model models/bottle_hog_svm.yml
```

訓練流程會從標註框擷取寶特瓶正樣本，並從不重疊區域隨機擷取負樣本，再訓練線性 SVM 分類器。由於訓練資料本身包含不同亮度條件，模型會直接從真實拍攝資料中學習光線充足與低光源場景下的寶特瓶外觀。

## 偵測

偵測單張圖片：

```bash
python scripts/detect.py \
  --model models/bottle_hog_svm.yml \
  --source dataset/images/IMG_8931.jpg \
  --output outputs/result.jpg
```

使用筆電或 USB 攝影機即時偵測：

```bash
python scripts/detect.py --model models/bottle_hog_svm.yml --source 0
```

視窗開啟後按 `q` 結束。若要錄製展示影片，可使用：

```bash
python scripts/detect.py \
  --model models/bottle_hog_svm.yml \
  --source 0 \
  --output outputs/demo.mp4
```

## 評估

使用標註資料計算 Precision、Recall 與 F1：

```bash
python scripts/evaluate.py \
  --images dataset/images \
  --labels dataset/labels \
  --model models/bottle_hog_svm.yml
```

輸出範例：

```text
Labeled images: 45
TP: 38  FP: 12  FN: 7
Precision: 0.760
Recall: 0.844
F1: 0.800
```
