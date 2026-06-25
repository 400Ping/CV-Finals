# 元智大學不同光照條件垃圾分類輔助系統：飲料紙杯物件偵測

本專案以元智大學校園中的飲料紙杯為偵測目標，使用自行拍攝的校園影像建立資料集。資料集包含光線充足、低光源等不同亮度條件下的飲料紙杯照片。本專案有兩個部分，分別是以 OpenCV HOG + SVM 建立傳統物件偵測器，以及以 YOLO 訓練主要偵測模型，比較傳統方法與深度學習方法在不同光照條件下的偵測表現。

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
├── dataset.yaml           # YOLO 資料集設定
├── scripts/
│   ├── convert_heic_macos.py # 將 HEIC 轉成 JPG
│   ├── annotate.py        # 互動式標註工具
│   ├── train_hog_svm.py   # 訓練 HOG + SVM 偵測器
│   ├── detect.py          # HOG + SVM 圖片、影片、攝影機偵測
│   ├── evaluate.py        # HOG + SVM Precision / Recall / F1 評估
│   ├── train_yolo.py      # 訓練 YOLO 偵測器
│   ├── detect_yolo.py     # YOLO 圖片、影片、攝影機偵測
│   └── evaluate_yolo.py   # YOLO Precision / Recall / mAP 評估
├── requirements.txt       # HOG + SVM、Modal 與資料處理依賴
└── requirements-yolo.txt  # 本機 YOLO 評估與偵測依賴
```

## 環境安裝

建議使用 Python 3.10 或 3.11。若要使用 Modal，建議使用 Python 3.11

本專案可以分成兩種環境使用：

- `cv-cup-detector` conda 環境：用於 Modal CLI、資料處理與 HOG + SVM。
- 本機 YOLO 評估/展示：使用已安裝 `ultralytics` 的 Anaconda base Python，或另建 YOLO-only 環境；不要用 `cv-cup-detector` 跑本機 YOLO，該環境在本機 PyTorch / Matplotlib 初始化時可能 segmentation fault。

Conda 安裝方式：

```bash
conda env create -f environment.yml
conda activate cv-cup-detector
```

本機 YOLO 評估或攝影機展示請使用目前可正常執行的 Anaconda base Python：

```bash
/opt/anaconda3/bin/python scripts/evaluate_yolo.py
/opt/anaconda3/bin/python scripts/detect_yolo.py --source 0
```

若要另建 YOLO-only 環境，可只安裝：

```bash
pip install -r requirements-yolo.txt
```

Venv 安裝方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

## 資料集與標註

`Data/` 內保留原始 HEIC 照片；`dataset/images/` 內為轉換後的訓練 JPG，`dataset/test/images/` 內為獨立測試 JPG。目前資料集包含 62 張訓練影像與 14 張測試影像，且每張影像皆有對應 YOLO 標註。這些照片由元智大學校園實際拍攝而來，包含日間光線充足、陰影處、室內或光線較不充足等不同亮度條件。若需要重新轉換，可在 macOS 執行：

```bash
python scripts/convert_heic_macos.py --input Data --output dataset/images
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

## HOG + SVM Baseline

完成標註後執行：

```bash
python scripts/train_hog_svm.py \
  --images dataset/images \
  --labels dataset/labels \
  --model models/cup_hog_svm.yml
```

訓練流程會從標註框擷取飲料紙杯正樣本，並從不重疊區域隨機擷取負樣本，再訓練線性 SVM 分類器。由於訓練資料本身包含不同亮度條件，模型會直接從真實拍攝資料中學習光線充足與低光源場景下的飲料紙杯外觀。

本專案針對飲料紙杯外型使用 `128x128` HOG 視窗，較符合紙杯接近方形的比例。訓練腳本也會執行 hard negative mining：先訓練初始模型，再掃描訓練圖片，把被誤判為紙杯的背景區域加入負樣本後重新訓練，以降低背景誤判。

若 hard negative mining 造成模型過度保守，可關閉此步驟重新訓練：

```bash
python scripts/train_hog_svm.py --hard-negative-rounds 0
```

### HOG + SVM 偵測

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

### HOG + SVM 評估

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

## YOLO 方法

YOLO 使用同一份 YOLO 格式標註資料，不需要重新標註。預設使用 `yolov8n.pt` 預訓練權重進行 fine-tuning；第一次執行時需要網路下載權重。

訓練 YOLO：

```bash
python scripts/train_yolo.py \
  --data dataset.yaml \
  --model yolov8n.pt \
  --epochs 80 \
  --imgsz 640 \
  --device cpu
```

使用 Modal GPU 訓練 YOLO：

```bash
modal run modal_train_yolo.py --epochs 80 --imgsz 640 --batch 8
```

訓練結果會存在 Modal Volume `drink-cup-yolo-runs`。訓練完成後下載最佳權重：

```bash
mkdir -p models/yolo/drink_cup/weights
modal volume get drink-cup-yolo-runs \
  yolo/drink_cup/weights/best.pt \
  models/yolo/drink_cup/weights/best.pt
```

訓練完成後，最佳權重會存放在：

```text
models/yolo/drink_cup/weights/best.pt
```

評估 YOLO：

```bash
python scripts/evaluate_yolo.py \
  --data dataset.yaml \
  --weights models/yolo/drink_cup/weights/best.pt
```

本機評估建議使用目前可正常執行的 Anaconda base Python：

```bash
/opt/anaconda3/bin/python scripts/evaluate_yolo.py
```

YOLO 評估會輸出 Precision、Recall、mAP50 與 mAP50-95。

使用 YOLO 偵測單張圖片：

```bash
/opt/anaconda3/bin/python scripts/detect_yolo.py \
  --weights models/yolo/drink_cup/weights/best.pt \
  --source dataset/test/images/IMG_9107.jpg
```

使用 YOLO 即時攝影機偵測：

```bash
/opt/anaconda3/bin/python scripts/detect_yolo.py \
  --weights models/yolo/drink_cup/weights/best.pt \
  --source 0
```

若本機執行時出現 macOS 視窗或 Matplotlib 相關 segmentation fault，可先關閉即時顯示，只輸出偵測結果到 `outputs/yolo/`：

```bash
/opt/anaconda3/bin/python scripts/detect_yolo.py \
  --weights models/yolo/drink_cup/weights/best.pt \
  --source 0 \
  --no-show
```
