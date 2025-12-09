# AI CUP 2025 秋季賽 - 電腦斷層心臟肌肉影像分割 (TEAM_7983)

本專案為 **AI CUP 2025 秋季賽－電腦斷層心臟肌肉影像分割競賽** 的完整解決方案與技術報告。

本團隊是使用 nnU-Net v2 進行三種方式的優化分割結果，最後將 fold 的 score 大於 0.796 的做 ensemble + 後處理(Keep largest connected component )。

1.  **方法一**：
- 前處理策略：採用自定義多通道特徵提取結合 nnU-Net v2
自動化前處理，原始CT影像先經由軟組織、血管、骨骼三種物理窗值轉換，合成為三通道特徵圖，再輸入nnU-Net v2 進行重採樣與標準化。
- 驗證策略：採用標準的5-Fold Cross-Validation，將資料集隨機分為5份，輪流作為驗證集，以確保評估結果的基礎可靠性。

2.  **方法二**：
- 前處理策略：同方法一，維持自定義多通道特徵提取加自動化前處理的組合。
- 驗證策略：採用更為嚴謹的10-Fold Cross-Validation。將資料切分為10份，每次僅使用10%數據進行驗證，90%進行訓練。
- 目的：透過增加驗證次數與訓練數據比例，減少隨機分組帶來的偏差，確認多通道策略在不同數據分佈下的性能是否一致且穩健。

3.  **方法三**：
- 此方法作為對照組，不使用多通道合成技術，而是專注於透過調整損失函數與採樣策略來解決類別不平衡問題。
- 前處理策略：僅使用nnU-Net v2自動化前處理，這意味著
模型直接讀取原始或單一通道的CT影像，依賴框架內建的 
CT窗值裁切（Clipping）與Z-score標準化。
- 驗證策略：採用標準5-Fold Cross-Validation。
- 參數優化：針對血管佔比極小的特性，導入了兩項關鍵修改：
(1) 損失函數(Loss Function)：
改用Tversky Loss結合 CrossEntropy，設定alpha = 0.3，beta = 0.7，透過提高beta 值賦予偽陰性（False Negative）更高的懲罰權重，強迫模型提升召回率（Recall）。
(2) 前景採樣(Foreground Sampling)：
將 oversample_foreground_percent從預設的0.33提升至0.5。保證50%的訓練Patch中心包含隨機選取的前景類別，以強迫模型關注稀疏的血管特徵。

# 方法一和方法二的多通道資料重現步驟 : 
- 執行 ```multichannel_ct.ipynb```

- 本程式是專為心臟 CT 影像分割任務設計的多通道影像處理工具，能夠將原始 CT 影像轉換為 nnU-Net v2 訓練格式的多通道影像，並支援視覺化功能。程式分為三個主要步驟：
### 窗值設定 (Window Level/Width)

程式使用三種預設窗值組合，針對心臟結構優化：

| 通道 | 窗位 (WL) | 窗寬 (WW) | 突顯結構 |
|------|----------|----------|---------|
| 0 | 40 | 400  | 心肌|
| 1 | 150 | 300  | 主動脈 |
| 2 | 300 | 1500  | 鈣化 |
- **Step 1**：將單通道 CT 影像轉換為多通道格式（應用三種不同窗值設定）
- **Step 2**：將多通道影像拆分為 nnU-Net v2標準格式（三個單通道檔案）
- **Step 3**：將 nnU-Net v2格式影像轉換為 PNG 圖片供視覺化檢查

## 功能概述
### Step 1: 多通道影像生成
- 讀取原始 CT 影像（NIfTI 格式）
- 應用三種窗值設定生成三個通道：
  - **通道 1 (軟組織窗)**：WL=40, WW=400 - 突顯心肌
  - **通道 2 (血管窗)**：WL=150, WW=300 - 突顯主動脈
  - **通道 3 (骨窗)**：WL=300, WW=1500 - 突顯高密度鈣化點
- 合併為單一多通道 NIfTI 檔案

### Step 2: nnU-Net 格式拆分
- 將多通道 NIfTI 檔案拆分為 nnU-Net 標準格式
- 為每個患者生成三個單通道檔案（_0000、_0001、_0002）

### Step 3: PNG 視覺化轉換
- 將 nnU-Net 格式影像轉換為 PNG 圖片
- 每個患者每個切片生成 3 張 PNG（對應 3 個通道）

### 1. 準備資料結構

確保您的資料夾結構如下：

```
您的專案資料夾/
├── imagesTs/              # 測試集原始 CT 影像
│   ├── patient0051_0000.nii.gz
│   ├── patient0052_0000.nii.gz
│   └── ...
├── imagesTr/              # 訓練集原始 CT 影像
│   ├── patient0001_0000.nii.gz
│   ├── patient0002_0000.nii.gz
│   └── ...
└── multichannel_ct.ipynb  # 本程式
```

**重要**：檔名必須遵循命名規範：`patient[ID]_0000.nii.gz`

---

## 使用指南

### Step 1: 生成多通道影像

#### 修改路徑

在 Step 1 程式碼區塊的主程式執行區，修改以下路徑：

```python
# --- 測試集資料夾路徑 ---
ct_images_dir_test = "/您的路徑/imagesTs"
output_dir_test = "/您的路徑/imagesTs_multichannel_3"

# --- 訓練集資料夾路徑 ---
ct_images_dir_train = "/您的路徑/imagesTr"
output_dir_train = "/您的路徑/imagesTr_multichannel_3"
```
生成的檔案結構：
```
imagesTr_multichannel_3/
├── patient0001_0000.nii.gz  # 包含 3 個通道
├── patient0002_0000.nii.gz
└── ...
```
---

### Step 2: 拆分多通道影像為 nnU-Net v2 格式

#### 修改路徑

在 Step 2 程式碼區塊中，修改以下路徑：
1. 修改 `multichannel_dir_test`：改為你的 Step 1 輸出資料夾（測試集多通道影像）
2. 修改 `nnunet_images_dir_test`：改為你想儲存 nnU-Net 格式影像的資料夾（測試集），請將這份資料夾放到nnUNet v2中的imagesTs
3. 修改 `multichannel_dir_train`：改為你的 Step 1 輸出資料夾（訓練集多通道影像）
4. 修改 `nnunet_images_dir_train`：改為你想儲存 nnU-Net 格式影像的資料夾（訓練集），請將這份資料夾放到nnUNet v2中的imagesTr

### Step 3: 轉換為 PNG 圖片（視覺化檢查）

#### 修改路徑
1. 修改`nnunet_images_dir_train` : 你step2 的nnunet_images_dir_train 的路徑
3. 修改 `nnunet_png_dirPNG` : png 輸出的資料夾
---
#### 檔案命名規範

- **輸入檔案（原始 CT）**：`patient[ID]_0000.nii.gz`
  - 例如：`patient0001_0000.nii.gz`
  
- **Step 1 輸出（多通道影像）**：`patient[ID]_0000.nii.gz`
  - 檔名不變，但內部包含 3 個通道
  - 例如：`patient0001_0000.nii.gz`（內含 3 通道）
  
- **Step 2 輸出（nnU-Net v2格式）**：
  - `patient[ID]_0000.nii.gz`（通道 0 - 心肌）
  - `patient[ID]_0001.nii.gz`（通道 1 - 主動脈瓣膜）
  - `patient[ID]_0002.nii.gz`（通道 2 - 鈣化）
  - 例如：`patient0001_0000.nii.gz`, `patient0001_0001.nii.gz`, `patient0001_0002.nii.gz`
  
- **Step 3 輸出（PNG 檔案）**：`slice_[切片編號]_channel[通道編號].png`
  - 例如：`slice_0000_channel0.png`, `slice_0000_channel1.png`, `slice_0000_channel2.png`
---
# 訓練重現步驟 :
- 我們是將三個方法中預測出來的 score 大於 0.796 的 fold 做 ensemble，之後再拿方法一產生的後處理(Keep largest connected component )做最後的後處理，所以我們需要先將三個方法中 score 大於 0.796 的 fold 都先訓練好之後再一起做 ensemble 和 後處理。
- 三個方法都必須先安裝 nnU-Net v2，我們是使用
- GeForce RTX 3090
- python = 3.10 
- Cuda版本12.4
```pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124```
- ```git clone https://github.com/MIC-DKFZ/nnUNet.git```

# 方法 1 :
- 使用預設的 nnU-Net v2 去訓練，有三個地方要做更改分別是```dataset.json``` 和```splits_final.json```有將檔案放在方法一的資料夾中，以及imagesTr，imagesTs 裡的病人資料要換成多通道的。
- 訓練完五個 fold 之後分別將五個fold做預測
- ```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 0```
- ```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 1```
- ```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 2```
- ```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 3```
- ```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 4```
- 再將五個 fold 做 ensemble 
```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 0 1 2 3 4``` 
- 之後做後處理
```nnUNetv2_apply_postprocessing -i <input_folder> -o <output_folder> -pp <postprocessing_json> --plans_json <plans_file>```
- 將 score 大於 0.796 的fold 0 和 fold 1 做 ensemble 之後，並將集成的預測權重存下來。
```nnUNetv2_predict -i {input_folder} -o {output_folder_method_1} -d 520 -c 3d_fullres -f 0 1 --save_probabilities``` 

# 方法 2 :
- 使用預設的 nnU-Net v2 去訓練，有三個地方要做更改分別是```dataset.json``` 內容和```splits_final.json```要做修改，有放在方法二的資料夾中，以及imagesTr，imagesTs 裡的病人資料要換成多通道的。
- 我們是訓練完十個 fold 之後分別將十個fold做預測，但因為我們只取 score 大於 0.796 的 fold ，所以根據我們的訓練結果你們只要取第0和第3個fold做訓練和預測就好。
- ```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 0```
- ```nnUNetv2_predict -i {input_folder} -o {output_folder} -d 520 -c 3d_fullres -f 3```
- 將 score 大於 0.796 的fold 0 和 fold 3 做 ensemble 之後，並將集成的預測權重存下來。
```nnUNetv2_predict -i {input_folder} -o {output_folder_method_2} -d 520 -c 3d_fullres -f 0 3 --save_probabilities```

# 方法 3 : 
- 使用預設的 nnU-Net v2 去訓練，有四個地方要做更改分別是```dataset.json``` 內容和```splits_final.json```要做修改，有放在方法三的資料夾中，以及將```nnunetv2/training/loss```裡的```compound_losses.py```替換成方法三資料夾中的，並新增一個```tversky.py```在```nnunetv2/training/loss```
- 將5個 fold 做 ensemble，並將集成的預測權重存下來。
```nnUNetv2_predict -i {input_folder} -o {output_folder_method_3} -d 520 -c 3d_fullres -f 0 1 2 3 4 --save_probabilities```
---
### 最後將 score 大於 0.796 的 fold 做 ensemble + 後處理(Keep largest connected component )

- ensemble 
```
nnUNetv2_ensemble -i \
{output_folder_method_1} \
{output_folder_method_2} \
{output_folder_method_3} \
- o {要儲存 ensemble 的資料夾}
```
- 後處理(Keep largest connected component )
```
nnUNetv2_apply_postprocessing \
-i {ensemble 的資料夾} \
-o {要儲存後處理的資料夾} \
-pp_pkl {方法一在做後處理之後會在nnUNet_results/Dataset520_trueLumen/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4會產生一個postprocessing.pkl，這邊放它的路徑} \
-plans_json {方法一在做後處理之後會在nnUNet_results/Dataset520_trueLumen/nnUNetTrainer__nnUNetPlans__3d_fullres/crossval_results_folds_0_1_2_3_4會產生一個plans.json，這邊放它的路徑} \
-np 8
```
以上就是我 Private leaderboard：0.806289 的重現。 

