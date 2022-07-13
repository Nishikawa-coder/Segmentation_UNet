# Segmentation with Unet

Unet で画像のセグメンテーションを学習する。

## 目的

本実験では白黒画像内にある`butu`のピクセル単位での検出を Unet で学習することを目的としている。

## 原理

U-Net はセマンティックセグメンテーション用のモデルで、エンコーダとデコーダから成る U 字型のモデルである。  
エンコーダでは入力された画像を何度か畳み込み、その画像の特徴を抽出する。一方、デコーダではエンコーダによって抽出された特徴を受け取り、逆畳み込みによって、入力画像と同じサイズの確率マップを出力する。

## 実験の方法

本実験では、白黒画像とアノテーションマスク（`butu`の位置情報）がセットになっている x3 というデータセットを用い、U-Net を用いて学習し、画像中の`butu`の検出を行った。  
また、学習したモデルに対して推論を行い、PR 曲線および Average Precision、処理時間により評価を行った。

### 準備

ファイルを下記のように置いてください。

```
[user_name]
│
├── data
│   └── x3
└── class3(ディレクトリ名（何でも良い）)
    ├── dataset.py
    ├── inference.ipynb
    ├── train.ipynb
    ├── unet_model.py
    └── unet_parts.py
```

接続しているマシンの`localtmp/users`に`rin(user_name)`を作成してください。

<訓練時>  
train.ipynb の全てのセルを再生してください。初期状態では、1 エポックだけ学習し、`log_output.csv`が生成されます。  
また、`/localtmp/users/rin(user_name)/Unet_1`が生成され`./weights/Unet_1`にファイルが保存されます。

<推論時>  
inference.ipynb の全てのセルを再生してください。  
`show_detection()`では推論結果の画像を表示します（緑：正しい推論、赤:間違った推論、青:推論されなかった gt）。また、最後に推論の処理時間が表示されます。

### 実験条件

データセット：x3(`butu`のみがある画像 101 枚を使用。訓練：80 枚、検証：10 枚、テスト：11 枚)  
エポック数：1000  
学習率：1e-3  
最適化関数：  

#### log_output_init_weight_1000_1000.csv
損失関数：loss_weight=(1,100)  
マシン：palkia 1  
バッチサイズ：4（batch_multiplier=6 なので実質24)  
color_mean = (0.485, 0.456, 0.406)  
color_std = (0.229, 0.224, 0.225)

#### log_output_v1.csv

損失関数：loss_weight=(1,70)  
マシン：palkia 1  
バッチサイズ：4（batch_multiplier=6 なので実質24)  
color_mean=0.18228737997050898  
color_std=0.15940997135888293  

#### log_output_v2.csv

損失関数：loss_weight=(1,50)  
マシン：palkia 0  
バッチサイズ：4（batch_multiplier=6 なので実質24)  
color_mean=0.18228737997050898  
color_std=0.15940997135888293

#### log_output_v3.csv
損失関数：loss_weight=(1,70)  
マシン：victini 0  
バッチサイズ：8（batch_multiplier=3 なので実質24)  
color_mean=0.18228737997050898  
color_std=0.15940997135888293


