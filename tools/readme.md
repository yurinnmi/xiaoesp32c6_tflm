# このフォルダについて

このフォルダのpythonスクリプトは、自作モデル用に作成したものです。  
エラーチェック等、足りていない部分は多々ありますが、AsIsのためご容赦ください。  
一部ChatGPTによる生成物を含みます。  

作業の流れ：  
　- 取得ログを本体サンプルプログラムに対し、不要部を取り除き、サンプル本体と同様のフィルタ処置を行う。(build_dataset.py)  
　- 学習（Keras）→ TFLM化（float/int8）　(train_gesture.py)  
　- C言語用に変換 (make_model_header.py)  
　- このモデルに必要な演算子を確認  (dump_tflm_ops_from_carray.py)  

<br><br>

# session_nm.csv
ログ取得結果ファイル

<br><br>

# build_dataset.py
## 使用方法
python build_dataset.py --in session1.csv --out session1.npz

## 生成物
Numpyバイナリである　*.npzファイル

## メモ
①raw -> g(0.004) -> DC除去(alpha=0.90) -> gain(1.5) -> clip(±2)  
②マーカー区間からサンプルを取り出し  
③128サンプルWindowsを stride=32 で生成  
④Window全体（128×3）でZ-score（mean/varは全要素）  
⑤X:(N,128,3) float32, y:(N,) int64 を npz に保存  


出力される npz の中身 :  
X : (N,128,3) float32（推論入力と同じ前処理＋窓Z-score済み）  
y : (N,) int64（W=0, RING=1, SLOPE=2, UNKNOWN=3）  

label_names :  
収集/前処理パラメータ（fs/window/stride/scale/alpha/gain/clip 等）  
headers（H行があれば）  
source  


収集品質のチェック（build_dataset.py の出力で見る）:  
実行後に表示される以下を確認。  
- bad_dt が 0 に近い事（10msからのズレが少ない） 
- class ... windows が極端に偏っていない事  
- UNKNOWNは多めでOK 
- intervals with 0 windows が大量に出ない事。
出る場合：START〜END が短すぎる（または pre/post が大きすぎる）

<br><br>

# train_gesture.py
## 使用方法
python train_gesture.py --train session_01.npz session_02.npz --test session_03.npz

## 生成物
gesture_model.keras (使用しない)
gesture_model_float32.tflite

## メモ
学習（Keras）→ TFLM化（float/int8）

<br><br>

# make_model_header.py
## 使用方法
python make_model_header.py --in gesture_model_float32.tflite --out gesture_model_float32.h --name g_gesture_model


## 生成物
gesture_model_float32.h
gesture_model_float32.cc

gesture_model_float32.tflite からcc, hを作る。(C言語で使えるように)

<br><br>

# dump_tflm_ops_from_carray.py
## 使用方法

1) テキスト出力  
python dump_tflm_ops_from_carray.py gesture_model_float32.c  

2) C配列から解析（Markdown表で出力）-> 未確認  
python dump_tflm_ops_from_carray.py gesture_model_float32.c --markdown  

3) 配列が複数ある／誤検出する場合（symbol指定が確実）-> 未確認  
python dump_tflm_ops_from_carray.py gesture_model_float32.h --symbol gesture_model_float32 --markdown  


## 生成物  

テキストの場合： 
```
[INFO] unique ops (count x builtin/name/version/custom):
    5 x builtin= 70 EXPAND_DIMS            ver=1 custom=- resolver=AddExpandDims
    5 x builtin= 22 RESHAPE                ver=1 custom=- resolver=AddReshape
    3 x builtin=  3 CONV_2D                ver=1 custom=- resolver=AddConv2D
    3 x builtin=  0 ADD                    ver=1 custom=- resolver=AddAdd
    2 x builtin= 17 MAX_POOL_2D            ver=1 custom=- resolver=AddMaxPool2D
    2 x builtin=  9 FULLY_CONNECTED        ver=1 custom=- resolver=AddFullyConnected
    1 x builtin= 40 MEAN                   ver=1 custom=- resolver=AddMean
    1 x builtin= 25 SOFTMAX                ver=1 custom=- resolver=AddSoftmax
```

## メモ
このモデルに必要な演算子とそのresolverを表示するツール。

