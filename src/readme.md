# XIAO ESP32C6と加速度センサーで動きを検知,BLEでiBeacon送信

## 仕様

Majorコード：0:静止、1:動作中

ツール：VSCode  
使用ライブラリ：NimBLE

対象デバイス：  

- CPU：XIAO esp32C6
- センサ：ADXL345

接続：  
VCC(3.3v)、GND、SDA、SDL端子をお互いに接続し、センサのSDO端とGNDとを接続。

## フローチャート

初期化：

```mermaid
flowchart TD
    I2C初期化 --> BLEビーコン開始 --> ADXL345初期化
```

</BR></BR>

処理ループ：

```mermaid
flowchart TD
    A[センサ値X,Y,Z取得] --> B[X,Y,Z値から動静判定]
    B -- 動いている --> C[現Major値?]
    C -- 0:静止であった --> Major値を1に変更 -->E[10ms待つ]
    C -- 1:動作中であった --> Major値変更なし-->E
    B -- 静止状態 --> D[現Major値?]
    D -- 0:静止であった --> Major値変更なし-->E
    D -- 1:動作中であった --> Major値を0に変更-->E
    E --> A

```

## その他

NOTE記載  

<https://note.com/yuzu_monaka_/n/n0ed9b26451bb>

<https://note.com/yuzu_monaka_/n/n42976252831a>

<https://note.com/yuzu_monaka_/n/n8208c410c1d1>

<https://note.com/yuzu_monaka_/n/n4d729d129264>
