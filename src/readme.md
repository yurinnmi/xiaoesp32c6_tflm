# TensorFlow Lite for Microcontrollers XIAO ESP32C と 加速度センサーで動きを検知

## Build

ツール：VSCode, Platform builder, Arduino Framewark  
使用ライブラリ：TensorFlow Lite for Microcontrollers　 (以降、TFLM)  

srcフォルダ以下に、PlatformIO用設定があるため、VSCodeでsrcフォルダを開く。一番上のフォルダを開くと、ビルドもダウンロードができない。  
以降は、ESP32に対する通常のビルド方法、ダウンロード方法。  


## シリアルポート(COM)転送速度
　921600 bps  

## シリアルポート　コマンド
  mode infer           : inference mode  
  mode log             : logging mode (100Hz raw stream)  
  start W|RING|SLOPE|UNK: emit START marker, set active label  
  end                  : emit END marker, clear active label  
  status               : print current status  
