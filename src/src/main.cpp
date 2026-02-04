#include <Arduino.h>

#include "drivers/adxl345.h"
#include "drivers/led.h"
#include "drivers/serialcom.h"
#include "tflm_if/tflm_if.h"

#define SCALE_FACTOR 0.004f // g/LSB
//#define SCALE_FACTOR 0.03923f * 0.1 // g/LSB

const uint32_t COOLDOWN_MS = 200;
const uint32_t kPeriodUs = 10000; // 10ms = 100Hz 固定スケジューリング

static bool initialized = false;
static uint32_t last_hit_ms = 0;
static bool timing_init = false;
static uint32_t next_tick_us = 0;
int log_count = 0;

// DC除去用（重力・オフセット追従）
static float ax = 0.0f, ay = 0.0f, az = 0.0f;
const float alpha = 0.90f;
const float gain  = 1.5f;

void wait_until_next(void){

  // ---- 100Hz周期維持（ここだけで待つ）----
  next_tick_us += kPeriodUs;
  int32_t wait = (int32_t)(next_tick_us - (uint32_t)micros());

  if (wait > 0) {
    // ArduinoのdelayMicrosecondsは長すぎると不正確になり得るが、10msなら許容
    delayMicroseconds((uint32_t)wait);
  } else {
    // 遅延が積み上がっている場合は追従（ドリフト抑制）
    next_tick_us = (uint32_t)micros();
  }
}

void output_data(int16_t x_raw, int16_t y_raw, int16_t z_raw, uint32_t t_ms){
  Serial.print("S,");
  Serial.print(t_ms);
  Serial.print(",");
  Serial.print(x_raw);
  Serial.print(",");
  Serial.print(y_raw);
  Serial.print(",");
  Serial.println(z_raw);
  log_count++;
  if(log_count > 180){
    log_count = 0;
    SerialCommands_emitMarker(t_ms, "END", SerialCommands_isActive(), SerialCommands_Getgid());
    SerialCommands_ClearActive();
    Serial.println("OK end");    
  }
}

void Preprocessing(int16_t x_raw, int16_t y_raw, int16_t z_raw, float *x, float *y, float *z){
  if((x != nullptr) && (y != nullptr) && (z != nullptr)){

  // ---- 前処理（推論モードでのみ必要、ただし安定のため毎回計算してもOK）----
  // 単位は現状コード踏襲（変数名は _g）
    float x_g = x_raw * SCALE_FACTOR;
    float y_g = y_raw * SCALE_FACTOR;
    float z_g = z_raw * SCALE_FACTOR;

  // DC除去
    ax = alpha * ax + (1.0f - alpha) * x_g;
    ay = alpha * ay + (1.0f - alpha) * y_g;
    az = alpha * az + (1.0f - alpha) * z_g;

    float x_tmp = (x_g - ax) * gain;
    float y_tmp = (y_g - ay) * gain;
    float z_tmp = (z_g - az) * gain;

    auto clip = [](float v) {
      if (v > 2.0f) return 2.0f;
      if (v < -2.0f) return -2.0f;
      return v;
    };
    *x = clip(x_tmp); *y = clip(y_tmp); *z = clip(z_tmp);
  }
}

void Do_something(int type){
  if(type==0){
    Serial.println("<< W >>");
  }
  else if(type==1){
    Serial.println("<< RING >>");
  }
  else if(type==2){
    Serial.println("<< SLOPE >>");
  }
}

void setup() {
  Serial.begin(921600); 

  adxl345_init();
  led_init();
  SerialCommands_init();

  initialized = tflm_init();

  delay(1000);

  if(initialized == true){
    Serial.println("************************************************");
    Serial.println("Start TF DEMO or Log gettting DEMO");
    Serial.println(" Command");
    Serial.println("   mode log   : Capture Logs for creating model");
    Serial.println("   mode infer : Run inference");
    Serial.println("************************************************");

    delay(100);
  }
}

void loop() {
  if (millis() - last_hit_ms < COOLDOWN_MS) return;

  // ---- 固定スケジューリング ----
  if (!timing_init) {
    next_tick_us = (uint32_t)micros();
    timing_init = true;
  }

  // シリアルコマンド処理受付
  SerialCommands_poll();

  // センサ値取得
  int16_t x_raw = 0;
  int16_t y_raw = 0;
  int16_t z_raw = 0;
  adxl345_getdata(&x_raw, &y_raw, &z_raw);

    // タイムスタンプ取得
  uint32_t t_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);

  if (SerialCommands_GetMode() == MODE_LOG) { // ---- 収集モード：RAWを毎サンプル出力 ----
    if(SerialCommands_isActive() != L_NONE){    // S,t_ms,x_raw,y_raw,z_raw
      output_data(x_raw, y_raw, z_raw, t_ms);
    }
  }
  else{  //推論モード（INFERモード）
    if (initialized != true) return;

    float x = 0;
    float y = 0;
    float z = 0;
    Preprocessing(x_raw, y_raw, z_raw, &x, &y, &z);

    //推論
    int maj = -1;
    tflm_infer(x, y, z, &maj);
    if( (maj < 3) && (maj >= 0)  ){
      Do_something(maj);
      last_hit_ms = millis();
    }
  }
  wait_until_next();
}
