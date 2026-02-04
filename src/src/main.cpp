#include <Arduino.h>
#include <Wire.h>
#include <HardwareSerial.h>

//For Tensorflow
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/core/api/error_reporter.h"

//#include "magic_wand_model_data.h"
#include "gesture_model_float32.h"
//#include "tensorflow/lite/version.h"

//#define DEBUG_LOG_OUTPUT

// ADXL345 I2Cアドレス（ALT ADDRESS=GNDの場合）
#define ADXL345_ADDR 0x53

// ADXL345 レジスタ定義
#define REG_BW_RATE    0x2C
#define REG_POWER_CTL  0x2D
#define REG_DATA_FORMAT 0x31
#define REG_DATAX0     0x32

#define SCALE_FACTOR 0.004f // g/LSB
//#define SCALE_FACTOR 0.03923f * 0.1 // g/LSB
//#define SCALE_FACTOR 4.000f // mg/LSB

int log_count = 0;

// ===== TFLM arena =====
constexpr int kTensorArenaSize = 240 * 1024;
//constexpr int kTensorArenaSize = 120 * 1024;
alignas(16) uint8_t DRAM_ATTR tensor_arena[kTensorArenaSize];
// --

void writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void readRegisters(uint8_t reg, uint8_t count, uint8_t *buf) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.endTransmission(false);
  Wire.requestFrom(ADXL345_ADDR, count);
  for (uint8_t i = 0; i < count; i++) {
    if (Wire.available()) {
      buf[i] = Wire.read();
    }
  }
}

class SerialErrorReporter : public tflite::ErrorReporter {
 public:
  int Report(const char* format, va_list args) override {
    char buf[256];
    vsnprintf(buf, sizeof(buf), format, args);
    Serial.println(buf);
    return 0;
  }
};

// ===== TFLM =====
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;

  // まずは大きめに。後で削る
//  constexpr int kTensorArenaSize = 40 * 1024;
 // alignas(16) uint8_t tensor_arena[kTensorArenaSize];

  // magic_wand は float32 入力が一般的
  tflite::MicroMutableOpResolver<16> resolver;
}

static void PrintModelOps(const tflite::Model* model) {
  if (!model || !model->subgraphs()) {
    Serial.println("Model or subgraphs is null");
    return;
  }
  const auto* subgraphs = model->subgraphs();
  if (subgraphs->size() == 0) {
    Serial.println("No subgraphs");
    return;
  }
  const auto* g = subgraphs->Get(0);
  const auto* ops = g->operators();
  const auto* opcodes = model->operator_codes();

  Serial.print("Operators count: ");
  Serial.println(ops ? ops->size() : 0);

  if (!ops || !opcodes) return;

  for (uint32_t i = 0; i < ops->size(); i++) {
    const auto* op = ops->Get(i);
    const int opcode_index = op->opcode_index();
    const auto* code = opcodes->Get(opcode_index);

    // builtin_code は enum (BuiltinOperator)。ここでは数値で出します。
    int builtin = (int)code->builtin_code();
    int custom = code->custom_code() ? 1 : 0;

    Serial.print("op[");
    Serial.print(i);
    Serial.print("] builtin=");
    Serial.print(builtin);
    Serial.print(" custom=");
    Serial.println(custom);
  }
}

void setup() {
  Serial.begin(921600); 
  Wire.begin();  // SDA/SCL ピンはESP32C6のデフォルトピンを使用

  delay(1000);

  Serial.println("************************************************");
  Serial.println("Start TF DEMO or Log gettting DEMO");
  Serial.println(" Command");
  Serial.println("   mode log   : Capture Logs for creating model");
  Serial.println("   mode infer : Run inference");
  Serial.println("************************************************");

  delay(100);
  
  // レジスタ設定
  writeRegister(REG_BW_RATE, 0x0A);      // 出力データレート設定 100MHz
  writeRegister(REG_DATA_FORMAT, 0x29);  // ±4g, FULL_RES, right-justified
  writeRegister(REG_POWER_CTL, 0x08);    // 測定モード有効
#ifdef DEBUG_LOG_OUTPUT
  Serial.println("ADXL345 Initialized");
#endif

  resolver.AddConv2D();          // builtin=3
  resolver.AddReshape();         // builtin=22
  resolver.AddAdd();             // builtin=0
  resolver.AddMean();            // builtin=40
  resolver.AddFullyConnected();  // builtin=9
  resolver.AddSoftmax();         // builtin=25

  // あなたのモデルだと MaxPool/ExpandDims/Squeeze が入っている可能性が高いので、
  // すでに使っているなら残してOK（入れても害はありません）
  resolver.AddMaxPool2D();       // builtin=17（出てくることが多い）
  resolver.AddExpandDims();      // builtin=70（出てくることが多い）
  resolver.AddSqueeze();         // builtin=43（出てくることが多い）

  // ===== TFLM init =====
//  model = tflite::GetModel(g_magic_wand_model_data);
  model = tflite::GetModel(g_gesture_model);

//  if (model->version() != TFLITE_SCHEMA_VERSION) {
//    Serial.println("Model schema mismatch");
//    while (1) delay(1000);
//  }

#ifdef DEBUG_LOG_OUTPUT
  PrintModelOps(model);
#endif

  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    delay(1000);
    while (1) delay(1000);
  }
#ifdef DEBUG_LOG_OUTPUT
  Serial.print("arena used bytes = ");
  Serial.print(interpreter->arena_used_bytes());
  Serial.println("");
  delay(500);
#endif
  input = interpreter->input(0);

#ifdef DEBUG_LOG_OUTPUT
  Serial.print("Input type: "); Serial.println(input->type);
  Serial.print("Input bytes: "); Serial.println(input->bytes);
  Serial.print("input dims="); 
  for (int i = 0; i < input->dims->size; i++) {
    Serial.print(input->dims->data[i]); Serial.print(i+1<input->dims->size ? "x" : "\n");
  }
  Serial.println("");
#endif

pinMode(LED_BUILTIN, OUTPUT); 
digitalWrite(LED_BUILTIN, HIGH);

Serial.println("Initialization Finished.");
    //For TLFM [end]

}

// ====== 追加：モード/ラベル管理 ======
enum RunMode : uint8_t { MODE_INFER = 0, MODE_LOG = 1 };
static RunMode g_mode = MODE_INFER;
//static RunMode g_mode = MODE_LOG;

enum LabelId : int8_t { L_NONE=-1, L_W=0, L_RING=1, L_SLOPE=2, L_UNK=3 };
static LabelId g_label_active = L_NONE;
static int g_gesture_id = 0;

// コマンド受信バッファ
static char g_cmd_buf[64];
static int  g_cmd_len = 0;

static const char* labelToStr(LabelId l) {
  switch (l) {
    case L_W: return "W";
    case L_RING: return "RING";
    case L_SLOPE: return "SLOPE";
    case L_UNK: return "UNK";
    default: return "NONE";
  }
}

static LabelId strToLabel(const char* s) {
  if (!s) return L_NONE;
  // 大文字小文字は雑に吸収（必要なら厳密化してください）
  if (!strcasecmp(s, "W")) return L_W;
  if (!strcasecmp(s, "RING")) return L_RING;
  if (!strcasecmp(s, "SLOPE")) return L_SLOPE;
  if (!strcasecmp(s, "UNK") || !strcasecmp(s, "UNKNOWN") || !strcasecmp(s, "NONE")) return L_UNK;
  return L_NONE;
}

// マーカー行出力：M,t_ms,label,gesture_id,START/END
static void emitMarker(uint64_t t_ms, const char* ev, LabelId label, int gesture_id) {
  Serial.print("M,");
  Serial.print((uint32_t)t_ms); // 32bitで足りない場合は分割出力にする
  Serial.print(",");
  Serial.print(labelToStr(label));
  Serial.print(",");
  Serial.print(gesture_id);
  Serial.print(",");
  Serial.println(ev);
}

static void printHelp() {
  Serial.println("Commands:");
  Serial.println("  mode infer           : inference mode");
  Serial.println("  mode log             : logging mode (100Hz raw stream)");
  Serial.println("  start W|RING|SLOPE|UNK: emit START marker, set active label");
  Serial.println("  end                  : emit END marker, clear active label");
  Serial.println("  status               : print current status");
}

static void printStatus() {
  Serial.print("STATUS mode=");
  Serial.print(g_mode == MODE_LOG ? "LOG" : "INFER");
  Serial.print(" label_active=");
  Serial.print(labelToStr(g_label_active));
  Serial.print(" gesture_id=");
  Serial.println(g_gesture_id);
}

// 1行コマンド処理（簡易パーサ）
static void handleCommandLine(char* line) {
  // 先頭/末尾空白除去（簡易）
  while (*line == ' ' || *line == '\t') line++;
  if (*line == 0) return;

  // トークナイズ
  char* cmd = strtok(line, " \t\r\n");
  if (!cmd) return;

  if (!strcasecmp(cmd, "help")) { printHelp(); return; }
  if (!strcasecmp(cmd, "status")) { printStatus(); return; }

  if (!strcasecmp(cmd, "mode")) {
    char* arg = strtok(nullptr, " \t\r\n");
    if (arg && !strcasecmp(arg, "log")) {
      g_mode = MODE_LOG;
      Serial.println("OK mode=LOG");
    } else if (arg && (!strcasecmp(arg, "infer") || !strcasecmp(arg, "inference"))) {
      g_mode = MODE_INFER;
      Serial.println("OK mode=INFER");
    } else {
      Serial.println("ERR mode {log|infer}");
    }
    return;
  }

  if (!strcasecmp(cmd, "start")) {
    char* lab = strtok(nullptr, " \t\r\n");
    LabelId l = strToLabel(lab);
    if (l == L_NONE) {
      Serial.println("ERR start {W|RING|SLOPE|UNK}");
      return;
    }
    g_gesture_id++;
    g_label_active = l;
    uint32_t t_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);
    emitMarker(t_ms, "START", g_label_active, g_gesture_id);
    Serial.println("OK start");
    return;
  }

  if (!strcasecmp(cmd, "end")) {
    if (g_label_active == L_NONE) {
      Serial.println("ERR end (no active label)");
      return;
    }
    uint64_t t_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);
    emitMarker(t_ms, "END", g_label_active, g_gesture_id);
    g_label_active = L_NONE;
    Serial.println("OK end");
    return;
  }

  Serial.println("ERR unknown command (type 'help')");
}

// シリアル受信（ノンブロッキング）
static void pollSerialCommands() {
  while (Serial.available() > 0) {
    int c = Serial.read();
    if (c < 0) break;

    if (c == '\n' || c == '\r') {
      if (g_cmd_len > 0) {
        g_cmd_buf[g_cmd_len] = 0;
        handleCommandLine(g_cmd_buf);
        g_cmd_len = 0;
      }
    } else {
      if (g_cmd_len < (int)sizeof(g_cmd_buf) - 1) {
        g_cmd_buf[g_cmd_len++] = (char)c;
      }
    }
  }
}

void Do_something(int num){
  if(num==0){
    digitalWrite(LED_BUILTIN, LOW);
    delay(1000);
    digitalWrite(LED_BUILTIN, HIGH);
  }
  else if(num==1){
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(500);
    digitalWrite(LED_BUILTIN, LOW);
    delay(500);
    digitalWrite(LED_BUILTIN, HIGH);
  }
  else if(num==2){
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
    digitalWrite(LED_BUILTIN, HIGH);
    delay(250);
    digitalWrite(LED_BUILTIN, LOW);
    delay(250);
    digitalWrite(LED_BUILTIN, HIGH);
  }

}

// ====== loop() 置換：100Hz固定サンプル + 推論間引き + 収集モード ======
void loop() {
  static uint32_t last_hit_ms = 0;
  const uint32_t COOLDOWN_MS = 500;
  if (millis() - last_hit_ms < COOLDOWN_MS) return;

  // ---- 100Hz 固定スケジューリング ----
  static bool timing_init = false;
  static uint32_t next_tick_us = 0;
  const uint32_t kPeriodUs = 10000; // 10ms = 100Hz

  if (!timing_init) {
    next_tick_us = (uint32_t)micros();
    timing_init = true;
  }

  // ここでコマンド処理（周期内で吸収）
  pollSerialCommands();

  // ---- 既存：初期化（tensor確認）----
  static bool initialized = false;
  static int window_samples = 0;
  if (!initialized) {
    if (input == nullptr) {
      Serial.println("input is null");
      // 初期化失敗でも周期は回す
    } else {
      if (input->type != kTfLiteFloat32) {
        Serial.println("This example assumes float32 input.");
        // ここは致命的なので止めてもよいが、止めるなら while(1) 等
      }
      window_samples = input->bytes / (sizeof(float) * 3);
#ifdef DEBUG_LOG_OUTPUT
      Serial.print("window_samples = ");
      Serial.println(window_samples);
#endif
      if (window_samples <= 0 || window_samples > 256) {
        Serial.println("Invalid window_samples");
      }
      initialized = true;
    }
  }
  // ---- 状態（推論/前処理）----
  static int write_pos = 0;
  static int filled = 0;
  static int step = 0;

  // DC除去用（重力・オフセット追従）
  static float ax = 0.0f, ay = 0.0f, az = 0.0f;
  const float alpha = 0.90f;
  const float gain  = 1.5f;

  // 推論間引き
  const int kInvokeEvery = 32; // 100Hzなら 0.32s ごと。0.64sにしたいなら 64。

  // 運動量ヒステリシス
  const float TH_START = 0.08f;
  const float TH_STOP  = 0.02f;

  // 確信度しきい値
  const float SCORE_TH = 0.55f;

  const float MARGIN_TH = 0.15f;

  static int hist[3] = {-1, -1, -1};
  static int hist_pos = 0;

  static bool in_motion = false;

  // ---- ADXL345 read（毎回）----
  uint8_t buf[6];
  readRegisters(REG_DATAX0, 6, buf);

  int16_t x_raw = (int16_t)((buf[1] << 8) | buf[0]);
  int16_t y_raw = (int16_t)((buf[3] << 8) | buf[2]);
  int16_t z_raw = (int16_t)((buf[5] << 8) | buf[4]);

  // タイムスタンプ
  uint32_t t_ms = (uint32_t)(esp_timer_get_time() / 1000ULL);

  // ---- 収集モード：RAWを毎サンプル出力 ----
  if (g_mode == MODE_LOG) {
    if(g_label_active != L_NONE){
    // S,t_ms,x_raw,y_raw,z_raw
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
      emitMarker(t_ms, "END", g_label_active, g_gesture_id);
      g_label_active = L_NONE;
      Serial.println("OK end");    
      }
    // 推論は行わない（収集品質優先）
    }
  }

  // ---- 前処理（推論モードでのみ必要、ただし安定のため毎回計算してもOK）----
  // 単位は現状コード踏襲（変数名は _g）
  float x_g = x_raw * SCALE_FACTOR;
  float y_g = y_raw * SCALE_FACTOR;
  float z_g = z_raw * SCALE_FACTOR;

  // DC除去
  ax = alpha * ax + (1.0f - alpha) * x_g;
  ay = alpha * ay + (1.0f - alpha) * y_g;
  az = alpha * az + (1.0f - alpha) * z_g;

  float x = (x_g - ax) * gain;
  float y = (y_g - ay) * gain;
  float z = (z_g - az) * gain;

  auto clip = [](float v) {
    if (v > 2.0f) return 2.0f;
    if (v < -2.0f) return -2.0f;
    return v;
  };
  x = clip(x); y = clip(y); z = clip(z);

  // ---- リング（最大256想定）----
  static float ring[256 * 3];
  if (window_samples <= 0) window_samples = 128; // フォールバック
  ring[write_pos * 3 + 0] = x;
  ring[write_pos * 3 + 1] = y;
  ring[write_pos * 3 + 2] = z;

  write_pos = (write_pos + 1) % window_samples;
  if (filled < window_samples) filled++;
  step++;

  // ---- 推論（INFERモードのみ）----
  if (g_mode == MODE_INFER && initialized && input != nullptr && filled >= window_samples) {
    if ((step % kInvokeEvery) == 0) {
      // リング→入力テンソル（古→新）
      float* in = input->data.f;
      int p = write_pos;
      for (int i = 0; i < window_samples; i++) {
        const int rb = p * 3;
        const int ib = i * 3;
        in[ib + 0] = ring[rb + 0];
        in[ib + 1] = ring[rb + 1];
        in[ib + 2] = ring[rb + 2];
        p = (p + 1) % window_samples;
      }

      // RMS(pre)で運動量判定（窓があるときだけ）
      const int n = window_samples * 3;
      float rms = 0.0f;
      for (int i = 0; i < n; i++) rms += in[i] * in[i];
      rms = sqrtf(rms / n);

      // ヒステリシス更新
      if (!in_motion) {
        if (rms > TH_START) in_motion = true;
      } else {
        if (rms < TH_STOP)  in_motion = false;
      }

      // 静止なら推論スキップ（ただしサンプリングは継続）
      if (in_motion) {
        // 窓全体 Z-score 正規化
        float mean = 0.0f;
        for (int i = 0; i < n; i++) mean += in[i];
        mean /= n;

        float var = 0.0f;
        for (int i = 0; i < n; i++) {
          float d = in[i] - mean;
          var += d * d;
        }
        var /= n;
        float inv_std = 1.0f / sqrtf(var + 1e-6f);

        for (int i = 0; i < n; i++) {
          in[i] = (in[i] - mean) * inv_std;
        }

        // Invoke
        if (interpreter->Invoke() == kTfLiteOk) {
          TfLiteTensor* output = interpreter->output(0);
          float p[4] = {
            output->data.f[0],
            output->data.f[1],
            output->data.f[2],
            output->data.f[3],
          };

          Serial.print("out=");
          Serial.print(p[0], 4); Serial.print(" ");
          Serial.print(p[1], 4); Serial.print(" ");
          Serial.print(p[2], 4); Serial.print(" ");
          Serial.println(p[3], 4);
#ifdef DEBUG_LOG_OUTPUT
          Serial.print("output type="); Serial.println(output->type);
          Serial.print("output dims=");
          for (int i = 0; i < output->dims->size; i++) {
            Serial.print(output->dims->data[i]); Serial.print(i+1<output->dims->size ? "x" : "\n");
          }
          Serial.println("");
#endif
          if (output && output->type == kTfLiteFloat32) {
            int argmax = 0;
            float best = output->data.f[0];
            for (int i = 1; i < 4; i++) {
              if (output->data.f[i] > best) { best = output->data.f[i]; argmax = i; }
            }

            // 多数決（直近3回）
            hist[hist_pos] = argmax;
            hist_pos = (hist_pos + 1) % 3;

            int votes[4] = {0,0,0,0};
            for (int i = 0; i < 3; i++) {
              if (hist[i] >= 0 && hist[i] < 4) votes[hist[i]]++;
            }
            int maj = 0;
            for (int c = 1; c < 4; c++) {
              if (votes[c] > votes[maj]) maj = c;
            }

            // UNK(=3)はイベントにしない
            if (maj == 3) {
              // デバッグで見たいならここでログだけ出す
              // Serial.println("UNK");
              delay(10);
              return;
            }

            // majの確率（これが “score”）
            float pmaj = p[maj];
            // maj以外の最大確率を取り、marginを作る（maj 기준）
            float p2 = 0.0f;
            for (int i = 0; i < 4; i++) {
              if (i == maj) continue;
              if (p[i] > p2) p2 = p[i];
            }
            float margin = pmaj - p2;

            if (votes[maj] >= 2 && pmaj >= SCORE_TH && margin >= MARGIN_TH) {
              Serial.print("HIT class=");
              Serial.print(maj);
              Serial.print(" score=");
              Serial.print(pmaj, 4);
              Serial.print(" margin=");
              Serial.print(margin, 4);
              Serial.print(" rms=");
              Serial.println(rms, 5);
              Do_something(maj);
              last_hit_ms = millis();
            }
          }
        }
        else{
            Serial.println("Invoke failed");
            return;
        }
      }
    }
  }

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



