#include <Arduino.h>
#include "tflm_if.h"

//---Parameters----
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

//リングバッファ
static float ring[256 * 3];
static int write_pos = 0;
static int filled = 0;
static int step = 0;
//---Parameters----

alignas(16) uint8_t DRAM_ATTR tensor_arena[kTensorArenaSize];

static bool initialized = false;
static int window_samples = 0;

/* ---------------Utilities------------------*/
namespace {
  const tflite::Model* model = nullptr;
  tflite::MicroInterpreter* interpreter = nullptr;
  TfLiteTensor* input = nullptr;
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

void put_to_ring(float x, float y, float z){
    // ---- リングバッファ（最大256想定）----
  if (window_samples <= 0) window_samples = 128; // フォールバック

  ring[write_pos * 3 + 0] = x;
  ring[write_pos * 3 + 1] = y;
  ring[write_pos * 3 + 2] = z;

  write_pos = (write_pos + 1) % window_samples;
  if (filled < window_samples) filled++;
  step++;
}

/* ---------------Interface------------------*/

/*
*/
bool tflm_init(void){
  //resolver :このモデルが使う演算（OP）を登録しておくリスト
  //モデル（.tflite）の中には「この推論では Conv2D を使う」「Mean を使う」などの情報が入っている
  //しかし TFLM 側は、最初は何も演算を知らない状態なので、使うものだけ手動登録しておく
  resolver.AddConv2D();          // builtin=3 畳み込み（Convolution）「時間×チャンネル」を2Dと見てフィルタを当てる感じ
  resolver.AddReshape();         // builtin=22 配列の形（shape）を変えるだけの演算 例：(128,3) を (1,128,3,1)
  resolver.AddAdd();             // builtin=0 足し算
  resolver.AddMean();            // builtin=40 平均を取る演算
  resolver.AddFullyConnected();  // builtin=9 全結合層（Dense）
  resolver.AddSoftmax();         // builtin=25 確率化の演算 出力を [0..1] の確率っぽい値にして、合計が1になるようにするなど
  resolver.AddMaxPool2D();       // builtin=17 プーリング（最大値を拾う）
  resolver.AddExpandDims();      // builtin=70 次元を1つ増やす演算


/*
  resolver.AddSqueeze();         // builtin=43 サイズ1の次元を消す演算
  resolver.AddMul();     //乗算 係数を掛ける処理など
  resolver.AddDepthwiseConv2D();  //Depthwise畳み込み　普通のConv2Dより軽いタイプ
  resolver.AddRelu(); //ReLU活性化 マイナスを0に潰す処理
  resolver.AddQuantize();  //量子化：float→int8など
  resolver.AddDequantize();  //逆量子化：int8→float
  */

  //C配列からTFLMが読めるモデル構造に変換（参照するだけなので軽い）
  model = tflite::GetModel(g_gesture_model);

  //モデルを動かす実行エンジン(MicroInterpreter)作成
  //resolver（使える演算）と tensor_arena（作業用メモリ）を使って推論
  static tflite::MicroInterpreter static_interpreter(
      model, resolver, tensor_arena, kTensorArenaSize);
   interpreter = &static_interpreter;

  //モデルを実行するために必要な 作業領域（テンソル） を tensor_arena の中に配置
  //[配置されるもの]
  //  - 入力テンソル（(1,128,3)等）
  //  - 中間テンソル（Convの途中結果など）
  //  - 出力テンソル
  //OP登録不足,arena領域が小さすぎるとエラーになる。
  TfLiteStatus alloc_status = interpreter->AllocateTensors();
  if (alloc_status != kTfLiteOk) {
    Serial.println("AllocateTensors failed");
    delay(1000);
    while (1) delay(1000);
  }
  //モデルの 入力テンソルへのポインタを取得
  input = interpreter->input(0);

  //いろいろ確認
  if (input == nullptr) {
    Serial.println("FAILED: input is null");    // 初期化失敗
  } else {
    if (input->type != kTfLiteFloat32) {
      Serial.println("FAILED: This example assumes float32 input.");
      Serial.print("Input type: "); Serial.println(input->type);
    }
    else{
      window_samples = input->bytes / (sizeof(float) * 3);
#ifdef DEBUG_LOG_OUTPUT
      Serial.print("window_samples = ");
      Serial.println(window_samples);
#endif
      if (window_samples <= 0 || window_samples > 256) {
        Serial.println("FAILED: Invalid window_samples");
      }
      else {
        initialized = true;
      }
    }
  }

#ifdef DEBUG_LOG_OUTPUT
  if(initialized == true){
    PrintModelOps(model);

    Serial.print("arena used bytes = ");
    Serial.print(interpreter->arena_used_bytes());
    Serial.println("");
    delay(100);
    Serial.print("Input type: "); Serial.println(input->type);
    Serial.print("Input bytes: "); Serial.println(input->bytes);
    Serial.print("input dims="); 
    for (int i = 0; i < input->dims->size; i++) {
      Serial.print(input->dims->data[i]); Serial.print(i+1<input->dims->size ? "x" : "\n");
    }
    Serial.println("");
    delay(100);
  }
#endif

  return initialized;

}

void get_inputdata(float* in){
  if(in != nullptr) {
    int p = write_pos;
    for (int i = 0; i < window_samples; i++) {
      const int rb = p * 3;
      const int ib = i * 3;
      in[ib + 0] = ring[rb + 0];
      in[ib + 1] = ring[rb + 1];
      in[ib + 2] = ring[rb + 2];
      p = (p + 1) % window_samples;
    }
  }
}

void change_inmotion(float* in, int n){
  // RMS(pre)で運動量判定
  if(in != nullptr) {
    float rms = 0.0f;
    for (int i = 0; i < n; i++) rms += in[i] * in[i];
    rms = sqrtf(rms / n);

    // ヒステリシス更新
    if (!in_motion) {
      if (rms > TH_START) in_motion = true;
    } else {
      if (rms < TH_STOP)  in_motion = false;
    }
  }
} 

void Z_score_Normalization(float* in, int n){
  if(in != nullptr) {
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
  }
}

void tflm_infer(float x, float y, float z, int *maj_result){

  if(maj_result == nullptr) return;

  put_to_ring(x, y, z);

  if ((step % kInvokeEvery) == 0) {
    // RINGバッファから入力テンソル取得 （古⇒新）
      float* in = input->data.f;
      get_inputdata(in);

      //動静判定(in_motion)
      const int n = window_samples * 3;
      change_inmotion(in, n);

      // 静止なら推論スキップ（ただしサンプリングは継続）
      if (in_motion) {
        // Window全体 Z-score 正規化
        Z_score_Normalization(in, n);

        // Invoke
        if (interpreter->Invoke() == kTfLiteOk) {
          TfLiteTensor* output = interpreter->output(0);
          //各クラスのスコア
          float p[4] = {
            output->data.f[0],
            output->data.f[1],
            output->data.f[2],
            output->data.f[3],
          };
#ifdef DEBUG_LOG_OUTPUT
          Serial.print("out: 0=");
          Serial.print(p[0], 4); Serial.print(" 1=");
          Serial.print(p[1], 4); Serial.print(" 2=");
          Serial.print(p[2], 4); Serial.print(" 3=");
          Serial.println(p[3], 4);
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

            // UNKNOWN(=3)はイベントにしない
            if (maj == 3) {
              delay(10);
              *maj_result = 3; //UNKNOWN
              return; //TODO
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
#ifdef DEBUG_LOG_OUTPUT
              Serial.print("HIT class=");
              Serial.print(maj);
              Serial.print(" score=");
              Serial.print(pmaj, 4);
              Serial.print(" margin=");
              Serial.println(margin, 4);
#endif
              *maj_result  = maj;
            }
          }
        }
        else{
            Serial.println("Invoke failed");
        }
      }
  }
}

