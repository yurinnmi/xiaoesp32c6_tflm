#include <Arduino.h>
#include "drivers/serialcom.h"

// ====== 追加：モード/ラベル管理 ======
static LabelId g_label_active = L_NONE;
static int g_gesture_id = 0;

static RunMode g_mode = MODE_INFER;
//static RunMode g_mode = MODE_LOG;

// コマンド受信バッファ
static char g_cmd_buf[64];
static int  g_cmd_len = 0;

class SerialComClass {
public:
    SerialComClass(void);
    ~SerialComClass(void);
// シリアル受信（ノンブロッキング）
    static void pollSerialCommands(void);
    static void emitMarker(uint64_t t_ms, const char* ev, LabelId label, int gesture_id);

private:
    static const char* labelToStr(LabelId l);

    static LabelId strToLabel(const char* s);

    static void printHelp() ;

    static void printStatus();
    static void handleCommandLine(char* line);
};

SerialComClass::SerialComClass(void)
{
  g_mode = MODE_INFER;
}
//  デストラクタ
SerialComClass::~SerialComClass(void)
{
}

void SerialComClass::pollSerialCommands(void) {
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

const char* SerialComClass::labelToStr(LabelId l) {
  switch (l) {
    case L_W: return "W";
    case L_RING: return "RING";
    case L_SLOPE: return "SLOPE";
    case L_UNK: return "UNK";
    default: return "NONE";
  }
}

LabelId SerialComClass::strToLabel(const char* s) {
  if (!s) return L_NONE;
  // 大文字小文字は雑に吸収（必要なら厳密化してください）
  if (!strcasecmp(s, "W")) return L_W;
  if (!strcasecmp(s, "RING")) return L_RING;
  if (!strcasecmp(s, "SLOPE")) return L_SLOPE;
  if (!strcasecmp(s, "UNK") || !strcasecmp(s, "UNKNOWN") || !strcasecmp(s, "NONE")) return L_UNK;
  return L_NONE;
}

// マーカー行出力：M,t_ms,label,gesture_id,START/END
void SerialComClass::emitMarker(uint64_t t_ms, const char* ev, LabelId label, int gesture_id) {
  Serial.print("M,");
  Serial.print((uint32_t)t_ms); // 32bitで足りない場合は分割出力にする
  Serial.print(",");
  Serial.print(labelToStr(label));
  Serial.print(",");
  Serial.print(gesture_id);
  Serial.print(",");
  Serial.println(ev);
}

void SerialComClass::printHelp() {
  Serial.println("Commands:");
  Serial.println("  mode infer           : inference mode");
  Serial.println("  mode log             : logging mode (100Hz raw stream)");
  Serial.println("  start W|RING|SLOPE|UNK: emit START marker, set active label");
  Serial.println("  end                  : emit END marker, clear active label");
  Serial.println("  status               : print current status");
}

void SerialComClass::printStatus() {
  Serial.print("STATUS mode=");
  Serial.print(g_mode == MODE_LOG ? "LOG" : "INFER");
  Serial.print(" label_active=");
  Serial.print(labelToStr(g_label_active));
  Serial.print(" gesture_id=");
  Serial.println(g_gesture_id);
}

// 1行コマンド処理（簡易パーサ）
void SerialComClass::handleCommandLine(char* line) {
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

SerialComClass* pSC = 0;
void SerialCommands_init(void){
  pSC = new SerialComClass();
}

void SerialCommands_deinit(void){
  delete pSC;  
}

void SerialCommands_poll(void) {
  if(pSC){
    pSC->pollSerialCommands();
  }
}
void SerialCommands_emitMarker(uint64_t t_ms, const char* ev, LabelId label, int gesture_id) {
  if(pSC){
    pSC->emitMarker(t_ms, ev, label, gesture_id);
  }
}

int SerialCommands_Getgid(void) {
  return g_gesture_id;
}

RunMode SerialCommands_GetMode(void) {
  return g_mode;
}

LabelId SerialCommands_isActive(void) {
  return g_label_active;
}

void SerialCommands_ClearActive(void) {
  g_label_active = L_NONE;
}

