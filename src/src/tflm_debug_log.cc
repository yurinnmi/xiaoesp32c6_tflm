#include <Arduino.h>
#include <cstdarg>
#include <cstdio>

// TFLM が参照するシンボル
extern "C" void DebugLog(const char* s) {
  Serial.print(s);
}

// 可変引数ログを使うパスがある場合の保険（環境により参照されます）
extern "C" void DebugVsnprintf(char* buffer, size_t buffer_size,
                               const char* format, va_list args) {
  vsnprintf(buffer, buffer_size, format, args);
}