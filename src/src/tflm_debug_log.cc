#include <Arduino.h>
#include <cstdarg>
#include <cstdio>

extern "C" void DebugLog(const char* s) {
  Serial.print(s);
}

extern "C" void DebugVsnprintf(char* buffer, size_t buffer_size,
                               const char* format, va_list args) {
  vsnprintf(buffer, buffer_size, format, args);
}