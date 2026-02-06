
#ifndef TFLM_IF_H_
#define TFLM_IF_H_

//For Tensorflow
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/core/api/error_reporter.h"

#include "gesture_model_float32.h"
//#include "magic_wand_model_data.h"

//#define DEBUG_LOG_OUTPUT

// ===== TFLM arena =====
constexpr int kTensorArenaSize = 240 * 1024;

// --

bool tflm_init(void);
void tflm_infer(float x, float y, float z, int *maj);

#endif  // TFLM_IF_H_