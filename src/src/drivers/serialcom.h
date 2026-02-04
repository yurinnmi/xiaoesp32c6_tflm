
#ifndef SERIALCOM_H_
#define SERIALCOM_H_

enum RunMode : uint8_t { MODE_INFER = 0, MODE_LOG = 1 };
enum LabelId : int8_t { L_NONE=-1, L_W=0, L_RING=1, L_SLOPE=2, L_UNK=3 };

void SerialCommands_init(void);
void SerialCommands_deinit(void);
void SerialCommands_poll(void);
void SerialCommands_emitMarker(uint64_t t_ms, const char* ev, LabelId label, int gesture_id);

int SerialCommands_Getgid(void);
RunMode SerialCommands_GetMode(void);
LabelId SerialCommands_isActive(void);
void SerialCommands_ClearActive(void);

#endif  // SERIALCOM_H_