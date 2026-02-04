#include <Arduino.h>
#include <wire.h>

// ADXL345 I2Cアドレス（ALT ADDRESS=GNDの場合）
#define ADXL345_ADDR 0x53

// ADXL345 レジスタ定義
#define REG_BW_RATE    0x2C
#define REG_POWER_CTL  0x2D
#define REG_DATA_FORMAT 0x31
#define REG_DATAX0     0x32

class ADXL345Class {
public:
    ADXL345Class(void);
    ~ADXL345Class(void);
    static void init(void);
    static void getdata(int16_t *x_raw, int16_t *y_raw, int16_t *z_raw);

private:
    static void writeRegister(uint8_t reg, uint8_t value);
    static void readRegisters(uint8_t reg, uint8_t count, uint8_t *buf);
};

ADXL345Class::ADXL345Class(void)
{
}
//  デストラクタ
ADXL345Class::~ADXL345Class(void)
{
}

void ADXL345Class::writeRegister(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(ADXL345_ADDR);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

void ADXL345Class::readRegisters(uint8_t reg, uint8_t count, uint8_t *buf) {
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

void ADXL345Class::init(void){
  Wire.begin();

  writeRegister(REG_BW_RATE, 0x0A);      // 出力データレート設定 100MHz
  writeRegister(REG_DATA_FORMAT, 0x29);  // ±4g, FULL_RES, right-justified
  writeRegister(REG_POWER_CTL, 0x08);    // 測定モード有効
}

void ADXL345Class::getdata(int16_t *x_raw, int16_t *y_raw, int16_t *z_raw){
  if((x_raw != nullptr) && (y_raw != nullptr) && (z_raw != nullptr)){
    uint8_t buf[6];
    readRegisters(REG_DATAX0, 6, buf);

    *x_raw = (int16_t)((buf[1] << 8) | buf[0]);
    *y_raw = (int16_t)((buf[3] << 8) | buf[2]);
    *z_raw = (int16_t)((buf[5] << 8) | buf[4]);
  }
}

ADXL345Class* pADXLC = 0;
void adxl345_init(void){
  pADXLC = new ADXL345Class();

  if(pADXLC){
    pADXLC->init();
  }
}

void adxl345_deinit(void){
  delete pADXLC;  
}

void adxl345_getdata(int16_t *x_raw, int16_t *y_raw, int16_t *z_raw){
  if(pADXLC){
    pADXLC->getdata(x_raw, y_raw, z_raw);
  }
}