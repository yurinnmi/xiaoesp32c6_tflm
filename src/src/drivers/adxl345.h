
#ifndef ADXL345_H_
#define ADXL345_H_

void adxl345_init(void);
void adxl345_deinit(void);
void adxl345_getdata(int16_t *x_raw, int16_t *y_raw, int16_t *z_raw);

#endif  // ADXL345_H_