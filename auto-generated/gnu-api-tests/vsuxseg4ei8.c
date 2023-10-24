/* { dg-do compile } */
/* { dg-options "-march=rv64gcv_zvfh -mabi=lp64d -Wno-psabi -O3 -fno-schedule-insns -fno-schedule-insns2" } */

#include <riscv_vector.h>

typedef _Float16 float16_t;
typedef float float32_t;
typedef double float64_t;

void test_vsuxseg4ei8_v_f16mf4x4(float16_t *rs1, vuint8mf8_t vs2, vfloat16mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16mf4x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f16mf2x4(float16_t *rs1, vuint8mf4_t vs2, vfloat16mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f16m1x4(float16_t *rs1, vuint8mf2_t vs2, vfloat16m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f16m2x4(float16_t *rs1, vuint8m1_t vs2, vfloat16m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f32mf2x4(float32_t *rs1, vuint8mf8_t vs2, vfloat32mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f32mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f32m1x4(float32_t *rs1, vuint8mf4_t vs2, vfloat32m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f32m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f32m2x4(float32_t *rs1, vuint8mf2_t vs2, vfloat32m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f32m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f64m1x4(float64_t *rs1, vuint8mf8_t vs2, vfloat64m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f64m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f64m2x4(float64_t *rs1, vuint8mf4_t vs2, vfloat64m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f64m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8mf8x4(int8_t *rs1, vuint8mf8_t vs2, vint8mf8x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8mf8x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8mf4x4(int8_t *rs1, vuint8mf4_t vs2, vint8mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8mf4x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8mf2x4(int8_t *rs1, vuint8mf2_t vs2, vint8mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8m1x4(int8_t *rs1, vuint8m1_t vs2, vint8m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8m2x4(int8_t *rs1, vuint8m2_t vs2, vint8m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16mf4x4(int16_t *rs1, vuint8mf8_t vs2, vint16mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16mf4x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16mf2x4(int16_t *rs1, vuint8mf4_t vs2, vint16mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16m1x4(int16_t *rs1, vuint8mf2_t vs2, vint16m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16m2x4(int16_t *rs1, vuint8m1_t vs2, vint16m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i32mf2x4(int32_t *rs1, vuint8mf8_t vs2, vint32mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i32mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i32m1x4(int32_t *rs1, vuint8mf4_t vs2, vint32m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i32m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i32m2x4(int32_t *rs1, vuint8mf2_t vs2, vint32m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i32m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i64m1x4(int64_t *rs1, vuint8mf8_t vs2, vint64m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i64m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i64m2x4(int64_t *rs1, vuint8mf4_t vs2, vint64m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i64m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8mf8x4(uint8_t *rs1, vuint8mf8_t vs2, vuint8mf8x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8mf8x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8mf4x4(uint8_t *rs1, vuint8mf4_t vs2, vuint8mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8mf4x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8mf2x4(uint8_t *rs1, vuint8mf2_t vs2, vuint8mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8m1x4(uint8_t *rs1, vuint8m1_t vs2, vuint8m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8m2x4(uint8_t *rs1, vuint8m2_t vs2, vuint8m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16mf4x4(uint16_t *rs1, vuint8mf8_t vs2, vuint16mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16mf4x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16mf2x4(uint16_t *rs1, vuint8mf4_t vs2, vuint16mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16m1x4(uint16_t *rs1, vuint8mf2_t vs2, vuint16m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16m2x4(uint16_t *rs1, vuint8m1_t vs2, vuint16m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u32mf2x4(uint32_t *rs1, vuint8mf8_t vs2, vuint32mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u32mf2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u32m1x4(uint32_t *rs1, vuint8mf4_t vs2, vuint32m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u32m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u32m2x4(uint32_t *rs1, vuint8mf2_t vs2, vuint32m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u32m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u64m1x4(uint64_t *rs1, vuint8mf8_t vs2, vuint64m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u64m1x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u64m2x4(uint64_t *rs1, vuint8mf4_t vs2, vuint64m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u64m2x4(rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f16mf4x4_m(vbool64_t vm, float16_t *rs1, vuint8mf8_t vs2, vfloat16mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16mf4x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f16mf2x4_m(vbool32_t vm, float16_t *rs1, vuint8mf4_t vs2, vfloat16mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f16m1x4_m(vbool16_t vm, float16_t *rs1, vuint8mf2_t vs2, vfloat16m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f16m2x4_m(vbool8_t vm, float16_t *rs1, vuint8m1_t vs2, vfloat16m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f16m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f32mf2x4_m(vbool64_t vm, float32_t *rs1, vuint8mf8_t vs2, vfloat32mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f32mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f32m1x4_m(vbool32_t vm, float32_t *rs1, vuint8mf4_t vs2, vfloat32m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f32m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f32m2x4_m(vbool16_t vm, float32_t *rs1, vuint8mf2_t vs2, vfloat32m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f32m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f64m1x4_m(vbool64_t vm, float64_t *rs1, vuint8mf8_t vs2, vfloat64m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f64m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_f64m2x4_m(vbool32_t vm, float64_t *rs1, vuint8mf4_t vs2, vfloat64m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_f64m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8mf8x4_m(vbool64_t vm, int8_t *rs1, vuint8mf8_t vs2, vint8mf8x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8mf8x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8mf4x4_m(vbool32_t vm, int8_t *rs1, vuint8mf4_t vs2, vint8mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8mf4x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8mf2x4_m(vbool16_t vm, int8_t *rs1, vuint8mf2_t vs2, vint8mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8m1x4_m(vbool8_t vm, int8_t *rs1, vuint8m1_t vs2, vint8m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i8m2x4_m(vbool4_t vm, int8_t *rs1, vuint8m2_t vs2, vint8m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i8m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16mf4x4_m(vbool64_t vm, int16_t *rs1, vuint8mf8_t vs2, vint16mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16mf4x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16mf2x4_m(vbool32_t vm, int16_t *rs1, vuint8mf4_t vs2, vint16mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16m1x4_m(vbool16_t vm, int16_t *rs1, vuint8mf2_t vs2, vint16m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i16m2x4_m(vbool8_t vm, int16_t *rs1, vuint8m1_t vs2, vint16m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i16m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i32mf2x4_m(vbool64_t vm, int32_t *rs1, vuint8mf8_t vs2, vint32mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i32mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i32m1x4_m(vbool32_t vm, int32_t *rs1, vuint8mf4_t vs2, vint32m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i32m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i32m2x4_m(vbool16_t vm, int32_t *rs1, vuint8mf2_t vs2, vint32m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i32m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i64m1x4_m(vbool64_t vm, int64_t *rs1, vuint8mf8_t vs2, vint64m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i64m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_i64m2x4_m(vbool32_t vm, int64_t *rs1, vuint8mf4_t vs2, vint64m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_i64m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8mf8x4_m(vbool64_t vm, uint8_t *rs1, vuint8mf8_t vs2, vuint8mf8x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8mf8x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8mf4x4_m(vbool32_t vm, uint8_t *rs1, vuint8mf4_t vs2, vuint8mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8mf4x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8mf2x4_m(vbool16_t vm, uint8_t *rs1, vuint8mf2_t vs2, vuint8mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8m1x4_m(vbool8_t vm, uint8_t *rs1, vuint8m1_t vs2, vuint8m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u8m2x4_m(vbool4_t vm, uint8_t *rs1, vuint8m2_t vs2, vuint8m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u8m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16mf4x4_m(vbool64_t vm, uint16_t *rs1, vuint8mf8_t vs2, vuint16mf4x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16mf4x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16mf2x4_m(vbool32_t vm, uint16_t *rs1, vuint8mf4_t vs2, vuint16mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16m1x4_m(vbool16_t vm, uint16_t *rs1, vuint8mf2_t vs2, vuint16m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u16m2x4_m(vbool8_t vm, uint16_t *rs1, vuint8m1_t vs2, vuint16m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u16m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u32mf2x4_m(vbool64_t vm, uint32_t *rs1, vuint8mf8_t vs2, vuint32mf2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u32mf2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u32m1x4_m(vbool32_t vm, uint32_t *rs1, vuint8mf4_t vs2, vuint32m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u32m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u32m2x4_m(vbool16_t vm, uint32_t *rs1, vuint8mf2_t vs2, vuint32m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u32m2x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u64m1x4_m(vbool64_t vm, uint64_t *rs1, vuint8mf8_t vs2, vuint64m1x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u64m1x4_m(vm, rs1, vs2, vs3, vl);
}

void test_vsuxseg4ei8_v_u64m2x4_m(vbool32_t vm, uint64_t *rs1, vuint8mf4_t vs2, vuint64m2x4_t vs3, size_t vl) {
  return __riscv_vsuxseg4ei8_v_u64m2x4_m(vm, rs1, vs2, vs3, vl);
}
/* { dg-final { scan-assembler-times {vseti?vli\s+[a-z0-9]+,\s*[a-z0-9]+,\s*e[0-9]+,\s*mf?[1248],\s*t[au],\s*m[au]\s+vsuxseg4ei8\.[ivxfswum.]+\s+} 74 } } */