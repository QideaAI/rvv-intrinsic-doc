/* { dg-do compile } */
/* { dg-options "-march=rv64gcv_zvfh -mabi=lp64d -Wno-psabi -O3 -fno-schedule-insns -fno-schedule-insns2" } */

#include <riscv_vector.h>

typedef _Float16 float16_t;
typedef float float32_t;
typedef double float64_t;

vbool1_t test_vmsif_m_b1_mu(vbool1_t vm, vbool1_t vd, vbool1_t vs2, size_t vl) {
  return __riscv_vmsif_mu(vm, vd, vs2, vl);
}

vbool2_t test_vmsif_m_b2_mu(vbool2_t vm, vbool2_t vd, vbool2_t vs2, size_t vl) {
  return __riscv_vmsif_mu(vm, vd, vs2, vl);
}

vbool4_t test_vmsif_m_b4_mu(vbool4_t vm, vbool4_t vd, vbool4_t vs2, size_t vl) {
  return __riscv_vmsif_mu(vm, vd, vs2, vl);
}

vbool8_t test_vmsif_m_b8_mu(vbool8_t vm, vbool8_t vd, vbool8_t vs2, size_t vl) {
  return __riscv_vmsif_mu(vm, vd, vs2, vl);
}

vbool16_t test_vmsif_m_b16_mu(vbool16_t vm, vbool16_t vd, vbool16_t vs2, size_t vl) {
  return __riscv_vmsif_mu(vm, vd, vs2, vl);
}

vbool32_t test_vmsif_m_b32_mu(vbool32_t vm, vbool32_t vd, vbool32_t vs2, size_t vl) {
  return __riscv_vmsif_mu(vm, vd, vs2, vl);
}

vbool64_t test_vmsif_m_b64_mu(vbool64_t vm, vbool64_t vd, vbool64_t vs2, size_t vl) {
  return __riscv_vmsif_mu(vm, vd, vs2, vl);
}
/* { dg-final { scan-assembler-times {vseti?vli\s+[a-z0-9]+,\s*[a-z0-9]+,\s*e[0-9]+,\s*mf?[1248],\s*t[au],\s*m[au]\s+vmsif\.[ivxfswum.]+\s+} 7 } } */