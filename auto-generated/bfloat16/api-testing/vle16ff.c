#include <riscv_vector.h>
#include <stdint.h>

vbfloat16mf4_t test_vle16ff_v_bf16mf4(const __bf16 *rs1, size_t *new_vl,
                                      size_t vl) {
  return __riscv_vle16ff_v_bf16mf4(rs1, new_vl, vl);
}

vbfloat16mf2_t test_vle16ff_v_bf16mf2(const __bf16 *rs1, size_t *new_vl,
                                      size_t vl) {
  return __riscv_vle16ff_v_bf16mf2(rs1, new_vl, vl);
}

vbfloat16m1_t test_vle16ff_v_bf16m1(const __bf16 *rs1, size_t *new_vl,
                                    size_t vl) {
  return __riscv_vle16ff_v_bf16m1(rs1, new_vl, vl);
}

vbfloat16m2_t test_vle16ff_v_bf16m2(const __bf16 *rs1, size_t *new_vl,
                                    size_t vl) {
  return __riscv_vle16ff_v_bf16m2(rs1, new_vl, vl);
}

vbfloat16m4_t test_vle16ff_v_bf16m4(const __bf16 *rs1, size_t *new_vl,
                                    size_t vl) {
  return __riscv_vle16ff_v_bf16m4(rs1, new_vl, vl);
}

vbfloat16m8_t test_vle16ff_v_bf16m8(const __bf16 *rs1, size_t *new_vl,
                                    size_t vl) {
  return __riscv_vle16ff_v_bf16m8(rs1, new_vl, vl);
}

vbfloat16mf4_t test_vle16ff_v_bf16mf4_m(vbool64_t vm, const __bf16 *rs1,
                                        size_t *new_vl, size_t vl) {
  return __riscv_vle16ff_v_bf16mf4_m(vm, rs1, new_vl, vl);
}

vbfloat16mf2_t test_vle16ff_v_bf16mf2_m(vbool32_t vm, const __bf16 *rs1,
                                        size_t *new_vl, size_t vl) {
  return __riscv_vle16ff_v_bf16mf2_m(vm, rs1, new_vl, vl);
}

vbfloat16m1_t test_vle16ff_v_bf16m1_m(vbool16_t vm, const __bf16 *rs1,
                                      size_t *new_vl, size_t vl) {
  return __riscv_vle16ff_v_bf16m1_m(vm, rs1, new_vl, vl);
}

vbfloat16m2_t test_vle16ff_v_bf16m2_m(vbool8_t vm, const __bf16 *rs1,
                                      size_t *new_vl, size_t vl) {
  return __riscv_vle16ff_v_bf16m2_m(vm, rs1, new_vl, vl);
}

vbfloat16m4_t test_vle16ff_v_bf16m4_m(vbool4_t vm, const __bf16 *rs1,
                                      size_t *new_vl, size_t vl) {
  return __riscv_vle16ff_v_bf16m4_m(vm, rs1, new_vl, vl);
}

vbfloat16m8_t test_vle16ff_v_bf16m8_m(vbool2_t vm, const __bf16 *rs1,
                                      size_t *new_vl, size_t vl) {
  return __riscv_vle16ff_v_bf16m8_m(vm, rs1, new_vl, vl);
}