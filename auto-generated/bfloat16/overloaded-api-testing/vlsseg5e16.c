#include <riscv_vector.h>
#include <stdint.h>

vbfloat16mf4x5_t test_vlsseg5e16_v_bf16mf4x5_m(vbool64_t vm, const __bf16 *rs1,
                                               ptrdiff_t rs2, size_t vl) {
  return __riscv_vlsseg5e16(vm, rs1, rs2, vl);
}

vbfloat16mf2x5_t test_vlsseg5e16_v_bf16mf2x5_m(vbool32_t vm, const __bf16 *rs1,
                                               ptrdiff_t rs2, size_t vl) {
  return __riscv_vlsseg5e16(vm, rs1, rs2, vl);
}

vbfloat16m1x5_t test_vlsseg5e16_v_bf16m1x5_m(vbool16_t vm, const __bf16 *rs1,
                                             ptrdiff_t rs2, size_t vl) {
  return __riscv_vlsseg5e16(vm, rs1, rs2, vl);
}
