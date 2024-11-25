#include "common.h"
#include <riscv_vector.h>
#include <float.h>
#include "common.h"

#ifndef ARRAY_SIZE
#define ARRAY_SIZE 2048
#endif

_Float16 A[ARRAY_SIZE];
_Float16 B[ARRAY_SIZE];
_Float16 C_golden[ARRAY_SIZE] = {0.f};
_Float16 C_vec[ARRAY_SIZE] = {0.f};

//golden scalar functioon
void add_golden(_Float16 *a, _Float16 *b, _Float16 *c, int N) {
    int i;
    for(i = 0; i < N; i ++) {
        c[i] = a[i] + b[i];
    }
}

//vector add function
void add_vec(_Float16 *a, _Float16 *b, _Float16 *c, int N) {
    for (size_t vl; N > 0; N -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_a = __riscv_vle16_v_f16m1(a, vl);
        vfloat16m1_t vec_b = __riscv_vle16_v_f16m1(b, vl);

        vfloat16m1_t vec_c = __riscv_vfadd_vv_f16m1(vec_a, vec_b, vl);
        __riscv_vse16_v_f16m1(c, vec_c, vl);
    }
}

int fp16_eq(_Float16 reference, _Float16 actual, _Float16 relErr)
{
  // if near zero, do absolute error instead.
  _Float16 absErr = relErr * ((fabsf(reference) > relErr) ? fabsf(reference) : relErr);
  return fabsf(actual - reference) < absErr;
}

int main() {
    const uint32_t seed = 0xdeadbeef;
    srand(seed);

    int i;
    int N = ARRAY_SIZE;
    for (i = 0; i < N; ++i) {
        A[i] = rand() / (_Float16) RAND_MAX;
        B[i] = rand() / (_Float16) RAND_MAX;
    }

    //check Vector size
    size_t vlmax = __riscv_vsetvlmax_e16m1();
    printf("VLEN: %d\n", (int)vlmax);

    //compute golden
    add_golden(A, B, C_golden, N);

    //compute vec
#ifdef COUNT_CYCLE
    int count_start, count_end;
    count_start = read_perf_counter();
#endif

    add_vec(A, B, C_golden, N);

#ifdef COUNT_CYCLE
    count_end = read_perf_counter();
    printf("Performance counter start: %d\n", count_start);
    printf("Performance counter end: %d\n", count_end);
    printf("Cycle count: %d\n", count_end - count_start);
#endif

}