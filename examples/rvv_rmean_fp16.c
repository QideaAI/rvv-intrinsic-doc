#include "common.h"
#include <riscv_vector.h>
#include <float.h>
#include "common.h"

#ifndef ARRAY_SIZE
    #define ARRAY_SIZE 256
#endif

_Float16 A[ARRAY_SIZE] = {1.0f};
_Float16 C_golden;
_Float16 C;

//golden scalar function
_Float16 rmean_golden(_Float16 *a, int N) {
    int i;
	_Float16 s = 0.0f;
    for(i = 0; i < N; i ++) {
        s += a[i];
    }
		
    return (s/N);
}

//vector rmean function
_Float16 rmean_vec(_Float16 *a, int N) {
    // set vlmax and initialize variables
    size_t vlmax = __riscv_vsetvlmax_e16m1();
    vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
    vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(0, vlmax);

    //vector add
    _Float16 len = (_Float16)N;
    for (size_t vl; N > 0; N -= vl, a += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_a = __riscv_vle16_v_f16m1(a, vl);
        vec_s = __riscv_vfadd_vv_f16m1(vec_a, vec_s, vl);
    }

    //reduction add
    vfloat16m1_t vec_sum;
    vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_s, vec_zero, vlmax);

    //generate scalar mean
	_Float16 rmean = __riscv_vfmv_f_s_f16m1_f16(vec_sum)/len;
	return rmean;
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
        A[i] = (rand() / (float) RAND_MAX);
        //printf("A: %f\n", (float)A[i]);
    }

    //check Vector size
    size_t vlmax = __riscv_vsetvlmax_e16m1();
    printf("VLEN: %d\n", (int)vlmax);

    //compute golden
    C_golden = rmean_golden(A, N);

    //compute vec
#ifdef COUNT_CYCLE
    int count_start, count_end;
    count_start = read_perf_counter();
#endif

    C = rmean_vec(A, N);

#ifdef COUNT_CYCLE
    count_end = read_perf_counter();
    printf("Performance counter start: %d\n", count_start);
    printf("Performance counter end: %d\n", count_end);
    printf("Cycle count: %d\n", count_end - count_start);
#endif

    int pass = 1;
    if (!fp16_eq(C_golden, C, 1e-3)) {
        printf("fail, %f=!%f\n", (float)C_golden, (float)C);
        pass = 0;
    }
    
    if (pass) {
        printf("pass, %f==%f\n", (float)C_golden, (float)C);
    }
    return (pass == 0);
}
