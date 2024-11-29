/*
    This function computes the LayerNorm of a vector per the following equation:
    Y = (X - E(x))/sqrt(VAR(x) + epsilon) * gamma + beta
    where E(x) is the mean of the vector and VAR(x) is the variance of the vector.
*/

#include "common.h"
#include <riscv_vector.h>
#include <float.h>
#include <math.h>
#include "common.h"

#ifndef ARRAY_SIZE
    #define ARRAY_SIZE 32
#endif

_Float16 X[ARRAY_SIZE] = {1.0f};
_Float16 _epsilon = 1e-5f;
_Float16 _gamma = 1.0f;
_Float16 _beta = 0.0f;
_Float16 Y_golden[ARRAY_SIZE] = {0.f};
_Float16 Y[ARRAY_SIZE] = {0.f};

//golden MVN function
void mvn_golden(_Float16 *x, _Float16 *y, _Float16 epsilon, _Float16 gamma, _Float16 beta, int N) {
    int i;
	_Float16 s = 0.0f;

    //compute mean
    for(i = 0; i < N; i ++) {
        s += x[i];
    }
    _Float16 E = (s/N);

    //compute mean subtraction and variance
    _Float16 V = 0.0f;
    for(i = 0; i < N; i ++) {
        y[i] = (x[i] - E);
        V += y[i] * y[i];
    }

    //normalization
    _Float16 D = 1.0f / sqrt (V + epsilon);
    for(i = 0; i < N; i ++) {
        y[i] = y[i] * D * gamma + beta;
    }    

}

//vector MVN function
void mvn_vec(_Float16 *x, _Float16 *y, _Float16 epsilon, _Float16 gamma, _Float16 beta, int N) {
    // set vlmax and initialize variables
    size_t vlmax = __riscv_vsetvlmax_e16m1();
    vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
    vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(0, vlmax);
    vfloat16m1_t vec_epsilon = __riscv_vfmv_v_f_f16m1(epsilon, vlmax);
    vfloat16m1_t vec_gama    = __riscv_vfmv_v_f_f16m1(gamma, vlmax);
    vfloat16m1_t vec_beta    = __riscv_vfmv_v_f_f16m1(beta, vlmax);
    vfloat16m1_t vec_n = __riscv_vfmv_v_f_f16m1((_Float16)N, vlmax);

    //vectored mean
    for (size_t vl; N > 0; N -= vl, x += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x, vl);
        vec_s = __riscv_vfadd_vv_f16m1(vec_x, vec_s, vl);
    }
    vfloat16m1_t vec_sum, vec_mean;
    vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_s, vec_zero, vlmax);
    vec_mean = __riscv_vfdiv_vv_f16m1(vec_sum, vec_n, vlmax);

    //vectored mean subtraction and variance
    vec_s = __riscv_vfmv_v_f_f16m1(0, vlmax);
    for (size_t vl; N > 0; N -= vl, x += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x, vl);
        vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
        vec_y = __riscv_vfsub_vv_f16m1(vec_x, vec_mean, vl);
        vec_s = __riscv_vfmacc_vv_f16m1(vec_s, vec_y, vec_y, vl);
    }    

    //normalization
    vec_s = __riscv_vfadd_vv_f16m1(vec_s, vec_epsilon, vlmax);
    vec_s = __riscv_vfsqrt_v_f16m1(vec_s, vlmax);

    for (size_t vl; N > 0; N -= vl, x += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
        vec_y = __riscv_vfmul_vv_f16m1(vec_y, vec_s, vl);
        vec_y = __riscv_vfmul_vv_f16m1(vec_y, vec_gama, vl);
        vec_y = __riscv_vfadd_vv_f16m1(vec_y, vec_beta, vl);
        __riscv_vse16_v_f16m1(y, vec_y, vl);
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
        X[i] = (rand() / (float) RAND_MAX);
        //printf("X: %f\n", (float)X[i]);
    }

    //check Vector size
    size_t vlmax = __riscv_vsetvlmax_e16m1();
    printf("VLEN: %d\n", (int)vlmax);

    //compute golden
    mvn_golden(X, Y, _epsilon, _gamma, _beta, N);

    //compute vec
#ifdef COUNT_CYCLE
    int count_start, count_end;
    count_start = read_perf_counter();
#endif

     mvn_vec(X, Y, _epsilon, _gamma, _beta, N);

#ifdef COUNT_CYCLE
    count_end = read_perf_counter();
    printf("Performance counter start: %d\n", count_start);
    printf("Performance counter end: %d\n", count_end);
    printf("Cycle count: %d\n", count_end - count_start);
#endif

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (!fp16_eq(Y_golden[i], Y[i], 1e-5)) {
        printf("index %d fail, %f=!%f\n", i, (float)Y_golden[i], (float)Y[i]);
        pass = 0;
        }
    }
    if (pass) {
        for (i = 0; i < N; ++i) {
            //printf("Y %f\n", (float)Y[i]);
        }
        printf("pass\n");
    }
    return (pass == 0);
}