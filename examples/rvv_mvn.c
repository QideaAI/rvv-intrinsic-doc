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
    #define ARRAY_SIZE 1024
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
    printf("sum: %f, mean: %f\n", (float)s, (float)E);

    //compute mean subtraction and variance
    _Float16 V = 0.0f;
    for(i = 0; i < N; i ++) {
        y[i] = (x[i] - E);
        V += y[i] * y[i];
    }

    // //normalization
    _Float16 D = 1.0f / sqrt (V + epsilon);
    printf("var: %f, inv_sqrt_var: %f\n", (float)V, (float)D);

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
    vfloat16m1_t vec_gama    = __riscv_vfmv_v_f_f16m1(gamma, vlmax);
    vfloat16m1_t vec_beta    = __riscv_vfmv_v_f_f16m1(beta, vlmax);

    _Float16 *rx = x;
    _Float16 *ry = y;
    int rN = N;

    //vectored mean
    for (size_t vl; N > 0; N -= vl, x += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x, vl);
        vec_s = __riscv_vfadd_vv_f16m1(vec_x, vec_s, vl);
    }

    //generate scalar mean
    vfloat16m1_t vec_sum, vec_mean_var;
    vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_s, vec_zero, vlmax);
	_Float16 sum = __riscv_vfmv_f_s_f16m1_f16(vec_sum);
    _Float16 E = sum/(float)rN;
    vec_mean_var = __riscv_vfmv_v_f_f16m1(E, vlmax);
    //printf("sum: %f, mean: %f\n", (float)sum, (float)E);

    // vectored mean subtraction and variance
    vec_s = __riscv_vfmv_v_f_f16m1(0, vlmax);
    N = rN;
    x = rx;
    for (size_t vl; N > 0; N -= vl, x += vl, y += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x, vl);
        vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
        vec_y = __riscv_vfsub_vv_f16m1(vec_x, vec_mean_var, vl);
        vec_s = __riscv_vfmacc_vv_f16m1(vec_s, vec_y, vec_y, vl);
        __riscv_vse16_v_f16m1(y, vec_y, vl);
    }    

    //generate scalar variance and inv_sqrt_var
    vec_sum = __riscv_vfredusum_vs_f16m1_f16m1(vec_s, vec_zero, vlmax);
	_Float16 V = __riscv_vfmv_f_s_f16m1_f16(vec_sum);
    _Float16 D = 1.0f / sqrt (V + epsilon);
    //printf("var: %f, inv_sqrt_var: %f\n", (float)V, (float)D);

    //normalization
    N = rN;
    y = ry;
    vec_mean_var = __riscv_vfmv_v_f_f16m1(D, vlmax);
    for (size_t vl; N > 0; N -= vl, y += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
        vec_y = __riscv_vfmul_vv_f16m1(vec_y, vec_mean_var, vl);
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
    mvn_golden(X, Y_golden, _epsilon, _gamma, _beta, N);

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
        if (!fp16_eq(Y_golden[i], Y[i], 5e-2)) {
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