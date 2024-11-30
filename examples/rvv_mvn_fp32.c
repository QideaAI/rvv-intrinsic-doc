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
    #define ARRAY_SIZE 2048
#endif

float X[ARRAY_SIZE] = {1.0f};
float _epsilon = 1e-5f;
float _gamma = 1.0f;
float _beta = 0.0f;
float Y_golden[ARRAY_SIZE] = {0.f};
float Y[ARRAY_SIZE] = {0.f};

//golden MVN function
void mvn_golden(float *x, float *y, float epsilon, float gamma, float beta, int N) {
    int i;
	float s = 0.0f;

    //compute mean
    for(i = 0; i < N; i ++) {
        s += x[i];
    }
    float E = (s/N);
    printf("sum: %f, mean: %f\n", (float)s, (float)E);

    //compute mean subtraction and variance
    float V = 0.0f;
    for(i = 0; i < N; i ++) {
        y[i] = (x[i] - E);
        V += y[i] * y[i];
    }

    // //normalization
    float D = 1.0f / sqrt (V + epsilon);
    printf("var: %f, inv_sqrt_var: %f\n", (float)V, (float)D);

    for(i = 0; i < N; i ++) {
        y[i] = y[i] * D * gamma + beta;
    }    

}

//vector MVN function
void mvn_vec(float *x, float *y, float epsilon, float gamma, float beta, int N) {
    // set vlmax and initialize variables
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    vfloat32m1_t vec_zero = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
    vfloat32m1_t vec_gama    = __riscv_vfmv_v_f_f32m1(gamma, vlmax);
    vfloat32m1_t vec_beta    = __riscv_vfmv_v_f_f32m1(beta, vlmax);

    float *rx = x;
    float *ry = y;
    int rN = N;

    //vectored mean
    for (size_t vl; N > 0; N -= vl, x += vl) {
        vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
        vec_s = __riscv_vfadd_vv_f32m1(vec_x, vec_s, vl);
    }

    //generate scalar mean
    vfloat32m1_t vec_sum, vec_mean_var;
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
	float sum = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    float E = sum/(float)rN;
    vec_mean_var = __riscv_vfmv_v_f_f32m1(E, vlmax);
    //printf("sum: %f, mean: %f\n", (float)sum, (float)E);

    // vectored mean subtraction and variance
    vec_s = __riscv_vfmv_v_f_f32m1(0, vlmax);
    N = rN;
    x = rx;
    for (size_t vl; N > 0; N -= vl, x += vl, y += vl) {
        vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t vec_x = __riscv_vle32_v_f32m1(x, vl);
        vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(y, vl);
        vec_y = __riscv_vfsub_vv_f32m1(vec_x, vec_mean_var, vl);
        vec_s = __riscv_vfmacc_vv_f32m1(vec_s, vec_y, vec_y, vl);
        __riscv_vse32_v_f32m1(y, vec_y, vl);
    }    

    //generate scalar variance and inv_sqrt_var
    vec_sum = __riscv_vfredusum_vs_f32m1_f32m1(vec_s, vec_zero, vlmax);
	float V = __riscv_vfmv_f_s_f32m1_f32(vec_sum);
    float D = 1.0f / sqrt (V + epsilon);
    //printf("var: %f, inv_sqrt_var: %f\n", (float)V, (float)D);

    //normalization
    N = rN;
    y = ry;
    vec_mean_var = __riscv_vfmv_v_f_f32m1(D, vlmax);
    for (size_t vl; N > 0; N -= vl, y += vl) {
        vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t vec_y = __riscv_vle32_v_f32m1(y, vl);
        vec_y = __riscv_vfmul_vv_f32m1(vec_y, vec_mean_var, vl);
        vec_y = __riscv_vfmul_vv_f32m1(vec_y, vec_gama, vl);
        vec_y = __riscv_vfadd_vv_f32m1(vec_y, vec_beta, vl);
        __riscv_vse32_v_f32m1(y, vec_y, vl);
    }   
}

int fp_eq(float reference, float actual, float relErr)
{
  // if near zero, do absolute error instead.
  float absErr = relErr * ((fabsf(reference) > relErr) ? fabsf(reference) : relErr);
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
    size_t vlmax = __riscv_vsetvlmax_e32m1();
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
        if (!fp_eq(Y_golden[i], Y[i], 5e-4)) {
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