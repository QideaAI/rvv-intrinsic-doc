/*
    This function computes the softmax of a vector Y = softmax(X) peer the following formula:

    Y[i] = exp(X[i] - max(X))/sum[exp(X[i] - max(X))]
    
*/

#include "common.h"
#include <riscv_vector.h>
#include <float.h>
#include <math.h>
#include "common.h"

#ifndef ARRAY_SIZE
    #define ARRAY_SIZE 32
#endif

 #define max(a,b) \
    ({ __typeof__ (a) _a = (a); \
        __typeof__ (b) _b = (b); \
        _a > _b ? _a : _b; })

_Float16 X[ARRAY_SIZE] = {1.0f};
_Float16 Y_golden[ARRAY_SIZE] = {0.f};
_Float16 Y[ARRAY_SIZE] = {0.f};

//golden softmax function
//implement the refernce as 4 loops to ease debug
void softmax_golden(_Float16 *x, _Float16 *y, int N) {
    int i;
	_Float16 x_max = __FLT16_MIN__;

    //compute max
    for(i = 0; i < N; i ++) {
        x_max = max(x_max, x[i]);
    }  
    printf("ref x_max: %f\n", (float)x_max);

    //subtract max
    for(i = 0; i < N; i ++) {
        y[i] = (x[i] - x_max);
    }  

    // //compute exp and the sum of exp
    // _Float16 esum = 0.0f;
    // for(i = 0; i < N; i ++) {
    //     y[i] = exp(y[i]);
    //     esum += y[i];
    // }      
    // _Float16 inv_sum_exp = 1.0f / esum;
    // printf("inv_sum_exp: %f\n", (float)inv_sum_exp);

    // //final softmax normalization
    // for(i = 0; i < N; i ++) {
    //     y[i] = y[i] * inv_sum_exp;
    // } 
}

//vector softmax function
void softmax_vec(_Float16 *x, _Float16 *y, int N) {
    // set vlmax and initialize variables
    size_t vlmax = __riscv_vsetvlmax_e16m1();
    vfloat16m1_t vec_zero = __riscv_vfmv_v_f_f16m1(0, vlmax);
    vfloat16m1_t vec_max = __riscv_vfmv_v_f_f16m1(__FLT16_MIN__, vlmax);

    _Float16 *rx = x;
    _Float16 *ry = y;
    int rN = N;

    //vectored max
    for (size_t vl; N > 0; N -= vl, x += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x, vl);
        vec_max = __riscv_vfmax_vv_f16m1(vec_x, vec_max, vl);
    }

    //generate scalar max
    vec_max = __riscv_vfredmax_vs_f16m1_f16m1(vec_max, vec_zero, vlmax);
	_Float16 x_max = __riscv_vfmv_f_s_f16m1_f16(vec_max);
    vec_max = __riscv_vfmv_v_f_f16m1(x_max, vlmax);
    printf("imp x_max: %f\n", (float)x_max);

    // vectored max subtraction
    N = rN;
    x = rx;
    for (size_t vl; N > 0; N -= vl, x += vl, y += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_x = __riscv_vle16_v_f16m1(x, vl);
        vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
        vec_y = __riscv_vfsub_vv_f16m1(vec_x, vec_max, vl);
        __riscv_vse16_v_f16m1(y, vec_y, vl);
    }    

    // //final softmax normalization
    // N = rN;
    // y = ry;
    // vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(0, vlmax);

    // //generate scalar sum of exp and inverse
    // vec_s = __riscv_vfredusum_vs_f16m1_f16m1(vec_s, vec_zero, vlmax);
	// _Float16 sum = __riscv_vfmv_f_s_f16m1_f16(vec_s);
    // _Float16 inv_sum_exp = 1.0f / sum;
    // printf("inv_sum_exp: %f\n", (float)inv_sum_exp);

    // //normalization
    // N = rN;
    // y = ry;
    // vec_s = __riscv_vfmv_v_f_f16m1(inv_sum_exp, vlmax);
    // for (size_t vl; N > 0; N -= vl, y += vl) {
    //     vl = __riscv_vsetvl_e16m1(N);
    //     vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
    //     vec_y = __riscv_vfmul_vv_f16m1(vec_y, vec_s, vl);
    //     __riscv_vse16_v_f16m1(y, vec_y, vl);
    // }   
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
    softmax_golden(X, Y_golden, N);

    //compute vec
#ifdef COUNT_CYCLE
    int count_start, count_end;
    count_start = read_perf_counter();
#endif

     softmax_vec(X, Y, N);

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