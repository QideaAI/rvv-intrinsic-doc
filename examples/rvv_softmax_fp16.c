/*
    This function computes the softmax of a vector Y = softmax(X) peer the following formula:

    Y[i] = exp(X[i] - max(X))/sum[exp(X[i] - max(X))]
    
*/

#include <math.h>
#include <stddef.h>
#include <fenv.h>
#include <string.h>
#include <inttypes.h>

#include <riscv_vector.h>
#include "common.h"

#ifndef ARRAY_SIZE
    #define ARRAY_SIZE 256
#endif

 #define max(a,b) \
    ({ __typeof__ (a) _a = (a); \
        __typeof__ (b) _b = (b); \
        _a > _b ? _a : _b; })

_Float16 X[ARRAY_SIZE] = {1.0f};
_Float16 Y_golden[ARRAY_SIZE] = {0.f};
_Float16 Y[ARRAY_SIZE] = {0.f};

#ifndef POLY_DEGREE
#define POLY_DEGREE 7
#elif (POLY_DEGREE > 7)
#error "POLY_DEGREE MUST NOT EXCEED 7"
#endif

//exponential approximation
_Float16 quick_dirty_expf_fp16(_Float16 x) {
    // values determined using (python)sollya
    // >>> iln2 = sollya.round(1/sollya.log(2), sollya.binary32, sollya.RN)
    // >>> ln2 = sollya.round(sollya.log(2), sollya.binary32, sollya.RN)
    const _Float16 ln2 = 0x1.62e43p-1;    
    const _Float16 iln2 = 0x1.715476p0f;

    // argument reduction
    const int k = nearbyintf(x * iln2);
    const _Float16 r = fmaf(- k, ln2, x);

    // polynomial approximation exp(r)
    // coefficients determined using (python)sollya
    // >>> ln2ov2 = sollya.round(sollya.log(2), sollya.binary32, sollya.RN)
    // >>> approxInt = sollya.Interval(-ln2ov2, ln2ov2)
    // >>> approxFun = sollya.exp(sollya.x)
    // >>> degree = 7
    // >>> poly = sollya.fpminimax(approxFunc,
    //                             degree,
    //                             [1] + [sollya.binary32] * degree,
    //                             approxInterval)
    // 0x1p0 + _x_ * (0x1.000002p0 +
    //         _x_ * (0x1.00001p-1 +
    //         _x_ * (0x1.55546ep-3 +
    //         _x_ * (0x1.554854p-5 +
    //         _x_ * (0x1.114662p-7 +
    //         _x_ * (0x1.7209d4p-10 +
    //         _x_ * 0x1.94480ap-13))))))
    const _Float16 poly_coeffs[] = {
        0x1.p0,
        0x1.000002p0, 
        0x1.00001p-1, 
        0x1.55546ep-3, 
        0x1.554854p-5, 
        0x1.114662p-7, 
        0x1.7209d4p-10, 
        0x1.94480ap-13,
    };

    const int poly_degree = POLY_DEGREE;

    _Float16 poly_r = poly_coeffs[poly_degree];
    int i = 0;
    for (i = poly_degree - 1; i >= 0; i--) {
        // poly_r = poly_r * r + poly_coeffs[i];
        poly_r = fmaf(poly_r, r, poly_coeffs[i]);
    }
    // poly_r = 1.f + r * poly_r;
    // poly_r = fmaf(poly_r, r, 1.f);

    // typedef union { float f; uint32_t u; } f_u32_t;
    // NOTE: a proper cast should be done through memcopy and not an union
    // as I think accessing two different fields from the same variable
    // of a union type is undefined behavior.

    // quick and dirty (does not manage overflow/underflow/special values)
    // way to compute 2^k by injecting the biased exponent in the proper place
    // for IEEE-754 binary32 encoding.
    uint16_t exp2_k_u = (15 + k) << 10;
    _Float16 exp2_k;
    memcpy(&exp2_k, &exp2_k_u, sizeof(_Float16)); // hopefully this memcpy is removed by the compiler   
    // uint32_t exp2_k_u = (127 + k) << 23;
    // float exp2_k;
    // memcpy(&exp2_k, &exp2_k_u, sizeof(float)); // hopefully this memcpy is removed by the compiler        
    printf("x*ln2: %f, k: %f, r: %f, poly_r: %f, exp2_k: %f\n", (float)(x * iln2), (float)k, (float)r, (float)poly_r, (float)exp2_k);

    // result reconstruction
    _Float16 exp_x = poly_r * exp2_k;

    return exp_x;
}

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

    //compute exp and the sum of exp
    _Float16 esum = 0.0f;
    for(i = 0; i < N; i ++) {
        y[i] = quick_dirty_expf_fp16(y[i]);
        esum += y[i];
    }      
    _Float16 inv_sum_exp = 1.0f / esum;
    printf("esum: %f, inv_sum_exp: %f\n", (float)esum, (float)inv_sum_exp);

    //final softmax normalization
    for(i = 0; i < N; i ++) {
        y[i] = y[i] * inv_sum_exp;
    } 
}

//vector exp function
_Float16 exp_vec_fp16(_Float16 *x, int N) {
    // values determined using (python)sollya
    const _Float16 ln2 = 0x1.62e43p-1;    
    const _Float16 iln2 = 0x1.715476p0f;

    const size_t vlmax = __riscv_vsetvlmax_e16m1(); 
    const vfloat16m1_t vln2 = __riscv_vfmv_v_f_f16m1(ln2, vlmax);
    const vfloat16m1_t viln2 = __riscv_vfmv_v_f_f16m1(iln2, vlmax);

    // element-wise reduction accumulator
    vfloat16m1_t vsum = __riscv_vfmv_v_f_f16m1(0.f, vlmax);

    const vfloat16m1_t poly_c_0 = __riscv_vfmv_v_f_f16m1(0x1.p0, vlmax);
    const vfloat16m1_t poly_c_1 = __riscv_vfmv_v_f_f16m1(0x1.000002p0, vlmax);
    const vfloat16m1_t poly_c_2 = __riscv_vfmv_v_f_f16m1(0x1.00001p-1, vlmax);
    const vfloat16m1_t poly_c_3 = __riscv_vfmv_v_f_f16m1(0x1.55546ep-3, vlmax);
    const vfloat16m1_t poly_c_4 = __riscv_vfmv_v_f_f16m1(0x1.554854p-5, vlmax);
    const vfloat16m1_t poly_c_5 = __riscv_vfmv_v_f_f16m1(0x1.114662p-7, vlmax);
    const vfloat16m1_t poly_c_6 = __riscv_vfmv_v_f_f16m1(0x1.7209d4p-10, vlmax);
    const vfloat16m1_t poly_c_7 = __riscv_vfmv_v_f_f16m1(0x1.94480ap-13, vlmax);
  
    // we need to make sure round-to-nearest is set, because we need
    // it to be enforced for the conversion from vxiln2 to vk.
    fesetround(FE_TONEAREST);

    size_t avl = N;
    while (avl > 0) {
        size_t vl = __riscv_vsetvl_e16m1(avl);
        vfloat16m1_t vx = __riscv_vle16_v_f16m1(x, vl);

        // argument reduction
        vfloat16m1_t vxiln2 = __riscv_vfmul(vx, viln2, vl);
        vint16m1_t       vk = __riscv_vfcvt_x_f_v_i16m1(vxiln2, vl); // require round to nearest mode
        vfloat16m1_t    vfk = __riscv_vfcvt_f_x_v_f16m1(vk, vl);
        
        // using vfnmsac.vf to evaluate r = x - k * log(2)
        vfloat16m1_t     vr = __riscv_vfnmsac(vx, vln2, vfk, vl);
        //__riscv_vse16(y, vr, vl);
        
        // polynomial approximation exp(r)
        vfloat16m1_t poly_vr = poly_c_7;
        poly_vr = __riscv_vfmadd(poly_vr, vr, poly_c_6, vl);
        poly_vr = __riscv_vfmadd(poly_vr, vr, poly_c_5, vl);
        poly_vr = __riscv_vfmadd(poly_vr, vr, poly_c_4, vl);
        poly_vr = __riscv_vfmadd(poly_vr, vr, poly_c_3, vl);
        poly_vr = __riscv_vfmadd(poly_vr, vr, poly_c_2, vl);
        poly_vr = __riscv_vfmadd(poly_vr, vr, poly_c_1, vl);
        poly_vr = __riscv_vfmadd(poly_vr, vr, poly_c_0, vl);
        //__riscv_vse16(y, poly_vr, vl);

        // reconstruction
        const uint16_t exp_bias = 15;
        vint16m1_t vbiased_exp = __riscv_vadd(vk, exp_bias, vl);
        vint16m1_t vexp2_vk    = __riscv_vsll(vbiased_exp, 10, vl);
        vfloat16m1_t vfexp2_vk;
        vfexp2_vk = __riscv_vreinterpret_v_i16m1_f16m1(vexp2_vk);
        vfloat16m1_t vexp_vx  = __riscv_vfmul(poly_vr, vfexp2_vk, vl);
        __riscv_vse16(x, vexp_vx, vl);

        // element-size reduction with redution accumulator
        // tail-undisturbed is mandatory here to ensure that if vl is less
        // than VLMAX then unaffacted sum terms are not changed.
        vsum = __riscv_vfadd_vv_f16m1(vsum, vexp_vx, vl);

        avl -= vl;
        x += vl;
    }

    vfloat16m1_t vredsum = __riscv_vfmv_v_f_f16m1(0.f, vlmax);
    vredsum = __riscv_vfredusum_vs_f16m1_f16m1(vsum, vredsum, vlmax);

    return __riscv_vfmv_f_s_f16m1_f16(vredsum);
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
    //printf("imp x_max: %f\n", (float)x_max);

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

    //compute exp and the sum of exp
    N = rN;
    y = ry;
    _Float16 esum = exp_vec_fp16(y, N);
    _Float16 inv_sum_exp = 1.0f / esum;
    //printf("esum: %f, inv_sum_exp: %f\n", (float)esum, (float)inv_sum_exp);

    //normalization
    N = rN;
    y = ry;
    vfloat16m1_t vec_s = __riscv_vfmv_v_f_f16m1(inv_sum_exp, vlmax);
    for (size_t vl; N > 0; N -= vl, y += vl) {
        vl = __riscv_vsetvl_e16m1(N);
        vfloat16m1_t vec_y = __riscv_vle16_v_f16m1(y, vl);
        vec_y = __riscv_vfmul_vv_f16m1(vec_y, vec_s, vl);
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
        if (!fp16_eq(Y_golden[i], Y[i], 1e-2)) {
        printf("index %d fail, %f=!%f\n", i, (float)Y_golden[i], (float)Y[i]);
        pass = 0;
        }
    }
    if (pass) {
        for (i = 0; i < N; ++i) {
            printf("index %d ref, %f, imp: %f\n", i, (float)Y_golden[i], (float)Y[i]);
        }
        printf("pass\n");
    }
    return (pass == 0);
}