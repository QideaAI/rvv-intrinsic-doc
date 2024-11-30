#include <riscv_vector.h>
#include <float.h>
#include "common.h"

#ifndef ARRAY_SIZE
    #define ARRAY_SIZE 2048
#endif

float A[ARRAY_SIZE] = {1.0f};
float B[ARRAY_SIZE] = {1.0f};
float C_golden[ARRAY_SIZE] = {0.f};
float C[ARRAY_SIZE] = {0.f};

//golden scalar function
void add_golden(float *a, float *b, float *c, int N) {
    int i;
    for(i = 0; i < N; i ++) {
        c[i] = a[i] + b[i];
    }
}

//vector elementwise add function
void add_vec(float *a, float *b, float *c, int N) {
    for (size_t vl; N > 0; N -= vl, a += vl, b += vl, c += vl) {
        vl = __riscv_vsetvl_e32m1(N);
        vfloat32m1_t vec_a = __riscv_vle32_v_f32m1(a, vl);
        vfloat32m1_t vec_b = __riscv_vle32_v_f32m1(b, vl);

        vfloat32m1_t vec_c = __riscv_vfadd_vv_f32m1(vec_a, vec_b, vl);
        __riscv_vse32_v_f32m1(c, vec_c, vl);
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
        A[i] = (rand() / (float) RAND_MAX);
        B[i] = (rand() / (float) RAND_MAX);
        printf("A: %f, B: %f\n", (float)A[i], (float)B[i]);
    }

    //check Vector size
    size_t vlmax = __riscv_vsetvlmax_e32m1();
    printf("VLEN: %d\n", (int)vlmax);

    //compute golden
    add_golden(A, B, C_golden, N);

    //compute vec
#ifdef COUNT_CYCLE
    int count_start, count_end;
    count_start = read_perf_counter();
#endif

    add_vec(A, B, C, N);
    
#ifdef COUNT_CYCLE
    count_end = read_perf_counter();
#endif

    int pass = 1;
    for (int i = 0; i < N; i++) {
        if (!fp_eq(C_golden[i], C[i], 5e-6)) {
            printf("index %d fail, %f=!%f\n", i, (float)C_golden[i], (float)C[i]);
            pass = 0;
        }
    }

    if (!pass) {
        return -1;
    } else {
        for (i = 0; i < N; ++i) {
            printf("index %d ref, %f, imp: %f\n", i, (float)C_golden[i], (float)C[i]);
        }

        float l2_dist = l2_distance(C_golden, C, N);
        float cos_sim = cos_similarity(C_golden, C, N);
        printf("pass!\n");
        printf("L2 distance: %f, cos_similarity: %f\n", l2_dist, cos_sim);
#ifdef COUNT_CYCLE
        printf("Performance counter start: %d\n", count_start);
        printf("Performance counter end: %d\n", count_end);
        printf("Cycle count: %d\n", count_end - count_start);
#endif
    }

    return 0;
}
