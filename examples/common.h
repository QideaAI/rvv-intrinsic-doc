// common.h
// common utilities for the test code under exmaples/

#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <float.h>

#ifndef __COMMON__H__
#define __COMMON__H__
void gen_rand_1d(double *a, int n) {
  for (int i = 0; i < n; ++i)
    a[i] = (double)rand() / (double)RAND_MAX + (double)(rand() % 1000);
}

void gen_string(char *s, int n) {
  // char value range: -128 ~ 127
  for (int i = 0; i < n - 1; ++i)
    s[i] = (char)(rand() % 127) + 1;
  s[n - 1] = '\0';
}

void gen_rand_2d(double **ar, int n, int m) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      ar[i][j] = (double)rand() / (double)RAND_MAX + (double)(rand() % 1000);
}

void print_string(const char *a, const char *name) {
  printf("const char *%s = \"", name);
  int i = 0;
  while (a[i] != 0)
    putchar(a[i++]);
  printf("\"\n");
  puts("");
}

void print_array_1d(double *a, int n, const char *type, const char *name) {
  printf("%s %s[%d] = {\n", type, name, n);
  for (int i = 0; i < n; ++i) {
    printf("%06.2f%s", a[i], i != n - 1 ? "," : "};\n");
    if (i % 10 == 9)
      puts("");
  }
  puts("");
}

void print_array_2d(double **a, int n, int m, const char *type,
                    const char *name) {
  printf("%s %s[%d][%d] = {\n", type, name, n, m);
  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < m; ++j) {
      printf("%06.2f", a[i][j]);
      if (j == m - 1)
        puts(i == n - 1 ? "};" : ",");
      else
        putchar(',');
    }
  }
  puts("");
}

bool double_eq(double golden, double actual, double relErr) {
  return (fabs(actual - golden) < relErr);
}

bool compare_1d(double *golden, double *actual, int n) {
  for (int i = 0; i < n; ++i)
    if (!double_eq(golden[i], actual[i], 1e-6))
      return false;
  return true;
}

bool compare_string(const char *golden, const char *actual, int n) {
  for (int i = 0; i < n; ++i)
    if (golden[i] != actual[i])
      return false;
  return true;
}

bool compare_2d(double **golden, double **actual, int n, int m) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      if (!double_eq(golden[i][j], actual[i][j], 1e-6))
        return false;
  return true;
}

double **alloc_array_2d(int n, int m) {
  double **ret;
  ret = (double **)malloc(sizeof(double *) * n);
  for (int i = 0; i < n; ++i)
    ret[i] = (double *)malloc(sizeof(double) * m);
  return ret;
}

void init_array_one_1d(double *ar, int n) {
  for (int i = 0; i < n; ++i)
    ar[i] = 1;
}

void init_array_one_2d(double **ar, int n, int m) {
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < m; ++j)
      ar[i][j] = 1;
}

unsigned long read_perf_counter(void)
{
  unsigned long counter_value;
#if defined(COUNT_INSTRET)
#define PERF_METRIC "instruction"
  asm volatile ("rdinstret %0" : "=r" (counter_value));
#elif defined(COUNT_CYCLE)
#define PERF_METRIC "cycle"
  asm volatile ("rdcycle %0" : "=r" (counter_value));
#else
  // instret is also the default
#define PERF_METRIC "instruction"
  asm volatile ("rdinstret %0" : "=r" (counter_value));
#endif
  return counter_value;
}

float l2_distance(float *x, float *y, int n) {
	float l2_dist = 0.0f;

	int i;
	for(i = 0; i < n; i++) {
		l2_dist += (x[i] - y[i]) * (x[i] - y[i]);
	}

	l2_dist = sqrt(l2_dist);

	return l2_dist;
}

float cos_similarity(float *x, float *y, int n) {
	float cos_sim = 0.0f;

	int i;
	float x_dot_y = 0.0f, x_norm = 0.0f, y_norm = 0.0f;
	for(i = 0; i < n; i++) {
		x_dot_y += x[i] * y[i];
		x_norm += x[i] * x[i];
		y_norm += y[i] * y[i];
	}

	cos_sim = x_dot_y / sqrt(x_norm * y_norm);
	
	return cos_sim;
}

_Float16 l2_distance_fp16(_Float16 *x, _Float16 *y, int n) {
	float l2_dist = 0.0f;

	int i;
	for(i = 0; i < n; i++) {
		l2_dist += (x[i] - y[i]) * (x[i] - y[i]);
	}

	l2_dist = sqrt(l2_dist);

	return l2_dist;
}

_Float16 cos_similarity_fp16(_Float16 *x, _Float16 *y, int n) {
	float cos_sim = 0.0f;

	int i;
	float x_dot_y = 0.0f, x_norm = 0.0f, y_norm = 0.0f;
	for(i = 0; i < n; i++) {
		x_dot_y += x[i] * y[i];
		x_norm += x[i] * x[i];
		y_norm += y[i] * y[i];
	}

	cos_sim = x_dot_y / sqrt(x_norm * y_norm);
	
	return cos_sim;
}

#endif
