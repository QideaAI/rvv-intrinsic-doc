# Measure RVV Performance Using Spike Simulator

This document is added to the forked repo to describe reference steps to measure RVV cycle count using spike simulator.

## Dependency

The performance simulation depends are the RISCV toolchain, the Spike simulator and the procxy kernel to be setup.
Please follow the instructions from this repo to build and install the toolchain and the simulator.
[RISCV Toolchain and Simulator](https://github.com/zhengstake/riscv-toolchain) 

## Setup
Assume the toolchain and simulator have been installed under /opt/riscv, you can use the setup.sh script to setup the environment.

```bash
export RISCV=/opt/riscv
export PATH=${RISCV}/bin:${RISCV}/riscv64-unknown-elf/bin:${PATH}
export LD_LIBRARY_PATH=${RISCV}/lib:${LD_LIBRARY_PATH}
```

## Instrument Performance Counting

The following utility function is added to examples/common.h.

```C
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
```

The function can be used inside each RVV example to read out performance of the actual compute function.
For example, in examples/rvv_sgemm.c, the performance counter can be read before and after the function call to
sgemm_vec.

```C
  int count_start, count_end;
  count_start = read_perf_counter();
  printf("Performance counter start: %d\n", count_start);
  sgemm_vec(MLEN, NLEN, KLEN, a_array, KLEN, b_array, NLEN, c_array, NLEN);
  count_end = read_perf_counter();
  printf("Performance counter end: %d\n", count_end);
  printf("Cycle count: %d\n", count_end - count_start);
```

## Build Examples 
A build flow is created to build all examples using GNU compiler for RISCV riscv64-unknown-elf-gcc.
To enable RVV with performance counter, we added the compile flags '-march=rv64gcv_zicntr -mabi=lp64d'.

```bash
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

or
```bash
mkdir build; cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
make
```

Alternatively you can build each example directly by
```bash
riscv64-unknown-elf-gcc examples/rvv_sgemm.c -o rvv_sgemm -mabi=lp64d -march=rv64gcv_zicntr
```

## Run Examples

The example's binary can be simulated using spike simulator with proxy kernel.

```bash
spike --isa=rv64gcv_zicntr_zvfh pk build/rvv_sgemm
```

You can also creat an instruction trace by
```bash
spike --isa=rv64gcv_zicntr_zvfh -l pk build/rvv_sgemm  2> trace.txt
```

By default, spike simulator uses VLEN = 128b as the SIMD width. In order to simulate a wider SIMD width,
the 'zvl' specifier can be used. For example to simulate with 256b SIMD or 512b SIMD, the following can be used:
```bash
spike --isa=rv64gcv_zicntr_zvfh_zvl256b pk rvv_add
spike --isa=rv64gcv_zicntr_zvfh_zvl512b pk rvv_add
```
