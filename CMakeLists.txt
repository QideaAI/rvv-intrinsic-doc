cmake_minimum_required(VERSION 3.20)

if(NOT DEFINED ENV{RISCV})
  set(ENV{RISCV} "/opt/riscv")
endif()
set(RISCV $ENV{RISCV})

#
# compiler settings
#

# Set the C and C++ compilers
set(CMAKE_C_COMPILER ${RISCV}/bin/riscv64-unknown-elf-gcc)
set(CMAKE_CXX_COMPILER ${RISCV}/bin/riscv64-unknown-elf-g++)

# Set the target architecture to RISC-V
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Find and configure the RISC-V toolchain
set(CMAKE_C_FLAGS "-Wall -march=rv64gcv_zicntr_zvfh -mabi=lp64d")
set(CMAKE_CXX_FLAGS "-Wall -march=rv64gcv_zicntr_zvfh -mabi=lp64d")

set(CMAKE_C_FLAGS_DEBUG "-O0 -g")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_EXE_LINKER_FLAGS "-lm")

project(rvv-examples)
#
# subdirectories
#
include(CTest)

add_definitions(-DCOUNT_CYCLE)
file(GLOB SOURCE_FILES "${CMAKE_SOURCE_DIR}/examples/*.c")
foreach(SOURCE_FILE ${SOURCE_FILES})
	set(path ${SOURCE_FILE})
	cmake_path(GET path STEM EXE_TARGET)
	message("Adding target ${EXE_TARGET}")
	add_executable(${EXE_TARGET} ${SOURCE_FILE})
	target_link_libraries(${EXE_TARGET} m)
	add_test(NAME ${EXE_TARGET} COMMAND ${RISCV}/bin/spike --isa=rv64gcv_zicntr_zvfh ${RISCV}/riscv64-unknown-elf/bin/pk ${EXE_TARGET})
endforeach()

