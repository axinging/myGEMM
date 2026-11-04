cmake_minimum_required(VERSION 3.15)
project(myGEMM LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 11)

# OpenCL SDK location (override with -DOPENCL_SDK_DIR=... if installed elsewhere)
set(OPENCL_SDK_DIR "C:/tools/OpenCL-SDK-v2026.05.29-Win-x64" CACHE PATH "OpenCL SDK root")

# Optional: clBLAS as reference / comparison (not shipped with the OpenCL SDK).
# When OFF, a CPU reference SGEMM is used as the 'golden' result instead.
option(ENABLE_CLBLAS "Build the clBLAS comparison" OFF)
set(CLBLAS_DIR "" CACHE PATH "clBLAS root (only if ENABLE_CLBLAS=ON)")

# Optional: SYCL backend (sycl/mysycl.cpp). Needs a SYCL compiler, e.g. Intel oneAPI DPC++:
#   cmake -DENABLE_SYCL=ON -DCMAKE_CXX_COMPILER=icx ...      (use -G Ninja, run from a oneAPI shell)
option(ENABLE_SYCL "Build the SYCL backend and compare it with OpenCL" OFF)

# src/    : shared driver (main, settings, CPU reference)
# opencl/ : OpenCL backend (host code + kernels.cl)
# sycl/   : SYCL backend
set(SOURCES src/main.cpp src/libcpu.cpp opencl/clGEMM.cpp)
if(ENABLE_CLBLAS)
  list(APPEND SOURCES opencl/libclblas.cpp)
endif()
if(ENABLE_SYCL)
  list(APPEND SOURCES sycl/mysycl.cpp)
endif()

add_executable(myGEMM ${SOURCES})

target_include_directories(myGEMM PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src" "${OPENCL_SDK_DIR}/include")
target_link_libraries(myGEMM PRIVATE "${OPENCL_SDK_DIR}/lib/OpenCL.lib")

# keeps clCreateCommandQueue non-deprecated (only for the OpenCL sources, not for SYCL)
set_source_files_properties(opencl/clGEMM.cpp opencl/libclblas.cpp
                            PROPERTIES COMPILE_DEFINITIONS CL_TARGET_OPENCL_VERSION=120)

target_compile_definitions(myGEMM PRIVATE
  _CRT_SECURE_NO_WARNINGS
  # Kernel/PTX files are located via this path, so the working directory does not matter
  MYGEMM_ROOT="${CMAKE_CURRENT_SOURCE_DIR}")

if(ENABLE_SYCL)
  set_property(TARGET myGEMM PROPERTY CXX_STANDARD 17)
  target_compile_definitions(myGEMM PRIVATE ENABLE_SYCL)
  target_compile_options(myGEMM PRIVATE -fsycl)
  target_link_options(myGEMM PRIVATE -fsycl)
endif()

if(ENABLE_CLBLAS)
  target_compile_definitions(myGEMM PRIVATE ENABLE_CLBLAS)
  target_include_directories(myGEMM PRIVATE "${CLBLAS_DIR}/include")
  target_link_libraries(myGEMM PRIVATE "${CLBLAS_DIR}/lib64/import/clBLAS.lib")
endif()

# clGEMM.cpp writes the compiled kernel binary here
file(MAKE_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/bin")

if(MSVC)
  target_compile_options(myGEMM PRIVATE /W3)
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT myGEMM)
  set_target_properties(myGEMM PROPERTIES
    VS_DEBUGGER_WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endif()

# Make sure OpenCL.dll from the SDK is found at run time (the system ICD loader also works)
add_custom_command(TARGET myGEMM POST_BUILD
  COMMAND ${CMAKE_COMMAND} -E copy_if_different
          "${OPENCL_SDK_DIR}/bin/OpenCL.dll" $<TARGET_FILE_DIR:myGEMM>)
