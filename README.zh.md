# myGEMM:在 Windows + VS2022 下编译与调试

本文说明如何用 Visual Studio 2022 和 OpenCL SDK 编译、调试本项目,以及为此做了哪些改动。

## 怎么用

前提:OpenCL SDK 位于 `C:\tools\OpenCL-SDK-v2026.05.29-Win-x64`。如果装在别处,配置时加 `-DOPENCL_SDK_DIR=<路径>`。

- **用 sln:** 在项目根目录执行
  ```
  cmake -S . -B build -G "Visual Studio 17 2022" -A x64
  ```
  然后打开 `build\myGEMM.sln`。`myGEMM` 已设为启动项,直接 F5 调试。
- **用 Open Folder:** 在 VS2022 里直接打开项目文件夹,它会识别 `CMakeLists.txt`。

## 改动

- **CMakeLists.txt(新增)**
  - 头文件和 `OpenCL.lib` 指向 OpenCL SDK,编译后把 `OpenCL.dll` 复制到输出目录。
  - 设置 `CL_TARGET_OPENCL_VERSION=120`,避免 `clCreateCommandQueue` 报弃用警告。
  - 通过 `MYGEMM_ROOT` 宏传入源码目录,运行时的工作目录不影响查找 kernel 文件。
- **src/main.cpp**
  - `gettimeofday` 和 `sys/time.h` 在 MSVC 下不存在,计时改用 `std::chrono`。
  - 读取 kernel 文件改为二进制模式 `"rb"`。Windows 文本模式下 `ftell` 得到的长度与实际读到的字节数不一致,会让 kernel 源码末尾带垃圾字符。
- **src/clGEMM.cpp**
  - kernel、`settings.h`、PTX 输出路径都基于 `MYGEMM_ROOT`,`fopen` 失败时不再崩溃。
  - FLOPs 计数由 `long` 改为 `long long`。Windows 上 `long` 只有 32 位,会溢出,导致 GFLOPS 为负数或 0。
- **clBLAS 改为可选**
  - OpenCL SDK 不含 clBLAS,所以默认关闭。需要时加 `-DENABLE_CLBLAS=ON -DCLBLAS_DIR=<路径>`。
  - `src/libclblas.cpp` 和 `src/main.cpp` 中的相关代码加了 `#ifdef ENABLE_CLBLAS` 保护;`libclblas.cpp` 里的 FLOPs 计数同样改为 `long long`。
  - 关闭时,对照用的"标准答案"由新增的 `src/libcpu.cpp`(CPU 版 SGEMM)计算。
- **.gitignore**:加入 `build/`。

## 注意

- 默认 `CURRENT_DEVICE` 为 0,且只取第一个平台的 GPU。机器上有多个平台(如 Intel 加 NVIDIA)时,可能选到不是想要的那个。
- CPU 参考实现跑 4096³ 很慢,Debug 下可能要几分钟。建议用 Release,或调小 `src/common.h` 里的 `MAXSIZE`。
- 不要打开 `ENABLE_CUDA`:两个 `.cu` 文件没有加入 CMake。
- 输出里的 GPU 名称和峰值(`GPU_NAME`、`GPU_CLOCK`、`GPU_CORES`)是写死的 Tesla K40m,与实际显卡无关,峰值百分比没有参考意义。需要时请改成你的显卡参数。
