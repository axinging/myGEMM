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

## 目录结构与两个后端

- `src/`:共享部分,包括主程序 `main.cpp`、`settings.h`、`common.h`、CPU 参考实现 `libcpu.cpp`,以及 CUDA 相关文件。
- `opencl/`:OpenCL 后端(`clGEMM.cpp`、`kernels.cl`,可选 `libclblas.cpp`)。
- `sycl/`:SYCL 后端(`mysycl.cpp`)。它把 `kernels.cl` 里的 **KERNEL 1–11 以及 transpose、padding 辅助 kernel 全部一一移植**,数据布局、分块参数、kernel 调用顺序、计时方法都与 OpenCL 版一致。

运行时,每个后端的结果都会和 CPU 参考结果比较 L2 范数(正确性)。SYCL 的结果还会直接和 OpenCL 的结果比较,输出里显示为 `vs .cl`。每个后端都会输出耗时和 GFLOPS(性能)。每个后端的计时前都有一次不计时的预热运行,这样 SYCL 的 JIT 编译时间不会混进测量。

输出示例(SYCL 打开时):
```
## [ myGEMM.cl]  0.004 s -->  ... GFLOPS (..%), L2 norm: 2.3e-01
## [myGEMM.sycl] 0.004 s -->  ... GFLOPS (..%), L2 norm: 2.3e-01, vs .cl: 1.0e-03
```

## 安装 oneAPI(SYCL 编译器)

1. 安装前提:
   - Visual Studio 2022(要有 "使用 C++ 的桌面开发" 工作负载)。icx 在 Windows 上使用 MSVC 的头文件和链接器。
   - 显卡驱动。Intel GPU 装最新的 Intel Arc/Graphics 驱动,它自带 Level Zero 和 OpenCL 运行时。
2. 下载并安装 **Intel oneAPI Base Toolkit**(其中包含 DPC++/C++ Compiler 和 `sycl-ls`),或者只装独立的 "Intel oneAPI DPC++/C++ Compiler"。默认安装路径是 `C:\Program Files (x86)\Intel\oneAPI`。
3. 装好后打开开始菜单里的 **"Intel oneAPI command prompt for Intel 64 for Visual Studio 2022"**。也可以在普通 cmd 里执行:
   ```
   "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
   ```
   这一步会把 `icx`、`sycl-ls` 等加入 PATH。之后的命令都在这个窗口里执行。
4. 验证安装:
   ```
   icx --version
   sycl-ls
   ```
   `sycl-ls` 应该列出你的 GPU,例如 `[level_zero:gpu] Intel(R) Arc(TM) ...` 和 `[opencl:gpu] ...`。如果一个 GPU 都没有,说明驱动有问题,需要先修驱动。
5. 需要 CMake 和 Ninja(VS 生成器不能直接用 icx 编译 SYCL)。CMake 已装。Ninja 在 `C:\Program Files\ninja-win` 已经在 PATH 里;没有的话 `choco install ninja`。

## 编译与运行

### 只用 OpenCL(不需要 oneAPI,和之前一样)
```
cmake -S . -B build -G "Visual Studio 17 2022" -A x64
cmake --build build --config Release
build\Release\myGEMM.exe
```

### OpenCL + SYCL 对比(在 oneAPI 命令行里)
```
cmake -S . -B build-sycl -G Ninja -DCMAKE_MAKE_PROGRAM="C:/Program Files/ninja-win/ninja.exe" -DCMAKE_CXX_COMPILER=icx -DCMAKE_BUILD_TYPE=Release -DENABLE_SYCL=ON
cmake --build build-sycl
build-sycl\myGEMM.exe
```
`-DCMAKE_MAKE_PROGRAM` 用来指定真正的 `ninja.exe`。如果 PATH 里有别的名为 `ninja` 的东西排在前面(例如 depot_tools 里的包装脚本),CMake 会报 `%1 is not a valid Win32 application`。PATH 里只有正常的 ninja 时可以省略。

`ENABLE_SYCL=ON` 会加上 `-fsycl` 并编译 `sycl/mysycl.cpp`。修改 CMake 选项后如果行为不对,删掉 `build-sycl` 目录重新配置。

### 选择设备和 SYCL 运行时
- OpenCL 后端用第一个平台的第 `CURRENT_DEVICE` 个 GPU(`src/common.h`)。
- SYCL 默认选第一个 GPU。用环境变量 `ONEAPI_DEVICE_SELECTOR` 指定:
  ```
  set ONEAPI_DEVICE_SELECTOR=level_zero:gpu
  set ONEAPI_DEVICE_SELECTOR=opencl:gpu
  ```
  `level_zero:gpu` 是 Intel 的原生路径,通常最快。`opencl:gpu` 让 SYCL 走 OpenCL 运行时,和 OpenCL 后端用同一套驱动,适合做公平的对比。程序启动时会打印 SYCL 使用的设备名。
- 两个后端要在同一块 GPU 上才有可比性。机器上有多块 GPU 时,请确认两边选的是同一块。

### 切换 kernel
`KERNEL` 在 `src/settings.h` 里定义(默认 8),OpenCL 和 SYCL 都用这个值。改完后需要重新编译。

一次测完 KERNEL 1–11(会自动改 `settings.h`、编译、运行,最后恢复原文件):
```
powershell -ExecutionPolicy Bypass -File scriptsun_all_kernels.ps1
powershell -ExecutionPolicy Bypass -File scriptsun_all_kernels.ps1 -Kernels 6,7,8
```

### 怎么判断结果
- **正确性**:`L2 norm` 是和 CPU 参考的差,同一矩阵尺寸下 OpenCL 和 SYCL 应在同一个量级(float 累加误差,4096³ 大约 1e-1 量级)。`vs .cl` 是两个后端的直接差,应该很小(远小于 L2 norm 的 10 倍)。如果 L2 norm 很大(例如 1e+3 以上)或 `vs .cl` 很大,说明这个 kernel 有问题。
- **性能**:比较两行的 GFLOPS。如果 SYCL 明显慢,先试 `ONEAPI_DEVICE_SELECTOR=opencl:gpu`,看是不是运行时不同造成的差距。
- 部分 kernel 对 GPU 有限制:例如 KERNEL 1、2 的 work-group 是 32×32=1024,超过某些 GPU 的上限时会报错。这时改小 `settings.h` 里的 `TS`。
- KERNEL 8 里 OpenCL 版有 CUDA 专用的 `__ldg`/`__shfl`(默认不启用)。SYCL 版没有它们,所以 KERNEL 7 和 8 在 SYCL 里是同一份代码。
- 矩阵尺寸需要是各 kernel 分块大小的整数倍(KERNEL 10 除外)。默认的 1024/2048/4096 满足要求。

## 改动

- **CMakeLists.txt(新增)**
  - 头文件和 `OpenCL.lib` 指向 OpenCL SDK,编译后把 `OpenCL.dll` 复制到输出目录。
  - 设置 `CL_TARGET_OPENCL_VERSION=120`,避免 `clCreateCommandQueue` 报弃用警告。
  - 通过 `MYGEMM_ROOT` 宏传入源码目录,运行时的工作目录不影响查找 kernel 文件。
- **src/main.cpp**
  - `gettimeofday` 和 `sys/time.h` 在 MSVC 下不存在,计时改用 `std::chrono`。
  - 读取 kernel 文件改为二进制模式 `"rb"`。Windows 文本模式下 `ftell` 得到的长度与实际读到的字节数不一致,会让 kernel 源码末尾带垃圾字符。
- **opencl/clGEMM.cpp**(原 src/clGEMM.cpp)
  - kernel、`settings.h`、PTX 输出路径都基于 `MYGEMM_ROOT`,`fopen` 失败时不再崩溃。
  - FLOPs 计数由 `long` 改为 `long long`。Windows 上 `long` 只有 32 位,会溢出,导致 GFLOPS 为负数或 0。
- **clBLAS 改为可选**
  - OpenCL SDK 不含 clBLAS,所以默认关闭。需要时加 `-DENABLE_CLBLAS=ON -DCLBLAS_DIR=<路径>`。
  - `opencl/libclblas.cpp` 和 `src/main.cpp` 中的相关代码加了 `#ifdef ENABLE_CLBLAS` 保护;`libclblas.cpp` 里的 FLOPs 计数同样改为 `long long`。
  - 关闭时,对照用的"标准答案"由新增的 `src/libcpu.cpp`(CPU 版 SGEMM)计算。
- **.gitignore**:加入 `build/`。

## 注意

- 默认 `CURRENT_DEVICE` 为 0,且只取第一个平台的 GPU。机器上有多个平台(如 Intel 加 NVIDIA)时,可能选到不是想要的那个。
- CPU 参考实现跑 4096³ 很慢,Debug 下可能要几分钟。建议用 Release,或调小 `src/common.h` 里的 `MAXSIZE`。
- 不要打开 `ENABLE_CUDA`:两个 `.cu` 文件没有加入 CMake。
- 输出里的 GPU 名称和峰值(`GPU_NAME`、`GPU_CLOCK`、`GPU_CORES`)是写死的 Tesla K40m,与实际显卡无关,峰值百分比没有参考意义。需要时请改成你的显卡参数。
