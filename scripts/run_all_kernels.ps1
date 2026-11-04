# Build and run every myGEMM kernel (1 -- 11) with the OpenCL and SYCL backends and print the
# result lines (time, GFLOPS, L2 norm vs the CPU reference, and SYCL-vs-OpenCL difference).
#
# Usage (from a oneAPI command prompt, in the project root):
#   powershell -ExecutionPolicy Bypass -File scripts\run_all_kernels.ps1 [-BuildDir build-sycl] [-Kernels 1,2,8]
param(
    [string]$BuildDir = "build-sycl",
    [int[]]$Kernels = (1..11)
)
$ErrorActionPreference = "Stop"
$settings = "src\settings.h"
$original = Get-Content $settings -Raw
try {
    foreach ($k in $Kernels) {
        Write-Host "=================== KERNEL $k ==================="
        $text = $original -replace '(?m)^#define KERNEL \d+', "#define KERNEL $k"
        Set-Content $settings $text -NoNewline
        cmake --build $BuildDir | Out-Null
        if ($LASTEXITCODE -ne 0) { Write-Host "BUILD FAILED for KERNEL $k"; continue }
        & "$BuildDir\myGEMM.exe" | Select-String -Pattern "SYCL device|--- |\[ *myGEMM"
    }
}
finally {
    Set-Content $settings $original -NoNewline   # restore settings.h
}
