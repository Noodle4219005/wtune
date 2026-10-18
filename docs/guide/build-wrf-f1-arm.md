# 在 f1-arm 編 WRF

WTune 不編 WRF，只要求 `WRF_TREE` 底下有 `main/wrf.exe`、`main/real.exe` 和 `run/`。這頁記錄 f1-arm 上實際用過、編得出來的做法：WRF v4.8.0 原始碼不改，nvhpc 25.5，dmpar，ADIOS2 2.10.2 + netCDF。

已經編好的樹（直接拿來用就不必重編）：

| 樹 | 原始碼 | FCOPTIM | 說明 |
|---|---|---|---|
| `~/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar/WRF` | 上游 v4.8.0（`06d4240`），未改 | configure 原值 | 2026-10-14，job 15516；ADIOS2 輸出與 netCDF 逐位元相同 |
| `~/wrf-env/nvhpc-arm/wrf/nvhpc-adios2-tuned-dmpar/WRF` | `~/SC26-WRF/src/WRF`，含團隊的 `wrf_io.F90` patch | 原值加 `-tp=grace -Mstack_arrays -Mfprelaxed` | 2026-08-24；目前 `env.sh` 的 `WRF_TREE` 指這棵 |

!!! warning "預設的那棵改過原始碼"
    `nvhpc-adios2-tuned-dmpar` 是從團隊的 WRF 分支複製來的，`external/io_adios2/wrf_io.F90` 有改動。要用未改的 WRF，把 `~/wtune-f1arm/env.sh` 的 `WRF_TREE` 和 `~/.config/wtune/machine.yaml` 的 `wrf_tree` 都換成 `vanilla-v480-adios2-dmpar` 那棵。

## 整體順序

```mermaid
flowchart LR
  A[Spack env<br>函式庫] --> B[clone WRF<br>登入節點]
  B --> C[configure + compile<br>計算節點 sbatch]
  C --> D[驗證]
  D --> E[env.sh 與<br>machine.yaml 指過去]
```

計算節點沒有外網：要下載的東西（Spack 原始碼包、WRF 原始碼）都在登入節點 `nlgn01` 先準備好。編譯本身很重，一律用 sbatch 在計算節點跑。

## 1. 編譯器與 MPI

| 項目 | 版本 | 位置 |
|---|---|---|
| nvhpc | 25.5（nvfortran／nvc） | `/pkg_arm/arm/compiler/nvhpc/25.5/Linux_aarch64/25.5/compilers/bin`，或 `module load nvhpc/25.5` |
| Open MPI | 5.0.10，Spack 以 `%nvhpc@25.5` 編 | 函式庫 view 的 `bin/`（`mpif90`、`mpirun`） |

登入節點的 PATH 預設沒有 `nvfortran` 和 `mpirun`，編譯 job 要自己加。

!!! note "Open MPI 沒有 UCX"
    這個 Open MPI 只有 TCP 和共享記憶體，跨節點走 TCP。`campaign/sites/f1arm.sh` 設 `OMPI_MCA_btl_tcp_if_include=10.200.0.0/14`，讓它走 IPoIB 而不是 1 Gb 管理網。即使如此，conus12 在 2、4 節點仍比 1 節點慢（0.207、0.462 vs 0.173 s/step），所以小網格常選 N=1–2。

## 2. 函式庫（Spack）

| 路徑 | 內容 |
|---|---|
| `~/spack` | Spack 本體（`source ~/spack/share/spack/setup-env.sh`） |
| `~/wrf-env/nvhpc-arm/spack-env/spack.yaml` | 環境定義 |
| `~/wrf-env/nvhpc-arm/store` | 安裝樹 |
| `~/wrf-env/nvhpc-arm/mirror` | 本機 mirror（計算節點從這裡拿原始碼包） |
| `~/wrf-stack/nvhpc-arm` | view：`bin/`、`lib/`、`include/`；`NETCDF`、`HDF5`、`ADIOS2` 都指這裡 |
| `~/wrf-env/nvhpc-arm/bootstrap-prefix{,-rpath}` | 在 Spack 外用 GCC 15.2 編的 patch、libtool、numactl、cmake 3.31.12 |

`spack.yaml` 的 specs（`$HOME` 換成你的家目錄）：

```yaml
spack:
  specs:
  - openmpi@5.0.10 %nvhpc@25.5
  - netcdf-fortran@4.6.2 %nvhpc@25.5 ^hdf5@1.14.6+fortran+hl+mpi ^netcdf-c@4.10.0~blosc~dap+mpi~szip~zstd ^openmpi@5.0.10
  - adios2@2.10.2 +mpi +fortran +blosc2 ~bzip2 ~zfp ~png ~sz ~mgard ~sst ~libcatalyst ~xrootd ~python ~hdf5 %nvhpc@25.5 ^openmpi@5.0.10 ^c-blosc2 +zstd ~lz4 ~lizard ~snappy ~zlib %gcc@11.4.1 ^zstd %gcc@11.4.1
  view: $HOME/wrf-stack/nvhpc-arm
  concretizer:
    unify: true
  config:
    build_stage: $HOME/wrf-env/nvhpc-arm/spack-stage
    install_tree:
      root: $HOME/wrf-env/nvhpc-arm/store
  mirrors:
    wrf-local: file://$HOME/wrf-env/nvhpc-arm/mirror
  packages:
    cmake:
      externals:
      - spec: cmake@3.31.12
        prefix: $HOME/wrf-env/nvhpc-arm/bootstrap-prefix-rpath
      buildable: false
    numactl:
      externals:
      - spec: numactl@2.0.19
        prefix: $HOME/wrf-env/nvhpc-arm/bootstrap-prefix
      buildable: false
```

改了 `spack.yaml` 之後一定先重新 concretize，再到計算節點安裝（當初的安裝 job 是 1038903，`arm144`、72 核；下面三行是同樣的做法，沒有原樣重跑過）：

```bash
source ~/spack/share/spack/setup-env.sh
E=~/wrf-env/nvhpc-arm/spack-env
spack -e $E concretize -f
spack -e $E mirror create -d ~/wrf-env/nvhpc-arm/mirror --all   # 登入節點：補齊原始碼包
sbatch -A <account> -p arm144 -N 1 -c 72 -t 4:00:00 \
       --wrap "source ~/spack/share/spack/setup-env.sh; spack -e $E install"
```

!!! note "為什麼有 bootstrap-prefix"
    在這台機器上 Spack 自己編不出 patch、libtool、numactl、cmake：nvc 把 `__va_list_tag` 塞進 patch 的 make 命令；GCC 15.2 的 assembler 不接受 Spack 推出的 `-march=armv9-a+...`；系統 CMake 3.20 太舊，而 GCC 15 編的 CMake 在 Spack 清掉 `LD_LIBRARY_PATH` 後載入舊的 `libstdc++`。所以這四個工具在 Spack 外用 GCC 15.2、`-march=armv8.6-a` 編好（CMake 用 rpath），再當 external 註冊。它們已經裝好，重建函式庫時不用再做。

!!! note "ADIOS2 實際上沒有 Blosc2"
    spec 寫了 `+blosc2`，view 裡也有 `libblosc2`，但 `lib*/cmake/adios2/adios2-config-common.cmake` 的 `ADIOS2_HAVE_Blosc2` 是空的。結果是壓縮不生效：WRF 每開一個輸出檔印一次 `adios2 error in ext_adios2_open_for_write_begin ... error code 4`，資料照寫、不壓縮。不影響正確性。

## 3. 取得 WRF 原始碼（登入節點）

```bash
B=~/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar
mkdir -p $B && cd $B
git clone --depth 1 --recurse-submodules --branch v4.8.0 https://github.com/wrf-model/WRF.git
```

`--recurse-submodules` 不能省：`phys/noahmp` 等是 submodule，計算節點抓不到。

## 4. configure 與 compile（計算節點）

這是 job 15516 實際做的事，寫成一個 sbatch：

```bash
#!/bin/bash
#SBATCH -J wrf-build
#SBATCH -A <account>
#SBATCH -p arm144
#SBATCH -N 1
#SBATCH --ntasks-per-node=16
#SBATCH --mem=200G
#SBATCH -t 02:00:00
set -uo pipefail
V=$HOME/wrf-stack/nvhpc-arm
export PATH=/pkg_arm/arm/compiler/nvhpc/25.5/Linux_aarch64/25.5/compilers/bin:$V/bin:$PATH
export LD_LIBRARY_PATH=$V/lib:$V/lib64:${LD_LIBRARY_PATH:-}
export NETCDF=$V HDF5=$V ADIOS2=$V
export PNETCDF=                        # 一定要明確設成空的：view 裡沒有 PnetCDF
ulimit -s unlimited
cd $HOME/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar/WRF
printf '19\n1\n' | ./configure         # 19 = NVHPC (nvfortran/nvc) dmpar；1 = basic nesting
grep -q DADIOS2 configure.wrf || { echo "configure.wrf has no -DADIOS2"; exit 1; }
./compile -j 8 em_real > compile.log 2>&1
ls -l main/wrf.exe main/real.exe
```

| 步驟 | 要注意的地方 |
|---|---|
| `./configure` | 選項編號要對 aarch64 的選單：`NVHPC (nvfortran/nvc)` 的 `(dmpar)` 是 19。不確定時先在登入節點跑一次 `./configure`，看選單再中斷。WRF 看到 `ADIOS2` 環境變數就自動加 `-DADIOS2`，不用手改 `configure.wrf` |
| `PNETCDF=` | `~/wtune-f1arm/env.sh` 會設 `PNETCDF=$V`；編譯時要蓋成空值，否則 configure 會去找不存在的 PnetCDF |
| `-j 8` | `-j 20` 會 OOM，且 Fortran module 的相依順序會亂掉；`-j 8` 約 20 分鐘編完 |
| 中途失敗 | 不要 `./clean -a`；同一行 `./compile -j 8 em_real >> compile.log 2>&1` 再跑一次，接著編下去。還是缺 `.mod` 就改 `-j 1`，必要時跑兩三輪 |
| 調校旗標 | 想和 tuned 那棵一樣，configure 之後改 `configure.wrf` 的 `FCOPTIM`：`-O3 -fast -Mvect=noaltcode -Mprefetch=distance:8 -tp=grace -Mstack_arrays -Mfprelaxed`。`-Mfprelaxed` 會放寬浮點精度 |

也可以用現成的腳本（job 15516 就是這樣跑的）：

```bash
bash   ~/wrf-env/nvhpc-arm/vanilla-build.sbatch prep    # 登入節點：clone
sbatch ~/wrf-env/nvhpc-arm/vanilla-build.sbatch         # 計算節點：configure + compile
sbatch ~/wrf-env/nvhpc-arm/vanilla-build.sbatch recompile
```

它呼叫 `~/SC26-WRF/scripts/build-wrf.sbatch`，做的就是上面那幾步。

## 5. 驗證

```bash
cd ~/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar/WRF
ls -l main/wrf.exe main/real.exe
source ~/wtune-f1arm/env.sh
ldd main/wrf.exe | grep "not found"     # 要沒有輸出
grep -c DADIOS2 configure.wrf           # 要大於 0
```

最後用一個小個案跑一次：

```bash
cd ~/wtune-projects/<project>
wtune run --nodes 1
```

## 6. 讓 WTune 用新的樹

兩個地方都要改：

```bash
# ~/wtune-f1arm/env.sh
export WRF_TREE=$HOME/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar/WRF

# ~/.config/wtune/machine.yaml
wrf_tree: $HOME/wrf-env/nvhpc-arm/wrf/vanilla-v480-adios2-dmpar/WRF
```

`wtune doctor` 會印出它找到的 `wrf.exe`。換樹之後，之前的 `tune_report.json` 是用舊的執行檔量的；編譯旗標不同時要重新 tune。
