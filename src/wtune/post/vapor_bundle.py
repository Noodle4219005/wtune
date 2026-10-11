"""post/vapor_bundle.py -- a ready-to-copy NCAR VAPOR bundle per domain (`wtune post vapor-bundle`;
also a data product of `wtune post recipe`): <out>/vapor/d0N/ with the selected wrfout frames,
vapor_render.py (verbatim), README.md and the bundle.json manifest. VAPOR has no linux-aarch64
build, so it runs on the user's workstation; VAPOR's WRF reader wants raw wrfout, one time per file.

Frames are hard-linked (no disk); copying needs an explicit allow_copy (a silent copy filled a GPFS
quota and truncated NetCDF files without ENOSPC). Without it a failed link removes what this call
wrote and raises NotHardLinkable naming the --allow-copy command. Multi-frame wrfout
(frames_per_outfile > 1) is written as single-time files (mode "extract", needs allow_copy too);
ADIOS2 BP history is served from post.extract's netCDF cache copies (decision 0.8.2).
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

SCRIPT_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vapor", "vapor_render.py")
SCRIPT_NAME = "vapor_render.py"
MANIFEST_NAME = "bundle.json"
README_NAME = "README.md"


def bundle_name(domain: int, time_str: str) -> str:
    """wrfout_d0N_YYYY-MM-DD_HH_MM_SS -- WRF's own name with the colons replaced."""
    return "wrfout_d%02d_%s" % (int(domain), time_str.replace(":", "_"))


class NotHardLinkable(OSError):
    """os.link failed and copying was not allowed; str() is the user-facing message."""


def link_or_copy(src: str, dst: str, allow_copy: bool = False) -> str:
    """Hard-links src to dst (replacing a stale dst). Returns "hardlink", "kept" (dst already
    IS src: a re-run of the same bundle) or "copy" (linking failed and allow_copy). Without
    allow_copy a failed link re-raises its OSError (EXDEV: another filesystem; EPERM/ENOTSUP:
    no hard links there; EMLINK; ...)."""
    if os.path.exists(dst) and os.path.samefile(src, dst):
        return "kept"
    # link (or copy) to a temporary name, then os.replace: a failed link never destroys a dst
    # that an earlier bundle left there
    tmp = dst + ".wtune-tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    try:
        os.link(src, tmp)
        mode = "hardlink"
    except OSError:
        if not allow_copy:
            raise
        shutil.copy2(src, tmp)
        mode = "copy"
    os.replace(tmp, dst)
    return mode


def _wtune_provenance():
    """(version, git commit or None) of this WTune checkout -- read-only, best effort."""
    import subprocess
    try:
        import wtune.version
        version = wtune.version.__version__
    except ImportError:
        version = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                           # noqa: BLE001 -- no git, not a checkout, ...
        commit = None
    return version, commit


def _human(n: float) -> str:
    return "%.1f MB" % (n / 2.0 ** 20)


_README = """\
# WTune VAPOR bundle：{run_label} d{domain:02d}

由 `wtune post vapor-bundle` / `wtune post recipe` 於 {created} 產生。本資料夾可整包複製到你自己的
工作站，用 NCAR 官方 VAPOR 開啟。scc（aarch64）上沒有 VAPOR：VAPOR 沒有 linux-aarch64 build。

## 內容

| 檔案 | 說明 |
|---|---|
{file_rows}| `{script}` | 以 VAPOR Python API 算圖（地形 + QCLOUD/QICE isosurface），也可只存 session |
| `{manifest}` | manifest：原始 wrfout 路徑、bundle 內檔名、valid time、大小、hard link 或 copy |

- run：`{run_dir}`
- domain：d{domain:02d}；時間：{n} 個 frame（{cadence}）
- 檔案：{link_note}。同一檔案系統上一律 hard link（不佔額外空間）；跨檔案系統時 `wtune post
  recipe` 不複製（報告記為 missing，並給出指令），只有明確下 `wtune post vapor-bundle --allow-copy`
  才複製。
- 檔名把 WRF 原名的 `:` 換成 `_`（macOS/Windows 可用）。**假設**：VAPOR 從每個檔案的 `Times`
  變數讀時間，不看檔名——此點未在 VAPOR 上驗證。

## 1. 複製到工作站

```
rsync -av scc:{bundle_dir}/ ./
```

hard link 的檔案在 rsync 時照常傳完整內容；本機不需要 scc 上的原始 run。{link_detail}
{native_section}
## 2. 安裝 VAPOR（x86_64 Linux / macOS）

```
conda create -n vapor -c conda-forge -c ncar-vapor vapor
conda activate vapor
```

## 3. 用 Python 算圖（`{script}`）

```
python {script}                    # 每個時間一張 PNG + wtune_session.vs3，輸出到 ./vapor_png
python {script} --list             # 只列出變數名稱與 time step 數
python {script} --session-only     # 只存 .vs3（給 GUI 開），不算圖
python {script} --iso-cloud 2e-4 --iso-ice 1e-5 --width 1920 --height 1080
```

沒有螢幕的 Linux 主機可能要 `xvfb-run -a python {script}`（未驗證）。結尾的 summary 會列出
算出哪些 PNG、略過什麼（例如沒有 QICE 時的 QICE isosurface）、哪些未驗證的呼叫失敗。

## 4. 用 GUI 開

`vapor` 啟動後：

- File > Open Session，選 `vapor_png/wtune_session.vs3`（先跑過 `{script}` 或 `--session-only`），或
- File > Import > WRF-ARW，選取本資料夾**全部** `wrfout_*` 檔（VAPOR 需要整個時間序列的檔案清單）。

垂直放大倍率（vertical exaggeration）沒有 Python API，請在 GUI 的 Viewpoint / Scene 設定裡調。

## 未驗證的部分（本 bundle 產生時無法在 VAPOR 上執行）

- `Session.SetTimestep(i)`：文件未記載、有不穩定的回報。若被忽略，每張 PNG 可能都是**同一個時間**；
  `{script}` 無法偵測，只會在呼叫失敗時提示。此時請改用 GUI 的時間軸。
- 斜視相機：`Camera.GetPosition/GetTarget/SetPosition/SetDirection/SetUp`（把 ViewAll 的俯視改為由南方
  約 45° 斜看）。
- isosurface 顏色：`GetPrimaryTransferFunction().SetColorRGBList(...)`。
- 以上呼叫失敗時 `{script}` 只印 note 並繼續；其餘（Session、OpenDataset、GetDataVarNames、
  GetNumTimeSteps、NewRenderer、SetVariableName、SetIsoValues、SetHeightVariableName、SetResolution、
  GetCamera、ViewAll、Render、Save）是 VAPOR 文件記載的呼叫。
- `xvfb-run` 的需求、檔名不影響時間的假設，同樣未驗證。
"""


_NATIVE_BP = """
## 原生 history 是 ADIOS2 BP，這裡是轉出的 netCDF 複本

這個 run 的 history 原生格式是 ADIOS2 BP（`io_form_history=14`），VAPOR 讀不了 BP。bundle 內的
wrfout 是 `wtune post` 以 `src/tools/wrf_adios2_to_netcdf4/` 轉出的 netCDF4 **複本**，取自 netCDF
cache（{cache_dirs}），不是 run 目錄裡的原始檔；每個檔案對應的 BP 目錄記在 `{manifest}` 的
`native` 欄位。

- 同一檔案系統上，bundle 檔是 cache 轉檔的 hard link（不另佔空間）；此時刪掉 cache 並不會釋放
  這些空間，要連 bundle 一起刪。跨檔案系統時與上面相同：只有 `--allow-copy` 才複製。
- BP 之後若被改寫，cache 會以新的轉檔整個取代（不是就地改寫），bundle 內仍是舊版：重跑
  `wtune post vapor-bundle` 才會更新。
"""


def _write_readme(path, bundle_dir, run_dir, run_label, domain, frames, cadence, link_note,
                  created):
    rows = "".join("| `%s` | wrfout%s，valid time %s |\n" % (
        f["name"], "（ADIOS2 BP 轉出的 netCDF 複本）" if f.get("native") else "", f["valid_time"])
        for f in frames)
    native = [f for f in frames if f.get("native")]
    if native:
        link_detail = "bundle 內的檔案來自 netCDF cache，不是 run 的原始 history（見下節）。"
        native_section = _NATIVE_BP.format(manifest=MANIFEST_NAME, cache_dirs="、".join(
            "`%s`" % d for d in sorted(set(os.path.dirname(f["original"]) for f in native))))
    else:
        link_detail = ("hard link 與 run 的原始\nwrfout 是同一份資料：之後若就地改寫原檔，"
                       "bundle 內容也會跟著變。")
        native_section = ""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_README.format(run_label=run_label, domain=domain, created=created,
                                file_rows=rows, script=SCRIPT_NAME, manifest=MANIFEST_NAME,
                                run_dir=os.path.abspath(run_dir), n=len(frames),
                                cadence=cadence, link_note=link_note,
                                bundle_dir=os.path.abspath(bundle_dir),
                                link_detail=link_detail, native_section=native_section))


def _ntimes(path):
    """Number of valid times in a wrfout file (post.extract.file_times; 1 when unreadable)."""
    from wtune.post.extract import file_times
    return len(file_times(path) or [None])


def write_single_time(src: str, timeidx: int, dst: str) -> str:
    """Writes dst as a copy of wrfout src restricted to its Time index `timeidx`: same global
    attributes, dimensions (Time of length 1, still unlimited), variables with their attributes
    and compression; non-Time variables copied whole. Written to a temporary name, then
    os.replace. Returns "extract"."""
    import netCDF4
    tmp = dst + ".wtune-tmp"
    if os.path.lexists(tmp):
        os.remove(tmp)
    try:
        with netCDF4.Dataset(src) as si, netCDF4.Dataset(tmp, "w", format=si.data_model) as so:
            so.setncatts({k: si.getncattr(k) for k in si.ncattrs()})
            for name, dim in si.dimensions.items():
                so.createDimension(name, None if dim.isunlimited() else
                                   (1 if name == "Time" else len(dim)))
            for name, var in si.variables.items():
                var.set_auto_maskandscale(False)
                attrs = {k: var.getncattr(k) for k in var.ncattrs()}
                fill = attrs.pop("_FillValue", None)
                filt = var.filters() or {}
                kw = {}
                if filt.get("zlib"):
                    kw = dict(zlib=True, complevel=filt.get("complevel", 4),
                              shuffle=filt.get("shuffle", False))
                out = so.createVariable(name, var.dtype, var.dimensions, fill_value=fill, **kw)
                out.set_auto_maskandscale(False)
                out.setncatts(attrs)
                if var.dimensions and var.dimensions[0] == "Time":
                    out[0:1] = var[timeidx:timeidx + 1]
                else:
                    out[...] = var[...]
    except Exception as e:                      # noqa: BLE001 -- netCDF4 raises RuntimeError
        if os.path.lexists(tmp):
            os.remove(tmp)
        raise OSError("cannot extract time index %d of %s: %s" % (timeidx, src, e)) from e
    os.replace(tmp, dst)
    return "extract"


def _multi_frame_hint(run_dir, out_root, domain, every, times, multi):
    size = sum(os.path.getsize(p) / float(_ntimes(p)) for _t, p, _i in multi)
    files = sorted(set(os.path.basename(p) for _t, p, _i in multi))
    sel = (" --times %s" % ",".join(times) if times else " --every %s" % every if every else "")
    return ("vapor bundle needs one time per file, but %d selected time(s) sit inside "
            "multi-frame wrfout files (frames_per_outfile > 1: %s%s); run `wtune post "
            "vapor-bundle %s --domain %d%s --out %s --allow-copy` to extract %d single-time "
            "files (~%s)"
            % (len(multi), ", ".join(files[:3]), ", ..." if len(files) > 3 else "",
               os.path.abspath(run_dir), domain, sel, os.path.abspath(out_root), len(multi),
               _human(size)))


def _allow_copy_hint(run_dir, out_root, domain, every, times, selected, err):
    size = sum(os.path.getsize(p) for _t, p, _i in selected)
    sel = (" --times %s" % ",".join(times) if times else " --every %s" % every if every else "")
    return ("vapor bundle not hard-linkable here (%s); run `wtune post vapor-bundle %s "
            "--domain %d%s --out %s --allow-copy` to copy %d files (~%s)"
            % (err.strerror or err, os.path.abspath(run_dir), domain, sel,
               os.path.abspath(out_root), len(selected), _human(size)))


def _place_frames(run_dir, out_root, bundle_dir, new_dirs, domain, every, times, selected,
                  allow_copy):
    """Link/copy/extract each selected frame into bundle_dir -> the manifest's frame records. On
    failure removes what was written and the directories this call created, then raises
    NotHardLinkable (or the OSError itself with allow_copy)."""
    from wtune.post.extract import bp_origin
    frames, written = [], []
    for t, src, timeidx in selected:
        name = bundle_name(domain, t)
        dst = os.path.join(bundle_dir, name)
        single = _ntimes(src) <= 1
        try:
            mode = (link_or_copy(src, dst, allow_copy=allow_copy) if single
                    else write_single_time(src, timeidx, dst))
        except OSError as e:
            for path in written:
                os.remove(path)
            for d in reversed(new_dirs):
                try:
                    os.rmdir(d)
                except OSError:
                    pass                        # not empty: it held something before us
            if allow_copy:
                raise
            raise NotHardLinkable(_allow_copy_hint(run_dir, out_root, domain, every, times,
                                                   selected, e)) from e
        if mode != "kept":
            written.append(dst)
        frames.append(dict(original=os.path.abspath(src), name=name, valid_time=t,
                           timeidx=timeidx, size_bytes=os.path.getsize(dst),
                           mode="hardlink" if mode == "kept" else mode,
                           native=bp_origin(src)))     # the BP dir of a converted copy, or None
    return frames


def _remove_stale_frames(bundle_dir: str, domain: int, frames: list) -> None:
    """Drop this domain's frame files of an earlier --every selection: left in place they would
    join the time series VAPOR builds from the wrfout_* file list."""
    wanted = {f["name"] for f in frames}
    prefix = "wrfout_d%02d_" % domain
    for name in os.listdir(bundle_dir):
        if name.startswith(prefix) and name not in wanted:
            os.remove(os.path.join(bundle_dir, name))


def _link_note_of(frames: list) -> tuple[str, int, int]:
    """(link_note, total bytes, copied bytes)."""
    total = sum(f["size_bytes"] for f in frames)
    copied = sum(f["size_bytes"] for f in frames if f["mode"] in ("copy", "extract"))
    note = ("hard-linked (no extra space)" if not copied else
            "copied (%s)" % _human(copied) if copied == total else
            "partly copied (%s copied, the rest hard-linked)" % _human(copied))
    if any(f["native"] for f in frames):
        note += ", from the netCDF cache of the run's ADIOS2 BP history"
    return note, total, copied


def build_bundle(run_dir: str, out_root: str, domain: int, every=None, times=None,
                 run_label: str | None = None, allow_copy: bool = False) -> dict:
    """Write <out_root>/vapor/d0N/ -> dict(dir, domain, n_files, total_bytes, copied_bytes,
    link_note, readme, script, manifest, frames). ValueError when the domain has no frames;
    NotHardLinkable when a frame cannot be hard-linked without allow_copy, after removing every
    frame and directory this call created (a later-frame refusal, e.g. EMLINK, can only have
    replaced earlier frames' links)."""
    from wtune.post.cf_pressure import select_frames

    selected = select_frames(run_dir, domain, every=every, times=times)
    if not selected:
        raise ValueError("no wrfout_d%02d_* frames in %s" % (domain, run_dir))
    multi = [f for f in selected if _ntimes(f[1]) > 1]
    if multi and not allow_copy:                # refused before anything is written
        raise NotHardLinkable(_multi_frame_hint(run_dir, out_root, domain, every, times, multi))
    bundle_dir = os.path.join(out_root, "vapor", "d%02d" % domain)
    new_dirs = [d for d in (out_root, os.path.dirname(bundle_dir), bundle_dir)
                if not os.path.isdir(d)]
    os.makedirs(bundle_dir, exist_ok=True)

    # frames first, before anything else in the bundle is touched, so a refusal can be undone
    frames = _place_frames(run_dir, out_root, bundle_dir, new_dirs, domain, every, times,
                           selected, allow_copy)
    _remove_stale_frames(bundle_dir, domain, frames)
    link_note, total, copied = _link_note_of(frames)

    shutil.copy2(SCRIPT_SRC, os.path.join(bundle_dir, SCRIPT_NAME))
    created = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    cadence = ("--times %s" % ",".join(times) if times else "--every %s" % every if every
               else "last frame only (no --every/--times)")
    run_label = run_label or os.path.basename(os.path.normpath(run_dir))
    readme = os.path.join(bundle_dir, README_NAME)
    _write_readme(readme, bundle_dir, run_dir, run_label, domain, frames, cadence, link_note,
                  created)

    version, commit = _wtune_provenance()
    manifest = os.path.join(bundle_dir, MANIFEST_NAME)
    with open(manifest, "w") as fh:
        json.dump(dict(run_dir=os.path.abspath(run_dir), run_label=run_label, domain=domain,
                       selection=cadence, frames=frames, total_bytes=total, copied_bytes=copied,
                       link_note=link_note, wtune_version=version, wtune_commit=commit,
                       created_utc=created), fh, indent=2)
    return dict(dir=bundle_dir, domain=domain, n_files=len(frames), total_bytes=total,
                copied_bytes=copied, link_note=link_note, readme=readme,
                script=os.path.join(bundle_dir, SCRIPT_NAME), manifest=manifest, frames=frames)


def export(run_dir: str, out_root: str, domains, every=None, times=None,
           run_label: str | None = None, allow_copy: bool = False) -> list[dict]:
    """build_bundle() per domain; a domain that fails (incl. NotHardLinkable) becomes
    dict(domain, error) instead of stopping the others."""
    results = []
    for domain in domains:
        try:
            results.append(build_bundle(run_dir, out_root, domain, every=every, times=times,
                                        run_label=run_label, allow_copy=allow_copy))
        except (ValueError, OSError) as e:
            results.append(dict(domain=domain, error="%s: %s" % (type(e).__name__, e)))
    return results
