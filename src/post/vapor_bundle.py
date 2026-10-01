"""post/vapor_bundle.py -- a ready-to-copy NCAR VAPOR bundle per domain (`wtune post vapor-bundle`,
and an always-on data product of `wtune post recipe`): <out>/vapor/d0N/ holding the selected
wrfout frames, vapor_render.py (post/vapor/vapor_render.py, copied verbatim), README.md and a
bundle.json manifest.

Why a bundle and not a render: VAPOR has no linux-aarch64 build, so it cannot run on scc. It
runs on the user's own x86_64 Linux / macOS workstation (official conda-forge + ncar-vapor
packages); the bundle is what that machine needs, gathered in one directory for one rsync.

Why the raw wrfout and not post/cf_pressure.py's NetCDF: VAPOR's own WRF-ARW reader
(dataset.WRF) understands the native staggered, terrain-following, map-projected grid itself --
a regridded file would only lose resolution. That reader wants the time series as a list of
files, one time per file, so each selected frame is one file here.

Frame files: hard-linked from the run (os.link: no extra disk on the same filesystem). A copy
happens ONLY with an explicit allow_copy (`wtune post vapor-bundle --allow-copy`): the recipe
path is always on, and silently duplicating every selected wrfout into a different filesystem
is how a quota fills up mid-write (on GPFS an exhausted quota truncated NetCDF files without
an ENOSPC anyone saw). Without allow_copy a failed link stops that domain, removes whatever
this call had written, and raises NotHardLinkable whose message is the exact --allow-copy
command plus the file count and size it would copy. Names drop WRF's colons
(wrfout_d01_2005-08-28_06_00_00) so the bundle survives macOS/Windows filesystems and tools;
VAPOR is ASSUMED to take the time from each file's Times variable, not from its name (stated in
the README, since nothing here could be run against VAPOR).
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


def bundle_name(domain, time_str):
    """wrfout_d0N_YYYY-MM-DD_HH_MM_SS -- WRF's own name with the colons replaced."""
    return "wrfout_d%02d_%s" % (int(domain), time_str.replace(":", "_"))


class NotHardLinkable(OSError):
    """os.link failed and copying was not allowed; str() is the user-facing message."""


def link_or_copy(src, dst, allow_copy=False):
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
        import wtune_version
        version = wtune_version.__version__
    except ImportError:
        version = None
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"],
                                cwd=os.path.dirname(os.path.abspath(__file__)),
                                capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                           # noqa: BLE001 -- no git, not a checkout, ...
        commit = None
    return version, commit


def _human(n):
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

hard link 的檔案在 rsync 時照常傳完整內容；本機不需要 scc 上的原始 run。hard link 與 run 的原始
wrfout 是同一份資料：之後若就地改寫原檔，bundle 內容也會跟著變。

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


def _write_readme(path, bundle_dir, run_dir, run_label, domain, frames, cadence, link_note,
                  created):
    rows = "".join("| `%s` | wrfout，valid time %s |\n" % (f["name"], f["valid_time"])
                   for f in frames)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(_README.format(run_label=run_label, domain=domain, created=created,
                                file_rows=rows, script=SCRIPT_NAME, manifest=MANIFEST_NAME,
                                run_dir=os.path.abspath(run_dir), n=len(frames),
                                cadence=cadence, link_note=link_note,
                                bundle_dir=os.path.abspath(bundle_dir)))


def _allow_copy_hint(run_dir, out_root, domain, every, times, selected, err):
    size = sum(os.path.getsize(p) for _t, p in selected)
    sel = (" --times %s" % ",".join(times) if times else " --every %s" % every if every else "")
    return ("vapor bundle not hard-linkable here (%s); run `wtune post vapor-bundle %s "
            "--domain %d%s --out %s --allow-copy` to copy %d files (~%s)"
            % (err.strerror or err, os.path.abspath(run_dir), domain, sel,
               os.path.abspath(out_root), len(selected), _human(size)))


def build_bundle(run_dir, out_root, domain, every=None, times=None, run_label=None,
                 allow_copy=False):
    """Writes <out_root>/vapor/d0N/. Returns dict(dir, domain, n_files, total_bytes,
    copied_bytes, link_note, readme, script, manifest, frames). Raises ValueError when the
    domain has no frames, NotHardLinkable when a frame cannot be hard-linked and allow_copy is
    False -- after removing every frame file and directory this call created. A refusal on the
    first frame (the cross-filesystem case) therefore leaves the disk exactly as it was; a
    refusal on a later frame (e.g. EMLINK) can only have replaced earlier frames' links."""
    from post.cf_pressure import select_frames

    selected = select_frames(run_dir, domain, every=every, times=times)
    if not selected:
        raise ValueError("no wrfout_d%02d_* frames in %s" % (domain, run_dir))
    bundle_dir = os.path.join(out_root, "vapor", "d%02d" % domain)
    new_dirs = [d for d in (out_root, os.path.dirname(bundle_dir), bundle_dir)
                if not os.path.isdir(d)]
    os.makedirs(bundle_dir, exist_ok=True)

    # frames first, before anything else in the bundle is touched, so a refusal can be undone
    frames, written = [], []
    for t, src in selected:
        name = bundle_name(domain, t)
        dst = os.path.join(bundle_dir, name)
        try:
            mode = link_or_copy(src, dst, allow_copy=allow_copy)
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
                           size_bytes=os.path.getsize(src),
                           mode="hardlink" if mode == "kept" else mode))

    # a frame left over from an earlier, different --every selection would silently join the
    # time series VAPOR builds from the wrfout_* file list -- only this domain's frames, which
    # this module itself put here, are removed
    wanted = {f["name"] for f in frames}
    prefix = "wrfout_d%02d_" % domain
    for name in os.listdir(bundle_dir):
        if name.startswith(prefix) and name not in wanted:
            os.remove(os.path.join(bundle_dir, name))
    total = sum(f["size_bytes"] for f in frames)
    copied = sum(f["size_bytes"] for f in frames if f["mode"] == "copy")
    link_note = ("hard-linked (no extra space)" if not copied else
                 "copied (%s)" % _human(copied) if copied == total else
                 "partly copied (%s copied, the rest hard-linked)" % _human(copied))

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


def export(run_dir, out_root, domains, every=None, times=None, run_label=None,
           allow_copy=False):
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
